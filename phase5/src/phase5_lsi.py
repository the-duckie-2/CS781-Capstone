"""CS781 Phase 5 - Legal Statute Identification (LSI) from Phase 1 legal language models.

Multi-label classification: every case -> a set of statute IDs.

Every experiment starts from a Phase 1 checkpoint (decoder / encoder_decoder / hybrid / efficient); no other
pretrained weights are used.  Training uses only the official Phase 5 train/dev data from the course HF repo.

Long-document representation (hierarchical):
  * the case is tokenised with the Phase 1 tokenizer and cut into overlapping windows of 256 tokens (the Phase 1
    context length, stride 256-overlap);
  * ``--selection spread`` keeps K windows spread evenly over the WHOLE document (always the first and the last),
    because statutes can be grounded anywhere (facts, arguments, reasoning, operative part); ``headtail`` keeps
    the first + last K-1 windows; ``head`` the first K;
  * the backbone encodes every window -> [masked mean ; masked max ; last token] -> projection, plus an embedding
    of the window's relative position in the document (8 buckets, a cheap section-awareness signal);
  * document pooling (``--pooling``):
      labelwise  attention pooling + mean pooling -> MLP logits, PLUS label-wise attention (every statute has its
                 own query over the windows, so different statutes can rely on different parts of the case);
      attention  attention + mean pooling only;
      max        max + mean pooling;
      chunkmax   per-window statute logits aggregated by max (chunk-level prediction -> document aggregation).
Loss: BCE with per-statute positive weights ((neg/pos)^p, clipped) | focal | asymmetric loss (ASL).
Output biases start at the log-prior of each statute (stable start for rare labels).
Thresholds: global vs per-statute (shrunk toward the global value for rare statutes), each with/without a
"at least one statute per case" rule; the strategy is chosen by 2-fold cross-fitting on dev (honest estimate),
then refit on the full dev set for the test predictions.  A temperature is fitted on dev for calibration.
Adaptation: full fine-tuning, LORMA+, LoRA, IA3, partial (top-layer) fine-tuning, BitFit and knowledge
distillation (truncated decoder student trained on a teacher's per-statute logits).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase2_lner import LNERModel, load_phase1, make_student
from phase3_rr import locate_phase1_files, seed_everything
from phase4_cjpe import apply_method, leakage_report

REPO_ID = "Exploration-Lab/CS-781-Capstone"
PHASE_DIR = "Phase_5_LSI"
POSITION_BUCKETS = 8
GRID = [round(0.02 + 0.01 * i, 2) for i in range(94)]  # 0.02 .. 0.95


# ----------------------------------------------------------------------------- data

def obtain_data(destination: Path, data_dir: str | None) -> dict[str, Path]:
    """train.json / dev.json / test.json from --data-dir or the official HF dataset (always re-downloaded)."""
    names = ("train.json", "dev.json", "test.json")
    if data_dir:
        found = {}
        for folder, _, files in os.walk(data_dir, followlinks=True):
            for name in files:
                if name in names and name not in found:
                    found[name] = Path(folder) / name
        if len(found) == 3:
            return found
        raise FileNotFoundError(f"--data-dir lacks {set(names) - set(found)}")
    from huggingface_hub import hf_hub_download, list_repo_files

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing: add the Kaggle secret HF and enable it")
    repo_files = [f for f in list_repo_files(REPO_ID, repo_type="dataset", token=token) if f.startswith(PHASE_DIR + "/")]
    print("Phase 5 files on HF:", repo_files, flush=True)
    found = {}
    for name in names:
        matches = [f for f in repo_files if f.lower().endswith("/" + name)]
        if not matches:
            raise FileNotFoundError(f"{name} not found under {PHASE_DIR} on HF: {repo_files}")
        found[name] = Path(hf_hub_download(repo_id=REPO_ID, filename=matches[0], repo_type="dataset", token=token,
                                           force_download=True))
    return found


def _pick(record: dict, keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


LABEL_KEYS = ("labels", "label", "statutes", "statute_ids", "statute", "sections", "pred")


def _to_int(value, where):
    if isinstance(value, bool):
        raise ValueError(f"{where}: boolean statute id {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError(f"{where}: statute id {value!r} is not an integer (the submission needs integer ids)")


def load_cases(path: Path, require_labels: bool) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for wrapper in ("data", "cases", "documents", "examples"):
            if isinstance(raw.get(wrapper), (list, dict)):
                raw = raw[wrapper]
                break
    if isinstance(raw, dict):
        raw = [{**(v if isinstance(v, dict) else {"text": v}), "id": v.get("id", k) if isinstance(v, dict) else k}
               for k, v in raw.items()]
    cases = []
    for position, item in enumerate(raw):
        case_id = _pick(item, ("id", "case_id", "uid", "name", "doc_id"))
        text = _pick(item, ("text", "case_text", "document", "raw_text", "content", "facts"))
        if case_id is None or text is None:
            raise ValueError(f"{path.name} record #{position}: cannot find id/text, keys={list(item)}")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        labels = None
        if require_labels:
            labels = _pick(item, LABEL_KEYS)
            if labels is None:
                raise ValueError(f"{path.name}/{case_id}: no label field among {LABEL_KEYS}; keys={list(item)}")
            if not isinstance(labels, (list, tuple)):
                labels = [labels]
            labels = [_to_int(v, f"{path.name}/{case_id}") for v in labels]
        cases.append({"id": case_id, "text": str(text), "labels": labels})
    if require_labels and cases:
        lengths = {len(c["labels"]) for c in cases}
        values = {v for c in cases for v in c["labels"]}
        if len(lengths) == 1 and next(iter(lengths)) > 20 and values <= {0, 1}:
            print(f"{path.name}: labels look MULTI-HOT (fixed length {lengths}); converting to index lists", flush=True)
            for c in cases:
                c["labels"] = [i for i, v in enumerate(c["labels"]) if v == 1]
        for c in cases:
            c["labels"] = sorted(set(c["labels"]))
    print(f"{path.name}: {len(cases)} cases; keys of first record: {list(raw[0]) if raw else []}", flush=True)
    return cases


def describe(cases, name):
    counts = Counter(l for c in cases for l in c["labels"])
    card = [len(c["labels"]) for c in cases]
    words = sorted(len(c["text"].split()) for c in cases)
    info = {"cases": len(cases), "distinct_statutes": len(counts), "label_instances": sum(card),
            "mean_labels_per_case": round(sum(card) / max(1, len(card)), 3), "max_labels_per_case": max(card or [0]),
            "cases_without_labels": sum(x == 0 for x in card),
            "cardinality_hist": dict(sorted(Counter(min(x, 6) for x in card).items())),
            "words_p50_p90_max": [words[len(words) // 2], words[int(len(words) * 0.9)], words[-1]] if words else None,
            "statutes_with_<=5_cases": sum(v <= 5 for v in counts.values()),
            "top10": counts.most_common(10)}
    print(f"[{name}]", json.dumps(info), flush=True)
    return info


# ----------------------------------------------------------------------------- long-document encoding

def select_windows(ids, L, K, overlap, selection):
    n = len(ids)
    if n <= L:
        return [ids], [0]
    stride = max(1, L - overlap)
    starts = list(range(0, n - L + 1, stride))
    if starts[-1] != n - L:
        starts.append(n - L)
    if len(starts) > K:
        if selection == "spread":
            picks = sorted({round(i * (len(starts) - 1) / (K - 1)) for i in range(K)}) if K > 1 else [0]
            starts = [starts[i] for i in picks]
        elif selection == "headtail":
            starts = [starts[0]] + starts[-(K - 1):] if K > 1 else [starts[0]]
        elif selection == "head":
            starts = starts[:K]
        else:
            raise ValueError(selection)
    buckets = [min(POSITION_BUCKETS - 1, int(POSITION_BUCKETS * s / n)) for s in starts]
    return [ids[s:s + L] for s in starts], buckets


def encode_cases(cases, tokenizer, args):
    """-> list of (list[np.int32 window], list[position bucket]); tokenised in batches to bound memory."""
    out = []
    for s in range(0, len(cases), 512):
        texts = [" ".join(c["text"].split()[:args.max_words]) for c in cases[s:s + 512]]
        for enc in tokenizer.encode_batch(texts):
            ids = enc.ids or [args.eos_id]
            windows, buckets = select_windows(ids, args.chunk_len, args.max_chunks, args.overlap, args.selection)
            out.append(([np.asarray(w, dtype=np.int32) for w in windows], buckets))
    return out


def make_batch(items, pad_id, device):
    flat = [w for windows, _ in items for w in windows]
    length = max(len(w) for w in flat)
    ids = torch.full((len(flat), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(flat), length), dtype=torch.bool)
    for r, w in enumerate(flat):
        ids[r, :len(w)] = torch.from_numpy(w.astype(np.int64))
        mask[r, :len(w)] = True
    K = max(len(windows) for windows, _ in items)
    valid = torch.zeros((len(items), K), dtype=torch.bool)
    positions = torch.zeros((len(items), K), dtype=torch.long)
    for i, (windows, buckets) in enumerate(items):
        valid[i, :len(windows)] = True
        positions[i, :len(buckets)] = torch.tensor(buckets)
    return ids.to(device), mask.to(device), positions.to(device), valid.to(device)


# ----------------------------------------------------------------------------- model

class LSIModel(nn.Module):
    def __init__(self, backbone, model_name, num_labels, pooling="labelwise", dropout=0.1):
        super().__init__()
        self.encoder = LNERModel(backbone, model_name, 1)
        del self.encoder.classifier
        d = backbone.cfg.d_model
        self.pooling = pooling
        self.chunk_proj = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU())
        self.position = nn.Embedding(POSITION_BUCKETS, d)
        self.attn = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.doc_head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(dropout), nn.Linear(2 * d, d), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(d, num_labels))
        if pooling == "labelwise":
            self.lw_norm = nn.LayerNorm(d)
            self.label_query = nn.Parameter(torch.randn(num_labels, d) * 0.02)
            self.label_out = nn.Parameter(torch.zeros(num_labels, d))
        if pooling == "chunkmax":
            self.chunk_head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, num_labels))

    def output_biases(self):
        biases = [self.doc_head[-1].bias]
        if self.pooling == "chunkmax":
            biases = [self.chunk_head[-1].bias]
        return biases

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def forward(self, ids, mask, positions, valid):
        hidden = self.encoder.hidden(ids, mask).float()
        w = mask.unsqueeze(-1).float()
        mean = (hidden * w).sum(1) / w.sum(1).clamp_min(1)
        mx = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        last = hidden[torch.arange(hidden.size(0), device=hidden.device), mask.sum(1) - 1]
        vectors = self.chunk_proj(torch.cat((mean, mx, last), -1))
        B, K = valid.shape
        chunks = torch.zeros(B, K, vectors.size(-1), device=vectors.device, dtype=vectors.dtype)
        chunks[valid] = vectors
        chunks = (chunks + self.position(positions)) * valid.unsqueeze(-1)
        scores = self.attn(chunks).squeeze(-1).masked_fill(~valid, -1e4)
        alpha = scores.softmax(-1)
        mean_chunks = chunks.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        if self.pooling == "chunkmax":
            per_chunk = self.chunk_head(chunks).masked_fill(~valid.unsqueeze(-1), -1e4)
            return per_chunk.max(1).values, alpha
        if self.pooling == "max":
            pooled = chunks.masked_fill(~valid.unsqueeze(-1), -1e4).max(1).values
        else:
            pooled = (alpha.unsqueeze(-1) * chunks).sum(1)
        logits = self.doc_head(torch.cat((pooled, mean_chunks), -1))
        if self.pooling == "labelwise":
            h = self.lw_norm(chunks)
            s = torch.einsum("bkd,ld->blk", h, self.label_query) / math.sqrt(h.size(-1))
            a = s.masked_fill(~valid.unsqueeze(1), -1e4).softmax(-1)
            ctx = torch.einsum("blk,bkd->bld", a, h)
            logits = logits + (ctx * self.label_out).sum(-1)
        return logits, alpha


# ----------------------------------------------------------------------------- losses

def task_loss(logits, y, args, pos_weight):
    """Summed over statutes, averaged over cases."""
    if args.loss == "bce":
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none").sum(-1).mean()
    if args.loss == "focal":
        ce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        return (ce * (1 - pt).pow(args.focal_gamma)).sum(-1).mean()
    if args.loss == "asl":  # Ridnik et al. 2021, asymmetric loss with probability shifting for negatives
        p = torch.sigmoid(logits)
        p_neg = (p - args.asl_clip).clamp_min(0) if args.asl_clip > 0 else p
        eps = 1e-8
        pos = y * torch.log(p.clamp_min(eps)) * (1 - p).pow(args.asl_gamma_pos)
        neg = (1 - y) * torch.log((1 - p_neg).clamp_min(eps)) * p_neg.pow(args.asl_gamma_neg)
        if pos_weight is not None:
            pos = pos * pos_weight
        return -(pos + neg).sum(-1).mean()
    raise ValueError(args.loss)


def kd_loss(student, teacher, T):
    """Per-statute binary KD: BCE between tempered student logits and tempered teacher probabilities."""
    return F.binary_cross_entropy_with_logits(student / T, torch.sigmoid(teacher / T), reduction="none").sum(-1).mean() * T * T


# ----------------------------------------------------------------------------- metrics / thresholds

def ml_scores(gold: torch.Tensor, pred: torch.Tensor) -> dict:
    """gold/pred: (N, L) bool.  Macro over statutes with gold support (the evaluator's label dimension comes from
    the reference data); macro_union also counts statutes that were only predicted (as F1 = 0)."""
    tp = (gold & pred).sum(0).float()
    fp = (~gold & pred).sum(0).float()
    fn = (gold & ~pred).sum(0).float()
    prec = torch.where(tp + fp > 0, tp / (tp + fp).clamp_min(1), torch.zeros_like(tp))
    rec = torch.where(tp + fn > 0, tp / (tp + fn).clamp_min(1), torch.zeros_like(tp))
    f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp_min(1), torch.zeros_like(tp))
    support = gold.sum(0) > 0
    union = support | (pred.sum(0) > 0)
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    micro_p = float(TP / (TP + FP)) if TP + FP > 0 else 0.0
    micro_r = float(TP / (TP + FN)) if TP + FN > 0 else 0.0
    return {
        "macro_f1": float(f1[support].mean()) if support.any() else 0.0,
        "macro_precision": float(prec[support].mean()) if support.any() else 0.0,
        "macro_recall": float(rec[support].mean()) if support.any() else 0.0,
        "macro_f1_union": float(f1[union].mean()) if union.any() else 0.0,
        "micro_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0,
        "micro_precision": micro_p, "micro_recall": micro_r,
        "subset_accuracy": float((gold == pred).all(1).float().mean()),
        "mean_pred_labels": float(pred.sum(1).float().mean()), "mean_gold_labels": float(gold.sum(1).float().mean()),
        "empty_predictions": int((pred.sum(1) == 0).sum()), "labels_evaluated": int(support.sum()),
        "_per_label": (prec, rec, f1),
    }


def public(scores):
    return {k: v for k, v in scores.items() if not k.startswith("_")}


def apply_thresholds(probs, thr, min_one):
    pred = probs >= thr
    if min_one:
        empty = pred.sum(1) == 0
        if empty.any():
            best = (probs[empty] / thr.clamp_min(1e-6)).argmax(1)
            rows = empty.nonzero().squeeze(1)
            pred[rows, best] = True
    return pred


def tune_global(probs, gold, min_one):
    best_t, best_f = 0.5, -1.0
    L = probs.size(1)
    for t in GRID:
        f = ml_scores(gold, apply_thresholds(probs, torch.full((L,), t), min_one))["macro_f1"]
        if f > best_f + 1e-9:
            best_t, best_f = t, f
    return best_t, best_f


def tune_per_label(probs, gold, t_global, min_support, shrink):
    """Per-statute threshold maximising that statute's F1, shrunk toward the global threshold by support."""
    L = probs.size(1)
    best_f = torch.full((L,), -1.0)
    best_t = torch.full((L,), float(t_global))
    for t in GRID:
        pred = probs >= t
        tp = (gold & pred).sum(0).float()
        fp = (~gold & pred).sum(0).float()
        fn = (gold & ~pred).sum(0).float()
        f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        better = f1 > best_f + 1e-9
        best_f = torch.where(better, f1, best_f)
        best_t = torch.where(better, torch.full_like(best_t, t), best_t)
    support = gold.sum(0).float()
    shrunk = (support * best_t + shrink * t_global) / (support + shrink)
    return torch.where(support >= min_support, shrunk, torch.full_like(shrunk, float(t_global)))


STRATEGIES = ("global", "global+min1", "per_label", "per_label+min1")


def fit_strategy(name, probs, gold, args):
    min_one = name.endswith("+min1")
    t_g, _ = tune_global(probs, gold, min_one)
    L = probs.size(1)
    if name.startswith("global"):
        return torch.full((L,), t_g), min_one, t_g
    return tune_per_label(probs, gold, t_g, args.min_support, args.threshold_shrink), min_one, t_g


def choose_thresholds(probs, gold, args):
    """2-fold cross-fitting on dev to pick the strategy honestly, then refit the chosen one on all of dev."""
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(probs.size(0), generator=g)
    halves = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
    report = {}
    for name in STRATEGIES:
        folds = []
        for a, b in ((0, 1), (1, 0)):
            thr, min_one, _ = fit_strategy(name, probs[halves[a]], gold[halves[a]], args)
            folds.append(public(ml_scores(gold[halves[b]], apply_thresholds(probs[halves[b]], thr, min_one))))
        report[name] = {k: sum(f[k] for f in folds) / 2 for k in ("macro_f1", "micro_f1", "subset_accuracy",
                                                                  "macro_precision", "macro_recall")}
    chosen = max(STRATEGIES, key=lambda s: report[s]["macro_f1"])
    thr, min_one, t_g = fit_strategy(chosen, probs, gold, args)
    print("threshold strategies (2-fold cross-fitted dev):", json.dumps(report), "-> chosen", chosen, flush=True)
    return chosen, thr, min_one, t_g, report


def fit_temperature(logits, gold):
    best_T, best_loss = 1.0, float("inf")
    y = gold.float()
    for T in [0.5 + 0.05 * i for i in range(51)]:
        loss = float(F.binary_cross_entropy_with_logits(logits / T, y))
        if loss < best_loss:
            best_T, best_loss = T, loss
    return best_T, best_loss


def calibration(probs, gold, bins=10):
    y = gold.float().flatten()
    p = probs.flatten()
    ece = 0.0
    for b in range(bins):
        sel = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1)
        if sel.any():
            ece += float(sel.float().mean()) * abs(float(p[sel].mean()) - float(y[sel].mean()))
    pos = gold.any(0)
    return {"ece_all_pairs": ece, "brier": float(((p - y) ** 2).mean()),
            "mean_prob_on_gold": float(probs[gold].mean()) if gold.any() else None,
            "labels_with_gold": int(pos.sum())}


@torch.no_grad()
def predict(model, items, pad_id, device, batch_docs):
    model.eval()
    order = sorted(range(len(items)), key=lambda i: -sum(len(w) for w in items[i][0]))
    result = [None] * len(items)
    for s in range(0, len(order), batch_docs):
        rows = order[s:s + batch_docs]
        ids, mask, positions, valid = make_batch([items[r] for r in rows], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out, alpha = model(ids, mask, positions, valid)
        out, alpha = out.float().cpu(), alpha.float().cpu()
        for j, r in enumerate(rows):
            result[r] = (out[j], alpha[j, :len(items[r][0])])
    return torch.stack([r[0] for r in result]), [r[1] for r in result]


# ----------------------------------------------------------------------------- training

def train(model, train_items, Y, dev_items, Ydev, args, device, pad_id, pos_weight, soft=None, has_soft=None):
    head_ids = {id(p) for p in model.head_parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": args.head_lr}] + ([{"params": body, "lr": args.learning_rate}] if body else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    budget = args.time_budget_min * 60
    est_steps = max(50, args.epochs * math.ceil(len(train_items) / (args.batch_docs * args.grad_accum)))
    warmup = 30
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, max(0.05, 1 - s / est_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    trainable = [p for p in model.parameters() if p.requires_grad]
    rng = random.Random(args.seed)
    eval_rows = sorted(rng.sample(range(len(dev_items)), min(args.eval_docs, len(dev_items))))
    eval_items, eval_gold = [dev_items[i] for i in eval_rows], Ydev[eval_rows]
    start, last_eval = time.time(), time.time()
    best, best_state, history = {"macro_f1": -1}, None, []
    step, opt_steps, seen = 0, 0, 0
    pw = pos_weight.to(device) if pos_weight is not None else None

    def evaluate(tag):
        nonlocal best, best_state, last_eval
        logits, _ = predict(model, eval_items, pad_id, device, args.batch_docs * 2)
        probs = torch.sigmoid(logits)
        t, _ = tune_global(probs, eval_gold, True)
        scores = public(ml_scores(eval_gold, apply_thresholds(probs, torch.full((probs.size(1),), t), True)))
        row = {"tag": tag, "docs_seen": seen, "opt_steps": opt_steps, "minutes": round((time.time() - start) / 60, 2),
               "eval_global_threshold": t, **{k: round(scores[k], 4) for k in
                                               ("macro_f1", "micro_f1", "subset_accuracy", "mean_pred_labels")}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if scores["macro_f1"] > best["macro_f1"]:
            best = {**scores, "tag": tag}
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        model.train()
        last_eval = time.time()

    done = False
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_items)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for s in range(0, len(order), args.batch_docs):
            rows = order[s:s + args.batch_docs]
            ids, mask, positions, valid = make_batch([train_items[r] for r in rows], pad_id, device)
            y = Y[rows].float().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(ids, mask, positions, valid)
            logits = logits.float()
            loss = task_loss(logits, y, args, pw)
            if soft is not None:
                sel = has_soft[rows].to(device)
                if sel.any():
                    kd = kd_loss(logits[sel], soft[rows].to(device)[sel], args.temperature)
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
            scaler.scale(loss / args.grad_accum).backward()
            step += 1
            seen += len(rows)
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                sched.step()
                opt_steps += 1
                if opt_steps == 40:
                    rate = opt_steps / (time.time() - start)
                    est_steps = max(60, min(est_steps, int(rate * budget * 0.92)))
                    print(f"~{rate:.2f} opt steps/s -> schedule over {est_steps} steps "
                          f"(~{est_steps * args.batch_docs * args.grad_accum} docs, "
                          f"{est_steps * args.batch_docs * args.grad_accum / max(1, len(train_items)):.2f} epochs)",
                          flush=True)
            losses.append(float(loss))
            if step % 50 == 0:
                print(f"epoch {epoch} docs {seen} loss {sum(losses[-50:]) / len(losses[-50:]):.4f} "
                      f"elapsed {(time.time() - start) / 60:.1f} min", flush=True)
            if time.time() - start > budget:
                print("Time budget reached.", flush=True)
                done = True
                break
            if time.time() - last_eval > args.eval_every_min * 60:
                evaluate(f"epoch{epoch}")
        if done:
            break
        evaluate(f"end_epoch{epoch}")
    evaluate("final")
    model.load_state_dict(best_state)
    return best, history, time.time() - start


# ----------------------------------------------------------------------------- analysis

def analysis(label_space, train_counts, dev_cases, Ydev, pred, probs, thr, alphas, dev_items, Ytrain):
    prec, rec, f1 = ml_scores(Ydev, pred)["_per_label"]
    support = Ydev.sum(0)
    per_label = []
    for j, lab in enumerate(label_space):
        per_label.append({"statute": lab, "train_cases": int(train_counts[j]), "dev_cases": int(support[j]),
                          "dev_predicted": int(pred[:, j].sum()), "precision": round(float(prec[j]), 4),
                          "recall": round(float(rec[j]), 4), "f1": round(float(f1[j]), 4),
                          "threshold": round(float(thr[j]), 3)})
    evaluated = [r for r in per_label if r["dev_cases"] > 0]
    # frequency buckets by train frequency (tertiles over evaluated statutes)
    ranked = sorted(evaluated, key=lambda r: -r["train_cases"])
    n = len(ranked)
    buckets = {}
    for name, part in (("head", ranked[:n // 3]), ("mid", ranked[n // 3: 2 * n // 3]), ("tail", ranked[2 * n // 3:])):
        if part:
            buckets[name] = {"statutes": len(part), "train_cases_range": [part[-1]["train_cases"], part[0]["train_cases"]],
                             "macro_f1": round(sum(r["f1"] for r in part) / len(part), 4)}
    zero_f1 = [r["statute"] for r in evaluated if r["f1"] == 0]
    # cardinality
    by_card = {}
    card = Ydev.sum(1)
    for c in sorted(set(min(int(x), 4) for x in card)):
        rows = (card.clamp(max=4) == c)
        s = public(ml_scores(Ydev[rows], pred[rows]))
        by_card[f"{c}{'+' if c == 4 else ''}"] = {"cases": int(rows.sum()), "micro_f1": round(s["micro_f1"], 4),
                                                  "subset_accuracy": round(s["subset_accuracy"], 4),
                                                  "mean_pred_labels": round(s["mean_pred_labels"], 3)}
    # co-occurrence (train) and whether the model reproduces the pairs on dev
    co = (Ytrain.float().T @ Ytrain.float())
    co.fill_diagonal_(0)
    pairs = []
    N = Ytrain.size(0)
    flat = torch.triu(co, 1)
    top = torch.topk(flat.flatten(), min(15, flat.numel()))
    for value, index in zip(top.values.tolist(), top.indices.tolist()):
        if value <= 0:
            break
        i, j = divmod(index, co.size(0))
        both_gold = Ydev[:, i] & Ydev[:, j]
        pmi = math.log(max(value, 1e-9) * N / max(1, train_counts[i] * train_counts[j]))
        pairs.append({"pair": [label_space[i], label_space[j]], "train_cooccur": int(value), "pmi": round(pmi, 3),
                      "dev_cases_with_both": int(both_gold.sum()),
                      "both_predicted_when_both_gold": round(float((pred[both_gold][:, [i, j]].all(1)).float().mean()), 3)
                      if both_gold.any() else None})
    # confusions: in the same case, gold statute missed while another (non-gold) statute was predicted
    confusion = Counter()
    for r in range(Ydev.size(0)):
        missed = (Ydev[r] & ~pred[r]).nonzero().flatten().tolist()
        extra = (~Ydev[r] & pred[r]).nonzero().flatten().tolist()
        for a, b in itertools.product(missed, extra):
            confusion[(label_space[a], label_space[b])] += 1
    most_missed = sorted(per_label, key=lambda r: -(r["dev_cases"] - round(r["recall"] * r["dev_cases"])))[:15]
    # long-document: F1 by length + mean attention by relative position
    words = torch.tensor([len(c["text"].split()) for c in dev_cases])
    length_buckets = {}
    qs = torch.quantile(words.float(), torch.tensor([0.33, 0.66])).tolist() if len(words) > 3 else [0, 0]
    for name, sel in (("short", words <= qs[0]), ("medium", (words > qs[0]) & (words <= qs[1])), ("long", words > qs[1])):
        if sel.any():
            s = public(ml_scores(Ydev[sel], pred[sel]))
            length_buckets[name] = {"cases": int(sel.sum()), "macro_f1": round(s["macro_f1"], 4),
                                    "micro_f1": round(s["micro_f1"], 4)}
    att = torch.zeros(POSITION_BUCKETS)
    cnt = torch.zeros(POSITION_BUCKETS)
    for (windows, positions), al in zip(dev_items, alphas):
        for p, a in zip(positions, al.tolist()):
            att[p] += a
            cnt[p] += 1
    windows_hist = dict(sorted(Counter(len(w) for w, _ in dev_items).items()))
    return {"per_label": per_label, "frequency_buckets": buckets, "statutes_with_zero_dev_f1": zero_f1,
            "by_gold_cardinality": by_card, "top_train_cooccurrences": pairs,
            "top_confusions_missed_vs_predicted": [{"missed": a, "predicted_instead": b, "cases": c}
                                                   for (a, b), c in confusion.most_common(20)],
            "most_missed": [{k: r[k] for k in ("statute", "train_cases", "dev_cases", "recall", "f1")} for r in most_missed],
            "f1_by_document_length": length_buckets, "length_cut_words": qs,
            "mean_attention_by_relative_position": (att / cnt.clamp_min(1)).tolist(),
            "windows_per_case_hist": windows_hist}


# ----------------------------------------------------------------------------- main

def multi_hot(cases, index, L):
    Y = torch.zeros((len(cases), L), dtype=torch.bool)
    for r, c in enumerate(cases):
        for lab in c["labels"]:
            if lab in index:
                Y[r, index[lab]] = True
    return Y


def run(args):
    t0 = time.time()
    seed_everything(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"p5_{args.model}_{args.method}"
    out = Path(args.output_dir or f"/kaggle/working/{name}")
    out.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    files = obtain_data(out / "data", args.data_dir)
    train_cases = load_cases(files["train.json"], True)
    dev_cases = load_cases(files["dev.json"], True)
    test_cases = load_cases(files["test.json"], False)
    if len({c["id"] for c in test_cases}) != len(test_cases):
        raise RuntimeError("Duplicate ids in test.json")
    leak = leakage_report(train_cases, dev_cases, test_cases)
    if args.limit_train:
        train_cases = train_cases[:args.limit_train]
    stats = {"train": describe(train_cases, "train"), "dev": describe(dev_cases, "dev")}
    words = sorted(len(c["text"].split()) for c in test_cases)
    stats["test"] = {"cases": len(test_cases), "words_p50_p90_max": [words[len(words) // 2], words[int(len(words) * .9)], words[-1]]}
    print("[test]", json.dumps(stats["test"]), flush=True)

    # label space = every statute id seen in train or dev (the only valid ids we can know)
    label_space = sorted({l for c in train_cases + dev_cases for l in c["labels"]})
    index = {l: j for j, l in enumerate(label_space)}
    L = len(label_space)
    dev_only = sorted({l for c in dev_cases for l in c["labels"]} - {l for c in train_cases for l in c["labels"]})
    print(f"label space: {L} statutes; {len(dev_only)} appear only in dev: {dev_only[:20]}", flush=True)
    Y = multi_hot(train_cases, index, L)
    Ydev = multi_hot(dev_cases, index, L)
    train_counts = Y.sum(0)

    checkpoint, tokenizer_path, artifact = locate_phase1_files(args.model, args.artifact_zip, out / "phase1")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id, args.eos_id = tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<eos>")
    if pad_id is None or args.eos_id is None:
        raise ValueError("Phase 1 tokenizer must contain <pad> and <eos>")

    soft = has_soft = None
    if args.method == "distill":
        saved = torch.load(args.teacher_soft_labels, map_location="cpu")
        if list(saved["labels"]) != label_space:
            raise RuntimeError("Teacher label space differs from the student's - teacher must use the same data")
        tindex = {cid: r for r, cid in enumerate(saved["ids"])}
        soft = torch.zeros((len(train_cases), L))
        has_soft = torch.zeros(len(train_cases), dtype=torch.bool)
        for r, c in enumerate(train_cases):
            if c["id"] in tindex:
                soft[r] = saved["logits"][tindex[c["id"]]]
                has_soft[r] = True
        print(f"KD: teacher {saved['teacher']} logits for {int(has_soft.sum())}/{len(train_cases)} train cases", flush=True)
        if args.kd_only_covered:
            keep = has_soft.nonzero().flatten().tolist()
            train_cases = [train_cases[i] for i in keep]
            Y, soft, has_soft = Y[keep], soft[keep], has_soft[keep]

    enc_t = time.time()
    train_items = encode_cases(train_cases, tokenizer, args)
    dev_items = encode_cases(dev_cases, tokenizer, args)
    test_items = encode_cases(test_cases, tokenizer, args)
    print(f"tokenised in {time.time() - enc_t:.0f}s; windows/case histogram (train) "
          f"{dict(sorted(Counter(len(w) for w, _ in train_items).items()))}", flush=True)

    backbone, phase1_cfg, phase1_saved = load_phase1(checkpoint, args.model)
    phase1_params = sum(p.numel() for p in backbone.parameters())
    if args.method == "distill":
        if args.model != "decoder":
            raise ValueError("The distillation student recipe uses the decoder Phase 1 model")
        backbone = make_student(backbone, layers=args.student_layers)
    model = LSIModel(backbone, args.model, L, args.pooling)
    prior = ((train_counts.float() + 0.5) / (len(train_cases) + 1.0)).clamp(1e-4, 0.5)
    with torch.no_grad():
        for b in model.output_biases():
            b.copy_(torch.log(prior / (1 - prior)))
    apply_method(model, args.model, args.method, args.rank, args.unfreeze_layers)
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    neg = len(train_cases) - train_counts.float()
    pos_weight = None
    if args.pos_weight_power > 0:
        pos_weight = (neg / train_counts.float().clamp_min(1)).pow(args.pos_weight_power).clamp(1, args.max_pos_weight)
    summary = {"run": name, "model": args.model, "method": args.method, "phase1_artifact": artifact,
               "phase1_parameters": phase1_params, "parameters": total, "trainable_parameters": trainable,
               "trainable_fraction": trainable / total, "num_statutes": L, "leakage_check": leak,
               "data_stats": stats, "device": str(device), "config": vars(args)}
    print(json.dumps({k: v for k, v in summary.items() if k != "data_stats"}, indent=2, default=str), flush=True)

    best, history, seconds = train(model, train_items, Y, dev_items, Ydev, args, device, pad_id, pos_weight, soft, has_soft)
    (out / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if args.export_soft_labels:
        ex_t = time.time()
        order = list(range(len(train_cases)))
        random.Random(args.seed).shuffle(order)
        ids, logits = [], []
        for s in range(0, len(order), 512):
            rows = order[s:s + 512]
            lg, _ = predict(model, [train_items[r] for r in rows], pad_id, device, args.batch_docs * 2)
            ids += [train_cases[r]["id"] for r in rows]
            logits.append(lg)
            if time.time() - ex_t > args.export_budget_min * 60:
                break
        torch.save({"ids": ids, "logits": torch.cat(logits), "labels": label_space, "teacher": name}, out / "teacher_soft_labels.pt")
        print(f"Saved teacher logits for {len(ids)}/{len(train_cases)} train cases in {(time.time() - ex_t) / 60:.1f} min",
              flush=True)


    inf_t = time.time()
    dev_logits, dev_alpha = predict(model, dev_items, pad_id, device, args.batch_docs * 2)
    dev_seconds = time.time() - inf_t
    T, cal_loss = fit_temperature(dev_logits, Ydev)
    raw_probs = torch.sigmoid(dev_logits)
    dev_probs = torch.sigmoid(dev_logits / T)
    calib = {"temperature": T, "dev_bce_raw": float(F.binary_cross_entropy_with_logits(dev_logits, Ydev.float())),
             "dev_bce_calibrated": cal_loss, "raw": calibration(raw_probs, Ydev), "calibrated": calibration(dev_probs, Ydev)}
    print("calibration:", json.dumps(calib), flush=True)
    fixed05 = public(ml_scores(Ydev, dev_probs >= 0.5))
    strategy, thr, min_one, t_global, strategy_report = choose_thresholds(dev_probs, Ydev, args)
    dev_pred = apply_thresholds(dev_probs, thr, min_one)
    tuned = ml_scores(Ydev, dev_pred)
    print(f"dev macro-F1 @0.5 {fixed05['macro_f1']:.4f} | {strategy} cross-fitted "
          f"{strategy_report[strategy]['macro_f1']:.4f} | in-sample {tuned['macro_f1']:.4f} "
          f"(micro {tuned['micro_f1']:.4f}, subset acc {tuned['subset_accuracy']:.4f})", flush=True)

    test_t = time.time()
    test_logits, _ = predict(model, test_items, pad_id, device, args.batch_docs * 2)
    test_seconds = time.time() - test_t
    test_pred = apply_thresholds(torch.sigmoid(test_logits / T), thr, min_one)
    torch.save({"labels": label_space, "temperature": T, "dev_ids": [c["id"] for c in dev_cases],
                "dev_logits": dev_logits, "test_ids": [c["id"] for c in test_cases], "test_logits": test_logits,
                "run": name}, out / "logits.pt")  # for ensembling
    valid_ids = set(label_space)
    predictions = []
    for c, row in zip(test_cases, test_pred):
        ids = sorted({int(label_space[j]) for j in row.nonzero().flatten().tolist()})
        predictions.append({"id": c["id"], "pred": ids})
    assert len(predictions) == len(test_cases) == len({p["id"] for p in predictions})
    assert all(isinstance(x, int) and x in valid_ids for p in predictions for x in p["pred"])
    assert all(len(p["pred"]) == len(set(p["pred"])) for p in predictions)
    path = out / "prediction.json"
    path.write_text(json.dumps(predictions), encoding="utf-8")
    archive = out / f"submission_{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(path, arcname="prediction.json")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["prediction.json"]
        assert len(json.loads(bundle.read("prediction.json"))) == len(test_cases)

    try:  # analysis must never cost the submission
        details = analysis(label_space, train_counts.tolist(), dev_cases, Ydev, dev_pred, dev_probs, thr, dev_alpha,
                           dev_items, Y)
        (out / "dev_predictions.json").write_text(json.dumps([
            {"id": c["id"], "gold": c["labels"], "pred": [label_space[j] for j in dev_pred[r].nonzero().flatten().tolist()],
             "top5": [[label_space[j], round(float(dev_probs[r, j]), 4)] for j in dev_probs[r].topk(min(5, L)).indices.tolist()],
             "words": len(c["text"].split()), "window_attention": [round(a, 3) for a in dev_alpha[r].tolist()]}
            for r, c in enumerate(dev_cases)]), encoding="utf-8")
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print('ANALYSIS FAILED (submission ZIP is fine):', error, flush=True)
        details = {'frequency_buckets': None, 'analysis_error': repr(error)}

    if args.save_checkpoint:
        torch.save({"model": {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                    "phase1_config": asdict(phase1_cfg), "run_config": vars(args), "label_space": label_space,
                    "thresholds": thr, "min_one": min_one, "temperature": T}, out / "best_fp16.pt")
    (out / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "validation_metrics.json").write_text(json.dumps({
        "selected_checkpoint_eval_subset": public(best) | {"tag": best.get("tag")},
        "dev_at_0.5": fixed05, "threshold_strategies_crossfit": strategy_report, "chosen_strategy": strategy,
        "global_threshold": t_global, "dev_tuned_in_sample": public(tuned), "calibration": calib,
        **details}, indent=2, default=str), encoding="utf-8")
    manifest = {**summary, "chosen_threshold_strategy": strategy, "global_threshold": t_global,
                "dev_macro_f1_at_0.5": fixed05["macro_f1"],
                "dev_macro_f1_crossfit": strategy_report[strategy]["macro_f1"],
                "dev_micro_f1_crossfit": strategy_report[strategy]["micro_f1"],
                "dev_subset_acc_crossfit": strategy_report[strategy]["subset_accuracy"],
                "dev_macro_f1_in_sample": tuned["macro_f1"], "dev_micro_f1_in_sample": tuned["micro_f1"],
                "dev_subset_acc_in_sample": tuned["subset_accuracy"], "frequency_buckets": details["frequency_buckets"],
                "temperature": T, "test_mean_pred_labels": float(test_pred.sum(1).float().mean()),
                "test_empty_predictions": int((test_pred.sum(1) == 0).sum()),
                "train_minutes": seconds / 60, "dev_inference_seconds": dev_seconds, "test_inference_seconds": test_seconds,
                "test_cases_per_second": len(test_cases) / max(1e-6, test_seconds), "total_minutes": (time.time() - t0) / 60,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
                "phase1_steps": phase1_saved.get("total_steps"), "submission": str(archive)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("run", "num_statutes", "chosen_threshold_strategy", "dev_macro_f1_crossfit",
                                                "dev_micro_f1_crossfit", "dev_subset_acc_crossfit", "trainable_parameters",
                                                "test_mean_pred_labels", "total_minutes")}, indent=2))
    print("SUBMISSION ZIP:", archive, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("decoder", "encoder_decoder", "hybrid", "efficient"))
    p.add_argument("--method", required=True, choices=("full", "lorma", "lora", "ia3", "partial", "bitfit", "distill"))
    p.add_argument("--selection", default="spread", choices=("spread", "headtail", "head"))
    p.add_argument("--pooling", default="labelwise", choices=("labelwise", "attention", "max", "chunkmax"))
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--max-chunks", type=int, default=6)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--max-words", type=int, default=10000)
    p.add_argument("--loss", default="bce", choices=("bce", "focal", "asl"))
    p.add_argument("--pos-weight-power", type=float, default=0.5)
    p.add_argument("--max-pos-weight", type=float, default=30.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--asl-gamma-neg", type=float, default=4.0)
    p.add_argument("--asl-gamma-pos", type=float, default=0.0)
    p.add_argument("--asl-clip", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-docs", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--student-layers", type=int, default=8)
    p.add_argument("--teacher-soft-labels")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-only-covered", type=int, default=0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--export-soft-labels", action="store_true")
    p.add_argument("--export-budget-min", type=float, default=8.0)
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--threshold-shrink", type=float, default=5.0)
    p.add_argument("--time-budget-min", type=float, default=35.0)
    p.add_argument("--eval-every-min", type=float, default=7.0)
    p.add_argument("--eval-docs", type=int, default=1500)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--artifact-zip")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--run-name")
    p.add_argument("--seed", type=int, default=230338)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
