"""CS781 Phase 4 - Court Judgment Prediction (CJPE) from Phase 1 legal language models.

Every experiment starts from a Phase 1 checkpoint (decoder / encoder_decoder / hybrid / efficient);
no other pretrained weights are used.  Training uses only the official Phase 4 CJPE train/dev data.

Long-document strategy (hierarchical, section-aware):
  * the case is tokenised with the Phase 1 tokenizer and cut into chunks of 256 tokens (the Phase 1
    context length);
  * ``--selection headtail`` keeps the first chunk (parties / facts / background) plus the last
    K-1 chunks (arguments, analysis and the operative part, where the outcome is decided);
    ``--selection tail`` keeps the last K chunks;
  * the Phase 1 backbone encodes every chunk -> [masked mean ; last token] chunk vector
    + learned chunk-slot embedding;
  * attention pooling across chunks (+ mean of chunks) -> MLP -> 2 logits.
Adaptation: full fine-tuning, LORMA+, LoRA, IA3, partial (top-layer) fine-tuning, BitFit and
knowledge distillation (truncated 8-layer decoder student trained on a teacher's soft labels).
Post-processing: the decision threshold on P(label=1) is tuned on dev for Macro-F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase2_lner import LNERModel, load_phase1, make_student
from phase3_rr import apply_adapter, locate_phase1_files, seed_everything, transformer_layers

REPO_ID = "Exploration-Lab/CS-781-Capstone"
PHASE_DIR = "Phase_4_CJPE"


# ----------------------------------------------------------------------------- data

def obtain_data(destination: Path, data_dir: str | None) -> dict[str, Path]:
    """train.json / dev.json / test.json from --data-dir or the official HF dataset (always the latest version)."""
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
    print("Phase 4 files on HF:", repo_files, flush=True)
    found = {}
    for name in names:
        matches = [f for f in repo_files if f.lower().endswith("/" + name)]
        if not matches:
            raise FileNotFoundError(f"{name} not found under {PHASE_DIR} on HF: {repo_files}")
        path = Path(hf_hub_download(repo_id=REPO_ID, filename=matches[0], repo_type="dataset", token=token,
                                    force_download=True))  # never reuse a cached (pre-fix, leaked) copy
        found[name] = path
    return found


def _pick(record: dict, keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


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
        text = _pick(item, ("text", "case_text", "document", "raw_text", "content"))
        if case_id is None or text is None:
            raise ValueError(f"{path.name} record #{position}: cannot find id/text, keys={list(item)}")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        label = _pick(item, ("label", "labels", "decision", "outcome", "judgment", "pred"))
        if require_labels:
            if isinstance(label, list):
                label = label[0]
            label = int(label)
            if label not in (0, 1):
                raise ValueError(f"{path.name}/{case_id}: label {label} is not 0/1")
        else:
            label = None
        cases.append({"id": case_id, "text": str(text), "label": label})
    print(f"{path.name}: {len(cases)} cases; keys of first record: {list(raw[0]) if raw else []}", flush=True)
    return cases


def text_key(text: str) -> str:
    norm = " ".join(text.split()).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def leakage_report(train, dev, test) -> dict:
    """Exact-duplicate check (by id and by whitespace/case-normalised text). Overlaps are removed from
    train/dev so that nothing that also appears in the test set is ever trained on or used for selection."""
    test_ids = {c["id"] for c in test}
    test_keys = {text_key(c["text"]) for c in test}
    dev_keys = {text_key(c["text"]) for c in dev}
    report = {
        "train_test_id_overlap": sum(c["id"] in test_ids for c in train),
        "dev_test_id_overlap": sum(c["id"] in test_ids for c in dev),
        "train_test_text_overlap": sum(text_key(c["text"]) in test_keys for c in train),
        "dev_test_text_overlap": sum(text_key(c["text"]) in test_keys for c in dev),
        "train_dev_text_overlap": sum(text_key(c["text"]) in dev_keys for c in train),
    }
    train[:] = [c for c in train if c["id"] not in test_ids and text_key(c["text"]) not in test_keys
                and text_key(c["text"]) not in dev_keys]
    dev[:] = [c for c in dev if c["id"] not in test_ids and text_key(c["text"]) not in test_keys]
    report["train_after_filter"], report["dev_after_filter"] = len(train), len(dev)
    print("LEAKAGE CHECK:", json.dumps(report), flush=True)
    return report


def encode_cases(cases, tokenizer, args):
    """-> list of chunk lists (token ids). Only the needed head/tail words are tokenised (speed)."""
    L, K = args.chunk_len, args.max_chunks
    tail_words = int(K * L * 1.1) + 50
    head_words = int(L * 1.1) + 20
    heads, tails = [], []
    for case in cases:
        words = case["text"].split()
        heads.append(" ".join(words[:head_words]))
        tails.append(" ".join(words[-tail_words:]))
    tail_ids = [e.ids for e in tokenizer.encode_batch(tails)]
    head_ids = [e.ids for e in tokenizer.encode_batch(heads)] if args.selection == "headtail" else None
    out = []
    for i, case in enumerate(cases):
        ids = tail_ids[i]
        n_words = len(case["text"].split())
        if args.selection == "headtail" and n_words > tail_words:
            tail = ids[-(K - 1) * L:]
            chunks = [head_ids[i][:L]] + [tail[j:j + L] for j in range(0, len(tail), L)]
        else:
            ids = ids[-K * L:]
            chunks = [ids[j:j + L] for j in range(0, len(ids), L)]
        chunks = [c for c in chunks if c] or [[args.eos_id]]
        out.append(chunks[:K])
    return out


def make_batch(chunk_lists, pad_id, device):
    flat = [c for chunks in chunk_lists for c in chunks]
    length = max(len(c) for c in flat)
    ids = torch.full((len(flat), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(flat), length), dtype=torch.bool)
    for r, c in enumerate(flat):
        ids[r, :len(c)] = torch.tensor(c)
        mask[r, :len(c)] = True
    K = max(len(chunks) for chunks in chunk_lists)
    slot_valid = torch.zeros((len(chunk_lists), K), dtype=torch.bool)
    for i, chunks in enumerate(chunk_lists):
        slot_valid[i, :len(chunks)] = True
    return ids.to(device), mask.to(device), slot_valid.to(device)


# ----------------------------------------------------------------------------- model

class CJPEModel(nn.Module):
    def __init__(self, backbone, model_name, max_chunks, dropout=0.1):
        super().__init__()
        self.encoder = LNERModel(backbone, model_name, 1)
        del self.encoder.classifier
        d = backbone.cfg.d_model
        self.chunk_proj = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU())
        self.slot = nn.Embedding(max_chunks, d)
        self.attn = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(dropout), nn.Linear(2 * d, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 2))

    def head_parameters(self):
        return [*self.chunk_proj.parameters(), *self.slot.parameters(), *self.attn.parameters(), *self.head.parameters()]

    def forward(self, ids, mask, slot_valid):
        hidden = self.encoder.hidden(ids, mask).float()
        w = mask.unsqueeze(-1).float()
        mean = (hidden * w).sum(1) / w.sum(1).clamp_min(1)
        last = hidden[torch.arange(hidden.size(0), device=hidden.device), mask.sum(1) - 1]
        vectors = self.chunk_proj(torch.cat((mean, last), -1))
        B, K = slot_valid.shape
        chunks = torch.zeros(B, K, vectors.size(-1), device=vectors.device, dtype=vectors.dtype)
        chunks[slot_valid] = vectors
        chunks = chunks + self.slot(torch.arange(K, device=ids.device))[None]
        scores = self.attn(chunks).squeeze(-1).masked_fill(~slot_valid, -1e4)
        alpha = scores.softmax(-1)
        pooled = (alpha.unsqueeze(-1) * chunks).sum(1)
        mean_chunks = (chunks * slot_valid.unsqueeze(-1)).sum(1) / slot_valid.sum(1, keepdim=True).clamp_min(1)
        return self.head(torch.cat((pooled, mean_chunks), -1)), alpha


def apply_method(model: CJPEModel, model_name: str, method: str, rank: int, unfreeze: int):
    backbone = model.encoder.backbone
    if method in ("full", "distill"):
        return
    if method in ("lorma", "lora", "ia3"):
        apply_adapter(model, model_name, method, rank)
        return
    for p in backbone.parameters():
        p.requires_grad = False
    if method == "bitfit":
        for name, p in backbone.named_parameters():
            if name.endswith("bias"):
                p.requires_grad = True
    elif method == "partial":
        if model_name == "efficient":
            groups = [list(backbone.blocks)]
        elif model_name == "encoder_decoder":
            groups = [list(backbone.encoder.layers), list(backbone.decoder.layers)]
        else:
            groups = [transformer_layers(backbone, model_name)]
        for layers in groups:
            for layer in layers[-unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True
    else:
        raise ValueError(method)
    for module_name in ("norm", "fusion"):
        module = getattr(backbone, module_name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True


# ----------------------------------------------------------------------------- metrics / inference

def binary_scores(gold, pred) -> dict:
    out = {}
    for c in (0, 1):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[c] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    tp1 = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    return {
        "macro_f1": (out[0][2] + out[1][2]) / 2, "macro_precision": (out[0][0] + out[1][0]) / 2,
        "macro_recall": (out[0][1] + out[1][1]) / 2, "accuracy": sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold)),
        "f1_class0": out[0][2], "f1_class1": out[1][2], "precision_class1": out[1][0], "recall_class1": out[1][1],
        "false_positives": sum(g == 0 and p == 1 for g, p in zip(gold, pred)),
        "false_negatives": sum(g == 1 and p == 0 for g, p in zip(gold, pred)), "true_positives": tp1,
    }


@torch.no_grad()
def predict(model, chunk_lists, pad_id, device, batch_docs):
    model.eval()
    logits, attentions = [], []
    order = sorted(range(len(chunk_lists)), key=lambda i: -sum(map(len, chunk_lists[i])))
    result = [None] * len(chunk_lists)
    for s in range(0, len(order), batch_docs):
        rows = order[s:s + batch_docs]
        ids, mask, valid = make_batch([chunk_lists[r] for r in rows], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out, alpha = model(ids, mask, valid)
        out, alpha = out.float().cpu(), alpha.float().cpu()
        for j, r in enumerate(rows):
            result[r] = (out[j], alpha[j, :len(chunk_lists[r])])
    return torch.stack([r[0] for r in result]), [r[1] for r in result]


def tune_threshold(probs, gold):
    best_t, best_f = 0.5, binary_scores(gold, [int(p >= 0.5) for p in probs])["macro_f1"]
    for t in [x / 100 for x in range(10, 91)]:
        f = binary_scores(gold, [int(p >= t) for p in probs])["macro_f1"]
        if f > best_f + 1e-9:
            best_t, best_f = t, f
    return best_t, best_f


# ----------------------------------------------------------------------------- training

def train(model, train_chunks, train_labels, dev_chunks, dev_labels, args, device, pad_id, soft=None):
    counts = Counter(train_labels)
    largest = max(counts.values())
    weights = torch.tensor([(largest / max(1, counts[c])) ** args.class_weight_power for c in (0, 1)])
    weights = (weights / weights.mean()).to(device)
    print("train label counts", dict(counts), "class weights", weights.tolist(), flush=True)
    head_ids = {id(p) for p in model.head_parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": args.head_lr}] + ([{"params": body, "lr": args.learning_rate}] if body else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    budget = args.time_budget_min * 60
    # schedule over the number of steps the time budget is expected to allow (re-estimated after warm-up)
    est_steps = max(50, args.epochs * math.ceil(len(train_chunks) / (args.batch_docs * args.grad_accum)))
    warmup = 30
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, max(0.1, 1 - s / est_steps)))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    trainable = [p for p in model.parameters() if p.requires_grad]
    start, last_eval = time.time(), time.time()
    best, best_state, history = {"macro_f1": -1}, None, []
    step, opt_steps, seen = 0, 0, 0

    def evaluate(tag):
        nonlocal best, best_state, last_eval
        logits, _ = predict(model, dev_chunks, pad_id, device, args.batch_docs * 2)
        pred = logits.argmax(-1).tolist()
        scores = binary_scores(dev_labels, pred)
        row = {"tag": tag, "docs_seen": seen, "opt_steps": opt_steps, "minutes": round((time.time() - start) / 60, 2),
               **{k: scores[k] for k in ("macro_f1", "macro_precision", "macro_recall", "accuracy", "f1_class0", "f1_class1")}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if scores["macro_f1"] > best["macro_f1"]:
            best = scores
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        model.train()
        last_eval = time.time()

    done = False
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_chunks)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for s in range(0, len(order), args.batch_docs):
            rows = order[s:s + args.batch_docs]
            ids, mask, valid = make_batch([train_chunks[r] for r in rows], pad_id, device)
            y = torch.tensor([train_labels[r] for r in rows], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(ids, mask, valid)
            logits = logits.float()
            loss = F.cross_entropy(logits, y, weight=weights, label_smoothing=args.label_smoothing)
            if soft is not None:
                t = args.temperature
                teacher = soft[rows].to(device)
                kd = F.kl_div(F.log_softmax(logits / t, -1), F.softmax(teacher / t, -1), reduction="batchmean") * t * t
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
                if opt_steps == 40:  # re-estimate how many optimiser steps fit into the budget
                    rate = opt_steps / (time.time() - start)
                    est_steps = max(60, int(rate * budget * 0.92))
                    print(f"~{rate:.2f} opt steps/s -> schedule over {est_steps} steps "
                          f"(~{est_steps * args.batch_docs * args.grad_accum} docs)", flush=True)
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


# ----------------------------------------------------------------------------- main

def run(args):
    t0 = time.time()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"p4_{args.model}_{args.method}"
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
    print({"train": len(train_cases), "dev": len(dev_cases), "test": len(test_cases),
           "train_labels": dict(Counter(c["label"] for c in train_cases)),
           "dev_labels": dict(Counter(c["label"] for c in dev_cases))}, flush=True)

    checkpoint, tokenizer_path, artifact = locate_phase1_files(args.model, args.artifact_zip, out / "phase1")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id, args.eos_id = tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<eos>")
    if pad_id is None or args.eos_id is None:
        raise ValueError("Phase 1 tokenizer must contain <pad> and <eos>")

    soft = None
    if args.method == "distill":
        saved = torch.load(args.teacher_soft_labels, map_location="cpu")
        index = {cid: r for r, cid in enumerate(saved["ids"])}
        before = len(train_cases)
        train_cases = [c for c in train_cases if c["id"] in index]
        soft = saved["logits"][[index[c["id"]] for c in train_cases]]
        print(f"KD: teacher {saved['teacher']} soft labels for {len(train_cases)}/{before} train cases", flush=True)

    enc_t = time.time()
    train_chunks = encode_cases(train_cases, tokenizer, args)
    dev_chunks = encode_cases(dev_cases, tokenizer, args)
    test_chunks = encode_cases(test_cases, tokenizer, args)
    chunk_stats = Counter(len(c) for c in train_chunks)
    print(f"tokenised in {time.time() - enc_t:.0f}s; chunks/doc histogram {dict(sorted(chunk_stats.items()))}", flush=True)
    train_labels = [c["label"] for c in train_cases]
    dev_labels = [c["label"] for c in dev_cases]

    backbone, phase1_cfg, phase1_saved = load_phase1(checkpoint, args.model)
    phase1_params = sum(p.numel() for p in backbone.parameters())
    if args.method == "distill":
        if args.model != "decoder":
            raise ValueError("The distillation student recipe uses the decoder Phase 1 model")
        backbone = make_student(backbone, layers=args.student_layers)
    model = CJPEModel(backbone, args.model, args.max_chunks)
    apply_method(model, args.model, args.method, args.rank, args.unfreeze_layers)
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {"run": name, "model": args.model, "method": args.method, "phase1_artifact": artifact,
               "phase1_parameters": phase1_params, "parameters": total, "trainable_parameters": trainable,
               "trainable_fraction": trainable / total, "leakage_check": leak, "device": str(device),
               "config": vars(args)}
    print(json.dumps(summary, indent=2, default=str), flush=True)

    best, history, seconds = train(model, train_chunks, train_labels, dev_chunks, dev_labels, args, device, pad_id, soft)

    inf_t = time.time()
    dev_logits, _ = predict(model, dev_chunks, pad_id, device, args.batch_docs * 2)
    dev_probs = dev_logits.softmax(-1)[:, 1].tolist()
    argmax_scores = binary_scores(dev_labels, dev_logits.argmax(-1).tolist())
    threshold, tuned_f1 = tune_threshold(dev_probs, dev_labels) if args.tune_threshold else (0.5, None)
    tuned = binary_scores(dev_labels, [int(p >= threshold) for p in dev_probs])
    test_logits, test_alpha = predict(model, test_chunks, pad_id, device, args.batch_docs * 2)
    inference_seconds = time.time() - inf_t
    test_probs = test_logits.softmax(-1)[:, 1].tolist()
    print(f"dev macro-F1 argmax {argmax_scores['macro_f1']:.4f} | threshold {threshold:.2f} -> {tuned['macro_f1']:.4f}", flush=True)

    predictions = [{"id": c["id"], "pred": int(p >= threshold)} for c, p in zip(test_cases, test_probs)]
    assert len(predictions) == len(test_cases) and all(p["pred"] in (0, 1) for p in predictions)
    path = out / "prediction.json"
    path.write_text(json.dumps(predictions), encoding="utf-8")
    archive = out / f"submission_{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(path, arcname="prediction.json")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["prediction.json"]
        assert len(json.loads(bundle.read("prediction.json"))) == len(test_cases)

    # error analysis material: dev predictions with probabilities + attention over chunks
    _, dev_alpha = predict(model, dev_chunks, pad_id, device, args.batch_docs * 2)
    (out / "dev_predictions.json").write_text(json.dumps([
        {"id": c["id"], "gold": c["label"], "prob1": round(p, 4), "pred": int(p >= threshold),
         "words": len(c["text"].split()), "chunk_attention": [round(a, 3) for a in al.tolist()]}
        for c, p, al in zip(dev_cases, dev_probs, dev_alpha)]), encoding="utf-8")
    slot_attention = torch.zeros(args.max_chunks)
    slot_counts = torch.zeros(args.max_chunks)
    for al in dev_alpha:
        slot_attention[:len(al)] += al
        slot_counts[:len(al)] += 1
    mean_slot_attention = (slot_attention / slot_counts.clamp_min(1)).tolist()

    if args.export_soft_labels:
        ex_t = time.time()
        order = list(range(len(train_cases)))
        random.Random(args.seed).shuffle(order)
        ids, logits = [], []
        for s in range(0, len(order), 256):
            rows = order[s:s + 256]
            lg, _ = predict(model, [train_chunks[r] for r in rows], pad_id, device, args.batch_docs * 2)
            ids += [train_cases[r]["id"] for r in rows]
            logits.append(lg)
            if time.time() - ex_t > args.export_budget_min * 60:
                break
        torch.save({"ids": ids, "logits": torch.cat(logits), "teacher": name}, out / "teacher_soft_labels.pt")
        print(f"Saved teacher soft labels for {len(ids)} train cases in {(time.time() - ex_t) / 60:.1f} min", flush=True)

    if args.save_checkpoint:
        torch.save({"model": model.state_dict(), "phase1_config": asdict(phase1_cfg), "run_config": vars(args)},
                   out / "best.pt")
    (out / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "validation_metrics.json").write_text(json.dumps(
        {"selected_checkpoint": best, "argmax": argmax_scores, "threshold": threshold, "tuned": tuned}, indent=2),
        encoding="utf-8")
    manifest = {**summary, "best_dev_macro_f1_argmax": argmax_scores["macro_f1"], "threshold": threshold,
                "dev_macro_f1_tuned": tuned["macro_f1"], "dev_macro_precision": tuned["macro_precision"],
                "dev_macro_recall": tuned["macro_recall"], "dev_false_positives": tuned["false_positives"],
                "dev_false_negatives": tuned["false_negatives"], "mean_chunk_attention_by_slot": mean_slot_attention,
                "test_pred_distribution": dict(Counter(p["pred"] for p in predictions)),
                "train_minutes": seconds / 60, "inference_seconds_dev_plus_test": inference_seconds,
                "total_minutes": (time.time() - t0) / 60,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
                "phase1_steps": phase1_saved.get("total_steps"), "submission": str(archive)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("run", "best_dev_macro_f1_argmax", "threshold", "dev_macro_f1_tuned",
                                                "trainable_parameters", "test_pred_distribution", "total_minutes")}, indent=2))
    print("SUBMISSION ZIP:", archive, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("decoder", "encoder_decoder", "hybrid", "efficient"))
    p.add_argument("--method", required=True, choices=("full", "lorma", "lora", "ia3", "partial", "bitfit", "distill"))
    p.add_argument("--selection", default="headtail", choices=("headtail", "tail"))
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--max-chunks", type=int, default=6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-docs", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--student-layers", type=int, default=8)
    p.add_argument("--teacher-soft-labels")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--export-soft-labels", action="store_true")
    p.add_argument("--export-budget-min", type=float, default=6.0)
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--tune-threshold", type=int, default=1)
    p.add_argument("--time-budget-min", type=float, default=25.0)
    p.add_argument("--eval-every-min", type=float, default=6.0)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--artifact-zip")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--run-name")
    p.add_argument("--seed", type=int, default=230338)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
