"""CS781 Phase 3 - document-level (hierarchical) rhetorical-role tagger on Phase 1 features.

Stage 1: the Phase 1 backbone (frozen) encodes every sentence on its own -> [mean ; last] hidden state.
Stage 2: a 2-layer BiGRU runs over the sentence sequence of each document (plus a relative-position
and a CL/IT domain embedding) and labels every sentence.  Rhetorical roles come in long contiguous
blocks, so modelling the whole document sequence matters far more than per-sentence context windows.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase2_lner import LNERModel, load_phase1
from phase3_rr import (
    DOMAINS,
    check_labels,
    load_label_vocab,
    locate_phase1_files,
    macro_scores,
    normalize_documents,
    obtain_data,
    read_json,
    seed_everything,
)

POSITION_BINS = 20


@torch.no_grad()
def sentence_features(backbone, model_name, tokenizer, documents, device, max_tokens, batch_size, eos_id, pad_id):
    encoder = LNERModel(backbone, model_name, 1).to(device).eval()
    sentences = [s for doc in documents for s in doc["sentences"]]
    ids = [list(e.ids)[:max_tokens] or [eos_id] for e in tokenizer.encode_batch(sentences)]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    output = None
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        length = max(len(ids[r]) for r in rows)
        batch = torch.full((len(rows), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), length), dtype=torch.bool)
        for i, r in enumerate(rows):
            batch[i, :len(ids[r])] = torch.tensor(ids[r])
            mask[i, :len(ids[r])] = True
        batch, mask = batch.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = encoder.hidden(batch, mask).float()
        weights = mask.unsqueeze(-1).float()
        mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        last = hidden[torch.arange(len(rows), device=device), mask.sum(1) - 1]
        features = torch.cat((mean, last), -1).cpu()
        if output is None:
            output = torch.zeros(len(ids), features.size(1))
        output[rows] = features
    split, cursor = [], 0
    for doc in documents:
        split.append(output[cursor: cursor + len(doc["sentences"])])
        cursor += len(doc["sentences"])
    return split


class DocTagger(nn.Module):
    def __init__(self, feature_dim, label_count, hidden=192, dropout=0.3, bag_weight=None):
        super().__init__()
        self.input_dropout = nn.Dropout(0.2)
        self.project = nn.Sequential(nn.Linear(feature_dim, 2 * hidden), nn.GELU())
        self.bag = None
        if bag_weight is not None:  # bag of Phase 1 token embeddings (initialised from the Phase 1 embedding matrix)
            self.bag = nn.EmbeddingBag.from_pretrained(bag_weight.clone().float(), freeze=False, mode="mean")
            self.bag_project = nn.Sequential(nn.LayerNorm(bag_weight.size(1)), nn.Dropout(0.2),
                                             nn.Linear(bag_weight.size(1), 2 * hidden), nn.GELU())
        self.position = nn.Embedding(POSITION_BINS, 2 * hidden)
        self.domain = nn.Embedding(len(DOMAINS), 2 * hidden)
        self.gru = nn.GRU(2 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, label_count)

    def forward(self, features, positions, domains, lengths, bag=None):
        x = self.project(self.input_dropout(features)) + self.position(positions) + self.domain(domains)[:, None]
        if self.bag is not None and bag is not None:
            ids, offsets, valid = bag
            vectors = self.bag_project(self.bag(ids, offsets))
            extra = torch.zeros_like(x)
            extra[valid] = vectors.to(extra.dtype)
            x = x + extra
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.gru(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=features.size(1))
        return self.out(self.dropout(y))


def make_batch(items, device):
    length = max(item["x"].size(0) for item in items)
    dim = items[0]["x"].size(1)
    x = torch.zeros(len(items), length, dim)
    pos = torch.zeros(len(items), length, dtype=torch.long)
    y = torch.full((len(items), length), -100, dtype=torch.long)
    for i, item in enumerate(items):
        n = item["x"].size(0)
        x[i, :n] = item["x"]
        pos[i, :n] = torch.clamp((torch.arange(n) * POSITION_BINS) // max(1, n), max=POSITION_BINS - 1)
        if item["y"] is not None:
            y[i, :n] = item["y"]
    domains = torch.tensor([item["domain"] for item in items])
    lengths = torch.tensor([item["x"].size(0) for item in items])
    valid = torch.zeros(len(items), length, dtype=torch.bool)
    flat, offsets = [], []
    for i, item in enumerate(items):
        valid[i, :item["x"].size(0)] = True
        for tokens in item["tok"]:
            offsets.append(len(flat))
            flat.extend(tokens)
    bag = (torch.tensor(flat, device=device), torch.tensor(offsets, device=device), valid.to(device))
    return x.to(device), pos.to(device), domains.to(device), lengths, y.to(device), bag


@torch.no_grad()
def doc_logits(model, items, device, batch_docs=16):
    model.eval()
    results = []
    for start in range(0, len(items), batch_docs):
        chunk = items[start:start + batch_docs]
        x, pos, dom, lengths, _, bag = make_batch(chunk, device)
        logits = model(x, pos, dom, lengths, bag).float().cpu()
        results += [logits[i, :item["x"].size(0)] for i, item in enumerate(chunk)]
    return results


def score(items, logits, labels):
    gold, pred, by_domain = [], [], {d: ([], []) for d in DOMAINS}
    for item, logit in zip(items, logits):
        g = [labels[i] for i in item["y"].tolist()]
        p = [labels[i] for i in logit.argmax(-1).tolist()]
        gold += g
        pred += p
        by_domain[DOMAINS[item["domain"]]][0].extend(g)
        by_domain[DOMAINS[item["domain"]]][1].extend(p)
    result = macro_scores(gold, pred)
    for d, (g, p) in by_domain.items():
        if g:
            result[f"{d}_macro_f1"] = macro_scores(g, p)["macro_f1"]
    return result


def train_tagger(train_items, dev_items, labels, args, device, seed, bag_weight=None):
    seed_everything(seed)
    model = DocTagger(train_items[0]["x"].size(1), len(labels), hidden=args.hidden, bag_weight=bag_weight).to(device)
    counts = Counter(i for item in train_items for i in item["y"].tolist())
    largest = max(counts.values())
    weights = torch.tensor([min(10.0, (largest / max(1, counts[i])) ** args.class_weight_power)
                            for i in range(len(labels))])
    weights = (weights / weights.mean()).to(device)
    bag_params = list(model.bag.parameters()) if model.bag is not None else []
    bag_ids = {id(p) for p in bag_params}
    other = [p for p in model.parameters() if id(p) not in bag_ids]
    groups = [{"params": other, "lr": args.lr}]
    if bag_params:
        groups.append({"params": bag_params, "lr": args.bag_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    best, best_state, stale = {"macro_f1": -1}, None, 0
    for epoch in range(1, args.doc_epochs + 1):
        model.train()
        perm = torch.randperm(len(train_items)).tolist()
        for start in range(0, len(perm), args.batch_docs):
            chunk = [train_items[i] for i in perm[start:start + args.batch_docs]]
            x, pos, dom, lengths, y, bag = make_batch(chunk, device)
            if args.feature_noise > 0:
                x = x + args.feature_noise * torch.randn_like(x)
            logits = model(x, pos, dom, lengths, bag)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), weight=weights, ignore_index=-100,
                                   label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = score(dev_items, doc_logits(model, dev_items, device), labels)
        if result["macro_f1"] > best["macro_f1"]:
            best, stale = result, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch % 5 == 0:
            print(f"seed {seed} epoch {epoch} loss {float(loss):.3f} dev macro-F1 {result['macro_f1']:.4f} "
                  f"(best {best['macro_f1']:.4f})", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best


def fast_macro(gold, pred, count):
    present = torch.bincount(gold, minlength=count) + torch.bincount(pred, minlength=count) > 0
    tp = torch.bincount(gold[gold == pred], minlength=count).float()
    fp = torch.bincount(pred, minlength=count).float() - tp
    fn = torch.bincount(gold, minlength=count).float() - tp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    return float(f1[present].mean())


def tune_bias(items, probs, labels, rounds=3):
    """Greedy per-class additive log-prob bias that maximises dev Macro-F1 (helps rare roles)."""
    gold = torch.cat([item["y"] for item in items])
    logp = torch.log(torch.cat(probs) + 1e-9)
    bias = torch.zeros(len(labels))
    best = fast_macro(gold, logp.argmax(-1), len(labels))
    grid = [x / 4 for x in range(-8, 13)]
    for _ in range(rounds):
        for c in range(len(labels)):
            keep = bias[c].item()
            for value in grid:
                bias[c] = value
                f = fast_macro(gold, (logp + bias).argmax(-1), len(labels))
                if f > best + 1e-6:
                    best, keep = f, value
            bias[c] = keep
    return bias


def run(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"phase3_doc_{args.model}"
    output_dir = Path(args.output_dir or f"/kaggle/working/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, tokenizer_path, artifact = locate_phase1_files(args.model, args.artifact_zip, output_dir / "phase1")
    files = obtain_data(output_dir / "data", args.data_dir)
    labels = load_label_vocab(files["label_vocab.json"])
    label_to_id = {label: i for i, label in enumerate(labels)}
    docs = {}
    for split, needs in (("train", True), ("dev", True), ("test", False)):
        docs[split] = []
        for domain in DOMAINS:
            docs[split] += normalize_documents(read_json(files[f"{domain}_{split}.json"]), domain, needs)
    check_labels(docs["train"], labels)
    check_labels(docs["dev"], labels)
    test_ids = [d["id"] for d in docs["test"]]
    if len(set(test_ids)) != len(test_ids):
        raise RuntimeError("CL and IT test sets share document ids")
    print({k: (len(v), sum(len(d["sentences"]) for d in v)) for k, v in docs.items()}, flush=True)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id, eos_id = tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<eos>")
    backbone, _, saved = load_phase1(checkpoint, args.model)
    parameters = sum(p.numel() for p in backbone.parameters())
    feats = {}
    for split in docs:
        feats[split] = sentence_features(backbone, args.model, tokenizer, docs[split], device,
                                         args.max_tokens, args.batch_size, eos_id, pad_id)
        print(f"features {split}: {sum(f.size(0) for f in feats[split])} sentences, "
              f"{(time.time() - start_time) / 60:.1f} min", flush=True)
    bag_weight = backbone.embedding.weight.detach().cpu() if args.bow else None
    del backbone
    torch.cuda.empty_cache() if device.type == "cuda" else None
    stacked = torch.cat(feats["train"])
    mean, std = stacked.mean(0), stacked.std(0).clamp_min(1e-4)

    def items(split):
        out = []
        token_ids = {}
        for doc in docs[split]:
            encoded = tokenizer.encode_batch(doc["sentences"]) if doc["sentences"] else []
            token_ids[id(doc)] = [list(e.ids)[:args.max_tokens] or [eos_id] for e in encoded]
        for doc, f in zip(docs[split], feats[split]):
            y = torch.tensor([label_to_id[l] for l in doc["labels"]]) if doc["labels"] is not None else None
            out.append({"x": (f - mean) / std, "y": y, "domain": DOMAINS.index(doc["domain"]), "doc": doc,
                        "tok": token_ids[id(doc)]})
        return out

    train_items, dev_items, test_items = items("train"), items("dev"), items("test")
    models, seed_scores = [], []
    for k in range(args.seeds):
        model, best = train_tagger(train_items, dev_items, labels, args, device, args.seed + k, bag_weight)
        models.append(model)
        seed_scores.append(best["macro_f1"])
        print(f"seed {args.seed + k}: best dev macro-F1 {best['macro_f1']:.4f}", flush=True)

    def ensemble(items_):
        all_logits = [doc_logits(m, items_, device) for m in models]
        return [torch.stack([F.softmax(l[i], -1) for l in all_logits]).mean(0) for i in range(len(items_))]

    dev_probs = ensemble(dev_items)
    bias = torch.zeros(len(labels))
    before = score(dev_items, dev_probs, labels)["macro_f1"]
    if args.tune_bias:
        bias = tune_bias(dev_items, dev_probs, labels)
    dev_probs = [torch.log(p + 1e-9) + bias for p in dev_probs]
    dev_result = score(dev_items, dev_probs, labels)
    print(f"dev macro-F1 ensemble {before:.4f} -> after class-bias tuning {dev_result['macro_f1']:.4f}", flush=True)
    print(json.dumps({k: dev_result[k] for k in ("macro_f1", "macro_precision", "macro_recall", "accuracy",
                                                  "CL_macro_f1", "IT_macro_f1") if k in dev_result}), flush=True)
    predictions = {}
    for item, probs in zip(test_items, ensemble(test_items)):
        predicted = [labels[i] for i in (torch.log(probs + 1e-9) + bias).argmax(-1).tolist()]
        assert len(predicted) == len(item["doc"]["sentences"])
        predictions[item["doc"]["id"]] = predicted
    path = output_dir / "prediction.json"
    path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    archive = output_dir / f"submission_{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(path, arcname="prediction.json")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["prediction.json"]
    (output_dir / "validation_metrics.json").write_text(json.dumps(dev_result, indent=2), encoding="utf-8")
    manifest = {
        "run": name, "model": args.model, "method": "frozen Phase 1 features + document BiGRU tagger",
        "phase1_artifact": artifact, "phase1_steps": saved.get("total_steps"), "phase1_parameters": parameters,
        "tagger_parameters": sum(p.numel() for p in models[0].parameters()), "seeds": args.seeds,
        "seed_dev_macro_f1": seed_scores, "class_bias": bias.tolist(), "bag_of_phase1_embeddings": bool(args.bow), "best_dev_macro_f1": dev_result["macro_f1"],
        "dev_CL_macro_f1": dev_result.get("CL_macro_f1"), "dev_IT_macro_f1": dev_result.get("IT_macro_f1"),
        "minutes": (time.time() - start_time) / 60, "config": vars(args), "submission": str(archive),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {archive} ({len(predictions)} documents). Dev macro-F1 {dev_result['macro_f1']:.4f}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("decoder", "encoder_decoder", "hybrid", "efficient"))
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-docs", type=int, default=4)
    p.add_argument("--doc-epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--feature-noise", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--bow", type=int, default=1)
    p.add_argument("--bag-lr", type=float, default=3e-4)
    p.add_argument("--tune-bias", type=int, default=1)
    p.add_argument("--seed", type=int, default=230338)
    p.add_argument("--artifact-zip")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--run-name")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
