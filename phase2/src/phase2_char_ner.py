"""Exact-character LNER head over the Phase 1 foundation encoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from phase1_foundation import Config, LegalEncoderDecoder
from phase1_pipeline import EfficientLM, RunConfig


REPO_ID = "Exploration-Lab/CS-781-Capstone"
FILES = (
    "Phase_2_LNER/Train/fold1-text.json",
    "Phase_2_LNER/Train/fold1-gold.json",
    "Phase_2_LNER/Train/fold2-text.json",
    "Phase_2_LNER/Train/fold2-gold.json",
    "Phase_2_LNER/Train/labels.txt",
    "Phase_2_LNER/Test/fold3-text.json",
)
EXPECTED_LABELS = {
    "APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT",
    "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO",
}


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_one(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name}; found {matches}")
    return matches[0]


def safe_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def download_data(destination):
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Load the Kaggle secret HF into HF_TOKEN first")
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for filename in FILES:
        cached = Path(hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename, token=token,
        ))
        target = destination / Path(filename).name
        target.write_bytes(cached.read_bytes())
        result[target.name] = target
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def texts(raw):
    return {
        str(key): value if isinstance(value, str) else value["text"]
        for key, value in raw.items()
    }


def entity(value):
    if isinstance(value, dict):
        return (
            int(value.get("start", value.get("start_offset"))),
            int(value.get("end", value.get("end_offset"))),
            str(value.get("label", value.get("type"))),
        )
    return int(value[0]), int(value[1]), str(value[2])


def gold(raw):
    result = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            value = value.get("entities", value.get("spans", value.get("annotations")))
        result[str(key)] = [entity(item) for item in value]
    return result


def load_labels(path):
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        for chunk in line.split(","):
            candidate = chunk.strip().split()[0] if chunk.strip() else ""
            if candidate in EXPECTED_LABELS:
                found.append(candidate)
    if len(found) != 12 or set(found) != EXPECTED_LABELS:
        raise ValueError(found)
    return found


def tags(labels):
    values = ["O"] + [tag for label in labels for tag in (f"B-{label}", f"I-{label}")]
    return values, {value: index for index, value in enumerate(values)}


def character_labels(text, spans, tag_to_id):
    result = [tag_to_id["O"]] * len(text)
    for start, end, label in sorted(spans):
        if not 0 <= start < end <= len(text):
            raise ValueError((start, end, label, len(text)))
        if any(result[index] != tag_to_id["O"] for index in range(start, end)):
            raise ValueError(f"Overlapping gold span {(start, end, label)}")
        result[start] = tag_to_id[f"B-{label}"]
        for index in range(start + 1, end):
            result[index] = tag_to_id[f"I-{label}"]
    return result


class CharacterWindows(Dataset):
    def __init__(self, documents, tokenizer, max_tokens, stride, annotations=None, tag_to_id=None):
        self.documents = documents
        self.items = []
        step = max_tokens - stride
        for document_id, text in documents.items():
            encoded = tokenizer.encode(text)
            ids = list(encoded.ids)
            offsets = [tuple(map(int, value)) for value in encoded.offsets]
            doc_labels = character_labels(text, annotations[document_id], tag_to_id) if annotations else None
            for token_start in range(0, len(ids), step):
                token_end = min(token_start + max_tokens, len(ids))
                window_offsets = offsets[token_start:token_end]
                if not window_offsets:
                    continue
                char_start = window_offsets[0][0]
                char_end = window_offsets[-1][1]
                if char_end <= char_start:
                    continue
                mapping = [-1] * (char_end - char_start)
                for local_token, (left, right) in enumerate(window_offsets):
                    for position in range(max(left, char_start), min(right, char_end)):
                        mapping[position - char_start] = local_token
                previous = 0
                for index, value in enumerate(mapping):
                    if value >= 0:
                        previous = value
                    else:
                        mapping[index] = previous
                following = len(window_offsets) - 1
                for index in range(len(mapping) - 1, -1, -1):
                    if mapping[index] >= 0:
                        following = mapping[index]
                    else:
                        mapping[index] = following
                self.items.append({
                    "document_id": document_id,
                    "char_start": char_start,
                    "ids": ids[token_start:token_end],
                    "char_ids": [ord(char) % 4096 for char in text[char_start:char_end]],
                    "char_to_token": mapping,
                    "labels": doc_labels[char_start:char_end] if doc_labels is not None else None,
                })
                if token_end == len(ids):
                    break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    if len(batch) != 1:
        raise ValueError("This exact-character recipe requires batch size 1")
    item = batch[0]
    return {
        "document_id": item["document_id"],
        "char_start": item["char_start"],
        "ids": torch.tensor(item["ids"], dtype=torch.long).unsqueeze(0),
        "mask": torch.ones((1, len(item["ids"])), dtype=torch.bool),
        "char_ids": torch.tensor(item["char_ids"], dtype=torch.long).unsqueeze(0),
        "char_to_token": torch.tensor(item["char_to_token"], dtype=torch.long).unsqueeze(0),
        "labels": None if item["labels"] is None else torch.tensor(item["labels"], dtype=torch.long).unsqueeze(0),
    }


class CharacterNER(nn.Module):
    def __init__(self, backbone, tag_count, model_name):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        width = backbone.cfg.d_model
        self.char_embedding = nn.Embedding(4096, 64)
        self.input = nn.Linear(width + 64, 256)
        self.norm1 = nn.LayerNorm(256)
        self.conv1 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(256, tag_count)

    def head_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                yield parameter

    def forward(self, ids, mask, char_ids, char_to_token):
        if self.model_name == "foundation":
            token_hidden = self.backbone.encode(ids, mask)
        else:
            token_hidden = self.backbone.embedding(ids)
            for block in self.backbone.blocks:
                token_hidden = block(token_hidden)
            token_hidden = self.backbone.norm(token_hidden)
        indexes = char_to_token.unsqueeze(-1).expand(-1, -1, token_hidden.size(-1))
        character_hidden = token_hidden.gather(1, indexes)
        x = torch.cat((character_hidden, self.char_embedding(char_ids)), dim=-1)
        x = F.gelu(self.input(x))
        residual = x
        x = self.conv1(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x) + residual
        residual = x
        x = self.conv2(self.norm2(x).transpose(1, 2)).transpose(1, 2)
        return self.classifier(self.dropout(F.gelu(x) + residual))


def decode(tag_ids, tag_names):
    spans = []
    active_label = None
    start = None
    for position, tag_id in enumerate(tag_ids + [0]):
        tag = tag_names[tag_id]
        if tag == "O":
            if active_label is not None:
                spans.append([start, position, active_label])
            active_label, start = None, None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or label != active_label:
            if active_label is not None:
                spans.append([start, position, active_label])
            active_label, start = label, position
    return spans


def macro_f1(predictions, truth, labels):
    details = {}
    for label in labels:
        predicted = {(doc, s, e) for doc, spans in predictions.items() for s, e, value in spans if value == label}
        actual = {(doc, s, e) for doc, spans in truth.items() for s, e, value in spans if value == label}
        tp = len(predicted & actual)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(actual) if actual else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        details[label] = {"precision": precision, "recall": recall, "f1": score, "support": len(actual)}
    return sum(row["f1"] for row in details.values()) / len(labels), details


@torch.inference_mode()
def score_documents(model, loader, documents, tag_count, device):
    model.eval()
    totals = {key: torch.zeros((len(text), tag_count)) for key, text in documents.items()}
    counts = {key: torch.zeros(len(text)) for key, text in documents.items()}
    for batch in loader:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        char_ids = batch["char_ids"].to(device)
        mapping = batch["char_to_token"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(ids, mask, char_ids, mapping).float().cpu()[0]
        start = batch["char_start"]
        end = start + logits.size(0)
        totals[batch["document_id"]][start:end] += logits
        counts[batch["document_id"]][start:end] += 1
    return totals, counts


def predictions_from_scores(totals, counts, documents, tag_names, o_bias):
    result = {}
    for document_id, text in documents.items():
        scores = totals[document_id] / counts[document_id].clamp_min(1).unsqueeze(1)
        scores[:, 0] -= o_bias
        result[document_id] = decode(scores.argmax(1).tolist(), tag_names)
    return result


def run(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = download_data(output / "data")

    from tokenizers import Tokenizer
    tokenizer_name = "legal_bpe_24k.json" if args.model == "foundation" else "phase1_bpe.json"
    checkpoint_name = "foundation.pt" if args.model == "foundation" else "efficient.pt"
    tokenizer = Tokenizer.from_file(str(find_one(tokenizer_name)))
    checkpoint = safe_load(find_one(checkpoint_name))
    phase1_steps = checkpoint.get("total_steps", checkpoint.get("completed_steps", checkpoint.get("step")))
    if args.model == "foundation":
        cfg = Config(**checkpoint["config"])
        backbone = LegalEncoderDecoder(cfg)
    else:
        cfg = RunConfig(**checkpoint["config"])
        backbone = EfficientLM(cfg)
    backbone.load_state_dict(checkpoint["model"], strict=True)
    if args.model == "foundation":
        for parameter in backbone.decoder.parameters():
            parameter.requires_grad = False
        for parameter in backbone.decoder_norm.parameters():
            parameter.requires_grad = False
    else:
        for block in backbone.blocks:
            block.attention.bidirectional = True

    label_names = load_labels(paths["labels.txt"])
    tag_names, tag_to_id = tags(label_names)
    train_text = texts(read_json(paths["fold1-text.json"]))
    train_gold = gold(read_json(paths["fold1-gold.json"]))
    valid_text = texts(read_json(paths["fold2-text.json"]))
    valid_gold = gold(read_json(paths["fold2-gold.json"]))
    test_text = texts(read_json(paths["fold3-text.json"]))

    train_set = CharacterWindows(train_text, tokenizer, args.max_tokens, args.stride, train_gold, tag_to_id)
    valid_set = CharacterWindows(valid_text, tokenizer, args.max_tokens, args.stride, valid_gold, tag_to_id)
    test_set = CharacterWindows(test_text, tokenizer, args.max_tokens, args.stride)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate, num_workers=2)

    model = CharacterNER(backbone, len(tag_names), args.model).to(device)
    head = list(model.head_parameters())
    head_ids = {id(parameter) for parameter in head}
    encoder = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(json.dumps({
        "architecture": f"{args.model}_encoder_plus_exact_character_cnn",
        "phase1_steps": phase1_steps,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "train_windows": len(train_set),
        "validation_windows": len(valid_set),
        "test_windows": len(test_set),
        "device": str(device),
    }, indent=2))
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.backbone_lr},
        {"params": head, "lr": args.head_lr},
    ], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    counts = Counter(tag for item in train_set.items for tag in item["labels"])
    maximum = max(counts.values())
    weights = torch.tensor([
        0.03 if index == 0 else min(10.0, math.sqrt(maximum / max(1, counts[index])))
        for index in range(len(tag_names))
    ], device=device)

    best_score = -1.0
    best_path = output / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader, 1):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            char_ids = batch["char_ids"].to(device)
            mapping = batch["char_to_token"].to(device)
            targets = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(ids, mask, char_ids, mapping)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), weight=weights)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder + head, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) * args.grad_accum)

        totals, window_counts = score_documents(model, valid_loader, valid_text, len(tag_names), device)
        score, details, selected_bias = -1.0, None, 0.0
        for bias in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            predicted = predictions_from_scores(totals, window_counts, valid_text, tag_names, bias)
            candidate, candidate_details = macro_f1(predicted, valid_gold, label_names)
            if candidate > score:
                score, details, selected_bias = candidate, candidate_details, bias
        row = {"epoch": epoch, "loss": sum(losses) / len(losses), "strict_macro_f1": score, "o_bias": selected_bias}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "score": score, "o_bias": selected_bias, "per_label": details}, best_path)

    saved = safe_load(best_path)
    model.load_state_dict(saved["model"], strict=True)
    totals, window_counts = score_documents(model, test_loader, test_text, len(tag_names), device)
    predicted = predictions_from_scores(totals, window_counts, test_text, tag_names, saved["o_bias"])
    prediction_path = output / "prediction.json"
    prediction_path.write_text(json.dumps(predicted, ensure_ascii=False), encoding="utf-8")
    archive_path = output / f"submission_phase2_{args.model}_character.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname="prediction.json")
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["prediction.json"]
    (output / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "validation_metrics.json").write_text(json.dumps({
        "strict_macro_f1": saved["score"], "o_bias": saved["o_bias"], "per_label": saved["per_label"],
    }, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "architecture": f"{args.model}_encoder_plus_exact_character_cnn",
        "phase1_config": asdict(cfg),
        "phase1_steps": phase1_steps,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "backbone_learning_rate": args.backbone_lr,
        "head_learning_rate": args.head_lr,
        "epochs": args.epochs,
        "max_tokens": args.max_tokens,
        "stride": args.stride,
        "seed": args.seed,
        "train_windows": len(train_set),
        "validation_windows": len(valid_set),
        "test_windows": len(test_set),
        "best_validation_f1": saved["score"],
        "submission": str(archive_path),
    }, indent=2), encoding="utf-8")
    print("UPLOAD:", archive_path)


def self_test():
    names, mapping = tags(["COURT"])
    labels = character_labels("xHigh Court!", [(1, 11, "COURT")], mapping)
    assert decode(labels, names) == [[1, 11, "COURT"]]
    print("Exact-character self-check passed")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("foundation", "efficient"), default="foundation")
    parser.add_argument("--output-dir", default="/kaggle/working/phase2_foundation_character")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=230338)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    if args.self_test:
        self_test()
    else:
        run(args)
