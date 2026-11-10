"""CS781 Phase 2 legal NER training, validation, and submission pipeline."""

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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from phase1_pipeline import (
    DecoderOnlyLM,
    EfficientLM,
    EncoderDecoderLM,
    HybridLM,
    RunConfig,
    build_model,
    causal_mask,
    sinusoidal_positions,
)
from phase1_foundation import Config as FoundationConfig, LegalEncoderDecoder


REPO_ID = "Exploration-Lab/CS-781-Capstone"
DATA_FILES = (
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
ARTIFACT_NAMES = {
    "foundation": None,
    "decoder": "phase1_improved_decoder_artifacts.zip",
    "encoder_decoder": "phase1_ed_baseline_artifacts.zip",
    "hybrid": "phase1_hybrid_baseline_artifacts.zip",
    "efficient": "phase1_improved_efficient_artifacts.zip",
}
DEFAULT_METHOD = {
    "foundation": "full",
    "decoder": "distill",
    "encoder_decoder": "full",
    "hybrid": "partial",
    "efficient": "lorma",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_unique(filename: str, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    roots = [Path("/kaggle/input"), Path(__file__).resolve().parent / "inputs"]
    matches = [path for root in roots if root.exists() for path in root.rglob(filename)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename}; found {matches}")
    return matches[0]


def extract_phase1_artifact(archive: Path, model_name: str, destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"{model_name}.pt"
    tokenizer_name = "phase1_bpe.json"
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        missing = {checkpoint_name, tokenizer_name} - names
        if missing:
            raise ValueError(f"{archive.name} is missing {sorted(missing)}")
        checkpoint = destination / checkpoint_name
        tokenizer = destination / tokenizer_name
        checkpoint.write_bytes(bundle.read(checkpoint_name))
        tokenizer.write_bytes(bundle.read(tokenizer_name))
    return checkpoint, tokenizer


def locate_phase1_files(model_name: str, artifact_zip: str | None, destination: Path) -> tuple[Path, Path, str]:
    """Accept either the original ZIP or Kaggle's automatically unpacked dataset."""
    archive_name = ARTIFACT_NAMES[model_name]
    try:
        archive = find_unique(archive_name, artifact_zip)
    except FileNotFoundError:
        if artifact_zip:
            raise
        checkpoint = find_unique(f"{model_name}.pt")
        tokenizers = list(checkpoint.parent.rglob("phase1_bpe.json"))
        if len(tokenizers) != 1:
            raise FileNotFoundError(
                f"Found {checkpoint}, but expected one phase1_bpe.json beside it; found {tokenizers}"
            )
        return checkpoint, tokenizers[0], f"{checkpoint.parent.name} (Kaggle-unpacked)"
    checkpoint, tokenizer = extract_phase1_artifact(archive, model_name, destination)
    return checkpoint, tokenizer, archive.name


def download_data(destination: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    destination.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add a Kaggle secret named HF and load it first.")
    paths = {}
    for filename in DATA_FILES:
        downloaded = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=token,
        ))
        target = destination / Path(filename).name
        target.write_bytes(downloaded.read_bytes())
        paths[Path(filename).name] = target
    return paths


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def normalize_texts(raw) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise TypeError("Text JSON must be an object keyed by document ID")
    result = {}
    for document_id, value in raw.items():
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict) and isinstance(value.get("text"), str):
            text = value["text"]
        else:
            raise TypeError(f"Unsupported text entry for {document_id}: {type(value).__name__}")
        result[str(document_id)] = text
    return result


def normalize_entity(raw) -> tuple[int, int, str, str | None]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        start, end, label = raw[:3]
        entity_text = raw[3] if len(raw) > 3 and isinstance(raw[3], str) else None
    elif isinstance(raw, dict):
        start = raw.get("start", raw.get("start_offset"))
        end = raw.get("end", raw.get("end_offset"))
        label = raw.get("label", raw.get("type"))
        entity_text = raw.get("text", raw.get("entity_text"))
    else:
        raise TypeError(f"Unsupported entity annotation: {raw!r}")
    return int(start), int(end), str(label), entity_text


def normalize_gold(raw) -> dict[str, list[tuple[int, int, str]]]:
    if not isinstance(raw, dict):
        raise TypeError("Gold JSON must be an object keyed by document ID")
    result = {}
    for document_id, value in raw.items():
        if isinstance(value, dict):
            value = value.get("entities", value.get("spans", value.get("annotations")))
        if not isinstance(value, list):
            raise TypeError(f"Unsupported gold entry for {document_id}")
        result[str(document_id)] = [(s, e, label) for s, e, label, _ in map(normalize_entity, value)]
    return result


def load_labels(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    labels = []
    for line in raw.splitlines():
        for chunk in line.split(","):
            if chunk.strip():
                candidate = chunk.strip().split()[0]
                if candidate in EXPECTED_LABELS:
                    labels.append(candidate)
    if set(labels) != EXPECTED_LABELS or len(labels) != 12:
        raise ValueError(f"Unexpected labels: {labels}")
    return labels


def validate_gold(texts: dict[str, str], gold: dict[str, list[tuple[int, int, str]]], labels: list[str]) -> None:
    if set(texts) != set(gold):
        raise ValueError(f"Text/gold ID mismatch: text-only={set(texts)-set(gold)}, gold-only={set(gold)-set(texts)}")
    allowed = set(labels)
    overlaps = 0
    for document_id, text in texts.items():
        previous_end = -1
        for start, end, label in sorted(gold[document_id]):
            if not 0 <= start < end <= len(text):
                raise ValueError(f"Bad span {document_id}: {(start, end, label)} for length {len(text)}")
            if label not in allowed:
                raise ValueError(f"Unknown label {label!r} in {document_id}")
            overlaps += start < previous_end
            previous_end = max(previous_end, end)
    if overlaps:
        raise ValueError(f"BIO cannot represent the {overlaps} overlapping gold spans; use a span head")


def make_tags(labels: list[str]) -> tuple[list[str], dict[str, int]]:
    tags = ["O"] + [tag for label in labels for tag in (f"B-{label}", f"I-{label}")]
    return tags, {tag: index for index, tag in enumerate(tags)}


def encode_document(tokenizer, text: str):
    encoding = tokenizer.encode(text)
    if len(encoding.ids) != len(encoding.offsets):
        raise RuntimeError("Tokenizer returned inconsistent IDs and offsets")
    boundary = set(" \t\r\n.,;:!?()[]{}\\\"“”‘’")
    offsets = []
    for left, right in encoding.offsets:
        left, right = int(left), int(right)
        while left < right and text[left] in boundary:
            left += 1
        while right > left and text[right - 1] in boundary:
            right -= 1
        offsets.append((left, right))
    return list(encoding.ids), offsets


def align_tags(
    offsets: list[tuple[int, int]],
    entities: list[tuple[int, int, str]],
    tag_to_id: dict[str, int],
) -> list[int]:
    tags = [tag_to_id["O"]] * len(offsets)
    for start, end, label in sorted(entities):
        token_indexes = [i for i, (left, right) in enumerate(offsets) if right > start and left < end]
        if not token_indexes:
            raise ValueError(f"No tokens cover entity {(start, end, label)}")
        token_indexes = [index for index in token_indexes if tags[index] == tag_to_id["O"]]
        if not token_indexes:
            continue
        first = token_indexes[0]
        tags[first] = tag_to_id[f"B-{label}"]
        for index in token_indexes[1:]:
            tags[index] = tag_to_id[f"I-{label}"]
    return tags


class WindowDataset(Dataset):
    def __init__(
        self,
        texts: dict[str, str],
        tokenizer,
        max_length: int,
        stride: int,
        gold: dict[str, list[tuple[int, int, str]]] | None = None,
        tag_to_id: dict[str, int] | None = None,
    ):
        if not 0 <= stride < max_length:
            raise ValueError("stride must satisfy 0 <= stride < max_length")
        self.documents = {}
        self.windows = []
        step = max_length - stride
        for document_id, text in texts.items():
            ids, offsets = encode_document(tokenizer, text)
            labels = align_tags(offsets, gold[document_id], tag_to_id) if gold is not None else None
            self.documents[document_id] = {"text": text, "ids": ids, "offsets": offsets}
            if not ids:
                continue
            for start in range(0, len(ids), step):
                end = min(start + max_length, len(ids))
                self.windows.append({
                    "document_id": document_id,
                    "start": start,
                    "ids": ids[start:end],
                    "labels": labels[start:end] if labels is not None else None,
                })
                if end == len(ids):
                    break

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


def make_collate(pad_id: int):
    def collate(batch):
        length = max(len(item["ids"]) for item in batch)
        ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), length), dtype=torch.bool)
        has_labels = batch[0]["labels"] is not None
        labels = torch.full((len(batch), length), -100, dtype=torch.long) if has_labels else None
        for row, item in enumerate(batch):
            size = len(item["ids"])
            ids[row, :size] = torch.tensor(item["ids"], dtype=torch.long)
            mask[row, :size] = True
            if has_labels:
                labels[row, :size] = torch.tensor(item["labels"], dtype=torch.long)
        return {
            "ids": ids,
            "mask": mask,
            "labels": labels,
            "document_ids": [item["document_id"] for item in batch],
            "starts": [item["start"] for item in batch],
        }
    return collate


class LoRMAPlusLinear(nn.Module):
    """LoRMA+ output-space transform: (I + scale * BA) W0 x."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.out_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        frozen = self.base(inputs)
        return frozen + self.scale * self.b(self.a(frozen))


class LNERModel(nn.Module):
    def __init__(self, backbone: nn.Module, model_name: str, tag_count: int):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(backbone.cfg.d_model, tag_count)

    def hidden(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cfg = self.backbone.cfg
        if self.model_name == "foundation":
            return self.backbone.encode(ids, mask)
        if self.model_name in {"encoder_decoder", "hybrid"}:
            embedded = self.backbone.embedding(ids) + sinusoidal_positions(ids.size(1), cfg.d_model, ids.device)
            if self.model_name == "encoder_decoder":
                memory = self.backbone.encoder(embedded, src_key_padding_mask=~mask)
            else:
                lengths = mask.sum(1).cpu()
                packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
                memory, _ = self.backbone.recurrent(packed)
                memory, _ = pad_packed_sequence(memory, batch_first=True, total_length=ids.size(1))
            target = self.backbone.embedding(ids) + sinusoidal_positions(ids.size(1), cfg.d_model, ids.device)
            decoded = self.backbone.decoder(
                target,
                memory,
                tgt_mask=causal_mask(ids.size(1), ids.device),
                tgt_key_padding_mask=~mask,
                memory_key_padding_mask=~mask,
            )
            denominator = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
            summary = (memory * mask.unsqueeze(-1)).sum(1, keepdim=True) / denominator
            decoded = self.backbone.fusion(decoded, summary.expand_as(decoded))
            return self.backbone.norm(decoded)
        if self.model_name == "decoder":
            hidden = self.backbone.embedding(ids) + sinusoidal_positions(ids.size(1), cfg.d_model, ids.device)
            hidden = self.backbone.blocks(
                hidden,
                mask=causal_mask(ids.size(1), ids.device),
                src_key_padding_mask=~mask,
            )
            counts = mask.cumsum(1).clamp_min(1).unsqueeze(-1)
            summary = (hidden * mask.unsqueeze(-1)).cumsum(1) / counts
            return self.backbone.norm(self.backbone.fusion(hidden, summary))
        if self.model_name == "efficient":
            hidden = self.backbone.embedding(ids)
            for block in self.backbone.blocks:
                hidden = block(hidden)
            return self.backbone.norm(hidden)
        raise ValueError(self.model_name)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.hidden(ids, mask)))


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raise RuntimeError("This notebook requires a PyTorch version supporting torch.load(..., weights_only=True)")


def load_phase1(checkpoint: Path, expected_name: str):
    saved = safe_torch_load(checkpoint)
    if saved.get("model_name") not in (None, expected_name):
        raise ValueError(f"Checkpoint is {saved.get('model_name')}, expected {expected_name}")
    cfg = RunConfig(**saved["config"])
    model = build_model(expected_name, cfg)
    model.load_state_dict(saved["model"], strict=True)
    return model, cfg, saved


def load_foundation(checkpoint: Path):
    saved = safe_torch_load(checkpoint)
    cfg = FoundationConfig(**saved["config"])
    model = LegalEncoderDecoder(cfg)
    model.load_state_dict(saved["model"], strict=True)
    return model, cfg, saved


def make_student(full_decoder: DecoderOnlyLM, layers: int = 8) -> DecoderOnlyLM:
    if layers >= len(full_decoder.blocks.layers):
        raise ValueError("Student must have fewer layers than teacher/base")
    student = DecoderOnlyLM(full_decoder.cfg, layers=layers)
    student.embedding.load_state_dict(full_decoder.embedding.state_dict())
    student.fusion.load_state_dict(full_decoder.fusion.state_dict())
    student.norm.load_state_dict(full_decoder.norm.state_dict())
    indexes = torch.linspace(0, len(full_decoder.blocks.layers) - 1, layers).round().int().tolist()
    for target, source_index in zip(student.blocks.layers, indexes):
        target.load_state_dict(full_decoder.blocks.layers[source_index].state_dict())
    student.head.weight = student.embedding.weight
    return student


def apply_method(model: LNERModel, method: str, rank: int) -> None:
    if method in {"full", "distill"}:
        return
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if method == "partial":
        if model.model_name != "hybrid":
            raise ValueError("The provided partial configuration is for the hybrid model")
        for name, parameter in model.backbone.recurrent.named_parameters():
            if "_l4" in name:
                parameter.requires_grad = True
        for layer in model.backbone.decoder.layers[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for module in (model.backbone.fusion, model.backbone.norm):
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif method == "lorma":
        if model.model_name == "efficient":
            for block in model.backbone.blocks:
                block.attention.q = LoRMAPlusLinear(block.attention.q, rank=rank, alpha=float(rank))
                block.attention.v = LoRMAPlusLinear(block.attention.v, rank=rank, alpha=float(rank))
            for parameter in model.backbone.norm.parameters():
                parameter.requires_grad = True
        elif model.model_name == "hybrid":
            for layer in model.backbone.decoder.layers:
                layer.linear1 = LoRMAPlusLinear(layer.linear1, rank=rank, alpha=float(rank))
                layer.linear2 = LoRMAPlusLinear(layer.linear2, rank=rank, alpha=float(rank))
            for module in (model.backbone.fusion, model.backbone.norm):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        else:
            raise ValueError("LoRMA is configured for the efficient or hybrid model")
    else:
        raise ValueError(method)


def decode_tags(tags: list[int], offsets: list[tuple[int, int]], id_to_tag: list[str]):
    spans = []
    active_label = None
    active_start = None
    active_end = None

    def close():
        nonlocal active_label, active_start, active_end
        if active_label is not None and active_start is not None and active_end is not None:
            spans.append([int(active_start), int(active_end), active_label])
        active_label = active_start = active_end = None

    for tag_id, (start, end) in zip(tags, offsets):
        tag = id_to_tag[tag_id]
        if tag == "O" or end <= start:
            close()
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or label != active_label:
            close()
            active_label, active_start, active_end = label, start, end
        else:
            active_end = end
    close()
    return spans


def decode_score_maps(dataset, totals, counts, id_to_tag, o_bias=0.0):
    predictions = {}
    for document_id, record in dataset.documents.items():
        if len(record["ids"]) == 0:
            predictions[document_id] = []
            continue
        if torch.any(counts[document_id] == 0):
            raise RuntimeError(f"Uncovered tokens in {document_id}")
        scores = totals[document_id] / counts[document_id].unsqueeze(1)
        scores[:, 0] -= o_bias
        tags = scores.argmax(1).tolist()
        predictions[document_id] = decode_tags(tags, record["offsets"], id_to_tag)
    return predictions


@torch.inference_mode()
def predict(
    model, dataset: WindowDataset, loader: DataLoader, id_to_tag: list[str],
    device: torch.device, o_bias: float = 0.0, return_scores: bool = False,
):
    model.eval()
    tag_count = len(id_to_tag)
    totals = {
        document_id: torch.zeros((len(record["ids"]), tag_count), dtype=torch.float32)
        for document_id, record in dataset.documents.items()
    }
    counts = {
        document_id: torch.zeros(len(record["ids"]), dtype=torch.float32)
        for document_id, record in dataset.documents.items()
    }
    for batch in loader:
        ids, mask = batch["ids"].to(device), batch["mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(ids, mask).float().cpu()
        for row, (document_id, start) in enumerate(zip(batch["document_ids"], batch["starts"])):
            size = int(batch["mask"][row].sum())
            totals[document_id][start:start + size] += logits[row, :size]
            counts[document_id][start:start + size] += 1
    if return_scores:
        return totals, counts
    return decode_score_maps(dataset, totals, counts, id_to_tag, o_bias)


def strict_macro_f1(predictions, gold, labels):
    per_label = {}
    for label in labels:
        predicted = {
            (document_id, int(span[0]), int(span[1]))
            for document_id, spans in predictions.items()
            for span in spans if span[2] == label
        }
        actual = {
            (document_id, int(start), int(end))
            for document_id, spans in gold.items()
            for start, end, entity_label in spans if entity_label == label
        }
        tp = len(predicted & actual)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(actual) if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": len(actual)}
    return sum(row["f1"] for row in per_label.values()) / len(labels), per_label


def class_weights(dataset: WindowDataset, tag_count: int) -> torch.Tensor:
    counts = Counter(tag for window in dataset.windows for tag in (window["labels"] or []))
    maximum = max(counts.values())
    weights = torch.ones(tag_count)
    for tag in range(tag_count):
        weights[tag] = min(8.0, math.sqrt(maximum / max(1, counts[tag])))
    weights[0] = 0.05
    return weights


def build_teacher(path: Path, tag_count: int, device: torch.device):
    saved = safe_torch_load(path)
    config = saved["run_config"]
    if config["method"] != "full":
        raise ValueError("Use a full-fine-tuning Phase 2 checkpoint as the distillation teacher")
    if config["model_name"] == "foundation":
        cfg = FoundationConfig(**saved["phase1_config"])
        backbone = LegalEncoderDecoder(cfg)
    else:
        cfg = RunConfig(**saved["phase1_config"])
        backbone = build_model(config["model_name"], cfg)
    teacher = LNERModel(backbone, config["model_name"], tag_count)
    teacher.load_state_dict(saved["model"], strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def train(
    model,
    teacher,
    train_loader,
    validation_dataset,
    validation_loader,
    validation_gold,
    labels,
    id_to_tag,
    device,
    args,
    output_dir,
    phase1_config,
):
    weights = class_weights(train_loader.dataset, len(id_to_tag)).to(device)
    classifier = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    classifier_ids = {id(parameter) for parameter in classifier}
    backbone = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    trainable = backbone + classifier
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.learning_rate},
        {"params": classifier, "lr": max(5e-4, args.learning_rate * 25)},
    ], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_score = -1.0
    best_path = output_dir / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader, 1):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(ids, mask)
                hard_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), weight=weights, ignore_index=-100)
                if teacher is None:
                    loss = hard_loss
                else:
                    with torch.no_grad():
                        teacher_logits = teacher(ids, mask)
                    valid = targets != -100
                    temperature = args.temperature
                    soft_loss = F.kl_div(
                        F.log_softmax(logits[valid] / temperature, dim=-1),
                        F.softmax(teacher_logits[valid] / temperature, dim=-1),
                        reduction="batchmean",
                    ) * temperature * temperature
                    loss = 0.5 * hard_loss + 0.5 * soft_loss
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss) * args.grad_accum)
        totals, counts = predict(
            model, validation_dataset, validation_loader, id_to_tag, device,
            return_scores=True,
        )
        score, per_label, selected_bias = -1.0, None, 0.0
        for candidate_bias in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            predictions = decode_score_maps(
                validation_dataset, totals, counts, id_to_tag,
                o_bias=candidate_bias,
            )
            candidate_score, candidate_per_label = strict_macro_f1(predictions, validation_gold, labels)
            if candidate_score > score:
                score, per_label, selected_bias = candidate_score, candidate_per_label, candidate_bias
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "strict_macro_f1": score,
            "o_bias": selected_bias,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            torch.save({
                "model": model.state_dict(),
                "phase1_config": asdict(phase1_config),
                "run_config": {
                    "model_name": args.model,
                    "method": args.method,
                    "max_length": args.max_length,
                    "stride": args.stride,
                    "rank": args.rank,
                },
                "labels": labels,
                "tags": id_to_tag,
                "score": score,
                "per_label": per_label,
                "o_bias": selected_bias,
            }, best_path)
    saved = safe_torch_load(best_path)
    model.load_state_dict(saved["model"], strict=True)
    (output_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "validation_metrics.json").write_text(json.dumps({
        "strict_macro_f1": saved["score"], "per_label": saved["per_label"]
    }, indent=2), encoding="utf-8")
    return best_path


def validate_and_zip(predictions, test_texts, labels, output_dir):
    if set(predictions) != set(test_texts):
        raise ValueError("Prediction IDs do not exactly match test IDs")
    allowed = set(labels)
    cleaned = {}
    for document_id, spans in predictions.items():
        text_length = len(test_texts[document_id])
        unique = set()
        for span in spans:
            if len(span) != 3:
                raise ValueError(f"Bad prediction: {span}")
            start, end, label = int(span[0]), int(span[1]), str(span[2])
            if not 0 <= start < end <= text_length or label not in allowed:
                raise ValueError(f"Invalid prediction for {document_id}: {span}")
            unique.add((start, end, label))
        cleaned[document_id] = [list(span) for span in sorted(unique)]
    prediction_path = output_dir / "prediction.json"
    prediction_path.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
    archive_path = output_dir / f"submission_{output_dir.name}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname="prediction.json")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["prediction.json"]:
            raise RuntimeError("Submission ZIP has the wrong layout")
        json.loads(archive.read("prediction.json"))
    return archive_path


def self_check() -> None:
    tags, tag_to_id = make_tags(["COURT"])
    decoded = decode_tags(
        [tag_to_id["B-COURT"], tag_to_id["I-COURT"], tag_to_id["O"]],
        [(0, 5), (5, 10), (11, 12)],
        tags,
    )
    assert decoded == [[0, 10, "COURT"]]
    score, _ = strict_macro_f1({"d": decoded}, {"d": [(0, 10, "COURT")]}, ["COURT"])
    assert score == 1.0
    assert align_tags([(136, 153)], [(137, 153, "COURT")], tag_to_id) == [tag_to_id["B-COURT"]]
    assert align_tags([(7439, 7501)], [(7439, 7500, "COURT")], tag_to_id) == [tag_to_id["B-COURT"]]
    print("Phase 2 helper self-check passed")


def run(args):
    seed_everything(args.seed)
    args.method = args.method or DEFAULT_METHOD[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir or f"/kaggle/working/phase2_{args.model}_{args.method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "foundation":
        phase1_checkpoint = find_unique("foundation.pt", args.phase1_checkpoint)
        tokenizer_path = find_unique("legal_bpe_24k.json", args.phase1_tokenizer)
        artifact_label = phase1_checkpoint.name
    else:
        phase1_checkpoint, tokenizer_path, artifact_label = locate_phase1_files(
            args.model, args.artifact_zip, output_dir / "phase1"
        )
    data_paths = download_data(output_dir / "data")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Phase 1 tokenizer has no <pad> token")

    labels = load_labels(data_paths["labels.txt"])
    tags, tag_to_id = make_tags(labels)
    train_texts = normalize_texts(read_json(data_paths["fold1-text.json"]))
    train_gold = normalize_gold(read_json(data_paths["fold1-gold.json"]))
    validation_texts = normalize_texts(read_json(data_paths["fold2-text.json"]))
    validation_gold = normalize_gold(read_json(data_paths["fold2-gold.json"]))
    test_texts = normalize_texts(read_json(data_paths["fold3-text.json"]))
    validate_gold(train_texts, train_gold, labels)
    validate_gold(validation_texts, validation_gold, labels)

    train_dataset = WindowDataset(train_texts, tokenizer, args.max_length, args.stride, train_gold, tag_to_id)
    validation_dataset = WindowDataset(validation_texts, tokenizer, args.max_length, args.stride, validation_gold, tag_to_id)
    test_dataset = WindowDataset(test_texts, tokenizer, args.max_length, args.stride)
    collate = make_collate(pad_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)

    if args.model == "foundation":
        backbone, phase1_config, phase1_saved = load_foundation(phase1_checkpoint)
    else:
        backbone, phase1_config, phase1_saved = load_phase1(phase1_checkpoint, args.model)
    if args.model == "efficient" and args.bidirectional_efficient:
        for block in backbone.blocks:
            block.attention.bidirectional = True
    if args.method == "distill":
        if args.model != "decoder":
            raise ValueError("The provided student recipe uses the decoder Phase 1 model")
        backbone = make_student(backbone, layers=args.student_layers)
    model = LNERModel(backbone, args.model, len(tags))
    if args.model == "foundation":
        for parameter in model.backbone.decoder.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.decoder_norm.parameters():
            parameter.requires_grad = False
    apply_method(model, args.method, args.rank)
    model.to(device)

    teacher = None
    if args.method == "distill":
        if not args.teacher_checkpoint:
            raise ValueError("--teacher-checkpoint is required for distillation")
        teacher = build_teacher(Path(args.teacher_checkpoint), len(tags), device)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(json.dumps({
        "artifact": artifact_label,
        "model": args.model,
        "method": args.method,
        "phase1_steps": phase1_saved.get("total_steps"),
        "parameters": total,
        "trainable_parameters": trainable,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "test_windows": len(test_dataset),
        "device": str(device),
    }, indent=2))

    best_path = train(
        model, teacher, train_loader, validation_dataset, validation_loader,
        validation_gold, labels, tags, device, args, output_dir, phase1_config,
    )
    selected = safe_torch_load(best_path)
    predictions = predict(
        model, test_dataset, test_loader, tags, device,
        o_bias=float(selected.get("o_bias", 0.0)),
    )
    submission = validate_and_zip(predictions, test_texts, labels, output_dir)
    manifest = {
        "model": args.model,
        "method": args.method,
        "phase1_artifact": artifact_label,
        "phase1_steps": phase1_saved.get("total_steps"),
        "parameters": total,
        "trainable_parameters": trainable,
        "checkpoint": str(best_path),
        "submission": str(submission),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(ARTIFACT_NAMES))
    parser.add_argument("--method", choices=("full", "partial", "lorma", "distill"))
    parser.add_argument("--artifact-zip")
    parser.add_argument("--phase1-checkpoint")
    parser.add_argument("--phase1-tokenizer")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--student-layers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--bidirectional-efficient", action="store_true")
    parser.add_argument("--seed", type=int, default=230338)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_check()
    elif not arguments.model:
        raise SystemExit("--model is required unless --self-test is used")
    else:
        run(arguments)
