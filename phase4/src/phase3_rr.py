"""CS781 Phase 3 - Rhetorical Role (RR) prediction from Phase 1 legal language models.

Every experiment starts from a Phase 1 checkpoint (decoder / encoder_decoder / hybrid / efficient).
A sentence-level head is added on top of a pooled representation of the *target* sentence; the
target can be presented alone (``--context none``) or after its neighbouring sentences
(``--context window``).  Adaptation techniques: full fine-tuning, LORMA+, LoRA, IA3 and
knowledge distillation (truncated decoder student trained on a teacher's soft labels).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from phase2_lner import (
    ARTIFACT_NAMES,
    LNERModel,
    LoRMAPlusLinear,
    extract_phase1_artifact,
    load_phase1,
    make_student,
    safe_torch_load,
)


def walk_files(roots):
    """All files under the roots, following symlinks (Kaggle mounts inputs through symlinks)."""
    files = []
    for root in roots:
        if root and os.path.exists(root):
            for folder, _, names in os.walk(root, followlinks=True):
                files += [Path(folder) / name for name in names]
    return files


def show_inputs(root="/kaggle/input"):
    print(f"Files under {root}:")
    for path in walk_files([root])[:300]:
        try:
            print("  ", path, path.stat().st_size)
        except OSError:
            print("  ", path)


def locate_phase1_files(model_name: str, artifact_zip, destination: Path):
    """Find the Phase 1 checkpoint + tokenizer as a ZIP or as Kaggle's auto-extracted folder."""
    if artifact_zip:
        archive = Path(artifact_zip)
        checkpoint, tokenizer = extract_phase1_artifact(archive, model_name, destination)
        return checkpoint, tokenizer, archive.name
    files = walk_files(["/kaggle/input", str(Path(__file__).resolve().parent / "inputs")])
    archive_name = ARTIFACT_NAMES[model_name]
    zips = [f for f in files if f.name == archive_name]
    if not zips:  # tolerate renamed uploads: any zip that contains <model>.pt
        for f in files:
            if f.suffix == ".zip":
                try:
                    with zipfile.ZipFile(f) as bundle:
                        if f"{model_name}.pt" in bundle.namelist():
                            zips.append(f)
                except (zipfile.BadZipFile, OSError):
                    pass
    if zips:
        checkpoint, tokenizer = extract_phase1_artifact(zips[0], model_name, destination)
        return checkpoint, tokenizer, zips[0].name
    checkpoints = [f for f in files if f.name == f"{model_name}.pt"]
    for checkpoint in checkpoints:
        tokenizers = [f for f in files if f.name == "phase1_bpe.json" and f.parent == checkpoint.parent]
        if tokenizers:
            return checkpoint, tokenizers[0], f"{checkpoint.parent.name} (Kaggle-unpacked)"
    show_inputs()
    raise FileNotFoundError(
        f"Could not find {archive_name} or {model_name}.pt + phase1_bpe.json under /kaggle/input. "
        "Attach the dataset holding the Phase 1 artifact ZIP (right panel -> Add Input), then rerun this cell."
    )

REPO_ID = "Exploration-Lab/CS-781-Capstone"
TRAIN_FILES = ("CL_train.json", "CL_dev.json", "IT_train.json", "IT_dev.json", "label_vocab.json")
TEST_FILES = ("CL_test.json", "IT_test.json")
DOMAINS = ("CL", "IT")
POSITION_BUCKETS = 10
OFFICIAL_LABELS = [
    "Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent",
    "PrecedentReliedUpon", "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt",
    "RatioOfTheDecision", "RulingByPresentCourt", "None",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data

def obtain_data(destination: Path, data_dir: str | None) -> dict[str, Path]:
    """Use local copies (``--data-dir`` or an attached Kaggle dataset) or download from HF."""
    names = TRAIN_FILES + TEST_FILES
    roots = [Path(data_dir)] if data_dir else []
    roots.append(Path("/kaggle/input"))
    found = {}
    available = walk_files([str(root) for root in roots])
    for name in names:
        matches = sorted(f for f in available if f.name == name)
        if matches:
            found[name] = matches[0]
    missing = [name for name in names if name not in found]
    if missing:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is missing (add the Kaggle secret HF) and data was not found locally")
        destination.mkdir(parents=True, exist_ok=True)
        for name in missing:
            folder = "Train" if name in TRAIN_FILES else "Test"
            path = Path(hf_hub_download(
                repo_id=REPO_ID, filename=f"Phase_3_RR/{folder}/{name}", repo_type="dataset", token=token,
            ))
            target = destination / name
            target.write_bytes(path.read_bytes())
            found[name] = target
    return found


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _pick(record: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in record:
            return record[key]
    return None


def normalize_documents(raw, domain: str, require_labels: bool) -> list[dict]:
    """Accept a list of documents or an id->document mapping; keep the original order."""
    if isinstance(raw, dict):
        for wrapper in ("data", "documents", "docs"):
            if wrapper in raw and isinstance(raw[wrapper], (list, dict)):
                raw = raw[wrapper]
                break
    if isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            if isinstance(value, list):
                value = {"text": value}
            value = dict(value)
            value.setdefault("id", key)
            items.append(value)
    else:
        items = list(raw)
    documents = []
    for position, item in enumerate(items):
        doc_id = _pick(item, ("id", "doc_id", "document_id", "docid", "name"))
        if doc_id is None:
            raise ValueError(f"{domain} document #{position} has no id field: keys={list(item)}")
        sentences = _pick(item, ("text", "sentences", "sents", "sentence"))
        if not isinstance(sentences, list):
            raise ValueError(f"{domain}/{doc_id}: expected a list of sentences, got {type(sentences)}")
        sentences = [str(sentence) for sentence in sentences]
        labels = _pick(item, ("labels", "label", "roles", "rhetorical_roles"))
        if require_labels:
            if not isinstance(labels, list) or len(labels) != len(sentences):
                raise ValueError(f"{domain}/{doc_id}: labels missing or length mismatch")
            labels = [str(label) for label in labels]
        else:
            labels = None
        documents.append({"id": str(doc_id), "domain": domain, "sentences": sentences, "labels": labels})
    return documents


def load_label_vocab(path: Path) -> list[str]:
    raw = read_json(path)
    if isinstance(raw, dict):
        for wrapper in ("labels", "label_vocab", "vocab"):
            if wrapper in raw:
                raw = raw[wrapper]
                break
    if isinstance(raw, dict):
        if all(isinstance(value, int) for value in raw.values()):
            labels = [label for label, _ in sorted(raw.items(), key=lambda item: item[1])]
        else:
            labels = [raw[key] for key in sorted(raw, key=lambda key: int(key))]
    else:
        labels = list(raw)
    labels = [str(label) for label in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate labels in label_vocab.json")
    if set(labels) != set(OFFICIAL_LABELS):
        print("WARNING: label_vocab.json differs from the competition document list:",
              sorted(set(labels) ^ set(OFFICIAL_LABELS)), flush=True)
    return labels


def check_labels(documents: list[dict], labels: list[str]) -> None:
    allowed = set(labels)
    unknown = Counter(label for doc in documents for label in doc["labels"] if label not in allowed)
    if unknown:
        raise ValueError(f"Gold labels outside label_vocab.json: {dict(unknown)}")


# ----------------------------------------------------------------------------- examples

class SentenceDataset(Dataset):
    """One example per sentence; inputs are built from cached per-sentence token ids."""

    def __init__(self, documents, tokenizer, label_to_id, args, separator_id):
        self.documents = documents
        self.args = args
        self.separator_id = separator_id
        self.examples = []
        self.token_cache = []
        for doc_index, document in enumerate(documents):
            encodings = tokenizer.encode_batch(document["sentences"]) if document["sentences"] else []
            self.token_cache.append([list(encoding.ids) for encoding in encodings])
            total = len(document["sentences"])
            for sentence_index in range(total):
                label = None
                if document["labels"] is not None:
                    label = label_to_id[document["labels"][sentence_index]]
                self.examples.append({
                    "doc": doc_index,
                    "index": sentence_index,
                    "domain": DOMAINS.index(document["domain"]),
                    "bucket": min(POSITION_BUCKETS - 1, int(POSITION_BUCKETS * sentence_index / max(1, total))),
                    "label": label,
                    "key": f"{document['domain']}::{document['id']}::{sentence_index}",
                })

    def __len__(self):
        return len(self.examples)

    def build(self, doc_index: int, sentence_index: int) -> tuple[list[int], int]:
        """Layout: [previous context] SEP [next context] SEP target (target always last)."""
        args = self.args
        sentences = self.token_cache[doc_index]
        target = sentences[sentence_index][: args.target_tokens] or [self.separator_id]
        if args.context == "none":
            return target, len(target)
        budget = args.max_length - len(target) - 2
        previous, following = [], []
        for offset in range(1, args.context_sentences + 1):
            for side in ("prev", "next"):
                position = sentence_index - offset if side == "prev" else sentence_index + offset
                if not 0 <= position < len(sentences) or budget <= 0:
                    continue
                piece = sentences[position][: min(args.context_tokens, budget)]
                budget -= len(piece)
                if side == "prev":
                    previous.insert(0, piece)
                else:
                    following.append(piece)
        ids = [token for piece in previous for token in piece] + [self.separator_id]
        ids += [token for piece in following for token in piece] + [self.separator_id]
        ids += target
        return ids, len(target)

    def __getitem__(self, item):
        example = self.examples[item]
        ids, target_length = self.build(example["doc"], example["index"])
        return {**example, "ids": ids, "target_length": target_length, "row": item}


def make_collate(pad_id: int):
    def collate(batch):
        length = max(len(item["ids"]) for item in batch)
        ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), length), dtype=torch.bool)
        target = torch.zeros((len(batch), length), dtype=torch.bool)
        for row, item in enumerate(batch):
            size = len(item["ids"])
            ids[row, :size] = torch.tensor(item["ids"], dtype=torch.long)
            mask[row, :size] = True
            target[row, size - item["target_length"]: size] = True
        labels = None
        if batch[0]["label"] is not None:
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {
            "ids": ids,
            "mask": mask,
            "target": target,
            "last": torch.tensor([len(item["ids"]) - 1 for item in batch], dtype=torch.long),
            "domain": torch.tensor([item["domain"] for item in batch], dtype=torch.long),
            "bucket": torch.tensor([item["bucket"] for item in batch], dtype=torch.long),
            "labels": labels,
            "rows": [item["row"] for item in batch],
            "keys": [item["key"] for item in batch],
        }
    return collate


# ----------------------------------------------------------------------------- model

class RRModel(nn.Module):
    """Phase 1 backbone -> [mean(target tokens); last target token; position; domain] -> MLP."""

    def __init__(self, backbone: nn.Module, model_name: str, label_count: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = LNERModel(backbone, model_name, 1)
        del self.encoder.classifier
        dim = backbone.cfg.d_model
        self.position = nn.Embedding(POSITION_BUCKETS, 32)
        self.domain = nn.Embedding(len(DOMAINS), 16)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim + 48),
            nn.Dropout(dropout),
            nn.Linear(2 * dim + 48, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, label_count),
        )

    def head_parameters(self):
        return [*self.position.parameters(), *self.domain.parameters(), *self.head.parameters()]

    def forward(self, batch):
        hidden = self.encoder.hidden(batch["ids"], batch["mask"]).float()
        weights = batch["target"].unsqueeze(-1).float()
        mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        last = hidden[torch.arange(hidden.size(0), device=hidden.device), batch["last"]]
        features = torch.cat((mean, last, self.position(batch["bucket"]), self.domain(batch["domain"])), dim=-1)
        return self.head(features)


class LoRALinear(nn.Module):
    """LoRA: W0 x + (alpha / r) B A x, with W0 frozen."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        return self.base(inputs) + self.scale * self.b(self.a(self.dropout(inputs)))


class IA3Linear(nn.Module):
    """IA3: learned element-wise rescaling l * (W0 x) of a frozen projection."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.scale = nn.Parameter(torch.ones(base.out_features))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        return self.base(inputs) * self.scale


class LoRMAWrapped(LoRMAPlusLinear):
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def transformer_layers(backbone: nn.Module, model_name: str):
    if model_name == "decoder":
        return list(backbone.blocks.layers)
    if model_name == "encoder_decoder":
        return list(backbone.encoder.layers) + list(backbone.decoder.layers)
    if model_name == "hybrid":
        return list(backbone.decoder.layers)
    raise ValueError(model_name)


def apply_adapter(model: RRModel, model_name: str, method: str, rank: int) -> None:
    if method in {"full", "distill"}:
        return
    backbone = model.encoder.backbone
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    def wrap(linear: nn.Linear, role: str) -> nn.Module:
        if method == "lorma":
            return LoRMAWrapped(linear, rank=rank, alpha=float(rank))
        if method == "lora":
            return LoRALinear(linear, rank=rank, alpha=2.0 * rank)
        if method == "ia3":
            return IA3Linear(linear)
        raise ValueError(method)

    if model_name == "efficient":
        for block in backbone.blocks:
            attention = block.attention
            if method == "ia3":       # IA3 rescales keys, values and the FFN intermediate activation
                attention.k, attention.v, block.up = wrap(attention.k, "k"), wrap(attention.v, "v"), wrap(block.up, "ffn")
            else:
                attention.q, attention.v = wrap(attention.q, "q"), wrap(attention.v, "v")
                block.down = wrap(block.down, "ffn")
    else:
        for layer in transformer_layers(backbone, model_name):
            layer.linear1 = wrap(layer.linear1, "ffn")
            if method != "ia3":
                layer.linear2 = wrap(layer.linear2, "ffn")
    for module_name in ("norm", "fusion"):
        module = getattr(backbone, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


# ----------------------------------------------------------------------------- metrics

def macro_scores(gold: list[str], predicted: list[str]) -> dict:
    """sklearn-style macro P/R/F1 over labels appearing in gold or predictions."""
    labels = sorted(set(gold) | set(predicted))
    per_label = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    count = max(1, len(labels))
    return {
        "macro_f1": sum(row["f1"] for row in per_label.values()) / count,
        "macro_precision": sum(row["precision"] for row in per_label.values()) / count,
        "macro_recall": sum(row["recall"] for row in per_label.values()) / count,
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / max(1, len(gold)),
        "per_label": per_label,
    }


@torch.no_grad()
def predict_logits(model, loader, device) -> torch.Tensor:
    model.eval()
    outputs = torch.zeros(len(loader.dataset), model.head[-1].out_features)
    for batch in loader:
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(tensors)
        outputs[batch["rows"]] = logits.float().cpu()
    return outputs


def evaluate(dataset: SentenceDataset, logits: torch.Tensor, labels: list[str]) -> dict:
    predicted = [labels[index] for index in logits.argmax(-1).tolist()]
    gold = [labels[example["label"]] for example in dataset.examples]
    result = macro_scores(gold, predicted)
    for domain_index, domain in enumerate(DOMAINS):
        rows = [row for row, example in enumerate(dataset.examples) if example["domain"] == domain_index]
        if rows:
            scores = macro_scores([gold[row] for row in rows], [predicted[row] for row in rows])
            result[f"{domain}_macro_f1"] = scores["macro_f1"]
    confusion = defaultdict(Counter)
    for g, p in zip(gold, predicted):
        confusion[g][p] += 1
    result["confusion"] = {g: dict(row) for g, row in confusion.items()}
    return result


def class_weights(dataset: SentenceDataset, count: int, power: float) -> torch.Tensor:
    frequency = Counter(example["label"] for example in dataset.examples)
    largest = max(frequency.values())
    weights = torch.ones(count)
    for index in range(count):
        weights[index] = min(10.0, (largest / max(1, frequency[index])) ** power)
    return weights / weights.mean()


# ----------------------------------------------------------------------------- training

def train(model, train_loader, dev_loader, labels, device, args, soft_targets=None):
    weights = class_weights(train_loader.dataset, len(labels), args.class_weight_power).to(device)
    head = [p for p in model.head_parameters() if p.requires_grad]
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": args.head_lr}]
    if body:
        groups.append({"params": body, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_total = max(1, args.epochs * math.ceil(len(train_loader) / args.grad_accum))
    warmup = max(1, int(0.06 * steps_total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0.05, (steps_total - step) / max(1, steps_total - warmup))),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    trainable = [p for p in model.parameters() if p.requires_grad]
    start = time.time()
    budget = args.time_budget_min * 60
    best = {"macro_f1": -1.0}
    best_state, history, out_of_time = None, [], False

    def checkpoint(epoch, step, mean_loss):
        nonlocal best, best_state
        logits = predict_logits(model, dev_loader, device)
        scores = evaluate(dev_loader.dataset, logits, labels)
        row = {
            "epoch": epoch, "step": step, "train_loss": mean_loss,
            "dev_macro_f1": scores["macro_f1"], "dev_macro_precision": scores["macro_precision"],
            "dev_macro_recall": scores["macro_recall"], "dev_accuracy": scores["accuracy"],
            "CL_macro_f1": scores.get("CL_macro_f1"), "IT_macro_f1": scores.get("IT_macro_f1"),
            "minutes": (time.time() - start) / 60,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if scores["macro_f1"] > best["macro_f1"]:
            best = scores
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        model.train()

    evals_per_epoch = max(1, args.evals_per_epoch)
    eval_every = max(1, len(train_loader) // evals_per_epoch)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader, 1):
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tensors).float()
            hard = F.cross_entropy(logits, tensors["labels"], weight=weights, label_smoothing=args.label_smoothing)
            if soft_targets is None:
                loss = hard
            else:
                teacher = soft_targets[batch["rows"]].to(device)
                t = args.temperature
                soft = F.kl_div(
                    F.log_softmax(logits / t, dim=-1), F.softmax(teacher / t, dim=-1), reduction="batchmean",
                ) * t * t
                loss = args.kd_alpha * hard + (1.0 - args.kd_alpha) * soft
            scaler.scale(loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss))
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {sum(losses[-100:]) / len(losses[-100:]):.4f} "
                      f"elapsed {(time.time() - start) / 60:.1f} min", flush=True)
            if time.time() - start > budget:
                out_of_time = True
                print("Time budget reached; evaluating and stopping.", flush=True)
                break
            if evals_per_epoch > 1 and step % eval_every == 0 and step < len(train_loader):
                checkpoint(epoch, step, sum(losses) / len(losses))
        checkpoint(epoch, len(losses), sum(losses) / max(1, len(losses)))
        if out_of_time:
            break
    model.load_state_dict(best_state)
    return best, history, time.time() - start


# ----------------------------------------------------------------------------- outputs

def write_submission(test_datasets, test_logits, labels, output_dir: Path) -> Path:
    combined, per_domain = {}, {}
    collisions = []
    for domain, dataset in test_datasets.items():
        logits = test_logits[domain]
        domain_predictions = {}
        for document in dataset.documents:
            domain_predictions[document["id"]] = [None] * len(document["sentences"])
        for row, example in enumerate(dataset.examples):
            document = dataset.documents[example["doc"]]
            domain_predictions[document["id"]][example["index"]] = labels[int(logits[row].argmax())]
        for document in dataset.documents:
            predictions = domain_predictions[document["id"]]
            if len(predictions) != len(document["sentences"]) or any(p not in labels for p in predictions):
                raise RuntimeError(f"Invalid prediction list for {domain}/{document['id']}")
            if document["id"] in combined:
                collisions.append(document["id"])
            combined[document["id"]] = predictions
        per_domain[domain] = domain_predictions
        (output_dir / f"prediction_{domain}.json").write_text(json.dumps(domain_predictions), encoding="utf-8")
    if collisions:
        raise RuntimeError(f"CL and IT share document ids {collisions[:5]}; check the Codabench format")
    expected = sum(len(dataset.documents) for dataset in test_datasets.values())
    if len(combined) != expected:
        raise RuntimeError("Document count mismatch")
    path = output_dir / "prediction.json"
    path.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    archive = output_dir / f"submission_{output_dir.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(path, arcname="prediction.json")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["prediction.json"]
        loaded = json.loads(bundle.read("prediction.json"))
    print(f"Wrote {archive} with {len(loaded)} documents / {sum(map(len, loaded.values()))} sentences", flush=True)
    return archive


# ----------------------------------------------------------------------------- main

def run(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"phase3_{args.model}_{args.method}_{args.context}"
    output_dir = Path(args.output_dir or f"/kaggle/working/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    checkpoint, tokenizer_path, artifact = locate_phase1_files(args.model, args.artifact_zip, output_dir / "phase1")
    files = obtain_data(output_dir / "data", args.data_dir)
    labels = load_label_vocab(files["label_vocab.json"])
    label_to_id = {label: index for index, label in enumerate(labels)}

    def documents(split, require_labels):
        docs = []
        for domain in DOMAINS:
            docs += normalize_documents(read_json(files[f"{domain}_{split}.json"]), domain, require_labels)
        if args.limit_docs:
            docs = [d for domain in DOMAINS for d in [x for x in docs if x["domain"] == domain][: args.limit_docs]]
        return docs

    train_docs, dev_docs = documents("train", True), documents("dev", True)
    check_labels(train_docs, labels)
    check_labels(dev_docs, labels)
    test_docs = {domain: normalize_documents(read_json(files[f"{domain}_test.json"]), domain, False) for domain in DOMAINS}
    shared = {d["id"] for d in test_docs["CL"]} & {d["id"] for d in test_docs["IT"]}
    if shared:  # fail fast, before any training time is spent
        raise RuntimeError(f"CL and IT test sets share document ids {sorted(shared)[:5]}; one prediction.json cannot hold both")
    print({d: f"{len(v)} test docs" for d, v in test_docs.items()}, "| train docs", len(train_docs), "| dev docs", len(dev_docs), flush=True)
    print("Train label distribution:", dict(Counter(l for d in train_docs for l in d["labels"]).most_common()), flush=True)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tokenizer.token_to_id("<pad>")
    separator = tokenizer.token_to_id("<eos>")
    if pad_id is None or separator is None:
        raise ValueError("Phase 1 tokenizer must contain <pad> and <eos>")

    train_set = SentenceDataset(train_docs, tokenizer, label_to_id, args, separator)
    dev_set = SentenceDataset(dev_docs, tokenizer, label_to_id, args, separator)
    test_sets = {d: SentenceDataset(test_docs[d], tokenizer, label_to_id, args, separator) for d in DOMAINS}
    collate = make_collate(pad_id)
    loader = lambda ds, shuffle, bs: DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=2)
    train_loader = loader(train_set, True, args.batch_size)
    dev_loader = loader(dev_set, False, args.batch_size * 2)

    backbone, phase1_config, phase1_saved = load_phase1(checkpoint, args.model)
    phase1_parameters = sum(p.numel() for p in backbone.parameters())
    if args.method == "distill":
        if args.model != "decoder":
            raise ValueError("The distillation student recipe uses the decoder Phase 1 model")
        backbone = make_student(backbone, layers=args.student_layers)
    model = RRModel(backbone, args.model, len(labels))
    apply_adapter(model, args.model, args.method, args.rank)
    model.to(device)

    soft_targets = None
    if args.method == "distill":
        if not args.teacher_soft_labels:
            raise ValueError("--teacher-soft-labels is required for distillation")
        saved = torch.load(args.teacher_soft_labels, map_location="cpu")
        if saved["labels"] != labels:
            raise ValueError("Teacher label order differs")
        index = {key: row for row, key in enumerate(saved["keys"])}
        missing = [e["key"] for e in train_set.examples if e["key"] not in index]
        if missing:
            raise ValueError(f"Teacher soft labels miss {len(missing)} training sentences, e.g. {missing[:3]}")
        soft_targets = saved["logits"][[index[e["key"]] for e in train_set.examples]]
        print(f"Loaded teacher soft labels from {saved.get('teacher')} ({len(index)} sentences)", flush=True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        "run": name, "phase1_artifact": artifact, "model": args.model, "method": args.method,
        "context": args.context, "phase1_parameters": phase1_parameters, "parameters": total,
        "trainable_parameters": trainable, "trainable_fraction": trainable / total,
        "train_sentences": len(train_set), "dev_sentences": len(dev_set),
        "test_sentences": {d: len(s) for d, s in test_sets.items()}, "device": str(device),
        "config": vars(args),
    }
    print(json.dumps(summary, indent=2, default=str), flush=True)

    best, history, seconds = train(model, train_loader, dev_loader, labels, device, args, soft_targets)
    print(json.dumps({k: best[k] for k in ("macro_f1", "macro_precision", "macro_recall", "accuracy")}), flush=True)

    test_logits = {d: predict_logits(model, loader(s, False, args.batch_size * 2), device) for d, s in test_sets.items()}
    archive = write_submission(test_sets, test_logits, labels, output_dir)

    if args.export_soft_labels:
        train_eval_loader = loader(train_set, False, args.batch_size * 2)
        logits = predict_logits(model, train_eval_loader, device)
        torch.save({
            "keys": [e["key"] for e in train_set.examples], "logits": logits, "labels": labels, "teacher": name,
        }, output_dir / "teacher_soft_labels.pt")
        print("Saved", output_dir / "teacher_soft_labels.pt", flush=True)

    if args.save_checkpoint:
        torch.save({"model": model.state_dict(), "phase1_config": asdict(phase1_config),
                    "run_config": vars(args), "labels": labels}, output_dir / "best.pt")
    (output_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "validation_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    manifest = {
        **summary, "best_dev_macro_f1": best["macro_f1"], "best_dev_macro_precision": best["macro_precision"],
        "best_dev_macro_recall": best["macro_recall"], "dev_CL_macro_f1": best.get("CL_macro_f1"),
        "dev_IT_macro_f1": best.get("IT_macro_f1"), "train_minutes": seconds / 60,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "phase1_steps": phase1_saved.get("total_steps"), "submission": str(archive),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("run", "best_dev_macro_f1", "dev_CL_macro_f1", "dev_IT_macro_f1",
                                                "trainable_parameters", "train_minutes", "submission")}, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("decoder", "encoder_decoder", "hybrid", "efficient"))
    parser.add_argument("--method", required=True, choices=("full", "lorma", "lora", "ia3", "distill"))
    parser.add_argument("--context", default="window", choices=("none", "window"))
    parser.add_argument("--context-sentences", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=48)
    parser.add_argument("--target-tokens", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--student-layers", type=int, default=8)
    parser.add_argument("--teacher-soft-labels")
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--export-soft-labels", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--time-budget-min", type=float, default=30.0)
    parser.add_argument("--evals-per-epoch", type=int, default=1)
    parser.add_argument("--artifact-zip")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=230338)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
