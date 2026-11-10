"""Phase 1 training, generation, validation, and submission pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 230338
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
MODEL_NAMES = ("decoder", "encoder_decoder", "hybrid", "efficient")
TRAIN_FILES = (
    "train_acts.json",
    "train_high_courts.json",
    "train_supreme_court.json",
    "train_tribunals.json",
)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RunConfig:
    vocab_size: int = 4096
    d_model: int = 512
    n_heads: int = 8
    seq_len: int = 256
    batch_size: int = 2
    grad_accum: int = 4
    steps: int = 400
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    train_chars: int = 40_000_000
    train_tokens: int = 12_000_000
    tokenizer_chars: int = 10_000_000
    generation_batch_size: int = 8
    max_generation_tokens: int = 320
    target_words: int = 128


class TokenBlocks(torch.utils.data.Dataset):
    def __init__(self, ids: list[int], block: int, samples: int, seed: int):
        if len(ids) <= block:
            raise ValueError(f"Need more than {block} tokens; received {len(ids)}")
        self.ids = torch.tensor(ids, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        self.starts = torch.randint(0, len(ids) - block, (samples,), generator=generator)
        self.block = block

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = int(self.starts[index])
        return self.ids[start : start + self.block]


def make_generation_validation(texts: list[str], examples: int = 8):
    """Hold out complete documents for the official 512-word to 128-word task."""
    training_texts, prompts, truth = [], {}, {}
    for text in texts:
        words = text.split()
        if len(truth) < examples and len(words) >= 640:
            example_id = f"heldout_{len(truth) + 1}"
            prompts[example_id] = {"text": " ".join(words[:512])}
            truth[example_id] = {"text": " ".join(words[512:640])}
        else:
            training_texts.append(text)
    if len(truth) < examples:
        raise RuntimeError(f"Could construct only {len(truth)} held-out continuation examples")
    return training_texts, prompts, truth


def exclude_generation_validation(texts: list[str], prompts: dict, truth: dict) -> list[str]:
    signatures = {
        example_id: f"{prompts[example_id]['text']} {truth[example_id]['text']}"
        for example_id in truth
    }
    filtered = [
        text for text in texts
        if " ".join(text.split()[:640]) not in signatures.values()
    ]
    if len(filtered) != len(texts) - len(truth):
        raise RuntimeError("Could not exclude every reused held-out document from training")
    return filtered


def train_tokenizer(texts: list[str], path: Path, vocab_size: int):
    """Train a new byte-level BPE on only the selected training corpus."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, min_frequency=2)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(path))
    return tokenizer


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def collect_training_text(config: RunConfig) -> list[str]:
    """Stream a balanced slice from the four official Phase 1 sources."""
    import ijson
    import requests

    texts, chars = [], 0
    quota = config.train_chars // len(TRAIN_FILES)
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    for filename in TRAIN_FILES:
        url = (
            "https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone/resolve/main/"
            f"Phase_1_Pretraining/Train/{filename}"
        )
        source_chars = 0
        with requests.get(url, headers=headers, stream=True, timeout=(30, 900)) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            for record in ijson.items(response.raw, "item"):
                text = record.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    continue
                remaining = quota - source_chars
                texts.append(text[:remaining])
                taken = min(len(text), remaining)
                chars += taken
                source_chars += taken
                if source_chars >= quota:
                    break
        print(f"{filename}: {source_chars:,} characters")
    if chars < 100_000:
        raise RuntimeError("Official training stream returned too little text")
    print(f"Collected {chars:,} training characters in {len(texts):,} records")
    return texts


def tokenize_training_text(texts: list[str], tokenizer, config: RunConfig) -> list[int]:
    eos = tokenizer.token_to_id("<eos>")
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        ids.append(eos)
        if len(ids) >= config.train_tokens:
            break
    print(f"Prepared {len(ids):,} training tokens")
    return ids[: config.train_tokens]


def load_test_data() -> dict[str, dict[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Exploration-Lab/CS-781-Capstone",
        filename="Phase_1_Pretraining/Test/test.json",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not data:
        raise ValueError("Official test input is not a non-empty ID mapping")
    return data


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).float().unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    output = torch.zeros(length, dim, device=device)
    output[:, 0::2] = torch.sin(position * scale)
    output[:, 1::2] = torch.cos(position * scale)
    return output


class GatedContextFusion(nn.Module):
    """Inject a global or causal summary through a learned content-dependent gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.context = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, hidden: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((hidden, summary), dim=-1)))
        return hidden + gate * self.context(summary)


class DecoderOnlyLM(nn.Module):
    def __init__(self, cfg: RunConfig, layers: int = 16):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, 4 * cfg.d_model, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.fusion = GatedContextFusion(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids) + sinusoidal_positions(ids.size(1), self.cfg.d_model, ids.device)
        x = self.blocks(x, mask=causal_mask(ids.size(1), ids.device))
        denominator = torch.arange(1, ids.size(1) + 1, device=ids.device).view(1, -1, 1)
        x = self.fusion(x, x.cumsum(dim=1) / denominator)
        return self.head(self.norm(x))

    def loss(self, batch: torch.Tensor, bos: int) -> torch.Tensor:
        del bos
        return F.cross_entropy(self.logits(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())

    def next_logits(self, source: torch.Tensor, generated: torch.Tensor, bos: int) -> torch.Tensor:
        del bos
        context = torch.cat((source, generated), dim=1)[:, -self.cfg.seq_len :]
        return self.logits(context)[:, -1]


class EncoderDecoderLM(nn.Module):
    def __init__(self, cfg: RunConfig, encoder_layers: int = 7, decoder_layers: int = 7):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        encoder = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, 4 * cfg.d_model, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0,
        )
        decoder = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.n_heads, 4 * cfg.d_model, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder, encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder, decoder_layers)
        self.fusion = GatedContextFusion(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        x = self.embedding(source) + sinusoidal_positions(source.size(1), self.cfg.d_model, source.device)
        return self.encoder(x)

    def decode(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        y = self.embedding(target) + sinusoidal_positions(target.size(1), self.cfg.d_model, target.device)
        y = self.decoder(y, memory, tgt_mask=causal_mask(target.size(1), target.device))
        source_summary = memory.mean(dim=1, keepdim=True).expand(-1, y.size(1), -1)
        y = self.fusion(y, source_summary)
        return self.head(self.norm(y))

    def loss(self, batch: torch.Tensor, bos: int) -> torch.Tensor:
        split = batch.size(1) // 2
        source, target = batch[:, :split], batch[:, split:]
        decoder_input = torch.cat((torch.full_like(target[:, :1], bos), target[:, :-1]), dim=1)
        return F.cross_entropy(self.decode(decoder_input, self.encode(source)).flatten(0, 1), target.flatten())

    def next_logits(self, source: torch.Tensor, generated: torch.Tensor, bos: int) -> torch.Tensor:
        memory = self.encode(source[:, -self.cfg.seq_len :])
        target = generated[:, -self.cfg.seq_len :]
        if target.size(1) == 0:
            target = torch.full((source.size(0), 1), bos, dtype=torch.long, device=source.device)
        return self.decode(target, memory)[:, -1]


class HybridLM(EncoderDecoderLM):
    """Bidirectional recurrent encoder plus causal Transformer decoder."""

    def __init__(self, cfg: RunConfig, recurrent_layers: int = 5, decoder_layers: int = 11):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.recurrent = nn.GRU(
            cfg.d_model, cfg.d_model // 2, recurrent_layers,
            batch_first=True, bidirectional=True, dropout=0.0,
        )
        decoder = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.n_heads, 4 * cfg.d_model, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(decoder, decoder_layers)
        self.fusion = GatedContextFusion(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        x = self.embedding(source) + sinusoidal_positions(source.size(1), self.cfg.d_model, source.device)
        return self.recurrent(x)[0]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x: torch.Tensor) -> torch.Tensor:
    # x: batch, heads, time, head_dim
    length, dim = x.size(-2), x.size(-1)
    position = torch.arange(length, device=x.device).float()
    frequency = torch.exp(torch.arange(0, dim, 2, device=x.device).float() * (-math.log(10000.0) / dim))
    angle = torch.outer(position, frequency)
    cos, sin = angle.cos()[None, None], angle.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int = 2, window: int = 64):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.window = heads, kv_heads, dim // heads, window
        self.q = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = rope(self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2))
        k = rope(self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2))
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        positions = torch.arange(length, device=x.device)
        distance = positions[:, None] - positions[None, :]
        if getattr(self, "bidirectional", False):
            allowed = distance.abs() < self.window
        else:
            allowed = (distance >= 0) & (distance < self.window)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        output = torch.matmul(scores.softmax(-1), v).transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out(output)


class EfficientBlock(nn.Module):
    def __init__(self, cfg: RunConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        hidden += (-hidden) % 64
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attention = GroupedQueryAttention(cfg.d_model, cfg.n_heads)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        y = self.norm2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class EfficientLM(nn.Module):
    def __init__(self, cfg: RunConfig, layers: int = 19):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(EfficientBlock(cfg) for _ in range(layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def loss(self, batch: torch.Tensor, bos: int) -> torch.Tensor:
        del bos
        return F.cross_entropy(self.logits(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())

    def next_logits(self, source: torch.Tensor, generated: torch.Tensor, bos: int) -> torch.Tensor:
        del bos
        context = torch.cat((source, generated), dim=1)[:, -self.cfg.seq_len :]
        return self.logits(context)[:, -1]


def build_model(name: str, cfg: RunConfig) -> nn.Module:
    builders = {
        "decoder": DecoderOnlyLM,
        "encoder_decoder": EncoderDecoderLM,
        "hybrid": HybridLM,
        "efficient": EfficientLM,
    }
    return builders[name](cfg)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate_loss(model: nn.Module, loader, bos: int, device: torch.device, batches: int = 20) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                losses.append(float(model.loss(batch, bos)))
    model.train()
    return sum(losses) / len(losses)


def train_model(
    model: nn.Module,
    loader,
    validation_loader,
    cfg: RunConfig,
    bos: int,
    device: torch.device,
    recovery_path: Path,
    start_step: int = 0,
    prior_losses: list[float] | None = None,
) -> tuple[list[float], float]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup = max(1, cfg.steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup
        if step < warmup
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, cfg.steps - warmup))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    losses, iterator = list(prior_losses or []), iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, cfg.steps):
        running = 0.0
        for _ in range(cfg.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model.loss(batch, bos) / cfg.grad_accum
            scaler.scale(loss).backward()
            running += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(running)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step {step + 1:>3}/{cfg.steps}: loss={running:.4f}")
        if (step + 1) % 25 == 0 or step + 1 == cfg.steps:
            temporary = recovery_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": recovery_path.stem.removesuffix("_recovery"),
                    "completed_steps": step + 1,
                    "config": asdict(cfg),
                    "train_loss": losses,
                },
                temporary,
            )
            os.replace(temporary, recovery_path)
            print(f"recovery checkpoint saved: {recovery_path.name}")
    return losses, evaluate_loss(model, validation_loader, bos, device)


def sample_next(logits: torch.Tensor, temperature: float = 0.2, top_k: int = 20) -> torch.Tensor:
    logits = logits / temperature
    values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
    sampled = torch.multinomial(values.softmax(-1), 1)
    return indices.gather(1, sampled)


@torch.inference_mode()
def generate_predictions(model, tokenizer, test_data, cfg: RunConfig, device: torch.device) -> dict:
    model.eval()
    pad, bos, eos = (tokenizer.token_to_id(token) for token in SPECIAL[:3])
    items, predictions = list(test_data.items()), {}
    for start in range(0, len(items), cfg.generation_batch_size):
        batch = items[start : start + cfg.generation_batch_size]
        encoded = [tokenizer.encode(record["text"]).ids[-cfg.seq_len :] for _, record in batch]
        width = max(map(len, encoded))
        source = torch.full((len(encoded), width), pad, dtype=torch.long, device=device)
        for row, ids in enumerate(encoded):
            source[row, -len(ids) :] = torch.tensor(ids, device=device)
        generated = torch.empty((len(encoded), 0), dtype=torch.long, device=device)
        finished_text = [None] * len(encoded)
        # Encoder-decoder models reuse their source representation at every step.
        memory = model.encode(source) if isinstance(model, EncoderDecoderLM) else None
        for _ in range(cfg.max_generation_tokens):
            if memory is None:
                next_logits = model.next_logits(source, generated, bos)
            else:
                target = generated[:, -cfg.seq_len :]
                if target.size(1) == 0:
                    target = torch.full((source.size(0), 1), bos, dtype=torch.long, device=device)
                next_logits = model.decode(target, memory)[:, -1]
            next_id = sample_next(next_logits)
            next_id[next_id.eq(eos)] = bos
            generated = torch.cat((generated, next_id), dim=1)
            if generated.size(1) % 16 == 0:
                for row in range(len(encoded)):
                    words = tokenizer.decode(generated[row].tolist(), skip_special_tokens=True).split()
                    if len(words) >= cfg.target_words:
                        finished_text[row] = " ".join(words[: cfg.target_words])
                if all(text is not None for text in finished_text):
                    break
        for row, (example_id, _) in enumerate(batch):
            if finished_text[row] is None:
                words = tokenizer.decode(generated[row].tolist(), skip_special_tokens=True).split()
                if not words:
                    words = ["the"]
                words = (words * math.ceil(cfg.target_words / len(words)))[: cfg.target_words]
                finished_text[row] = " ".join(words)
            predictions[str(example_id)] = {"text": finished_text[row]}
        print(f"generated {min(start + len(batch), len(items))}/{len(items)}")
    return predictions


def validate_and_zip(predictions: dict, test_data: dict, output_dir: Path, name: str) -> Path:
    expected, actual = {str(key) for key in test_data}, set(predictions)
    if expected != actual:
        raise ValueError(f"ID mismatch: missing={expected - actual}, extra={actual - expected}")
    for example_id, record in predictions.items():
        if set(record) != {"text"} or len(record["text"].split()) != 128:
            raise ValueError(f"Invalid prediction for {example_id}")
    folder = output_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    prediction_path = folder / "prediction.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = output_dir / f"submission_{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["prediction.json"]:
            raise ValueError(f"Bad archive layout: {archive.namelist()}")
    return zip_path


def run(args) -> None:
    seed_everything()
    cfg = RunConfig(
        steps=args.steps,
        train_chars=args.train_chars,
        train_tokens=args.train_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "phase1_bpe.json"
    if args.generate_only:
        if args.model == "all" or not args.resume_checkpoint:
            raise ValueError("--generate-only requires one --model and --resume-checkpoint")
        source_tokenizer = Path(args.tokenizer_path) if args.tokenizer_path else tokenizer_path
        tokenizer = load_tokenizer(source_tokenizer)
        cfg.vocab_size = tokenizer.get_vocab_size()
        model = build_model(args.model, cfg)
        saved = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(saved["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_data = load_test_data()
        predictions = generate_predictions(model, tokenizer, test_data, cfg, device)
        zip_path = validate_and_zip(predictions, test_data, output_dir, args.model)
        print(f"created {zip_path}")
        return
    texts = collect_training_text(cfg)
    random.Random(SEED).shuffle(texts)
    if args.validation_prompts and args.validation_truth:
        validation_prompts = json.loads(Path(args.validation_prompts).read_text(encoding="utf-8"))
        validation_truth = json.loads(Path(args.validation_truth).read_text(encoding="utf-8"))
        texts = exclude_generation_validation(texts, validation_prompts, validation_truth)
    elif args.validation_prompts or args.validation_truth:
        raise ValueError("Provide both --validation-prompts and --validation-truth")
    else:
        texts, validation_prompts, validation_truth = make_generation_validation(texts)
    validation_prompts_path = output_dir / "validation_prompts.json"
    validation_truth_path = output_dir / "validation_truth.json"
    validation_prompts_path.write_text(
        json.dumps(validation_prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_truth_path.write_text(
        json.dumps(validation_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.tokenizer_path:
        tokenizer = load_tokenizer(Path(args.tokenizer_path))
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer_texts, used = [], 0
        for text in texts:
            tokenizer_texts.append(text)
            used += len(text)
            if used >= cfg.tokenizer_chars:
                break
        tokenizer = train_tokenizer(tokenizer_texts, tokenizer_path, cfg.vocab_size)
    cfg.vocab_size = tokenizer.get_vocab_size()
    token_ids = tokenize_training_text(texts, tokenizer, cfg)
    split = int(0.95 * len(token_ids))
    train_ids, validation_ids = token_ids[:split], token_ids[split:]
    samples = cfg.steps * cfg.grad_accum * cfg.batch_size
    dataset = TokenBlocks(train_ids, cfg.seq_len + 1, samples, SEED)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    validation = TokenBlocks(validation_ids, cfg.seq_len + 1, 20 * cfg.batch_size, SEED + 1)
    validation_loader = torch.utils.data.DataLoader(
        validation, batch_size=cfg.batch_size, shuffle=False, pin_memory=True
    )
    test_data = load_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bos = tokenizer.token_to_id("<bos>")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if args.models:
        selected = tuple(name.strip() for name in args.models.split(",") if name.strip())
        invalid = set(selected) - set(MODEL_NAMES)
        if invalid or not selected:
            raise ValueError(f"Invalid --models selection: {sorted(invalid)}")
    else:
        selected = MODEL_NAMES if args.model == "all" else (args.model,)
    for name in selected:
        seed_everything(SEED + MODEL_NAMES.index(name))
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        model = build_model(name, cfg)
        previous_steps = 0
        recovery_path = output_dir / f"{name}_recovery.pt"
        recovery_step = 0
        recovery_losses = []
        if args.resume_checkpoint:
            if len(selected) != 1:
                raise ValueError("--resume-checkpoint requires one selected model")
            saved = torch.load(args.resume_checkpoint, map_location="cpu")
            saved_name = saved.get("model_name")
            if saved_name and saved_name != name:
                raise ValueError(f"Checkpoint model is {saved_name}, not {name}")
            model.load_state_dict(saved["model"])
            previous_steps = int(saved.get("total_steps", saved.get("config", {}).get("steps", 0)))
            print(f"resumed {name} after {previous_steps:,} prior steps")
        elif recovery_path.exists():
            saved = torch.load(recovery_path, map_location="cpu")
            model.load_state_dict(saved["model"])
            recovery_step = int(saved["completed_steps"])
            recovery_losses = list(saved.get("train_loss", []))
            print(f"automatically resumed {name} at step {recovery_step}/{cfg.steps}")
        parameters = parameter_count(model)
        if not 50_000_000 <= parameters <= 300_000_000:
            raise ValueError(f"{name} has {parameters:,} parameters; must be 50M-300M")
        print(f"parameters: {parameters:,}")
        started = time.perf_counter()
        losses, validation_loss = train_model(
            model,
            loader,
            validation_loader,
            cfg,
            bos,
            device,
            recovery_path,
            recovery_step,
            recovery_losses,
        )
        elapsed = time.perf_counter() - started
        log_path = output_dir / f"training_log_{name}.json"
        log_path.write_text(
            json.dumps(
                {
                    "config": asdict(cfg),
                    "train_loss": losses,
                    "validation_loss": validation_loss,
                    "training_seconds": elapsed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        checkpoint = output_dir / f"{name}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "parameters": parameters,
                "model_name": name,
                "total_steps": previous_steps + cfg.steps,
                "train_loss": losses,
                "validation_loss": validation_loss,
            },
            checkpoint,
        )
        validation_predictions = generate_predictions(
            model, tokenizer, validation_prompts, cfg, device
        )
        validation_prediction_path = output_dir / f"validation_prediction_{name}.json"
        validation_prediction_path.write_text(
            json.dumps(validation_predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        predictions = generate_predictions(model, tokenizer, test_data, cfg, device)
        zip_path = validate_and_zip(predictions, test_data, output_dir, name)
        manifest[name] = {
            "category": {
                "decoder": "decoder-only",
                "encoder_decoder": "encoder-decoder",
                "hybrid": "encoder/hybrid",
                "efficient": "efficient/lightweight",
            }[name],
            "parameters": parameters,
            "steps_this_run": cfg.steps,
            "total_steps": previous_steps + cfg.steps,
            "final_loss": losses[-1],
            "validation_loss": validation_loss,
            "validation_perplexity": math.exp(min(validation_loss, 20.0)),
            "training_seconds": elapsed,
            "device": str(device),
            "checkpoint": str(checkpoint),
            "training_log": str(log_path),
            "validation_truth": str(validation_truth_path),
            "validation_prompts": str(validation_prompts_path),
            "validation_prediction": str(validation_prediction_path),
            "submission": str(zip_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        recovery_path.unlink(missing_ok=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("all",) + MODEL_NAMES, default="all")
    parser.add_argument("--models", help="Comma-separated subset for multi-GPU orchestration")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--train-chars", type=int, default=40_000_000)
    parser.add_argument("--train-tokens", type=int, default=12_000_000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--validation-prompts")
    parser.add_argument("--validation-truth")
    parser.add_argument("--output-dir", default="/kaggle/working/phase1")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
