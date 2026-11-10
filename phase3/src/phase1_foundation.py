"""Train the Phase-1 flagship legal encoder-decoder model from scratch.

This file deliberately reuses only the proven corpus/test download helpers from
phase1_pipeline.py.  The tokenizer, model, objective, checkpointing, validation,
generation, and submission path are implemented here.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import math
import os
import random
import time
from array import array
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import phase1_pipeline as legacy


SEED = 781
SPECIAL = ("<pad>", "<bos>", "<eos>", "<unk>", "<mask>")


def init_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size > torch.cuda.device_count():
        raise RuntimeError(f"Requested {world_size} ranks but found {torch.cuda.device_count()} GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    return True, rank, world_size, local_rank, torch.device("cuda", local_rank)


def distributed_barrier(distributed):
    if distributed:
        dist.barrier()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


@dataclass
class Config:
    vocab_size: int = 24_000
    d_model: int = 640
    heads: int = 10
    encoder_layers: int = 12
    decoder_layers: int = 8
    d_ff: int = 1_728
    source_len: int = 1_024
    target_len: int = 256
    global_block: int = 32
    dropout: float = 0.1
    steps: int = 60_000
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02
    train_chars: int = 500_000_000
    token_pool: int = 100_000_000
    tokenizer_chars: int = 80_000_000
    validation_examples: int = 256
    validation_batches: int = 24
    checkpoint_every: int = 100
    mlm_probability: float = 0.25
    mlm_mask_fraction: float = 0.15
    mlm_weight: float = 0.35
    generation_batch_size: int = 8
    max_generation_tokens: int = 384
    target_words: int = 128
    use_global_memory: bool = True


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_tokenizer(texts: list[str], path: Path, cfg: Config) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    used = 0

    def iterator():
        nonlocal used
        for text in texts:
            if used >= cfg.tokenizer_chars:
                break
            used += len(text)
            yield text

    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    tokenizer.save(str(path))
    return tokenizer


def make_validation(texts: list[str], count: int) -> tuple[list[str], list[str], dict, dict]:
    rng = random.Random(SEED + 11)
    order = list(range(len(texts)))
    rng.shuffle(order)
    used: set[int] = set()
    prompts, truth = {}, {}
    for index in order:
        words = texts[index].split()
        if len(words) < 640:
            continue
        start = rng.randrange(len(words) - 640 + 1)
        key = f"validation_{len(prompts):04d}"
        prompts[key] = {"text": " ".join(words[start : start + 512])}
        truth[key] = {"text": " ".join(words[start + 512 : start + 640])}
        used.add(index)
        if len(prompts) == count:
            break
    if len(prompts) < count:
        raise RuntimeError(f"Only {len(prompts)} documents could supply validation examples")
    training = [text for i, text in enumerate(texts) if i not in used]
    validation = [text for i, text in enumerate(texts) if i in used]
    return training, validation, prompts, truth


def tokenize_documents(
    texts: list[str],
    tokenizer: Tokenizer,
    cfg: Config,
    source_len: int | None = None,
    target_len: int | None = None,
):
    eos = tokenizer.token_to_id("<eos>")
    flat, spans = array("I"), []
    block = (source_len or cfg.source_len) + (target_len or cfg.target_len)
    for text in texts:
        ids = tokenizer.encode(text).ids
        if len(ids) < block:
            continue
        remaining = cfg.token_pool - len(flat)
        if remaining <= 0:
            break
        ids = ids[:remaining]
        if len(ids) < block:
            break
        start = len(flat)
        flat.extend(ids)
        flat.append(eos)
        spans.append((start, start + len(ids)))
    if not spans:
        raise RuntimeError("No documents were long enough for the configured source/target lengths")
    return flat, spans


class ContinuationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokens,
        spans,
        source_len,
        target_len,
        samples,
        seed,
        sample_offset=0,
        rank=0,
        world_size=1,
        samples_per_step=1,
    ):
        self.tokens, self.spans = tokens, spans
        self.source_len, self.target_len = source_len, target_len
        self.samples, self.seed, self.sample_offset = samples, seed, sample_offset
        self.rank, self.world_size = rank, world_size
        self.samples_per_step = samples_per_step
        block = source_len + target_len
        self.cumulative, total = [], 0
        for start, end in spans:
            total += end - start - block + 1
            self.cumulative.append(total)
        self.total = total

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        step, within_step = divmod(index, self.samples_per_step)
        global_index = (
            self.sample_offset
            + step * self.samples_per_step * self.world_size
            + self.rank * self.samples_per_step
            + within_step
        )
        pick = random.Random(self.seed + global_index).randrange(self.total)
        doc = bisect.bisect_right(self.cumulative, pick)
        previous = self.cumulative[doc - 1] if doc else 0
        start = self.spans[doc][0] + pick - previous
        stop = start + self.source_len + self.target_len
        values = torch.tensor(self.tokens[start:stop], dtype=torch.long)
        return values[: self.source_len], values[self.source_len :]


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return x * scale * self.weight


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(q, k):
    length, width = q.size(-2), q.size(-1)
    positions = torch.arange(length, device=q.device, dtype=torch.float32)
    inverse = 1.0 / (10_000 ** (torch.arange(0, width, 2, device=q.device).float() / width))
    angles = torch.outer(positions, inverse)
    angles = torch.cat((angles, angles), dim=-1)[None, None]
    cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.d_model // cfg.heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, query, memory=None, key_mask=None, causal=False, rope=False):
        memory = query if memory is None else memory
        batch, query_len, _ = query.shape
        key_len = memory.size(1)

        def split(projection, values, length):
            return projection(values).view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q = split(self.q, query, query_len)
        k = split(self.k, memory, key_len)
        v = split(self.v, memory, key_len)
        if rope:
            q, k = apply_rope(q, k)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        if causal and mask is not None:
            causal_mask = torch.ones(query_len, key_len, device=query.device, dtype=torch.bool).tril()
            mask = mask & causal_mask[None, None]
            causal = False
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        result = result.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return self.out(result)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.value = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.out = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.value(x))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn, self.ff = Attention(cfg), SwiGLU(cfg)
        self.attn_norm, self.ff_norm = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        normalized = self.attn_norm(x)
        x = x + self.dropout(self.attn(normalized, key_mask=mask, rope=True))
        return x + self.dropout(self.ff(self.ff_norm(x)))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.self_attn, self.cross_attn, self.ff = Attention(cfg), Attention(cfg), SwiGLU(cfg)
        self.self_norm = RMSNorm(cfg.d_model)
        self.cross_norm = RMSNorm(cfg.d_model)
        self.ff_norm = RMSNorm(cfg.d_model)
        self.memory_gate = nn.Linear(cfg.d_model, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.use_global_memory = cfg.use_global_memory

    def forward(self, x, memory, global_memory, memory_mask=None, global_mask=None):
        normalized = self.self_norm(x)
        x = x + self.dropout(self.self_attn(normalized, causal=True, rope=True))
        normalized = self.cross_norm(x)
        token_context = self.cross_attn(normalized, memory, key_mask=memory_mask)
        if self.use_global_memory:
            global_context = self.cross_attn(normalized, global_memory, key_mask=global_mask)
            gate = torch.sigmoid(self.memory_gate(normalized))
            token_context = gate * token_context + (1.0 - gate) * global_context
        x = x + self.dropout(token_context)
        return x + self.dropout(self.ff(self.ff_norm(x)))


class LegalEncoderDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.encoder_layers))
        self.decoder = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        self.encoder_norm = RMSNorm(cfg.d_model)
        self.decoder_norm = RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def encode(self, source, source_mask=None):
        x = self.dropout(self.embedding(source))
        for layer in self.encoder:
            if self.training:
                x = checkpoint(layer, x, source_mask, use_reentrant=False)
            else:
                x = layer(x, source_mask)
        return self.encoder_norm(x)

    def pool_memory(self, memory, mask=None):
        block = self.cfg.global_block
        batch, length, width = memory.shape
        padding = (-length) % block
        if padding:
            memory = F.pad(memory, (0, 0, 0, padding))
            if mask is not None:
                mask = F.pad(mask, (0, padding), value=False)
        memory = memory.view(batch, -1, block, width)
        if mask is None:
            return memory.mean(2), None
        mask = mask.view(batch, -1, block)
        counts = mask.sum(2, keepdim=True).clamp_min(1)
        pooled = (memory * mask[..., None]).sum(2) / counts
        return pooled, mask.any(2)

    def decode(self, decoder_input, memory, source_mask=None):
        x = self.dropout(self.embedding(decoder_input))
        global_memory, global_mask = self.pool_memory(memory, source_mask)
        for layer in self.decoder:
            if self.training:
                x = checkpoint(
                    layer, x, memory, global_memory, source_mask, global_mask, use_reentrant=False
                )
            else:
                x = layer(x, memory, global_memory, source_mask, global_mask)
        return self.decoder_norm(x)

    def logits(self, hidden):
        return F.linear(hidden, self.embedding.weight)

    def forward(self, source, decoder_input, source_mask=None):
        memory = self.encode(source, source_mask)
        return self.logits(self.decode(decoder_input, memory, source_mask)), memory


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def corrupt_source(source, mask_id, pad_id, fraction):
    selected = torch.rand(source.shape, device=source.device) < fraction
    selected &= source.ne(pad_id)
    corrupted = source.clone()
    corrupted[selected] = mask_id
    return corrupted, selected


def cosine_multiplier(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def atomic_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate_loss(model, loader, bos, device, batches):
    model.eval()
    losses = []
    with torch.inference_mode():
        for number, (source, target) in enumerate(loader):
            if number == batches:
                break
            source, target = source.to(device), target.to(device)
            decoder_input = torch.cat((torch.full_like(target[:, :1], bos), target[:, :-1]), 1)
            logits, _ = model(source, decoder_input)
            losses.append(F.cross_entropy(logits.flatten(0, 1), target.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def train(
    model,
    loader,
    validation_loader,
    tokenizer,
    cfg,
    output_dir,
    device,
    resume=None,
    run_steps=None,
    distributed=False,
    rank=0,
    world_size=1,
    max_hours=None,
    generation_reserve_hours=0.0,
    session_started=None,
    finalize_on_time_limit=False,
):
    if cfg.grad_accum % world_size:
        raise ValueError("--grad-accum must be divisible by the number of DDP processes")
    local_grad_accum = cfg.grad_accum // world_size
    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
    )
    warmup = int(cfg.steps * cfg.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_multiplier(step, cfg.steps, warmup)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    start_step, losses = 0, []
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        state = saved["model"]
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        raw_model.load_state_dict(state)
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
        start_step = int(saved.get("completed_steps", saved.get("total_steps", 0)))
        losses = list(saved.get("train_loss", []))
        rng_states = saved.get("rng_states")
        if rng_states and rank < len(rng_states):
            rng = rng_states[rank]
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            if device.type == "cuda" and rng.get("cuda") is not None:
                torch.cuda.set_rng_state(rng["cuda"], device)
        else:
            seed_everything(SEED + rank + start_step)
        if rank == 0:
            print(f"resumed at step {start_step}/{cfg.steps}")
    else:
        seed_everything(SEED + rank)
    bos = tokenizer.token_to_id("<bos>")
    pad = tokenizer.token_to_id("<pad>")
    mask_id = tokenizer.token_to_id("<mask>")
    iterator = iter(loader)
    recovery = output_dir / "foundation_recovery.pt"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    end_step = min(cfg.steps, start_step + run_steps) if run_steps else cfg.steps
    if max_hours and not 0 <= generation_reserve_hours < max_hours:
        raise ValueError("--generation-reserve-hours must be non-negative and below --max-hours")
    training_hours = max_hours - generation_reserve_hours if max_hours else None
    deadline = (
        (session_started if session_started is not None else time.perf_counter())
        + training_hours * 3600
        if training_hours
        else None
    )
    if rank == 0 and max_hours:
        print(
            f"total runtime budget: {max_hours:.2f}h; training/data budget: "
            f"{training_hours:.2f}h; generation reserve: {generation_reserve_hours:.2f}h",
            flush=True,
        )
    completed = start_step
    stopped_by_time = False

    def save_recovery():
        local_rng = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        if distributed:
            rng_states = [None] * world_size
            dist.all_gather_object(rng_states, local_rng)
        else:
            rng_states = [local_rng]
        if rank == 0:
            atomic_save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "completed_steps": completed,
                    "config": asdict(cfg),
                    "parameters": parameter_count(raw_model),
                    "train_loss": losses,
                    "rng_states": rng_states,
                    "world_size": world_size,
                },
                recovery,
            )
            print(f"checkpoint saved: {recovery}", flush=True)
        distributed_barrier(distributed)

    for step in range(start_step, end_step):
        running = 0.0
        for micro_step in range(local_grad_accum):
            try:
                source, target = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                source, target = next(iterator)
            source, target = source.to(device), target.to(device)
            decoder_input = torch.cat((torch.full_like(target[:, :1], bos), target[:, :-1]), 1)
            use_mlm = random.random() < cfg.mlm_probability
            model_source, selected = (
                corrupt_source(source, mask_id, pad, cfg.mlm_mask_fraction)
                if use_mlm
                else (source, None)
            )
            synchronize = micro_step == local_grad_accum - 1
            sync_context = (
                contextlib.nullcontext()
                if synchronize or not distributed
                else model.no_sync()
            )
            with sync_context:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    logits, memory = model(model_source, decoder_input)
                    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
                    if selected is not None and selected.any():
                        encoder_logits = raw_model.logits(memory[selected])
                        loss = loss + cfg.mlm_weight * F.cross_entropy(
                            encoder_logits, source[selected]
                        )
                    loss = loss / local_grad_accum
                scaler.scale(loss).backward()
            running += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if distributed:
            reduced = torch.tensor(running, dtype=torch.float32, device=device)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            running = reduced.item() / world_size
        losses.append(running)
        completed = step + 1
        if rank == 0 and (completed == 1 or completed % 10 == 0):
            tokens = completed * cfg.batch_size * cfg.grad_accum * (cfg.source_len + cfg.target_len)
            elapsed = time.perf_counter() - (session_started or time.perf_counter())
            rate = elapsed / max(1, completed - start_step)
            print(
                f"step {completed}/{cfg.steps}: loss={running:.4f}, tokens={tokens:,}, "
                f"elapsed={elapsed / 3600:.2f}h, seconds/step={rate:.2f}",
                flush=True,
            )
        time_limit = bool(deadline and time.perf_counter() >= deadline)
        if distributed:
            stop = torch.tensor(int(time_limit), device=device)
            dist.broadcast(stop, src=0)
            time_limit = bool(stop.item())
        if completed % cfg.checkpoint_every == 0 or completed == end_step or time_limit:
            save_recovery()
        if time_limit:
            stopped_by_time = True
            break
    should_finalize = completed >= cfg.steps or (stopped_by_time and finalize_on_time_limit)
    if not should_finalize:
        if rank == 0:
            reason = "time limit" if stopped_by_time else "chunk limit"
            print(f"{reason} reached at {completed}/{cfg.steps}; resume from {recovery}")
        return recovery, None, False, completed, "time_limit" if stopped_by_time else "chunk_limit"
    distributed_barrier(distributed)
    validation_loss = None
    if rank == 0:
        validation_loss = evaluate_loss(
            raw_model, validation_loader, bos, device, cfg.validation_batches
        )
    if distributed:
        value = torch.tensor(
            float("nan") if validation_loss is None else validation_loss,
            dtype=torch.float64,
            device=device,
        )
        dist.broadcast(value, src=0)
        validation_loss = value.item()
    final = output_dir / "foundation.pt"
    if rank == 0:
        atomic_save(
            {
                "model": raw_model.state_dict(),
                "total_steps": completed,
                "target_steps": cfg.steps,
                "config": asdict(cfg),
                "parameters": parameter_count(raw_model),
                "train_loss": losses,
                "validation_loss": validation_loss,
            },
            final,
        )
    distributed_barrier(distributed)
    reason = "target_steps" if completed >= cfg.steps else "time_limit_finalized"
    return final, validation_loss, True, completed, reason


@torch.inference_mode()
def generate(model, tokenizer, records, cfg, device):
    model.eval()
    pad, bos = tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<bos>")
    items, predictions = list(records.items()), {}
    truncated = 0
    for start in range(0, len(items), cfg.generation_batch_size):
        batch = items[start : start + cfg.generation_batch_size]
        encoded = []
        for _, record in batch:
            ids = tokenizer.encode(record["text"]).ids
            truncated += int(len(ids) > cfg.source_len)
            encoded.append(ids[-cfg.source_len :])
        width = max(map(len, encoded))
        source = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        source_mask = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
        for row, ids in enumerate(encoded):
            source[row, : len(ids)] = torch.tensor(ids, device=device)
            source_mask[row, : len(ids)] = True
        memory = model.encode(source, source_mask)
        generated = torch.full((len(batch), 1), bos, dtype=torch.long, device=device)
        finished = [None] * len(batch)
        for _ in range(cfg.max_generation_tokens):
            hidden = model.decode(generated, memory, source_mask)
            next_id = model.logits(hidden[:, -1:])[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat((generated, next_id), 1)
            if generated.size(1) % 8 == 0:
                for row in range(len(batch)):
                    words = tokenizer.decode(generated[row, 1:].tolist(), skip_special_tokens=True).split()
                    if len(words) >= cfg.target_words and finished[row] is None:
                        finished[row] = " ".join(words[: cfg.target_words])
                if all(finished):
                    break
        for row, (key, _) in enumerate(batch):
            words = tokenizer.decode(generated[row, 1:].tolist(), skip_special_tokens=True).split()
            if not words:
                words = ["the"]
            if len(words) < cfg.target_words:
                words = (words * math.ceil(cfg.target_words / len(words)))[: cfg.target_words]
            predictions[str(key)] = {"text": finished[row] or " ".join(words[: cfg.target_words])}
        print(f"generated {min(start + len(batch), len(items))}/{len(items)}")
    print(f"prompts truncated to {cfg.source_len} tokens: {truncated}/{len(items)}")
    return predictions


def distributed_generate(model, tokenizer, records, cfg, device, distributed, rank, world_size):
    if not distributed:
        return generate(model, tokenizer, records, cfg, device)
    local_records = dict(list(records.items())[rank::world_size])
    local_predictions = generate(model, tokenizer, local_records, cfg, device)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_predictions)
    if rank != 0:
        return None
    predictions = {}
    for shard in gathered:
        predictions.update(shard)
    return predictions


def build_legacy_config(cfg):
    result = legacy.RunConfig(
        steps=cfg.steps,
        train_chars=cfg.train_chars,
        train_tokens=cfg.token_pool,
        seq_len=cfg.source_len,
        batch_size=cfg.batch_size,
        grad_accum=cfg.grad_accum,
    )
    return result


def preparation_signature(cfg):
    return {
        "train_chars": cfg.train_chars,
        "token_pool": cfg.token_pool,
        "tokenizer_chars": cfg.tokenizer_chars,
        "validation_examples": cfg.validation_examples,
        "source_len": cfg.source_len,
        "target_len": cfg.target_len,
        "vocab_size": cfg.vocab_size,
    }


def prepare_data(cfg, output, tokenizer_path, tokenizer_source=None):
    prepared_path = output / "prepared_data.pt"
    expected = preparation_signature(cfg)
    if prepared_path.exists() and tokenizer_path.exists():
        prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
        if prepared.get("signature") != expected:
            raise ValueError(
                f"{prepared_path} was built with different data settings; use another output "
                "directory or remove that cache"
            )
        return prepared

    texts = legacy.collect_training_text(build_legacy_config(cfg))
    random.Random(SEED).shuffle(texts)
    texts, validation_texts, validation_prompts, validation_truth = make_validation(
        texts, cfg.validation_examples
    )
    if tokenizer_source:
        tokenizer = Tokenizer.from_file(tokenizer_source)
        if Path(tokenizer_source) != tokenizer_path:
            tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = train_tokenizer(texts, tokenizer_path, cfg)
    cfg.vocab_size = tokenizer.get_vocab_size()
    expected["vocab_size"] = cfg.vocab_size
    tokens, spans = tokenize_documents(texts, tokenizer, cfg)
    validation_source_len, validation_target_len = 512, 128
    validation_tokens, validation_spans = tokenize_documents(
        validation_texts, tokenizer, cfg, validation_source_len, validation_target_len
    )
    prepared = {
        "signature": expected,
        "tokens": tokens,
        "spans": spans,
        "validation_tokens": validation_tokens,
        "validation_spans": validation_spans,
        "validation_prompts": validation_prompts,
        "validation_truth": validation_truth,
    }
    atomic_save(prepared, prepared_path)
    (output / "validation_prompts.json").write_text(
        json.dumps(validation_prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "validation_truth.json").write_text(
        json.dumps(validation_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return prepared


def main(args):
    session_started = time.perf_counter()
    distributed, rank, world_size, local_rank, device = init_distributed()
    try:
        seed_everything()
        cfg = Config(
            steps=args.steps,
            train_chars=args.train_chars,
            token_pool=args.token_pool,
            validation_examples=args.validation_examples,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            use_global_memory=not args.no_global_memory,
        )
        output = Path(args.output_dir)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        distributed_barrier(distributed)
        tokenizer_path = output / "legal_bpe_24k.json"

        if args.generate_only:
            if not args.checkpoint or not args.tokenizer:
                raise ValueError("--generate-only requires --checkpoint and --tokenizer")
            tokenizer = Tokenizer.from_file(args.tokenizer)
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            cfg = Config(**saved["config"])
            model = LegalEncoderDecoder(cfg).to(device)
            model.load_state_dict(saved["model"])
            prepared_path = output / "prepared_data.pt"
            if args.validation_only and not prepared_path.exists():
                raise ValueError("--validation-only requires prepared_data.pt in --output-dir")
            if prepared_path.exists():
                prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
                validation_predictions = distributed_generate(
                    model,
                    tokenizer,
                    prepared["validation_prompts"],
                    cfg,
                    device,
                    distributed,
                    rank,
                    world_size,
                )
                if rank == 0:
                    (output / "validation_prediction_foundation.json").write_text(
                        json.dumps(validation_predictions, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            if args.validation_only:
                if rank == 0:
                    print("created validation_prediction_foundation.json; skipped official test generation")
                return
            test_holder = [legacy.load_test_data() if rank == 0 else None]
            if distributed:
                dist.broadcast_object_list(test_holder, src=0)
            test = test_holder[0]
            predictions = distributed_generate(
                model, tokenizer, test, cfg, device, distributed, rank, world_size
            )
            if rank == 0:
                zip_path = legacy.validate_and_zip(predictions, test, output, "foundation")
                print(f"created {zip_path}")
            return

        if args.checkpoint and not args.tokenizer and not tokenizer_path.exists():
            raise ValueError("An external --checkpoint must be paired with its --tokenizer")
        tokenizer_source = args.tokenizer or (str(tokenizer_path) if tokenizer_path.exists() else None)
        if rank == 0:
            prepare_data(cfg, output, tokenizer_path, tokenizer_source)
        distributed_barrier(distributed)
        prepared = torch.load(
            output / "prepared_data.pt", map_location="cpu", weights_only=False
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        cfg.vocab_size = tokenizer.get_vocab_size()
        if prepared.get("signature") != preparation_signature(cfg):
            raise ValueError("Prepared-data cache does not match the requested configuration")
        validation_prompts = prepared["validation_prompts"]
        validation_truth = prepared["validation_truth"]

        resume = args.checkpoint
        automatic = output / "foundation_recovery.pt"
        if resume is None and automatic.exists():
            resume = str(automatic)
        completed_steps = 0
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=False)
            completed_steps = int(saved.get("completed_steps", saved.get("total_steps", 0)))
            saved_cfg = saved.get("config", {})
            architecture_fields = (
                "vocab_size", "d_model", "heads", "encoder_layers", "decoder_layers",
                "d_ff", "source_len", "target_len", "global_block", "use_global_memory",
            )
            mismatched = [
                name for name in architecture_fields
                if name in saved_cfg and saved_cfg[name] != getattr(cfg, name)
            ]
            if mismatched:
                raise ValueError(f"Checkpoint architecture mismatch: {', '.join(mismatched)}")
            schedule_fields = ("steps", "batch_size", "grad_accum")
            changed_schedule = [
                name for name in schedule_fields
                if name in saved_cfg and saved_cfg[name] != getattr(cfg, name)
            ]
            if changed_schedule:
                raise ValueError(
                    "Checkpoint training schedule mismatch: "
                    f"{', '.join(changed_schedule)}. Resume with the original values."
                )
        if completed_steps >= cfg.steps:
            raise ValueError(
                f"Checkpoint is already at {completed_steps} steps, not below target {cfg.steps}"
            )
        if cfg.grad_accum % world_size:
            raise ValueError("--grad-accum must be divisible by the number of DDP processes")
        local_grad_accum = cfg.grad_accum // world_size
        sample_offset = completed_steps * cfg.batch_size * cfg.grad_accum
        local_samples = (
            (cfg.steps - completed_steps) * cfg.batch_size * local_grad_accum
        )
        samples_per_step = cfg.batch_size * local_grad_accum
        training = ContinuationDataset(
            prepared["tokens"],
            prepared["spans"],
            cfg.source_len,
            cfg.target_len,
            local_samples,
            SEED,
            sample_offset,
            rank,
            world_size,
            samples_per_step,
        )
        validation_source_len, validation_target_len = 512, 128
        validation = ContinuationDataset(
            prepared["validation_tokens"],
            prepared["validation_spans"],
            validation_source_len,
            validation_target_len,
            cfg.validation_batches * cfg.batch_size,
            SEED + 1_000_000,
        )
        loader = torch.utils.data.DataLoader(
            training, batch_size=cfg.batch_size, pin_memory=True
        )
        validation_loader = torch.utils.data.DataLoader(
            validation, batch_size=cfg.batch_size, pin_memory=True
        )
        model = LegalEncoderDecoder(cfg).to(device)
        parameters = parameter_count(model)
        if rank == 0:
            print(
                f"parameters: {parameters:,}; GPUs: {world_size}; "
                f"local accumulation: {local_grad_accum}; global accumulation: {cfg.grad_accum}"
            )
        if not 100_000_000 <= parameters <= 150_000_000:
            raise ValueError("Flagship model must remain inside the recommended 100M-150M range")
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
            )
        checkpoint_path, validation_loss, complete, completed_steps, stop_reason = train(
            model,
            loader,
            validation_loader,
            tokenizer,
            cfg,
            output,
            device,
            resume,
            args.run_steps,
            distributed,
            rank,
            world_size,
            args.max_hours,
            args.generation_reserve_hours,
            session_started,
            args.finalize_on_time_limit,
        )
        model = unwrap_model(model)
        if not complete:
            if rank == 0:
                partial = {
                    "status": "training_incomplete",
                    "stop_reason": stop_reason,
                    "completed_steps": completed_steps,
                    "checkpoint": str(checkpoint_path),
                    "tokenizer": str(tokenizer_path),
                    "prepared_data": str(output / "prepared_data.pt"),
                    "target_steps": cfg.steps,
                    "world_size": world_size,
                }
                (output / "manifest_partial.json").write_text(
                    json.dumps(partial, indent=2), encoding="utf-8"
                )
                print(json.dumps(partial, indent=2))
            return
        validation_predictions = distributed_generate(
            model, tokenizer, validation_prompts, cfg, device, distributed, rank, world_size
        )
        if rank == 0:
            (output / "validation_prediction_foundation.json").write_text(
                json.dumps(validation_predictions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        manifest = {
            "architecture": "12-layer bidirectional encoder + 8-layer causal decoder",
            "original_component": "gated token-level and pooled global cross-attention memory",
            "parameters": parameters,
            "completed_steps": completed_steps,
            "target_steps": cfg.steps,
            "stop_reason": stop_reason,
            "world_size": world_size,
            "training_tokens_processed": completed_steps * cfg.batch_size * cfg.grad_accum * (cfg.source_len + cfg.target_len),
            "validation_loss": validation_loss,
            "training_seconds": time.perf_counter() - session_started,
            "checkpoint": str(checkpoint_path),
            "recovery_checkpoint": str(output / "foundation_recovery.pt"),
            "tokenizer": str(tokenizer_path),
            "submission": None,
            "config": asdict(cfg),
        }
        if args.validation_only:
            if rank == 0:
                (output / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                print(json.dumps(manifest, indent=2))
                print("validation-only run complete; skipped official test generation")
            return
        test_holder = [legacy.load_test_data() if rank == 0 else None]
        if distributed:
            dist.broadcast_object_list(test_holder, src=0)
        test = test_holder[0]
        predictions = distributed_generate(
            model, tokenizer, test, cfg, device, distributed, rank, world_size
        )
        if rank != 0:
            return
        zip_path = legacy.validate_and_zip(predictions, test, output, "foundation")
        manifest["submission"] = str(zip_path)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--run-steps", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--generation-reserve-hours", type=float, default=0.0)
    parser.add_argument("--finalize-on-time-limit", action="store_true")
    parser.add_argument("--train-chars", type=int, default=500_000_000)
    parser.add_argument("--token-pool", type=int, default=100_000_000)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--output-dir", default="/kaggle/working/phase1_foundation")
    parser.add_argument("--tokenizer")
    parser.add_argument("--checkpoint")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--no-global-memory", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
