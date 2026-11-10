"""CS781 Phase 6 - Abstractive legal summarization from Phase 1 legal language models.

Every experiment starts from a Phase 1 checkpoint (decoder / encoder_decoder / hybrid / efficient); no other
pretrained weights are used for training or inference.  Only the official Phase 6 train.json is used for training;
a held-out validation split is carved from it (Phase 6 has no public dev split).

Pipeline of one run:
  0. (optional, --tapt-minutes) task-adaptive continued pretraining of the Phase 1 model with its OWN Phase 1
     objective on the train-split documents + summaries (no test text, no validation summaries);
  1. long-document input:  sentences are scored by TF-IDF centrality to the whole document + a position prior and the
     most salient ones are kept, in document order, up to the source budget (``--selection salience``; ``lead`` keeps
     the beginning).  Two architectures consume it differently:
       * causal models (decoder, efficient): prefix LM  [source][<bos> as separator][summary][<eos>], loss only on the
         summary; the efficient model's sliding-window attention can be widened to global attention in every N-th layer
         (``--global-every``);
       * encoder-decoder / hybrid: Fusion-in-Decoder - the source is cut into K chunks of 256 tokens (the Phase 1
         context), each chunk is encoded independently, and the decoder cross-attends to all K x 256 states;
  2. adaptation: full fine-tuning, Houlsby adapters, LoRA, LORMA+, or knowledge distillation (smaller student trained on
     the reference + the teacher's per-token distributions; all Phase 1 models share one tokenizer);
  3. checkpoint = lowest validation loss;  decoding = batched greedy search with KV cache, no-repeat-n-gram and
     repetition penalty; the decoding config and the output length (words) are chosen on validation documents with
     BERTScore (roberta-large, as the course evaluator; ROUGE-L if bert-score is unavailable);
  4. test generation -> prediction.json  [{"id": ..., "summary": ...}] -> submission ZIP, plus analysis files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from phase2_lner import load_phase1, make_student
from phase3_rr import IA3Linear, LoRALinear, LoRMAWrapped, locate_phase1_files, seed_everything

REPO_ID = "Exploration-Lab/CS-781-Capstone"
PHASE_DIR = "Phase_6_Summ"
BIG = 1 << 30


# ============================================================================= data

def obtain_data(data_dir):
    names = ("train.json", "test.json")
    if data_dir:
        found = {}
        for folder, _, files in os.walk(data_dir, followlinks=True):
            for name in files:
                if name in names and name not in found:
                    found[name] = Path(folder) / name
        if len(found) == 2:
            return found
        raise FileNotFoundError(f"--data-dir lacks {set(names) - set(found)}")
    from huggingface_hub import hf_hub_download, list_repo_files

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing: add the Kaggle secret HF and enable it")
    repo = [f for f in list_repo_files(REPO_ID, repo_type="dataset", token=token) if f.startswith(PHASE_DIR + "/")]
    print("Phase 6 files on HF:", repo, flush=True)
    found = {}
    for name in names:
        matches = [f for f in repo if f.lower().endswith("/" + name)]
        if not matches:
            raise FileNotFoundError(f"{name} not found under {PHASE_DIR}: {repo}")
        found[name] = Path(hf_hub_download(repo_id=REPO_ID, filename=matches[0], repo_type="dataset", token=token,
                                           force_download=True))
    return found


def _pick(record, keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _join(value):
    if isinstance(value, list):
        return " ".join(_join(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_join(v) for v in value.values())
    return str(value)


def load_records(path: Path, require_summary: bool):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for wrapper in ("data", "documents", "examples", "records"):
            if isinstance(raw.get(wrapper), (list, dict)):
                raw = raw[wrapper]
                break
    if isinstance(raw, dict):
        raw = [{**(v if isinstance(v, dict) else {"document": v}), "id": v.get("id", k) if isinstance(v, dict) else k}
               for k, v in raw.items()]
    out = []
    for i, item in enumerate(raw):
        rid = _pick(item, ("id", "doc_id", "case_id", "uid", "name"))
        doc = _pick(item, ("document", "text", "judgment", "judgement", "doc", "source", "case_text", "input"))
        if rid is None or doc is None:
            raise ValueError(f"{path.name} record #{i}: cannot find id/document, keys={list(item)}")
        summ = None
        if require_summary:
            summ = _pick(item, ("summary", "abstract", "reference", "target", "headnote", "output"))
            if summ is None:
                raise ValueError(f"{path.name}/{rid}: no summary field; keys={list(item)}")
            summ = " ".join(_join(summ).split())
        out.append({"id": rid, "document": " ".join(_join(doc).split()), "summary": summ})
    print(f"{path.name}: {len(out)} records; keys of first record: {list(raw[0]) if raw else []}", flush=True)
    return out


def text_key(text):
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


def pct(values, qs=(0.5, 0.9, 1.0)):
    v = sorted(values)
    return [v[min(len(v) - 1, int(q * (len(v) - 1)))] for q in qs] if v else None


# ============================================================================= long-document content selection

_SENT = re.compile(r"(?<=[.?!;])\s+(?=[\"'(\[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9]+")


def split_sentences(text, max_words=60):
    out = []
    for s in _SENT.split(text):
        words = s.split()
        for j in range(0, len(words), max_words):
            piece = " ".join(words[j:j + max_words])
            if piece:
                out.append(piece)
    return out


def select_source(text, budget_words, method):
    """Importance-based context selection (TF-IDF centrality + position prior), kept in document order."""
    words = text.split()
    if len(words) <= budget_words:
        return text
    if method == "lead":
        return " ".join(words[:budget_words])
    sents = split_sentences(text)
    toks = [_WORD.findall(s.lower()) for s in sents]
    df = Counter(t for ts in toks for t in set(ts))
    n = len(sents)
    idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
    centroid = Counter()
    vecs = []
    for ts in toks:
        tf = Counter(ts)
        v = {t: c * idf[t] for t, c in tf.items()}
        vecs.append(v)
        centroid.update(v)
    cnorm = math.sqrt(sum(x * x for x in centroid.values())) or 1.0
    scores = []
    for i, v in enumerate(vecs):
        vnorm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        cos = sum(x * centroid[t] for t, x in v.items()) / (vnorm * cnorm)
        position = 1.0 - i / max(1, n - 1)
        scores.append(cos + 0.15 * position + (0.3 if i < 3 else 0.0) + (0.1 if i >= n - 3 else 0.0))
    chosen, used = set(), 0
    for i in sorted(range(n), key=lambda i: -scores[i]):
        w = len(sents[i].split())
        if used + w > budget_words and chosen:
            continue
        chosen.add(i)
        used += w
        if used >= budget_words:
            break
    return " ".join(sents[i] for i in sorted(chosen))


# ============================================================================= model internals (custom forward + cache)

def sinusoid_at(pos, dim):
    """Same formula as phase1_pipeline.sinusoidal_positions, at arbitrary (B, L) position ids."""
    scale = torch.exp(torch.arange(0, dim, 2, device=pos.device).float() * (-math.log(10000.0) / dim))
    angle = pos.float().unsqueeze(-1) * scale
    out = torch.zeros(*pos.shape, dim, device=pos.device)
    out[..., 0::2] = torch.sin(angle)
    out[..., 1::2] = torch.cos(angle)
    return out


def rope_at(x, pos):
    """Same rotation as phase1_pipeline.rope, at (B, L) position ids.  x: (B, H, L, D)."""
    dim = x.size(-1)
    freq = torch.exp(torch.arange(0, dim, 2, device=x.device).float() * (-math.log(10000.0) / dim))
    angle = pos.float()[:, None, :, None] * freq
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _safe(allowed):
    """Rows with no allowed key (padding queries) attend to everything -> avoids NaN; their outputs are ignored."""
    empty = ~allowed.any(-1, keepdim=True)
    return allowed | empty


def _sdpa(q, k, v, allowed):
    dt = v.dtype
    return F.scaled_dot_product_attention(q.to(dt), k.to(dt), v, attn_mask=_safe(allowed)[:, None])


class Adapter(nn.Module):
    """Houlsby bottleneck adapter: h + W_up(GELU(W_down(LN(h)))), W_up zero-initialised."""

    def __init__(self, dim, bottleneck):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.gelu(self.down(self.norm(h))))


def _ad(adapters, i, which, h):
    if adapters is None:
        return h
    return adapters[2 * i + which](h)


def mha_self(mha, x, pos, valid, cache, causal=True):
    """nn.MultiheadAttention self-attention computed from its weights (batch_first), with KV cache."""
    B, L, d = x.shape
    H = mha.num_heads
    q, k, v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, -1)
    q, k, v = (t.view(B, L, H, d // H).transpose(1, 2) for t in (q, k, v))
    kpos, kvalid = pos, valid
    if cache is not None:
        if "k" in cache:
            k = torch.cat((cache["k"], k), 2)
            v = torch.cat((cache["v"], v), 2)
            kpos = torch.cat((cache["pos"], pos), 1)
            kvalid = torch.cat((cache["valid"], valid), 1)
        cache.update(k=k, v=v, pos=kpos, valid=kvalid)
    allowed = kvalid[:, None, :].expand(B, L, -1)
    if causal:
        allowed = allowed & (kpos[:, None, :] <= pos[:, :, None])
    out = _sdpa(q, k, v, allowed).transpose(1, 2).reshape(B, L, d)
    return mha.out_proj(out)


def mha_cross(mha, x, mem_k, mem_v, mem_valid):
    B, L, d = x.shape
    H = mha.num_heads
    q = F.linear(x, mha.in_proj_weight[:d], None if mha.in_proj_bias is None else mha.in_proj_bias[:d])
    q = q.view(B, L, H, d // H).transpose(1, 2)
    allowed = mem_valid[:, None, :].expand(B, L, -1)
    return mha.out_proj(_sdpa(q, mem_k, mem_v, allowed).transpose(1, 2).reshape(B, L, d))


def cross_kv(mha, memory):
    B, M, d = memory.shape
    H = mha.num_heads
    w, b = mha.in_proj_weight, mha.in_proj_bias
    k = F.linear(memory, w[d:2 * d], None if b is None else b[d:2 * d]).view(B, M, H, d // H).transpose(1, 2)
    v = F.linear(memory, w[2 * d:], None if b is None else b[2 * d:]).view(B, M, H, d // H).transpose(1, 2)
    return k, v


def ffn(layer, h):
    return layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))


def enc_layer(layer, x, pos, valid, cache, causal, adapters=None, i=0):
    """norm_first nn.TransformerEncoderLayer."""
    x = x + layer.dropout1(_ad(adapters, i, 0, mha_self(layer.self_attn, layer.norm1(x), pos, valid, cache, causal)))
    return x + layer.dropout2(_ad(adapters, i, 1, ffn(layer, layer.norm2(x))))


def dec_layer(layer, y, pos, valid, cache, mem_kv, mem_valid, adapters=None, i=0):
    """norm_first nn.TransformerDecoderLayer."""
    y = y + layer.dropout1(_ad(adapters, i, 0, mha_self(layer.self_attn, layer.norm1(y), pos, valid, cache, True)))
    y = y + layer.dropout2(mha_cross(layer.multihead_attn, layer.norm2(y), mem_kv[0], mem_kv[1], mem_valid))
    return y + layer.dropout3(_ad(adapters, i, 1, ffn(layer, layer.norm3(y))))


def gqa(att, x, pos, valid, cache):
    """phase1 GroupedQueryAttention (RoPE + sliding window) at arbitrary positions, with KV cache."""
    B, L, _ = x.shape
    H, KV, D = att.heads, att.kv_heads, att.head_dim
    q = rope_at(att.q(x).view(B, L, H, D).transpose(1, 2), pos)
    k = rope_at(att.k(x).view(B, L, KV, D).transpose(1, 2), pos)
    v = att.v(x).view(B, L, KV, D).transpose(1, 2)
    kpos, kvalid = pos, valid
    if cache is not None:
        if "k" in cache:
            k = torch.cat((cache["k"], k), 2)
            v = torch.cat((cache["v"], v), 2)
            kpos = torch.cat((cache["pos"], pos), 1)
            kvalid = torch.cat((cache["valid"], valid), 1)
        cache.update(k=k, v=v, pos=kpos, valid=kvalid)
    rep = H // KV
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    dist = pos[:, :, None] - kpos[:, None, :]
    allowed = (dist >= 0) & (dist < att.window) & kvalid[:, None, :]
    return att.out(_sdpa(q, k, v, allowed).transpose(1, 2).reshape(B, L, -1))


class Summarizer(nn.Module):
    """Wraps a Phase 1 LM.  kind == 'prefix' (decoder, efficient) or 'seq2seq' (encoder_decoder, hybrid, FiD)."""

    def __init__(self, lm, arch, adapter_dim=0, global_every=0, dropout=0.0):
        super().__init__()
        self.lm, self.arch = lm, arch
        self.kind = "prefix" if arch in ("decoder", "efficient") else "seq2seq"
        self.d = lm.cfg.d_model
        if arch == "efficient":
            n_layers = len(lm.blocks)
            if global_every:
                for i, block in enumerate(lm.blocks):
                    if (i + 1) % global_every == 0:
                        block.attention.window = BIG
        elif arch == "decoder":
            n_layers = len(lm.blocks.layers)
        else:
            n_layers = len(lm.decoder.layers)
        self.adapters = nn.ModuleList(Adapter(self.d, adapter_dim) for _ in range(2 * n_layers)) if adapter_dim else None
        for m in lm.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout

    # ---- prefix LM -------------------------------------------------------------------------------------------
    def prefix_hidden(self, ids, pos, valid, caches=None, fusion_state=None):
        lm = self.lm
        if self.arch == "efficient":
            x = lm.embedding(ids)
            for i, block in enumerate(lm.blocks):
                c = caches[i] if caches is not None else None
                x = x + _ad(self.adapters, i, 0, gqa(block.attention, block.norm1(x), pos, valid, c))
                y = block.norm2(x)
                x = x + _ad(self.adapters, i, 1, block.down(F.silu(block.gate(y)) * block.up(y)))
            return lm.norm(x)
        x = lm.embedding(ids) + sinusoid_at(pos, self.d).to(lm.embedding.weight.dtype)
        for i, layer in enumerate(lm.blocks.layers):
            x = enc_layer(layer, x, pos, valid, caches[i] if caches is not None else None, True, self.adapters, i)
        w = valid.unsqueeze(-1).float()
        csum, ccount = (x.float() * w).cumsum(1), w.cumsum(1)
        if fusion_state is not None:
            if "sum" in fusion_state:
                csum = csum + fusion_state["sum"][:, None]
                ccount = ccount + fusion_state["count"][:, None]
            fusion_state["sum"], fusion_state["count"] = csum[:, -1], ccount[:, -1]
        x = lm.fusion(x, (csum / ccount.clamp_min(1)).to(x.dtype))
        return lm.norm(x)

    # ---- FiD encoder-decoder -----------------------------------------------------------------------------------
    def encode_chunks(self, chunks, chunk_valid):
        """chunks: (B, K, L) ids, chunk_valid: (B, K, L) bool -> memory (B, K*L, d), mem_valid (B, K*L)."""
        lm = self.lm
        B, K, L = chunks.shape
        ids = chunks.view(B * K, L)
        valid = chunk_valid.view(B * K, L)
        rows = valid.any(1)
        pos = torch.arange(L, device=ids.device).expand(B * K, L)
        x = lm.embedding(ids) + sinusoid_at(pos, self.d).to(lm.embedding.weight.dtype)
        out = torch.zeros_like(x)
        if rows.any():
            xs, vs, ps = x[rows], valid[rows], pos[rows]
            if self.arch == "encoder_decoder":
                for layer in lm.encoder.layers:
                    xs = enc_layer(layer, xs, ps, vs, None, False)
            else:
                lengths = vs.sum(1).clamp_min(1).cpu()
                packed = pack_padded_sequence(xs, lengths, batch_first=True, enforce_sorted=False)
                xs, _ = lm.recurrent(packed)
                xs, _ = pad_packed_sequence(xs, batch_first=True, total_length=L)
            out[rows] = xs.to(out.dtype)
        return out.view(B, K * L, -1), valid.view(B, K * L)

    def decoder_hidden(self, tgt, pos, valid, memory, mem_valid, mem_kv=None, caches=None):
        lm = self.lm
        if mem_kv is None:
            mem_kv = [cross_kv(layer.multihead_attn, memory) for layer in lm.decoder.layers]
        y = lm.embedding(tgt) + sinusoid_at(pos, self.d).to(lm.embedding.weight.dtype)
        for i, layer in enumerate(lm.decoder.layers):
            y = dec_layer(layer, y, pos, valid, caches[i] if caches is not None else None, mem_kv[i], mem_valid,
                          self.adapters, i)
        w = mem_valid.unsqueeze(-1).to(memory.dtype)
        summary = ((memory * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)).expand(-1, y.size(1), -1)
        return lm.norm(lm.fusion(y, summary.to(y.dtype)))

    def n_layers(self):
        if self.arch == "efficient":
            return len(self.lm.blocks)
        if self.arch == "decoder":
            return len(self.lm.blocks.layers)
        return len(self.lm.decoder.layers)

    def head(self, h):
        return self.lm.head(h)

    def target_logits(self, batch):
        """Teacher-forced logits at the summary positions only (B*T_valid, V) in batch['label_flat'] order."""
        if self.kind == "prefix":
            h = self.prefix_hidden(batch["ids"], batch["pos"], batch["valid"])
        else:
            memory, mem_valid = self.encode_chunks(batch["chunks"], batch["chunk_valid"])
            h = self.decoder_hidden(batch["dec_in"], batch["dec_pos"], batch["dec_valid"], memory, mem_valid)
        return self.head(h[batch["label_mask"]])


# ============================================================================= adaptation

def wrap_linears(summ, method, rank):
    lm = summ.lm

    def wrap(linear):
        if method == "lora":
            return LoRALinear(linear, rank=rank, alpha=2.0 * rank)
        if method == "lorma":
            return LoRMAWrapped(linear, rank=rank, alpha=float(rank))
        if method == "ia3":
            return IA3Linear(linear)
        raise ValueError(method)

    if summ.arch == "efficient":
        for block in lm.blocks:
            att = block.attention
            att.q, att.v, att.out = wrap(att.q), wrap(att.v), wrap(att.out)
            block.gate, block.up, block.down = wrap(block.gate), wrap(block.up), wrap(block.down)
    else:
        layers = list(lm.blocks.layers) if summ.arch == "decoder" else list(lm.decoder.layers)
        if summ.arch == "encoder_decoder":
            layers += list(lm.encoder.layers)
        for layer in layers:
            layer.linear1, layer.linear2 = wrap(layer.linear1), wrap(layer.linear2)


def apply_method(summ, method, rank):
    if method in ("full", "distill"):
        return
    for p in summ.lm.parameters():
        p.requires_grad = False
    if method in ("lora", "lorma", "ia3"):
        wrap_linears(summ, method, rank)
    elif method != "adapter":
        raise ValueError(method)
    for m in summ.lm.modules():  # norms + fusion gate stay trainable in every PEFT method
        if isinstance(m, (nn.LayerNorm,)) or m.__class__.__name__ in ("RMSNorm", "GatedContextFusion"):
            for p in m.parameters():
                p.requires_grad = True
    if summ.adapters is not None:
        for p in summ.adapters.parameters():
            p.requires_grad = True


# ============================================================================= batching

class Encoder:
    def __init__(self, tokenizer, args, summ_kind):
        self.tok = tokenizer
        self.args = args
        self.kind = summ_kind
        self.pad, self.bos, self.eos = (tokenizer.token_to_id(t) for t in ("<pad>", "<bos>", "<eos>"))
        self.unk = tokenizer.token_to_id("<unk>")

    def source_ids(self, docs):
        a = self.args
        budget = a.src_tokens if self.kind == "prefix" else a.fid_chunks * a.chunk_len
        texts = [select_source(d, int(budget * a.words_per_token), a.selection) for d in docs]
        out = []
        for s in range(0, len(texts), 256):
            out += [e.ids[:budget] or [self.eos] for e in self.tok.encode_batch(texts[s:s + 256])]
        return out

    def target_ids(self, summaries):
        out = []
        for s in range(0, len(summaries), 256):
            for e in self.tok.encode_batch(summaries[s:s + 256]):
                ids = e.ids[:self.args.tgt_tokens]
                out.append(ids + ([self.eos] if len(e.ids) <= self.args.tgt_tokens else []))
        return out

    def train_batch(self, src, tgt, device):
        if self.kind == "prefix":
            seqs = [s + [self.bos] + t for s, t in zip(src, tgt)]
            inputs = [q[:-1] for q in seqs]
            L = max(map(len, inputs))
            B = len(seqs)
            ids = torch.full((B, L), self.pad, dtype=torch.long)
            valid = torch.zeros((B, L), dtype=torch.bool)
            label_mask = torch.zeros((B, L), dtype=torch.bool)
            labels = []
            for r, (q, s) in enumerate(zip(seqs, src)):
                n = len(q) - 1
                ids[r, :n] = torch.tensor(q[:-1])
                valid[r, :n] = True
                label_mask[r, len(s):n] = True  # predicting t[0] from the separator onwards
                labels += q[len(s) + 1:]
            pos = torch.arange(L).expand(B, L).clone()
            batch = {"ids": ids, "pos": pos, "valid": valid}
        else:
            K, Lc = self.args.fid_chunks, self.args.chunk_len
            B = len(src)
            Kb = max(1, max(math.ceil(len(s) / Lc) for s in src))
            chunks = torch.full((B, Kb, Lc), self.pad, dtype=torch.long)
            chunk_valid = torch.zeros((B, Kb, Lc), dtype=torch.bool)
            for r, s in enumerate(src):
                for j in range(0, len(s), Lc):
                    piece = s[j:j + Lc]
                    chunks[r, j // Lc, :len(piece)] = torch.tensor(piece)
                    chunk_valid[r, j // Lc, :len(piece)] = True
            T = max(map(len, tgt))
            dec_in = torch.full((B, T), self.pad, dtype=torch.long)
            dec_valid = torch.zeros((B, T), dtype=torch.bool)
            labels = []
            for r, t in enumerate(tgt):
                inp = [self.bos] + t[:-1]
                dec_in[r, :len(inp)] = torch.tensor(inp)
                dec_valid[r, :len(inp)] = True
                labels += t
            batch = {"chunks": chunks, "chunk_valid": chunk_valid, "dec_in": dec_in,
                     "dec_pos": torch.arange(T).expand(B, T).clone(), "dec_valid": dec_valid}
            label_mask = dec_valid.clone()
        batch["label_mask"] = label_mask
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return {k: v.to(device) for k, v in batch.items()}


# ============================================================================= generation

@torch.no_grad()
def generate(models, enc, src_list, device, cfg, batch_size):
    """Batched greedy decoding with KV cache for one model or an ensemble of same-kind models (log-prob average)."""
    for m in models:
        m.eval()
    kind = models[0].kind
    pad, bos, eos, unk = enc.pad, enc.bos, enc.eos, enc.unk
    order = sorted(range(len(src_list)), key=lambda i: -len(src_list[i]))
    results = [None] * len(src_list)
    n_rep = cfg["no_repeat_ngram"]
    for s in range(0, len(order), batch_size):
        rows = order[s:s + batch_size]
        B = len(rows)
        states = []
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if kind == "prefix":
                prompts = [src_list[r] + [bos] for r in rows]
                L = max(map(len, prompts))
                ids = torch.full((B, L), pad, dtype=torch.long, device=device)
                valid = torch.zeros((B, L), dtype=torch.bool, device=device)
                for j, p in enumerate(prompts):
                    ids[j, L - len(p):] = torch.tensor(p, device=device)
                    valid[j, L - len(p):] = True
                pos = (valid.long().cumsum(1) - 1).clamp_min(0)
                next_pos = pos[:, -1] + 1
                logps = []
                for m in models:
                    caches = [dict() for _ in range(m.n_layers())]
                    fstate = {}
                    h = m.prefix_hidden(ids, pos, valid, caches, fstate)
                    logps.append(F.log_softmax(m.head(h[:, -1]).float(), -1))
                    states.append((caches, fstate))
            else:
                batch = enc.train_batch([src_list[r] for r in rows], [[eos]] * B, device)
                logps = []
                for m in models:
                    memory, mem_valid = m.encode_chunks(batch["chunks"], batch["chunk_valid"])
                    mem_kv = [cross_kv(layer.multihead_attn, memory) for layer in m.lm.decoder.layers]
                    caches = [dict() for _ in range(m.n_layers())]
                    tgt = torch.full((B, 1), bos, dtype=torch.long, device=device)
                    p0 = torch.zeros((B, 1), dtype=torch.long, device=device)
                    v0 = torch.ones((B, 1), dtype=torch.bool, device=device)
                    h = m.decoder_hidden(tgt, p0, v0, memory, mem_valid, mem_kv, caches)
                    logps.append(F.log_softmax(m.head(h[:, -1]).float(), -1))
                    states.append((caches, memory, mem_valid, mem_kv))
                next_pos = torch.ones(B, dtype=torch.long, device=device)
        generated = [[] for _ in range(B)]
        seen = [dict() for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for step in range(cfg["max_new"]):
            logp = sum(logps) / len(logps)
            logp[:, pad] = -1e9
            logp[:, bos] = -1e9
            if unk is not None:
                logp[:, unk] = -1e9
            if step < cfg["min_new"]:
                logp[:, eos] = -1e9
            if cfg["repetition_penalty"] != 1.0:
                for j in range(B):
                    if generated[j]:
                        idx = torch.tensor(sorted(set(generated[j])), device=device)
                        logp[j, idx] = logp[j, idx] * cfg["repetition_penalty"]
            if n_rep:
                for j in range(B):
                    g = generated[j]
                    if len(g) >= n_rep - 1:
                        banned = seen[j].get(tuple(g[len(g) - n_rep + 1:]))
                        if banned:
                            logp[j, list(banned)] = -1e9
            nxt = logp.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            for j, t in enumerate(nxt.tolist()):
                if done[j]:
                    continue
                g = generated[j]
                g.append(t)
                if n_rep and len(g) >= n_rep:
                    seen[j].setdefault(tuple(g[-n_rep:-1]), set()).add(g[-1])
            done = done | (nxt == eos)
            if bool(done.all()):
                break
            step_ids = nxt[:, None]
            step_pos = next_pos[:, None]
            step_valid = torch.ones((B, 1), dtype=torch.bool, device=device)
            logps = []
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                for m, st in zip(models, states):
                    if kind == "prefix":
                        h = m.prefix_hidden(step_ids, step_pos, step_valid, st[0], st[1])
                    else:
                        h = m.decoder_hidden(step_ids, step_pos, step_valid, st[1], st[2], st[3], st[0])
                    logps.append(F.log_softmax(m.head(h[:, -1]).float(), -1))
            next_pos = next_pos + 1
        for j, r in enumerate(rows):
            g = generated[j]
            if eos in g:
                g = g[:g.index(eos)]
            results[r] = g
    return results


def clean_text(text):
    text = " ".join(text.replace("�", " ").split())
    return text


def truncate_words(text, n_words):
    """Cut to at most n_words, preferring to end at a sentence boundary in the last 40% of the budget."""
    words = text.split()
    if n_words is None or len(words) <= n_words:
        return text
    cut = " ".join(words[:n_words])
    end = max(cut.rfind(". "), cut.rfind(".\""), cut.rfind("; "))
    if end > 0.6 * len(cut):
        return cut[:end + 1]
    return cut


# ============================================================================= metrics

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(t):
    return _TOK.findall(t.lower())


def rouge_n(c, r, n=1):
    ct, rt = _tokens(c), _tokens(r)
    cg = Counter(tuple(ct[i:i + n]) for i in range(len(ct) - n + 1))
    rg = Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
    overlap = sum((cg & rg).values())
    if not overlap:
        return 0.0
    p, rc = overlap / max(1, sum(cg.values())), overlap / max(1, sum(rg.values()))
    return 2 * p * rc / (p + rc)


def rouge_l(c, r):
    a, b = _tokens(c), _tokens(r)
    if not a or not b:
        return 0.0
    if len(a) * len(b) > 4_000_000:  # keep it cheap for very long texts
        a, b = a[:2000], b[:2000]
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, rc = lcs / len(a), lcs / len(b)
    return 2 * p * rc / (p + rc)


class BertScorer:
    """bert_score.score(lang='en') = roberta-large, the course evaluator's setting.  Evaluation only."""

    def __init__(self, enabled, device):
        self.ok = False
        if not enabled:
            return
        try:
            from bert_score import BERTScorer
            self.bs = BERTScorer(lang="en", batch_size=16, device=str(device))  # roberta-large, layer 17, no rescale
            self.ok = True
        except Exception as error:  # noqa: BLE001
            print("bert-score unavailable -> ROUGE-L is used for tuning:", error, flush=True)

    def __call__(self, cands, refs):
        if not self.ok:
            return None
        try:
            _, _, f = self.bs.score([c or "." for c in cands], refs)
            return float(f.mean())
        except Exception as error:  # noqa: BLE001
            print("bert-score failed -> ROUGE-L is used:", error, flush=True)
            self.ok = False
            return None


def text_stats(cands, refs, docs):
    def ngrams(t, n):
        w = _tokens(t)
        return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]

    rep3, novel3, num_recall = [], [], []
    for c, r, d in zip(cands, refs, docs):
        g = ngrams(c, 3)
        rep3.append(1 - len(set(g)) / len(g) if g else 0.0)
        dg = set(ngrams(d, 3))
        novel3.append(sum(x not in dg for x in g) / len(g) if g else 0.0)
        rnums = set(re.findall(r"\b\d+[a-z]?\b", r.lower()))
        cnums = set(re.findall(r"\b\d+[a-z]?\b", c.lower()))
        if rnums:
            num_recall.append(len(rnums & cnums) / len(rnums))
    m = lambda v: round(sum(v) / len(v), 4) if v else None
    return {"repeated_trigram_rate": m(rep3), "novel_trigram_rate_vs_source": m(novel3),
            "number_and_section_recall": m(num_recall), "mean_words": m([len(c.split()) for c in cands]),
            "ref_mean_words": m([len(r.split()) for r in refs])}


def score_set(cands, refs, scorer):
    out = {"rouge1_f1": sum(rouge_n(c, r, 1) for c, r in zip(cands, refs)) / max(1, len(refs)),
           "rougeL_f1": sum(rouge_l(c, r) for c, r in zip(cands, refs)) / max(1, len(refs))}
    bs = scorer(cands, refs)
    if bs is not None:
        out["bertscore_f1"] = bs
    out["selection_score"] = out.get("bertscore_f1", out["rougeL_f1"])
    return out


# ============================================================================= TAPT (continued Phase 1 objective)

def tapt(lm, arch, tokenizer, texts, args, device):
    eos, bos = tokenizer.token_to_id("<eos>"), tokenizer.token_to_id("<bos>")
    pieces = []
    for s in range(0, len(texts), 500):
        for e in tokenizer.encode_batch(texts[s:s + 500]):
            pieces.append(np.asarray(e.ids + [eos], dtype=np.int32))
    stream = np.concatenate(pieces)
    block = 2 * args.chunk_len if arch in ("encoder_decoder", "hybrid") else args.chunk_len + 1
    n = len(stream) // block
    blocks = torch.from_numpy(stream[:n * block].astype(np.int64)).view(n, block)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(32, n // 200)
    val, tr = blocks[perm[:n_val]], blocks[perm[n_val:]]
    print(f"TAPT: {len(texts)} texts, {len(stream) / 1e6:.1f}M tokens, {len(tr)} blocks of {block}", flush=True)
    lm.to(device)
    bs = args.tapt_batch

    @torch.no_grad()
    def vloss():
        lm.eval()
        out = []
        for s in range(0, min(len(val), 256), bs):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out.append(float(lm.loss(val[s:s + bs].to(device), bos)))
        lm.train()
        return sum(out) / len(out)

    before = vloss()
    print(f"TAPT held-out loss before {before:.4f}", flush=True)
    opt = torch.optim.AdamW(lm.parameters(), lr=args.tapt_lr, weight_decay=0.01)
    budget = args.tapt_minutes * 60
    total = max(100, len(tr) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 100, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lm.train()
    start, step, ptr = time.time(), 0, 0
    order = torch.randperm(len(tr), generator=g)
    while step < total and time.time() - start < budget:
        if ptr + bs > len(order):
            order, ptr = torch.randperm(len(tr), generator=g), 0
        batch = tr[order[ptr:ptr + bs]].to(device)
        ptr += bs
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = lm.loss(batch, bos)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step == 50:
            rate = step / (time.time() - start)
            total = max(100, min(total, int(rate * budget * 0.95)))
            print(f"TAPT ~{rate:.2f} steps/s -> {total} steps ({total * bs / len(tr):.2f} passes)", flush=True)
        if step % 500 == 0:
            print(f"TAPT step {step} loss {float(loss):.4f} {(time.time() - start) / 60:.1f} min", flush=True)
    after = vloss()
    print(f"TAPT held-out loss after {after:.4f} (before {before:.4f}); {step} steps", flush=True)
    return {"val_loss_before": before, "val_loss_after": after, "steps": step, "tokens": int(len(stream)),
            "minutes": (time.time() - start) / 60}


# ============================================================================= training

def load_teacher(path, device, artifact_zip, workdir):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**saved["run_config"])
    checkpoint, _, _ = locate_phase1_files(a.model, artifact_zip or a.artifact_zip, workdir)
    lm, _, _ = load_phase1(checkpoint, a.model)
    t = Summarizer(lm, a.model, a.adapter_dim if a.method == "adapter" else 0, a.global_every)
    apply_method(t, a.method, a.rank)
    t.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in saved["model"].items()})
    for p in t.parameters():
        p.requires_grad = False
    return t.to(device).half().eval(), a


def train(model, enc, train_src, train_tgt, val_src, val_tgt, args, device, teacher=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    budget = args.time_budget_min * 60
    est = max(50, args.epochs * math.ceil(len(train_src) / (args.batch_docs * args.grad_accum)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 50, max(0.05, 1 - s / est)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start, last_eval = time.time(), time.time()
    best, best_state, history = float("inf"), None, []
    step, opt_steps, seen = 0, 0, 0

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot, cnt = 0.0, 0
        for s in range(0, len(val_src), args.batch_docs * 2):
            b = enc.train_batch(val_src[s:s + args.batch_docs * 2], val_tgt[s:s + args.batch_docs * 2], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                lg = model.target_logits(b).float()
            tot += float(F.cross_entropy(lg, b["labels"], reduction="sum"))
            cnt += int(b["labels"].numel())
        model.train()
        return tot / max(1, cnt)

    def evaluate(tag):
        nonlocal best, best_state, last_eval
        v = val_loss()
        row = {"tag": tag, "docs_seen": seen, "opt_steps": opt_steps, "minutes": round((time.time() - start) / 60, 2),
               "val_loss": round(v, 4), "val_ppl": round(math.exp(min(v, 20)), 2)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if v < best:
            best = v
            best_state = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
        last_eval = time.time()

    evaluate("start")
    model.train()
    done = False
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_src)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for s in range(0, len(order), args.batch_docs):
            rows = order[s:s + args.batch_docs]
            b = enc.train_batch([train_src[r] for r in rows], [train_tgt[r] for r in rows], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                lg = model.target_logits(b).float()
                tl = teacher.target_logits(b).float() if teacher is not None else None
            loss = F.cross_entropy(lg, b["labels"], label_smoothing=args.label_smoothing)
            if tl is not None:
                T = args.temperature
                kd = F.kl_div(F.log_softmax(lg / T, -1), F.log_softmax(tl / T, -1), log_target=True,
                              reduction="batchmean") * T * T
                loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
            scaler.scale(loss / args.grad_accum).backward()
            step += 1
            seen += len(rows)
            if step % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                opt_steps += 1
                if opt_steps == 30:
                    rate = opt_steps / (time.time() - start)
                    est = max(60, min(est, int(rate * budget * 0.93)))
                    print(f"~{rate:.3f} opt steps/s -> schedule over {est} steps "
                          f"({est * args.batch_docs * args.grad_accum / max(1, len(train_src)):.2f} epochs)", flush=True)
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
                model.train()
        if done:
            break
        evaluate(f"end_epoch{epoch}")
        model.train()
    evaluate("final")
    model.load_state_dict(best_state)
    return best, history, time.time() - start


# ============================================================================= decode/length tuning

def decode_configs(args, tgt_lengths):
    med = int(np.median(tgt_lengths)) if tgt_lengths else 200
    max_new = min(args.tgt_tokens, int(np.percentile(tgt_lengths, 90)) + 16) if tgt_lengths else args.tgt_tokens
    min_new = min(int(0.5 * med), max_new // 2)
    base = {"max_new": max(32, max_new), "min_new": max(8, min_new)}
    configs = [dict(base, name="nr4_rp1.0", no_repeat_ngram=4, repetition_penalty=1.0),
               dict(base, name="nr3_rp1.15", no_repeat_ngram=3, repetition_penalty=1.15)]
    return configs[:args.n_decode_configs]


def tune_decoding(models, enc, tokenizer, val_docs, val_src, val_refs, args, device, scorer):
    lengths = sorted(set(int(x) for x in args.length_grid.split(",")))
    report, best = [], None
    for cfg in decode_configs(args, [len(t) for t in args._train_tgt_lengths]):
        t0 = time.time()
        ids = generate(models, enc, val_src, device, cfg, args.gen_batch)
        secs = time.time() - t0
        texts = [clean_text(tokenizer.decode(g, skip_special_tokens=True)) for g in ids]
        for w in lengths + [0]:
            cands = [truncate_words(t, w if w else None) for t in texts]
            sc = score_set(cands, val_refs, scorer)
            row = {"decode": cfg["name"], "max_words": w or "untruncated", **{k: round(v, 4) for k, v in sc.items()},
                   "mean_words": round(sum(len(c.split()) for c in cands) / max(1, len(cands)), 1),
                   "gen_seconds": round(secs, 1)}
            report.append(row)
            print(json.dumps(row), flush=True)
            if best is None or sc["selection_score"] > best[0]:
                best = (sc["selection_score"], cfg, w or None, texts, sc)
    return best, report


# ============================================================================= main

def build_model(args, device, tokenizer, texts_for_tapt):
    checkpoint, tokenizer_path, artifact = locate_phase1_files(args.model, args.artifact_zip, Path(args.output_dir) / "phase1")
    lm, cfg, saved = load_phase1(checkpoint, args.model)
    phase1_params = sum(p.numel() for p in lm.parameters())
    tapt_info = None
    if args.tapt_minutes > 0:
        tapt_info = tapt(lm, args.model, tokenizer, texts_for_tapt, args, device)
    if args.method == "distill":
        if args.model != "decoder":
            raise ValueError("The distillation student recipe uses the decoder Phase 1 model")
        lm = make_student(lm.cpu(), layers=args.student_layers)
    summ = Summarizer(lm, args.model, args.adapter_dim if args.method == "adapter" else 0, args.global_every,
                      args.dropout)
    apply_method(summ, args.method, args.rank)
    return summ.to(device), cfg, saved, artifact, phase1_params, tapt_info


def load_split(args):
    """Official train/test, leakage filter, seeded validation split (identical for every run with the same seed)."""
    files = obtain_data(args.data_dir)
    records = load_records(files["train.json"], True)
    test = load_records(files["test.json"], False)
    if len({t["id"] for t in test}) != len(test):
        raise RuntimeError("Duplicate ids in test.json")
    test_ids = {t["id"] for t in test}
    test_keys = {text_key(t["document"]) for t in test}
    before = len(records)
    leak = {"train_test_id_overlap": sum(r["id"] in test_ids for r in records),
            "train_test_document_overlap": sum(text_key(r["document"]) in test_keys for r in records)}
    records = [r for r in records if r["id"] not in test_ids and text_key(r["document"]) not in test_keys]
    records = [r for r in records if r["summary"] and r["document"]]
    leak.update(train_before=before, train_after_filter=len(records))
    print("LEAKAGE CHECK:", json.dumps(leak), flush=True)
    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    n_val = min(args.val_docs, max(20, len(records) // 10))
    val = [records[i] for i in order[:n_val]]
    train_recs = [records[i] for i in order[n_val:]]
    if args.limit_train:
        train_recs = train_recs[:args.limit_train]
    gen_val = val[:args.val_gen_docs]
    stats = {"train": len(train_recs), "val": len(val), "val_generation": len(gen_val), "test": len(test),
             "doc_words_p50_p90_max": pct([len(r["document"].split()) for r in records]),
             "summary_words_p50_p90_max": pct([len(r["summary"].split()) for r in records]),
             "test_doc_words_p50_p90_max": pct([len(r["document"].split()) for r in test])}
    print("DATA:", json.dumps(stats), flush=True)

    return train_recs, val, gen_val, test, leak, stats


def run(args):
    t0 = time.time()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"p6_{args.model}_{args.method}"
    out = Path(args.output_dir or f"/kaggle/working/{name}")
    args.output_dir = str(out)
    out.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_recs, val, gen_val, test, leak, stats = load_split(args)

    from tokenizers import Tokenizer
    _, tokenizer_path, _ = locate_phase1_files(args.model, args.artifact_zip, out / "phase1")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tok_md5 = hashlib.md5(Path(tokenizer_path).read_bytes()).hexdigest()

    tapt_texts = [r["document"] for r in train_recs] + [r["summary"] for r in train_recs]
    summ, phase1_cfg, phase1_saved, artifact, phase1_params, tapt_info = build_model(args, device, tokenizer, tapt_texts)
    enc = Encoder(tokenizer, args, summ.kind)

    t_enc = time.time()
    train_src = enc.source_ids([r["document"] for r in train_recs])
    train_tgt = enc.target_ids([r["summary"] for r in train_recs])
    val_src = enc.source_ids([r["document"] for r in val])
    val_tgt = enc.target_ids([r["summary"] for r in val])
    test_src = enc.source_ids([r["document"] for r in test])
    args._train_tgt_lengths = train_tgt
    print(f"tokenised in {time.time() - t_enc:.0f}s; source tokens p50/p90/max {pct([len(s) for s in train_src])}; "
          f"target tokens p50/p90/max {pct([len(t) for t in train_tgt])} (cap {args.tgt_tokens}); "
          f"summaries longer than cap: {sum(len(t) >= args.tgt_tokens for t in train_tgt)}", flush=True)

    teacher = None
    if args.method == "distill":
        teacher, targs = load_teacher(args.teacher_checkpoint, device, args.teacher_artifact_zip, out / "teacher_phase1")
        for k in ("src_tokens", "tgt_tokens", "selection", "words_per_token"):
            if getattr(targs, k) != getattr(args, k):
                raise ValueError(f"teacher/student input mismatch on {k}: {getattr(targs, k)} vs {getattr(args, k)}")
        if teacher.kind != summ.kind:
            raise ValueError("teacher and student must both be prefix models (same input layout)")
        print(f"KD teacher: {targs.run_name} ({sum(p.numel() for p in teacher.parameters()):,} params)", flush=True)

    total = sum(p.numel() for p in summ.parameters())
    trainable = sum(p.numel() for p in summ.parameters() if p.requires_grad)
    summary = {"run": name, "model": args.model, "method": args.method, "kind": summ.kind, "phase1_artifact": artifact,
               "phase1_parameters": phase1_params, "parameters": total, "trainable_parameters": trainable,
               "trainable_fraction": trainable / total, "tokenizer_md5": tok_md5, "leakage_check": leak,
               "data_stats": stats, "tapt": tapt_info, "device": str(device),
               "config": {k: v for k, v in vars(args).items() if not k.startswith("_")}}
    print(json.dumps({k: v for k, v in summary.items() if k not in ("data_stats", "config")}, indent=1, default=str), flush=True)

    best_loss, history, train_seconds = train(summ, enc, train_src, train_tgt, val_src, val_tgt, args, device, teacher)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    (out / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    ckpt = {"model": {k: (v.half() if v.is_floating_point() else v) for k, v in summ.state_dict().items()},
            "phase1_config": asdict(phase1_cfg), "run_config": {k: v for k, v in vars(args).items() if not k.startswith("_")}}
    torch.save(ckpt, out / "best_fp16.pt")

    scorer = BertScorer(args.bertscore, device)
    gen_docs = [r["document"] for r in gen_val]
    gen_refs = [r["summary"] for r in gen_val]
    (best_score, cfg, max_words, val_texts, val_scores), tuning = tune_decoding(
        [summ], enc, tokenizer, gen_docs, val_src[:len(gen_val)], gen_refs, args, device, scorer)
    print(f"chosen decoding {cfg['name']} max_words {max_words} -> {json.dumps(val_scores)}", flush=True)
    val_final = [truncate_words(t, max_words) for t in val_texts]
    lead = [truncate_words(d, max_words or int(np.median([len(r.split()) for r in gen_refs]))) for d in gen_docs]
    lead_scores = score_set(lead, gen_refs, scorer)
    analysis = {"generated": text_stats(val_final, gen_refs, gen_docs), "lead_baseline_same_length": lead_scores,
                "length_and_decoding_grid": tuning}

    t_test = time.time()
    test_ids_gen = generate([summ], enc, test_src, device, cfg, args.gen_batch)
    test_seconds = time.time() - t_test
    predictions, empty = [], 0
    for r, g in zip(test, test_ids_gen):
        text = truncate_words(clean_text(tokenizer.decode(g, skip_special_tokens=True)), max_words)
        if not text.strip():
            empty += 1
            text = split_sentences(r["document"])[0] if r["document"] else "The matter was decided by the court."
        predictions.append({"id": r["id"], "summary": text})
    assert len(predictions) == len(test) and len({p["id"] for p in predictions}) == len(test)
    path = out / "prediction.json"
    path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    archive = out / f"submission_{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ["prediction.json"] and len(json.loads(z.read("prediction.json"))) == len(test)

    (out / "val_generations.json").write_text(json.dumps([
        {"id": r["id"], "generated": c, "reference": r["summary"], "doc_words": len(r["document"].split())}
        for r, c in zip(gen_val, val_final)], ensure_ascii=False, indent=1), encoding="utf-8")
    ckpt["decode"] = {"config": cfg, "max_words": max_words}
    torch.save(ckpt, out / "best_fp16.pt")
    gen_tokens = sum(len(g) for g in test_ids_gen)
    manifest = {**summary, "best_val_loss": best_loss, "best_val_ppl": math.exp(min(best_loss, 20)),
                "chosen_decoding": cfg, "chosen_max_words": max_words, "val_scores": val_scores,
                "val_bertscore_f1": val_scores.get("bertscore_f1"), "val_rouge1_f1": val_scores["rouge1_f1"],
                "val_rougeL_f1": val_scores["rougeL_f1"], "analysis": analysis,
                "test_empty_generations_replaced": empty,
                "test_mean_words": sum(len(p["summary"].split()) for p in predictions) / len(predictions),
                "train_minutes": train_seconds / 60, "test_generation_seconds": test_seconds,
                "test_generated_tokens_per_second": gen_tokens / max(1e-6, test_seconds),
                "total_minutes": (time.time() - t0) / 60,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
                "phase1_steps": phase1_saved.get("total_steps"), "submission": str(archive)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("run", "best_val_ppl", "val_bertscore_f1", "val_rouge1_f1", "val_rougeL_f1",
                                                "chosen_max_words", "test_mean_words", "trainable_parameters",
                                                "total_minutes")}, indent=1, default=str))
    print("SUBMISSION ZIP:", archive, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("decoder", "encoder_decoder", "hybrid", "efficient"))
    p.add_argument("--method", required=True, choices=("full", "adapter", "lora", "lorma", "ia3", "distill"))
    p.add_argument("--selection", default="salience", choices=("salience", "lead"))
    p.add_argument("--src-tokens", type=int, default=768, help="source budget of prefix models")
    p.add_argument("--fid-chunks", type=int, default=8, help="FiD: number of 256-token source chunks")
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--tgt-tokens", type=int, default=560, help="summary tokens trained/generated (BERTScore window)")
    p.add_argument("--words-per-token", type=float, default=0.72)
    p.add_argument("--global-every", type=int, default=0, help="efficient: global attention in every N-th layer")
    p.add_argument("--adapter-dim", type=int, default=128)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-docs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--student-layers", type=int, default=8)
    p.add_argument("--teacher-checkpoint")
    p.add_argument("--teacher-artifact-zip")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--tapt-minutes", type=float, default=0.0)
    p.add_argument("--tapt-lr", type=float, default=1e-4)
    p.add_argument("--tapt-batch", type=int, default=16)
    p.add_argument("--time-budget-min", type=float, default=90.0)
    p.add_argument("--eval-every-min", type=float, default=20.0)
    p.add_argument("--val-docs", type=int, default=300)
    p.add_argument("--val-gen-docs", type=int, default=150)
    p.add_argument("--gen-batch", type=int, default=24)
    p.add_argument("--n-decode-configs", type=int, default=2)
    p.add_argument("--length-grid", default="120,180,250,320,400,500")
    p.add_argument("--bertscore", type=int, default=1)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--artifact-zip")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--run-name")
    p.add_argument("--seed", type=int, default=230338)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
