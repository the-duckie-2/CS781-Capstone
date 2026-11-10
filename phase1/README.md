# Phase 1 — Legal Language Model Pretraining

**Task.** Continue each official 512-word legal prompt with exactly 128 words (prompt not repeated).
**Metric.** BERTScore-F1 (primary); ROUGE-L F1, BLEU-4. **Best:** 0.8060 (rank 135).

## Data
Four official training files (`train_acts`, `train_high_courts`, `train_supreme_court`, `train_tribunals`) streamed
with `ijson`. An equal character quota is taken from each source and the documents are shuffled *before*
tokenizer fitting and token-pool construction (an early source-ordered sample over-represented the first file).
Eight complete documents (≥ 640 words) are held out in the test format (512 prompt + 128 reference words) and
excluded from training.

## Tokenizer
Byte-level BPE, 4,096 tokens, NFC normalisation, no lowercasing, special tokens `<pad> <bos> <eos> <unk>`,
fitted on 10M characters. Shared by all four compliance models.

## Architectures (`src/phase1_pipeline.py`, d = 512, 8 heads, tied embeddings)

| Model | Params | Design |
|---|---|---|
| Decoder-only | 53.32M | 16 pre-LN GELU blocks, sinusoidal positions, **causal gated context fusion**: each position mixes in the running mean of earlier states through a content-dependent sigmoid gate |
| Encoder–decoder | 54.38M | 7 + 7 Transformer layers; the same gate fuses the mean encoder memory into every decoder state |
| Hybrid | 55.04M | 5-layer bidirectional GRU encoder (256/direction) + 11-layer Transformer decoder |
| Efficient | 55.66M | 19 blocks: grouped-query attention (8 query / 2 KV heads), 64-token causal sliding window, RoPE, RMSNorm, SwiGLU (hidden 1,408) |

A runtime assertion aborts any model outside 50–300M parameters.

**Foundation model** (`src/phase1_foundation.py`): 12-layer bidirectional encoder + 8-layer decoder, width 640,
24k BPE, 1,024-token source / 256-token target, RoPE, RMSNorm, SwiGLU, decoder cross-attention gated between
token-level memory and 32-token pooled global memory, auxiliary masked-source reconstruction loss (weight 0.35),
resumable multi-GPU training. Used as an extra bidirectional backbone in Phase 2.

## Training
- Stage 1 (all four): 400 steps, context 256, batch 2 × 4 accumulation, AdamW, peak LR 3e-4, 5% warmup,
  cosine decay to 0.1×, weight decay 0.1, FP16, clipping 1.0, checkpoints every 25 steps.
- Stage 2 (efficient, decoder): +6,000 steps at context 512, batch 1 × 8, 40M-token pool.

| Model | Steps | Held-out loss | PPL |
|---|---|---|---|
| Decoder | 400 | 9.663 | 15,726 |
| Enc–dec | 400 | 9.572 | 14,354 |
| Hybrid | 400 | 9.346 | 11,453 |
| Efficient | 400 | 9.318 | 11,140 |
| Decoder (continued) | 6,400 | 3.205 | 24.7 |
| Efficient (continued) | 6,400 | **2.948** | **19.1** |

**Decoding.** Prompt left-truncated to the context; temperature 0.2, top-k 20, ≤ 320 BPE tokens; output cut or
extended to exactly 128 words.

## Submissions

| Codabench ID | File | Model | BERTScore-F1 |
|---|---|---|---|
| 965167 | `submission_efficient_improved.zip` | Efficient, 6,400 steps | **0.8060** (leaderboard) |
| 965084 | `submission_decoder_eff.zip` | Decoder, 6,400 steps | 0.7926 |
| 964608 | `submission_efficient.zip` | Efficient, 400 steps | 0.7620 |
| 964557 | `submission_decoder.zip` | Decoder, 400 steps | 0.7595 |
| 964513 | `submission_hybrid.zip` | Hybrid, 400 steps | 0.7611 |
| 964498 | `submission_encoder_decoder.zip` | Encoder–decoder, 400 steps | 0.7204 |

## Files
- `src/phase1_pipeline.py` – data, tokenizer, four architectures, training, generation, validation, ZIP.
- `src/phase1_foundation.py` – foundation model.
- `src/evaluate_phase1.py` – official local evaluator (unchanged).
- `src/build_notebook.py`, `src/build_foundation_notebook.py` – notebook builders.
- `notebooks/CS781_Phase1_Training.ipynb` – Stage 1 (two models per T4) and Stage 2 continuation.
- `notebooks/CS781_Phase1_Foundation.ipynb` – foundation model training.

## Run
Import `CS781_Phase1_Training.ipynb` into Kaggle, enable GPU + Internet, add the `HF` secret, run setup and
Stage 1 (on T4×2: `decoder,hybrid` on GPU 0 and `encoder_decoder,efficient` on GPU 1), download the artifact
ZIPs, then run Stage 2. The local-evaluation cell scores held-out documents with `evaluate_phase1.py`.
