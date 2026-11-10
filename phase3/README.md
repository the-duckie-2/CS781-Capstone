# Phase 3 — Rhetorical Role Prediction

**Task.** Assign one of 13 rhetorical roles to every sentence of Corporate-Law (CL) and IT judgments.
**Metric.** Macro-F1. **Best:** 0.4236 (rank 115).

## Data
CL: 40 train / 5 dev / 5 test documents; IT: 40 / 5 / 5 (16,845 training sentences). Roles occur in long
contiguous blocks; `Fact` and `RatioOfTheDecision` cover ~50% of sentences, while `Issue` (107) and
`PrecedentOverruled` (3) are rare. The provided sentence lists are used unchanged, so predictions stay aligned.
Both domains are trained jointly with a domain embedding.

## Method

### Sentence-context classifier (`src/phase3_rr.py`)
- Input `[prev ≤ 2 sentences] <eos> [next ≤ 2] <eos> [target]`; the target goes last so causal backbones see all
  context (`--context none` = target only).
- Head: mean + last target-token state, 10-bin document-position embedding, CL/IT domain embedding → 2-layer MLP.
- Loss: sqrt-inverse-frequency class-weighted CE, label smoothing 0.05; KD = 0.5·CE + 0.5·T²·KL, T = 2, with
  per-sentence teacher logits.
- Checkpoint by combined CL+IT dev Macro-F1, evaluated twice per epoch.

### Document-level tagger (`src/phase3_doc.py`) — final system
- A frozen Phase 1 backbone encodes each sentence (128 tokens) into `[mean; last]`.
- A 2-layer BiGRU (hidden 192, dropout 0.3) runs over the whole document with relative-position and domain
  embeddings and labels every sentence.
- v2 adds a mean bag of Phase 1 token embeddings per sentence, Gaussian feature noise (0.1), an 8-seed ensemble,
  and per-class bias tuning on dev.

## Submissions

| Codabench ID | File | Backbone / method | Macro-F1 |
|---|---|---|---|
| 994176 | `submission_p3_J_decoder_docgru_v2.zip` | Decoder, document BiGRU v2 | **0.4236** (leaderboard) |
| 993970 | `submission_p3_D_decoder_kd_ctx.zip` | Decoder-8 student, KD, ±2 context | 0.3895 |
| 994177 | `submission_p3_H_hybrid_docgru_v2.zip` | Hybrid, document BiGRU v2 | 0.3853 |
| 994083 / 994055 | `submission_p3_J_decoder_docgru.zip` | Decoder, document BiGRU v1 | 0.3812 |
| 993950 | `submission_p3_A_efficient_full_ctx.zip` | Efficient, full FT, ±2 context | 0.3785 |
| 994082 | `submission_p3_I_encoder_decoder_doc.zip` | Encoder–decoder, document BiGRU v1 | 0.3602 |
| 994081 | `submission_p3_H_hybrid_docgru.zip` | Hybrid, document BiGRU v1 | 0.2696 |
| 993967 | `submission_p3_F_efficient_ia3_ctx.zip` | Efficient, IA³, ±2 context | failed |
| 994092 | `submission_p3_G_efficient_docgru.zip` | Efficient, document BiGRU v1 | failed |
| 993909 | `submission_p3_C_encdec_lora_ctx.zip` | Encoder–decoder, LoRA, ±2 context | not scored |

## Notebooks
| Notebook | Runs |
|---|---|
| `CS781_Phase3_NB1_Efficient_Full_then_Decoder_KD.ipynb` | A efficient full FT (teacher) → D decoder-8 KD |
| `CS781_Phase3_NB2_Hybrid_LORMA_then_Efficient_SentenceOnly.ipynb` | B hybrid LORMA+ r16, E efficient LORMA+ sentence-only |
| `CS781_Phase3_NB3_EncDec_LoRA_then_Efficient_IA3.ipynb` | C encoder–decoder LoRA, F efficient IA³ |
| `CS781_Phase3_NB4_DocLevel_All4.ipynb` | G/H/I/J document BiGRU v1 on all four backbones |
| `CS781_Phase3_NB5_DocLevel_v2.ipynb` | G/H/I/J document BiGRU v2 |
| `CS781_Phase3_NB1_Strong_and_KD.ipynb`, `CS781_Phase3_NB2_LORMA_and_LoRA.ipynb` | earlier versions of NB1/NB2 |

`src/smoke_test.py` runs every recipe on CPU with fake data.

## Run
Attach the Phase 1 artifact ZIPs, add the `HF` secret, enable GPU + Internet, run all cells. Each run saves
`prediction.json`, `submission_<run>.zip`, per-domain predictions, `manifest.json` and `validation_metrics.json`
(per-label P/R/F1 and confusion matrix).
