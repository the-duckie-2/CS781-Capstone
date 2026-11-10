# Phase 4 — Court Judgment Prediction

**Task.** Predict whether an appeal is accepted (0/1) for every case.
**Metric.** Macro-F1. **Best:** 0.3779 (rank 82).

## Data
Only the re-released split is used: 37,387 train (41% accepted), 1,988 dev (balanced), 1,517 test; median case
length ~2,800 words. The original test set (a subset of the old dev set) was never used for training. Train/dev
cases duplicating a test case are removed before training.

## Method (`src/phase4_cjpe.py`)
- **Long documents:** cases are cut into 256-token chunks; `--selection headtail` keeps the first chunk (parties,
  facts) plus the last K−1 chunks (analysis and operative order), K = 6.
- **Encoder:** the Phase 1 backbone encodes each chunk → `[mean; last]` + learned chunk-slot embedding;
  attention pooling + mean pooling over chunks → MLP → 2 logits.
- **Loss:** class-weighted CE (power 0.5), label smoothing 0.05; decision threshold on P(y = 1) tuned on dev.
- **Adaptation:** full FT, LORMA+, LoRA, IA³, partial FT, BitFit, KD (8-layer decoder student on teacher soft
  labels, T = 2, α = 0.5).

## Submissions

| Codabench ID | File | Backbone / method | Macro-F1 |
|---|---|---|---|
| 1009617 | `submission_p4_D_decoder_kd.zip` | Decoder-8 student, KD from A | **0.3779** |
| 1009255 | `submission_p4_C_encdec_partial.zip` | Encoder–decoder, partial FT (top 2 layers) | **0.3779** |
| 1009616 | `submission_p4_A_efficient_full.zip` | Efficient, full FT (teacher) | 0.3470 |
| 1009259 | `submission_p4_B_hybrid_lorma.zip` | Hybrid, LORMA+ r16 | 0.3358 |

## Notebooks and commands
| Notebook | Runs |
|---|---|
| `CS781_Phase4_NB1_Efficient_Full_then_Decoder_KD.ipynb` | A: `--model efficient --method full --learning-rate 5e-5 --export-soft-labels`; D: `--model decoder --method distill --student-layers 8 --learning-rate 5e-5` |
| `CS781_Phase4_NB2_Hybrid_LORMA.ipynb` | B: `--model hybrid --method lorma --rank 16 --learning-rate 5e-4` |
| `CS781_Phase4_NB3_EncDec_Partial.ipynb` | C: `--model encoder_decoder --method partial --unfreeze-layers 2 --learning-rate 1e-4` |

All runs use `--selection headtail --max-chunks 6 --chunk-len 256 --batch-docs 4 --grad-accum 2`.

## Run
Attach the Phase 1 artifact ZIPs, add the `HF` secret, enable GPU + Internet, run all cells. D requires A's
`teacher_soft_labels.pt`.
