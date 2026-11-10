# Phase 5 — Legal Statute Identification

**Task.** Predict the set of relevant statute ids for every case (multi-label).
**Metric.** Macro-F1 (primary); Micro-F1, subset accuracy. **Best:** 0.2371 (rank 94).

## Data
42,750 train / 10,181 dev / 13,019 test cases; 100 statutes, 3.7 statutes per case on average; statute frequency
ranges from 272 to 11,731 training cases; median case length ~430 words. Train/dev cases duplicating a test case
are removed. The label space is every statute id in train ∪ dev.

## Method (`src/phase5_lsi.py`)
- **Windows:** the case is cut into 256-token windows (overlap 32); up to 6 windows spread evenly over the case
  (first and last always kept) + 8-bucket section embedding. `--selection headtail` is the ablation.
- **Window vector:** masked mean + max + last-token state → projection.
- **Heads:** attention + mean pooling → MLP, plus **label-wise attention** (one query per statute over the
  windows). Output biases initialised to each statute's log-prior.
- **Loss:** BCE with per-statute positive weights (neg/pos)^0.5, clipped at 30 (focal and ASL also implemented).
- **KD:** 0.5·BCE(gold) + 0.5·T²·BCE(student/T, sigmoid(teacher/T)), T = 2.
- **Calibration and thresholds:** temperature fitted on dev; four strategies (global, global + at-least-one,
  per-statute shrunk to global, per-statute + at-least-one) compared by 2-fold cross-fitting on dev; the winner is
  refit on all of dev.
- `src/phase5_dapt.py` (task-adaptive pretraining) and `src/phase5_ensemble.py` (probability ensembling) support
  the TAPT + ensemble configuration.

## Submissions

| Codabench ID | File | Backbone / method | Macro-F1 |
|---|---|---|---|
| 1016743 | `submission_p5_D_decoder_kd.zip` | Decoder-8 student, KD from A | **0.2371** |
| 1016602 | `submission_p5_A_efficient_full.zip` | Efficient, full FT (teacher) | 0.1603 |
| 1016604 | `submission_p5_C_encdec_partial.zip` | Encoder–decoder, partial FT (top 2 enc + 2 dec) | 0.1025 |
| 1016603 | `submission_p5_B_hybrid_lorma.zip` | Hybrid, LORMA+ r16 | 0.0982 |
| 1016744 | `submission_p5_F_efficient_lorma_hea.zip` | Efficient, LORMA+, head+tail windows | 0.0981 |

## Notebooks
| Notebook | Runs |
|---|---|
| `CS781_Phase5_NB1_Efficient_Full_then_Decoder_KD.ipynb` | A efficient full FT (`--learning-rate 5e-5 --pooling labelwise --selection spread`) → D decoder-8 KD |
| `CS781_Phase5_NB2_Hybrid_LORMA.ipynb` | B hybrid LORMA+ r16 (optional E: efficient LORMA+) |
| `CS781_Phase5_NB3_EncDec_Partial.ipynb` | C encoder–decoder partial FT, F efficient LORMA+ head+tail |

`src/smoke_test_p5.py` tests the recipes on CPU.

## Run
Attach the Phase 1 artifact ZIPs, add the `HF` secret, enable GPU + Internet, run all cells. Each run saves
`prediction.json`, `submission_<run>.zip`, `manifest.json` (dev scores, chosen threshold strategy, head/mid/tail
Macro-F1) and `validation_metrics.json` (per-statute P/R/F1, co-occurrence, calibration).
