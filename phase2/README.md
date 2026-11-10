# Phase 2 — Legal Named Entity Recognition

**Task.** Predict exact character spans `[start, end, label]` for 12 labels (`APP, RESP, A.COUNSEL, R.COUNSEL,
JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO`).
**Metric.** Strict Macro-F1 (document, both offsets and label must match). **Best:** 0.2227 (rank 127).

## Data
Fold 1 (35 documents, 4,120 spans) for training, fold 2 (35 documents, 4,273 spans) for model selection,
fold 3 (35 documents) for test. Labels are imbalanced (fold 1: `DATE` 813 spans vs. `WIT` 67, `A.COUNSEL` 70).
Inputs are validated: text/gold ids match, spans in bounds, labels in the official set, no overlapping gold spans.

## Method

### Token-level BIO head (`src/phase2_lner.py`)
- 25 classes (`O` + `B-/I-` for 12 labels) on Phase 1 hidden states.
- Overlapping windows (256/64 or 512/128 tokens); logits averaged over overlaps; every token must be covered.
- Offsets: tokenizer offsets trimmed of whitespace/punctuation/quotes; a token is aligned to an entity if their
  intervals overlap (first token `B-`, later `I-`).
- Class weights `w_c = min(8, sqrt(n_max / n_c))`, `O` down-weighted to 0.05.
- Validation-selected `O`-logit penalty ∈ {0, 0.5, 1, 1.5, 2, 3}; best checkpoint by strict Macro-F1 on fold 2.
- Adaptation: full FT, selective FT (hybrid: top GRU layer + top two decoder layers), LORMA+ (rank 8; efficient
  `q`/`v`, hybrid decoder FFN), KD into the 8-layer decoder from a full-FT teacher (T = 2, 50/50 hard/soft),
  and a bidirectional local-attention variant of the efficient model (`|i − j| < 64`).

### Exact-character head (`src/phase2_char_ner.py`) — final system
Every character receives the hidden state of its covering token concatenated with a 64-d code-point embedding;
a two-layer residual 1D-CNN (width 256, kernel 5, dropout 0.15) predicts a BIO tag per character, so span
boundaries are exact by construction. 384/96 token windows, 10 epochs, backbone LR 3e-5, head LR 1e-3,
batch 1 × 2 accumulation, character logits averaged over overlapping windows.

## Submissions

| Codabench ID | File | Backbone / method | Strict Macro-F1 |
|---|---|---|---|
| 978513 | `submission_phase2_efficient_charact.zip` | Efficient, exact-character head | **0.2227** (leaderboard) |
| 978123 | `submission_phase2_foundation_optimi.zip` | Foundation 12E/8D, optimised full FT | 0.1444 |
| 978122 | `submission_phase2_efficient_bidirec.zip` | Efficient, bidirectional local attention, full FT | 0.1403 |
| 977959 | `submission_phase2_efficient_full.zip` | Efficient, token head, full FT | 0.0796 |
| 977958 | `submission_phase2_foundation_full.zip` | Foundation, token head, full FT | 0.0669 |
| 977962 | `submission_phase2_decoder_distill.zip` | Decoder-8 student, KD | 0.0604 |
| 977960 | `submission_phase2_hybrid_lorma.zip` | Hybrid, LORMA+ | failed on Codabench |

## Notebooks
| Notebook | Runs |
|---|---|
| `CS781_Phase2_TOP1_Foundation.ipynb` | foundation full FT |
| `CS781_Phase2_TOP2_Efficient.ipynb` | efficient full FT (also the KD teacher) |
| `CS781_Phase2_FAST_Compliance.ipynb` | hybrid LORMA+, decoder-8 KD |
| `CS781_Phase2_OPTIMIZED_Foundation.ipynb` | foundation, 8 epochs, layer-wise learning rates |
| `CS781_Phase2_OPTIMIZED_Efficient_Bidirectional.ipynb` | efficient with bidirectional local attention |
| `CS781_Phase2_Efficient_Exact_Character.ipynb` | efficient exact-character head |
| `CS781_Phase2_Foundation_Exact_Character.ipynb` | foundation exact-character head |
| `CS781_Phase2_LNER.ipynb` | first main-pipeline runs (enc–dec full, hybrid partial, efficient LORMA+, KD) |

`src/build_*.py` rebuild the notebooks from the source files.

## Run
Attach the Phase 1 artifact ZIPs as a private Kaggle dataset, add the `HF` secret, enable GPU + Internet, and run
the notebook. For KD, first run a full-FT teacher, publish its `best.pt` as a dataset and attach it to the student
run. Each run writes `best.pt`, `manifest.json`, `training_log.json`, `validation_metrics.json`,
`prediction.json` and `submission_phase2_<run>.zip`.
