# Phase 7 — Prior Case Retrieval

**Task.** For each query judgment, return exactly 7 unique candidate ids from the same split's candidate pool,
ranked best to worst.
**Metric.** Micro-F1@7 (primary); Micro-P@7, Micro-R@7. **Best:** 0.4607 (rank 24).

## Data
Train 827 queries / 4,320 candidates, dev 118 / 1,023, test 237 / 1,727; 6.4 relevant candidates per training
query on average; queries median ~4,500 words. Citations in queries are masked as `<CITATION>`.

## Method (`src/phase7_pcr.py`)
1. **Lexical stage** (built from each split's own pool): BM25 on hashed word 1–3-grams and TF-IDF cosine on
   1–2-grams, k1/b tuned on dev, merged by reciprocal-rank fusion into a top-150 candidate list. The query itself
   and exact duplicates are never returned.
2. **Dense Phase 1 bi-encoder:**
   - query windows of 256 tokens centred on each `<CITATION>` marker plus the opening window; 8 candidate windows
     spread over the judgment (training samples 4 query / 3 candidate windows);
   - window vector = masked mean of hidden states + separate query/candidate projections;
   - document score = attention + mean pooling with cosine, or **maxsim** late interaction (mean over query windows
     of the best-matching candidate window);
   - InfoNCE with learned temperature over 1 positive + 3 BM25 hard negatives (from the top 50) + in-batch
     candidates; other gold candidates of the query are masked;
   - checkpoint chosen by dense-only rerank F1@7 on dev.
3. **KD:** teacher A exports scores over each training query's positives + top-50 hard negatives; student loss
   = 0.5·InfoNCE + 0.5·T²·KL(teacher‖student), T = 2; 8-layer decoder student.
4. **Fusion ranker:** LightGBM LambdaMART or logistic regression over per-query z-scores, ranks and gaps of every
   system plus section/article overlap, judgment-year difference, candidate length, train citation popularity
   and pool centrality. Fitted on dev only and chosen by 5-fold query-grouped cross-fitting among methods that
   include the Phase 1 retriever.
5. **Ensembles:** one fusion ranker over the dense scores of several runs.

## Submissions

| Codabench ID | File | Backbone / method | Micro-F1@7 |
|---|---|---|---|
| 1042535 | `submission_p7_E_efficient_full_maxs.zip` | E: efficient, full FT, maxsim | **0.4607** (leaderboard) |
| 1042753 | `submission_p7_E3_efficient_full_max.zip` | E3: maxsim, 5 hard negatives | 0.4601 |
| 1042432 | `submission_p7_A_efficient_full.zip` | A: efficient, full FT, attention pooling (teacher) | 0.4584 |
| 1042750 | `submission_p7_C_efficient_partial_m.zip` | C: efficient, partial FT (top 6 blocks), maxsim | 0.4584 |
| 1042531 | `submission_p7_ENS_NB1.zip` | fusion over A + D + E | 0.4584 |
| 1042555 | `submission_p7_B_efficient_lorma.zip` | B: efficient, LORMA+ r32 | 0.4572 |
| 1042752 | `submission_p7_E2_efficient_full_max.zip` | E2: maxsim, 12 candidate windows | 0.4560 |
| 1042562 | `submission_p7_ENS_NB2.zip` | fusion over B + F | 0.4554 |
| 1042527 | `submission_p7_D_decoder8_kd.zip` | D: decoder-8 student, KD from A | 0.4542 |
| 1042556 | `submission_p7_F_efficient_ia3.zip` | F: efficient, IA³ | 0.4530 |
| 1042751 | `submission_p7_G_decoder_full.zip` | G: decoder-16, full FT | 0.4530 |

## Notebooks
| Notebook | Runs |
|---|---|
| `CS781_Phase7_NB1_Full_Teacher_KD_Student.ipynb` | A (GPU 0) → D; E (GPU 1); ENS_NB1 |
| `CS781_Phase7_NB1b_D_KD_and_E_without_A.ipynb` | D and E re-run variant |
| `CS781_Phase7_NB1c_Extra_Maxsim_W12.ipynb` | E2 (12 windows), E3 (5 hard negatives) |
| `CS781_Phase7_NB2_LORMA_IA3.ipynb` | B (GPU 0), F (GPU 1), ENS_NB2 |
| `CS781_Phase7_NB3_Partial_Maxsim_DecoderFull.ipynb` | C (GPU 0), G (GPU 1) |
| `CS781_Phase7_FUSE_ALL.ipynb` | CPU fusion over every run's `dense_scores.npz` |

`src/smoke_test_p7.py` runs every recipe on CPU with fake data.

## Run
Attach the Phase 1 artifact ZIPs, add the `HF` secret, enable GPU (T4×2) + Internet. Run the cells up to
*step 1 (prepare)* interactively to cache the data, then *Save Version → Save & Run All*. Each run saves
`prediction.json`, `submission_<run>.zip`, `manifest.json` (lexical grid, fusion report, top features, F1 by query
length), `dev_predictions.json` and `dense_scores.npz`.
