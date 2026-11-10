# Phase 6 — Abstractive Summarisation

**Task.** Generate an abstractive summary for every test judgment.
**Metric.** BERTScore-F1 (primary); ROUGE-1, ROUGE-L. **Best:** 0.8101 (rank 124).

## Data
7,030 training documents (median 3,128 words; reference summaries median 638 words), 100 test documents. There is
no public dev split, so a fixed 150-document validation split is carved from train and used for all decisions.
The course scorer (`bert_score`, roberta-large) truncates at 510 tokens, so only the first ~400 words of a summary
affect the primary metric.

## Method (`src/phase6_summ.py`)
- **Content selection:** sentences scored by TF-IDF cosine to the document centroid plus a position prior; the
  best are kept in document order up to the source budget (`--selection salience`; `lead` is the ablation).
- **Decoder and efficient (prefix LM):** `[source] <bos> [summary] <eos>`, loss on summary tokens only;
  768 source tokens, 560 summary tokens.
- **Encoder–decoder and hybrid (Fusion-in-Decoder):** 8 chunks × 256 tokens encoded separately; the decoder
  cross-attends to all 2,048 encoder states. These 400-step backbones first get 60 min of task-adaptive
  pretraining with their own Phase 1 objective on train documents and summaries.
- **Decoding:** batched greedy with KV cache, no-repeat-4-gram or repetition penalty 1.15; decoding mode and a
  120–500-word budget chosen by BERTScore on the validation split with the same scorer as Codabench.
- **KD:** 0.5·CE + 0.5·T²·KL on every summary token, T = 2 (all backbones share the tokenizer).
- Re-implemented forward passes (KV cache, padding, adapters, global attention) match the original Phase 1 models
  to ~1e-5.

## Submissions

| Codabench ID | File | Backbone / method | BERTScore-F1 |
|---|---|---|---|
| 1030929 | `submission_p6_A_efficient_adapter.zip` | Efficient, Houlsby adapters (dim 128), teacher | **0.8101** (leaderboard) |
| 1030970 | `submission_p6_D_decoder_kd.zip` | Decoder-8 student, token-level KD from A | 0.7711 |
| 1030933 | `submission_p6_C_encdec_lora.zip` | Encoder–decoder, TAPT + LoRA, FiD | 0.7670 |
| 1030932 | `submission_p6_B_hybrid_lorma.zip` | Hybrid, TAPT + LORMA+, FiD | 0.7655 |

## Notebooks
| Notebook | Runs |
|---|---|
| `CS781_Phase6_NB1_Efficient_Adapters_then_Decoder_KD.ipynb` | A: `--method adapter --adapter-dim 128 --learning-rate 3e-4`; D: `--method distill --student-layers 8 --learning-rate 1e-4` |
| `CS781_Phase6_NB2_Hybrid_LORMA.ipynb` | B: hybrid TAPT 60 min + LORMA+ |
| `CS781_Phase6_NB3_EncDec_LoRA.ipynb` | C: encoder–decoder TAPT 60 min + LoRA |

`src/phase6_ensemble.py` ensembles members by averaging per-step log-probabilities; `src/smoke_test_p6.py` runs
every notebook command on CPU with fake data.

## Run
Attach the Phase 1 artifact ZIPs, add the `HF` secret, enable GPU (T4×2) + Internet, then
*Save Version → Save & Run All*. Each run saves `prediction.json` (`[{"id", "summary"}]`),
`submission_<run>.zip`, `manifest.json` (validation BERTScore/ROUGE, decoding grid, lead baseline, repetition and
novel-trigram rates) and `val_generations.json`.
