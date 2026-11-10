# CS781 Capstone — In-House Legal Intelligence System

**Author:** Daksh Niranjan Shettar (230338), IIT Kanpur
**Course:** CS781 Large Language Models (Dr. Ashutosh Modi)
**Codabench username:** `d_230338`

A self-hosted legal-NLP stack for Indian court documents, built under the capstone constraints: every model is
trained from random initialisation on the official `Exploration-Lab/CS-781-Capstone` corpus, has 50–300M
parameters, uses no pretrained language-model weights or commercial APIs, and was trained on Kaggle T4×2 / P100 GPUs.

Phase 1 pretrains four architecturally different legal language models plus a larger encoder–decoder.
Phases 2–7 adapt those checkpoints to six downstream tasks with full, partial and parameter-efficient fine-tuning
(LORMA+, LoRA, IA³, Houlsby adapters) and knowledge distillation into an 8-layer decoder student.

## Results (best valid Codabench submission per phase)

| Phase | Task | Metric | Best score | Rank | Best system |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore-F1 | **0.8060** | 135 | Efficient GQA decoder, 6,400 steps |
| 2 | Legal NER | Strict Macro-F1 | **0.2227** | 127 | Efficient backbone + exact-character CNN head |
| 3 | Rhetorical roles | Macro-F1 | **0.4236** | 115 | Frozen decoder + document-level BiGRU (v2) |
| 4 | Judgment prediction | Macro-F1 | **0.3779** | 82 | Decoder-8 KD student / Enc–dec partial FT |
| 5 | Statute identification | Macro-F1 | **0.2371** | 94 | Decoder-8 KD student, label-wise attention |
| 6 | Abstractive summarisation | BERTScore-F1 | **0.8101** | 124 | Efficient backbone + Houlsby adapters |
| 7 | Prior-case retrieval | Micro-F1@7 | **0.4607** | 24 | BM25 → efficient maxsim bi-encoder → LambdaMART |

Every submission of every phase is listed, with its Codabench ID and score, in that phase's README.

## Repository layout

```
phase1/  Legal LM pretraining (4 compliance architectures + 12E/8D foundation model)
phase2/  Legal NER (token-level BIO heads, exact-character head)
phase3/  Rhetorical role prediction (sentence-context and document-level taggers)
phase4/  Court judgment prediction (hierarchical head+tail encoder)
phase5/  Legal statute identification (spread windows, label-wise attention, cross-fitted thresholds)
phase6/  Abstractive summarisation (prefix LM / Fusion-in-Decoder)
phase7/  Prior-case retrieval (lexical → dense → learning-to-rank cascade)
```

Each phase folder contains:

- `src/` – the Python source. Later phases re-use the Phase 1–3 modules (`phase1_pipeline.py`,
  `phase1_foundation.py`, `phase2_lner.py`, `phase3_rr.py`, …), so each `src/` is self-contained.
- `notebooks/` – the exact Kaggle notebooks that produced the submissions. The `build_*.py` scripts in `src/`
  embed the source files into these notebooks; edit the `.py` files and rebuild rather than editing notebooks.
- `README.md` – task, method, every submission, and run instructions.

## Shared design

- **Tokenizer.** One byte-level BPE (4,096 tokens, NFC, no lowercasing) trained on the official corpus and
  shared by the four Phase 1 models, which makes token-level distillation possible across architectures.
- **Phase 1 artifacts as interfaces.** Each Phase 1 run is saved as an artifact ZIP
  (`phaseN_*_artifacts.zip`: checkpoint, tokenizer, manifest, logs). Later phases rebuild the exact architecture
  from the saved config.
- **Adaptation library** (`phase2_lner.py`, `phase3_rr.py`): full FT, partial FT, LORMA+
  (`y = (I + α/r·BA)·W₀x`, B initialised to 0), LoRA, IA³, Houlsby adapters, and KD with
  `L = 0.5·L_task + 0.5·T²·KL(teacher‖student)`, T = 2, into the Phase 1 decoder truncated from 16 to 8 layers.
- **Wall-clock budgets.** Every run has `--time-budget-min`; when it runs out the run evaluates on dev and
  always writes a validated `submission_*.zip` (`prediction.json` at the ZIP root).
- **Leakage control.** From Phase 4 on, train/dev documents that duplicate a test document (by id or normalised
  text) are removed and counted before training.

## Running

Requirements: Python 3.10+, PyTorch 2.x, `tokenizers`, `huggingface_hub`, `ijson`, `numpy`; Phase 6 also uses
`bert-score`, Phase 7 uses `lightgbm` and `scikit-learn`.

1. Accept access to the gated `Exploration-Lab/CS-781-Capstone` dataset on Hugging Face and create a read token.
2. In Kaggle, import the phase notebook, enable GPU (T4×2) and Internet, and add the token as a secret named `HF`.
3. For Phases 2–7, attach a private Kaggle dataset containing the Phase 1 artifact ZIPs
   (`phase1_improved_efficient_artifacts.zip`, `phase1_improved_decoder_artifacts.zip`,
   `phase1_ed_baseline_artifacts.zip`, `phase1_hybrid_baseline_artifacts.zip`).
4. Run the notebook (long runs: *Save Version → Save & Run All*). Each run writes its outputs and a
   `SUBMIT/submission_*.zip` for Codabench.

Model weights are not stored in this repository (they exceed GitHub limits).

## Data

Only the official dataset was used: Phase 1 corpus (Acts, High Courts, Supreme Court, Tribunals), Phase 2 folds
1–3, Phase 3 CL/IT, the re-released Phase 4 CJPE split (the original leaked test set was never used), Phase 5 LSI,
Phase 6 summarisation and Phase 7 PCR with per-split candidate pools. No external corpora or synthetic data.
