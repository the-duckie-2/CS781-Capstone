"""Build the Kaggle notebooks for CS781 Phase 6 (abstractive summarization):
3 notebooks (one per Kaggle account) for the required submissions + 1 max-score notebook."""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
src = root / "src"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root
SOURCES = ["phase1_pipeline.py", "phase1_foundation.py", "phase2_lner.py", "phase3_rr.py", "phase6_summ.py",
           "phase6_ensemble.py"]


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def flags(args):
    return " ".join(f"--{k.replace('_', '-')}" if v is True else f"--{k.replace('_', '-')} {v}" for k, v in args.items())


SUMMARY_CELL = (
    "import json, shutil\nfrom pathlib import Path\nfrom IPython.display import FileLink, display\n"
    "run = Path('/kaggle/working/{name}')\n"
    "zips = sorted(run.glob('submission_*.zip'))\n"
    "if not zips:\n"
    "    print('!!! No submission ZIP for {name} - read its log above / {name}.log')\n"
    "else:\n"
    "    Path('/kaggle/working/SUBMIT').mkdir(exist_ok=True)\n"
    "    shutil.copy(zips[0], '/kaggle/working/SUBMIT/')\n"
    "    m = json.loads((run / 'manifest.json').read_text())\n"
    "    print(json.dumps({{k: m.get(k) for k in ('run', 'method', 'trainable_parameters', 'parameters', 'best_val_ppl', "
    "'val_bertscore_f1', 'val_rouge1_f1', 'val_rougeL_f1', 'chosen_max_words', 'test_mean_words', 'total_minutes', "
    "'peak_gpu_memory_gb')}}, indent=1, default=str))\n"
    "    print('lead baseline (same length):', m['analysis']['lead_baseline_same_length'])\n"
    "    display(FileLink(str(zips[0].relative_to('/kaggle/working'))))\n"
    "    print('UPLOAD THIS ZIP TO CODABENCH:', zips[0])\n"
)


def run_cells(label, args, optional=False):
    name = args["run_name"]
    head = f"## {label}\n" + ("\n*Optional extra experiment - the 4 required submissions do not depend on it. Delete these "
                              "two cells before Save Version if you don't want it.*\n" if optional else "")
    return [md(head),
            code(f"!cd /kaggle/working && python phase6_summ.py {flags(args)} --output-dir /kaggle/working/{name} "
                 f"2>&1 | tee /kaggle/working/{name}.log\n"),
            code(SUMMARY_CELL.format(name=name))]


SETUP = ("!nvidia-smi\n!pip install -q tokenizers huggingface_hub bert-score\nimport os\n"
         "from kaggle_secrets import UserSecretsClient\n"
         "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n"
         "print('HF token loaded; value intentionally hidden')\n"
         "found = [os.path.join(f, n) for f, _, ns in os.walk('/kaggle/input', followlinks=True) for n in ns "
         "if n.endswith(('.zip', '.pt'))]\nprint('\\n'.join(found))\n"
         "assert found, 'NOTHING attached under /kaggle/input - Add Input -> your Phase 1 artifact dataset'\n")

DATA_NOTE = ("Data: only the official Phase 6 `train.json` / `test.json` from `Exploration-Lab/CS-781-Capstone` "
             "(re-downloaded with `force_download`). Train documents identical to a test document (by id or normalised "
             "text) are removed (`LEAKAGE CHECK`). A seeded validation split (300 docs) is held out from train; "
             "checkpoints, decoding and output length are chosen on it only. Continued pretraining (TAPT) uses only the "
             "train-split documents + summaries. BERTScore (roberta-large, `bert_score` with `lang='en'`, the course "
             "evaluator's setting) is used **only to evaluate** validation outputs, never for training or generation.\n")


def notebook(title, intro, runs, attach, extra_cells=()):
    cells = [md(f"# {title}\n\n{intro}\n\n"
                f"**Setup:** GPU on, Internet on, secret `HF` enabled, Phase 1 artifact dataset attached with **{attach}**. "
                "These runs take hours, so use **Save Version -> Save & Run All (Commit)**; when it finishes, open the "
                "version and download the ZIPs from Output -> `SUBMIT/`.\n\n" + DATA_NOTE),
             code(SETUP)]
    for name in SOURCES:
        cells.append(code(f"%%writefile /kaggle/working/{name}\n" + (src / name).read_text(encoding="utf-8")))
    for r in runs:
        cells += run_cells(*r)
    cells += list(extra_cells)
    cells.append(md("## Collect outputs\nDownload `phase6_outputs.zip` (logs, manifests, val generations) and the "
                    "`best_fp16.pt` checkpoints (model archive).\n"))
    cells.append(code("!ls -la /kaggle/working/SUBMIT\n!cd /kaggle/working && zip -qr phase6_outputs.zip p6_* *.log "
                      "-x '*/phase1*/*' '*/teacher_phase1/*' '*/best_fp16.pt' && ls -la phase6_outputs.zip\n"
                      "!ls -la /kaggle/working/p6_*/best_fp16.pt\n"))
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}


# shared recipe: salience-selected source, 560-token summaries (the BERTScore window), batch 2 x accum 8
COMMON = {"selection": "salience", "src_tokens": 768, "fid_chunks": 8, "tgt_tokens": 560, "batch_docs": 2,
          "grad_accum": 8, "epochs": 3, "eval_every_min": 20}

books = {
    "CS781_Phase6_NB1_Efficient_Adapters_then_Decoder_KD.ipynb": notebook(
        "CS781 Phase 6 (Summarization) - Notebook 1 (Kaggle account 1)",
        "Run A: Phase 1 **efficient** model + **Houlsby adapters** (bottleneck 128 after every attention and FFN "
        "sub-layer; backbone frozen) as a prefix LM over the salience-selected source (KD teacher).\n\n"
        "Run D: Phase 1 **decoder** cut from 16 to 8 layers, **knowledge distillation**: loss = 0.5 CE(reference) + "
        "0.5 T^2 KL(teacher || student) over every summary token, T = 2, with teacher A's per-token distributions "
        "computed on the fly (same tokenizer, same input). ~4.5 h in total.",
        [("Run A - efficient / Houlsby adapters / prefix LM (teacher)", {
            "model": "efficient", "method": "adapter", "adapter_dim": 128, "learning_rate": "3e-4",
            "time_budget_min": 100, "run_name": "p6_A_efficient_adapter", **COMMON}),
         ("Run D - decoder student (8 layers) / knowledge distillation from Run A", {
            "model": "decoder", "method": "distill", "student_layers": 8, "learning_rate": "1e-4",
            "teacher_checkpoint": "/kaggle/working/p6_A_efficient_adapter/best_fp16.pt", "kd_alpha": 0.5,
            "temperature": 2.0, "time_budget_min": 80, "run_name": "p6_D_decoder_kd", **COMMON})],
        "phase1_improved_efficient_artifacts.zip + phase1_improved_decoder_artifacts.zip"),
    "CS781_Phase6_NB2_Hybrid_LORMA.ipynb": notebook(
        "CS781 Phase 6 (Summarization) - Notebook 2 (Kaggle account 2)",
        "Run B: Phase 1 **hybrid** model (GRU encoder + Transformer decoder). The Phase 1 hybrid checkpoint is barely "
        "trained (400 steps), so it first gets 60 min of **TAPT** (its own Phase 1 objective on the train-split text), "
        "then **LORMA+** (rank 16 on the decoder FFNs, backbone frozen) with Fusion-in-Decoder over 8 x 256-token "
        "source chunks. ~3 h.\n\n"
        "Optional Run E: **efficient + LORMA+** (rank 16 on q/v/out + FFN). Compared with Run A, it isolates adapters vs "
        "LORMA on the same backbone. +2 h.",
        [("Run B - hybrid / TAPT then LORMA+ / Fusion-in-Decoder", {
            "model": "hybrid", "method": "lorma", "rank": 16, "learning_rate": "5e-4", "tapt_minutes": 60,
            "tapt_lr": "2e-4", "time_budget_min": 100, "run_name": "p6_B_hybrid_lorma", **COMMON}),
         ("Run E - efficient / LORMA+ (adapters vs LORMA ablation against Run A)", {
            "model": "efficient", "method": "lorma", "rank": 16, "learning_rate": "5e-4", "time_budget_min": 90,
            "run_name": "p6_E_efficient_lorma", **COMMON}, True)],
        "phase1_hybrid_baseline_artifacts.zip (+ phase1_improved_efficient_artifacts.zip for optional Run E)"),
    "CS781_Phase6_NB3_EncDec_LoRA.ipynb": notebook(
        "CS781 Phase 6 (Summarization) - Notebook 3 (Kaggle account 3)",
        "Run C: Phase 1 **encoder-decoder** model (barely trained in Phase 1, 400 steps): 60 min of **TAPT**, then "
        "**LoRA** (rank 16 on all encoder + decoder FFNs, backbone frozen) with Fusion-in-Decoder over 8 x 256-token "
        "chunks. ~3 h.\n\n"
        "Optional Run F: **efficient + Houlsby adapters with LEAD source selection** (first 768 tokens instead of "
        "salience selection). Compared with Run A, it is the long-document content-selection ablation. +2 h.",
        [("Run C - encoder-decoder / TAPT then LoRA / Fusion-in-Decoder", {
            "model": "encoder_decoder", "method": "lora", "rank": 16, "learning_rate": "5e-4", "tapt_minutes": 60,
            "tapt_lr": "2e-4", "time_budget_min": 100, "run_name": "p6_C_encdec_lora", **COMMON}),
         ("Run F - efficient / adapters / LEAD selection (content-selection ablation vs Run A)", {
            "model": "efficient", "method": "adapter", "adapter_dim": 128, "learning_rate": "3e-4",
            "time_budget_min": 90, "run_name": "p6_F_efficient_adapter_lead",
            **{**COMMON, "selection": "lead"}}, True)],
        "phase1_ed_baseline_artifacts.zip (+ phase1_improved_efficient_artifacts.zip for optional Run F)"),
}

# ----------------------------------------------------------------------------- max-score notebook
BEST = {"selection": "salience", "src_tokens": 1024, "tgt_tokens": 560, "batch_docs": 2, "grad_accum": 8, "epochs": 5,
        "eval_every_min": 25, "method": "full", "tapt_minutes": 90, "tapt_lr": "1e-4", "learning_rate": "5e-5"}
M1 = flags({"model": "efficient", "global_every": 4, "run_name": "p6_M1_efficient_full", **BEST})
M2 = flags({"model": "decoder", "run_name": "p6_M2_decoder_full", **BEST})
best_cells = [
    md("# CS781 Phase 6 (Summarization) - max-score run: TAPT + 2 full fine-tunes + ensemble decoding\n\n"
       "1. Two members train **in parallel, one per T4 GPU**, both prefix LMs over the same 1,024-token salience-selected "
       "source with 560-token summaries:\n"
       "   * **M1 = efficient** (the strongest Phase 1 LM, perplexity 19): 90 min TAPT, then full fine-tuning with "
       "**global attention in every 4th layer** (the other layers keep the Phase 1 64-token sliding window), so summary "
       "tokens can attend to the whole source;\n"
       "   * **M2 = decoder** (perplexity 25, full attention): 90 min TAPT, then full fine-tuning.\n"
       "   Each member tunes decoding + length on validation with BERTScore and writes its own submission.\n"
       "2. **Ensemble decoding:** both members decode together (average of per-step token log-probabilities - they share "
       "one tokenizer). It is tuned on the same validation documents, and `BEST_submission.zip` is whichever of {M1, M2, "
       "ensemble} has the higher validation BERTScore.\n\n"
       "**Setup:** Accelerator **GPU T4 x2** (on a single GPU the members run one after the other at half budget), "
       "Internet on, secret `HF`, dataset with `phase1_improved_efficient_artifacts.zip` + "
       "`phase1_improved_decoder_artifacts.zip`. **Save Version -> Save & Run All (Commit)**: ~8-9 h (Kaggle limit "
       "12 h).\n\n" + DATA_NOTE),
    code(SETUP)]
for name in SOURCES:
    best_cells.append(code(f"%%writefile /kaggle/working/{name}\n" + (src / name).read_text(encoding="utf-8")))
best_cells += [
    md("## 1. Two members in parallel (~7 h)\nLogs: `p6_M1.log`, `p6_M2.log`.\n"),
    code("%%bash\nset +e\ncd /kaggle/working\nNGPU=$(nvidia-smi -L | wc -l); echo \"GPUs: $NGPU\"\n"
         "if [ \"$NGPU\" -ge 2 ]; then\n"
         f"  CUDA_VISIBLE_DEVICES=0 python phase6_summ.py {M1} --time-budget-min 300 --output-dir /kaggle/working/p6_M1_efficient_full > p6_M1.log 2>&1 &\n"
         "  sleep 180\n"
         f"  CUDA_VISIBLE_DEVICES=1 python phase6_summ.py {M2} --time-budget-min 295 --output-dir /kaggle/working/p6_M2_decoder_full > p6_M2.log 2>&1 &\n"
         "  wait\nelse\n"
         f"  python phase6_summ.py {M1} --tapt-minutes 45 --time-budget-min 150 --output-dir /kaggle/working/p6_M1_efficient_full > p6_M1.log 2>&1\n"
         f"  python phase6_summ.py {M2} --tapt-minutes 45 --time-budget-min 150 --output-dir /kaggle/working/p6_M2_decoder_full > p6_M2.log 2>&1\n"
         "fi\n"
         "for f in p6_M1.log p6_M2.log; do echo \"===== $f =====\"; grep -E 'LEAKAGE|DATA:|TAPT held|opt steps/s|chosen decoding|SUBMISSION|Error|error' $f | tail -n 20; tail -n 15 $f; done\nexit 0\n"),
    md("## 2. Ensemble decoding + pick the best (~30-60 min)\n"),
    code("!cd /kaggle/working && CUDA_VISIBLE_DEVICES=0 python phase6_ensemble.py --runs p6_M1_efficient_full "
         "p6_M2_decoder_full --output-dir /kaggle/working/p6_ENSEMBLE 2>&1 | tee p6_ensemble.log\n"
         "import shutil, glob\nfrom pathlib import Path\nfrom IPython.display import FileLink, display\n"
         "Path('/kaggle/working/SUBMIT').mkdir(exist_ok=True)\n"
         "for z in glob.glob('/kaggle/working/p6_*/submission_*.zip') + glob.glob('/kaggle/working/p6_ENSEMBLE/BEST_submission.zip'):\n"
         "    shutil.copy(z, '/kaggle/working/SUBMIT/'); print(z)\n"
         "for z in sorted(glob.glob('/kaggle/working/SUBMIT/*.zip')):\n"
         "    display(FileLink(z.replace('/kaggle/working/', '')))\n"
         "print('UPLOAD SUBMIT/BEST_submission.zip')\n"),
    code("!cd /kaggle/working && zip -qr phase6_best_outputs.zip p6_* *.log -x '*/phase1*/*' '*/best_fp16.pt' "
         "&& ls -la phase6_best_outputs.zip SUBMIT\n"),
]
books["CS781_Phase6_BEST_TAPT_Full_Ensemble.ipynb"] = {
    "cells": best_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                      "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}

for name, book in books.items():
    (out / name).write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / name)
