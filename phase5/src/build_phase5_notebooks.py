"""Build the three Kaggle notebooks (one per Kaggle account) for CS781 Phase 5 (Legal Statute Identification)."""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
src = root / "src"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root
SOURCES = ["phase1_pipeline.py", "phase1_foundation.py", "phase2_lner.py", "phase3_rr.py", "phase4_cjpe.py",
           "phase5_lsi.py"]


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def run_cells(label, args, optional=False):
    name = args["run_name"]
    flags = " ".join(f"--{k.replace('_', '-')}" if v is True else f"--{k.replace('_', '-')} {v}" for k, v in args.items())
    head = f"## {label}\n" + ("\n*Optional extra experiment: run it only if the GPU quota allows. The 4 required "
                              "submissions do not depend on it.*\n" if optional else "")
    return [
        md(head),
        code(f"!cd /kaggle/working && python phase5_lsi.py {flags} --output-dir /kaggle/working/{name} 2>&1 | tee /kaggle/working/{name}.log\n"),
        code(
            "import json, shutil\nfrom pathlib import Path\nfrom IPython.display import FileLink, display\n"
            f"run = Path('/kaggle/working/{name}')\n"
            "zips = sorted(run.glob('submission_*.zip'))\n"
            "assert zips, 'No submission ZIP produced - read the log above'\n"
            "Path('/kaggle/working/SUBMIT').mkdir(exist_ok=True)\n"
            "shutil.copy(zips[0], '/kaggle/working/SUBMIT/')\n"
            "m = json.loads((run / 'manifest.json').read_text())\n"
            "print(json.dumps({k: m[k] for k in ('run', 'num_statutes', 'chosen_threshold_strategy', 'dev_macro_f1_at_0.5', "
            "'dev_macro_f1_crossfit', 'dev_micro_f1_crossfit', 'dev_subset_acc_crossfit', 'frequency_buckets', "
            "'trainable_parameters', 'parameters', 'test_mean_pred_labels', 'test_empty_predictions', 'total_minutes', "
            "'peak_gpu_memory_gb')}, indent=1))\n"
            "display(FileLink(str(zips[0].relative_to('/kaggle/working'))))\n"
            "print('UPLOAD THIS ZIP TO CODABENCH NOW:', zips[0])\n"
        ),
    ]


def notebook(title, intro, runs, attach):
    cells = [
        md(f"# {title}\n\n{intro}\n\n"
           "**Setup:** GPU on (T4 x2 or P100; one GPU is used), Internet on, secret `HF` enabled, and the Phase 1 artifact "
           f"dataset attached with **{attach}**. Use **Run All in the interactive editor** (not *Save Version*). "
           "Finished ZIPs are copied to `/kaggle/working/SUBMIT/`.\n\n"
           "Data: only the official Phase 5 LSI `train.json` / `dev.json` / `test.json` from `Exploration-Lab/CS-781-Capstone` "
           "(always re-downloaded with `force_download`, so the latest corrected version is used). Before training, the "
           "script removes any train/dev case whose id or normalised text also appears in test (and train cases duplicated "
           "in dev) and prints the counts (`LEAKAGE CHECK`). Test labels are never available or used. Thresholds and "
           "checkpoints are chosen on dev only.\n"),
        code(
            "!nvidia-smi\n"
            "!pip install -q tokenizers huggingface_hub\n"
            "import os\n"
            "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')\n"
            "from kaggle_secrets import UserSecretsClient\n"
            "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n"
            "print('HF token loaded; value intentionally hidden')\n"
            "found = []\n"
            "for folder, _, names in os.walk('/kaggle/input', followlinks=True):\n"
            "    for n in names:\n"
            "        if n.endswith(('.zip', '.pt')): found.append(os.path.join(folder, n)); print(found[-1])\n"
            "assert found, 'NOTHING attached under /kaggle/input - Add Input -> your Phase 1 artifact dataset'\n"
        ),
    ]
    for name in SOURCES:
        cells.append(code(f"%%writefile /kaggle/working/{name}\n" + (src / name).read_text(encoding="utf-8")))
    for run in runs:
        cells += run_cells(*run)
    cells.append(md("## Collect outputs\nDownload `phase5_outputs.zip` (logs, manifests, metrics, dev predictions, teacher "
                    "logits) and, for the model archive, the `best_fp16.pt` files.\n"))
    cells.append(code("!ls -la /kaggle/working/SUBMIT\n!for z in /kaggle/working/SUBMIT/*.zip; do echo $z; unzip -l $z; done\n"
                      "!cd /kaggle/working && zip -qr phase5_outputs.zip p5_* -x '*/phase1/*' '*/data/*' '*/best_fp16.pt' "
                      "&& ls -la phase5_outputs.zip\n!ls -la /kaggle/working/p5_*/best_fp16.pt\n"))
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}


# shared long-document / head / loss recipe: 6 windows of 256 tokens spread over the whole case (overlap 32),
# label-wise attention pooling, BCE with sqrt(neg/pos) positive weights, dev-tuned thresholds.
COMMON = {"selection": "spread", "pooling": "labelwise", "max_chunks": 6, "chunk_len": 256, "overlap": 32,
          "loss": "bce", "pos_weight_power": 0.5, "batch_docs": 4, "grad_accum": 2, "eval_every_min": 7,
          "save_checkpoint": True}

books = {
    "CS781_Phase5_NB1_Efficient_Full_then_Decoder_KD.ipynb": notebook(
        "CS781 Phase 5 (LSI) - Notebook 1 (Kaggle account 1)",
        "Run A: Phase 1 **efficient** model, **full fine-tuning**, 6 windows spread over the case + label-wise attention "
        "(also the KD teacher: exports per-statute logits for the train cases).\n\n"
        "Run D: Phase 1 **decoder** cut from 16 to 8 layers, **knowledge distillation** from Run A's per-statute logits "
        "(loss = 0.5 BCE + 0.5 T^2 * BCE(student/T, sigmoid(teacher/T)), T = 2). Run D needs Run A's output, so run the "
        "cells in order.",
        [
            ("Run A - efficient / full fine-tuning / spread windows + label-wise attention (teacher)", {
                "model": "efficient", "method": "full", "learning_rate": "5e-5", "time_budget_min": 35,
                "export_soft_labels": True, "export_budget_min": 8, "run_name": "p5_A_efficient_full", **COMMON}),
            ("Run D - decoder student (8 layers) / knowledge distillation from Run A", {
                "model": "decoder", "method": "distill", "learning_rate": "5e-5",
                "teacher_soft_labels": "/kaggle/working/p5_A_efficient_full/teacher_soft_labels.pt",
                "student_layers": 8, "temperature": 2.0, "kd_alpha": 0.5, "time_budget_min": 30,
                "run_name": "p5_D_decoder_kd", **COMMON}),
        ],
        "phase1_improved_efficient_artifacts.zip + phase1_improved_decoder_artifacts.zip",
    ),
    "CS781_Phase5_NB2_Hybrid_LORMA.ipynb": notebook(
        "CS781 Phase 5 (LSI) - Notebook 2 (Kaggle account 2)",
        "Run B: Phase 1 **hybrid** model with **LORMA+** (frozen backbone, multiplicative low-rank adapters on the decoder "
        "FFNs, rank 16), same long-document recipe.\n\n"
        "Optional Run E: **efficient + LORMA+**, the same setup as Run A except for the technique, so A vs E isolates "
        "*PEFT vs full fine-tuning* on one backbone.",
        [
            ("Run B - hybrid / LORMA+ / spread windows + label-wise attention", {
                "model": "hybrid", "method": "lorma", "learning_rate": "5e-4", "rank": 16, "time_budget_min": 45,
                "run_name": "p5_B_hybrid_lorma", **COMMON}),
            ("Run E - efficient / LORMA+ (PEFT vs full-FT ablation against Run A)", {
                "model": "efficient", "method": "lorma", "learning_rate": "5e-4", "rank": 16, "time_budget_min": 35,
                "run_name": "p5_E_efficient_lorma", **COMMON}, True),
        ],
        "phase1_hybrid_baseline_artifacts.zip (+ phase1_improved_efficient_artifacts.zip for optional Run E)",
    ),
    "CS781_Phase5_NB3_EncDec_Partial.ipynb": notebook(
        "CS781 Phase 5 (LSI) - Notebook 3 (Kaggle account 3)",
        "Run C: Phase 1 **encoder-decoder** model with **partial / layer-wise fine-tuning** (top 2 encoder + top 2 decoder "
        "layers, fusion and norm trainable, the rest frozen), same long-document recipe.\n\n"
        "Optional Run F: **efficient + LORMA+** with **head+tail windows and plain attention pooling** (no label-wise "
        "attention). Compared with optional Run E, it is the long-document representation ablation.",
        [
            ("Run C - encoder-decoder / partial fine-tuning / spread windows + label-wise attention", {
                "model": "encoder_decoder", "method": "partial", "unfreeze_layers": 2, "learning_rate": "1e-4",
                "time_budget_min": 45, "run_name": "p5_C_encdec_partial", **COMMON}),
            ("Run F - efficient / LORMA+ / head+tail windows, attention pooling (representation ablation vs Run E)", {
                "model": "efficient", "method": "lorma", "learning_rate": "5e-4", "rank": 16, "time_budget_min": 35,
                "run_name": "p5_F_efficient_lorma_headtail_attn",
                **{**COMMON, "selection": "headtail", "pooling": "attention"}}, True),
        ],
        "phase1_ed_baseline_artifacts.zip (+ phase1_improved_efficient_artifacts.zip for optional Run F)",
    ),
}
for name, book in books.items():
    (out / name).write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / name)
