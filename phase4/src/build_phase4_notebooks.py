"""Build the three Kaggle notebooks (one per Kaggle account) for CS781 Phase 4 (CJPE)."""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root
SOURCES = ["phase1_pipeline.py", "phase1_foundation.py", "phase2_lner.py", "phase3_rr.py", "phase4_cjpe.py"]


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def run_cells(label, args):
    name = args["run_name"]
    flags = " ".join(f"--{k.replace('_', '-')}" if v is True else f"--{k.replace('_', '-')} {v}" for k, v in args.items())
    return [
        md(f"## {label}\n"),
        code(f"!cd /kaggle/working && python phase4_cjpe.py {flags} --output-dir /kaggle/working/{name} 2>&1 | tee /kaggle/working/{name}.log\n"),
        code(
            "import json, shutil\nfrom pathlib import Path\nfrom IPython.display import FileLink, display\n"
            f"run = Path('/kaggle/working/{name}')\n"
            "zips = sorted(run.glob('submission_*.zip'))\n"
            "assert zips, 'No submission ZIP produced - read the log above'\n"
            "Path('/kaggle/working/SUBMIT').mkdir(exist_ok=True)\n"
            "shutil.copy(zips[0], '/kaggle/working/SUBMIT/')\n"
            "m = json.loads((run / 'manifest.json').read_text())\n"
            "print({k: m[k] for k in ('run', 'best_dev_macro_f1_argmax', 'threshold', 'dev_macro_f1_tuned', "
            "'trainable_parameters', 'parameters', 'test_pred_distribution', 'total_minutes')})\n"
            "display(FileLink(str(zips[0].relative_to('/kaggle/working'))))\n"
            "print('UPLOAD THIS ZIP TO CODABENCH NOW:', zips[0])\n"
        ),
    ]


def notebook(title, intro, runs):
    cells = [
        md(f"# {title}\n\n{intro}\n\n"
           "**Setup:** GPU on (T4 or P100), Internet on, secret `HF` enabled, Phase 1 artifact dataset attached. "
           "Use **Run All in the interactive editor** (not *Save Version*). Finished ZIPs are copied to `/kaggle/working/SUBMIT/`.\n\n"
           "Data: only the official Phase 4 CJPE train/dev from `Exploration-Lab/CS-781-Capstone` (always re-downloaded, so the "
           "corrected post-leak split is used). The script also removes any train/dev case that duplicates a test case and logs "
           "the counts (`LEAKAGE CHECK`).\n"),
        code(
            "!nvidia-smi\n"
            "!pip install -q tokenizers huggingface_hub\n"
            "import os\n"
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
        cells.append(code(f"%%writefile /kaggle/working/{name}\n" + (root / name).read_text(encoding="utf-8")))
    for label, args in runs:
        cells += run_cells(label, args)
    cells.append(code("!ls -la /kaggle/working/SUBMIT\n!for z in /kaggle/working/SUBMIT/*.zip; do echo $z; unzip -l $z; done\n"
                      "!cd /kaggle/working && zip -qr phase4_outputs.zip p4_* -x '*/phase1/*' '*/data/*' && ls -la phase4_outputs.zip\n"))
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}


COMMON = {"selection": "headtail", "max_chunks": 6, "chunk_len": 256, "batch_docs": 4, "grad_accum": 2, "eval_every_min": 6}

books = {
    "CS781_Phase4_NB1_Efficient_Full_then_Decoder_KD.ipynb": notebook(
        "CS781 Phase 4 - Notebook 1 (Kaggle account 1)",
        "Run A: Phase 1 **efficient** model, **full fine-tuning**, hierarchical head+tail chunks with attention pooling (KD teacher).\n"
        "Run D: Phase 1 **decoder** cut to 8 of 16 layers, **knowledge distillation** from Run A's soft labels.",
        [
            ("Run A - efficient / full fine-tuning / hierarchical (teacher)", {
                "model": "efficient", "method": "full", "learning_rate": "5e-5", "time_budget_min": 22,
                "export_soft_labels": True, "export_budget_min": 5, "run_name": "p4_A_efficient_full", **COMMON}),
            ("Run D - decoder student (8 layers) / knowledge distillation from Run A", {
                "model": "decoder", "method": "distill", "learning_rate": "5e-5",
                "teacher_soft_labels": "/kaggle/working/p4_A_efficient_full/teacher_soft_labels.pt",
                "student_layers": 8, "temperature": 2.0, "kd_alpha": 0.5, "time_budget_min": 18,
                "run_name": "p4_D_decoder_kd", **COMMON}),
        ],
    ),
    "CS781_Phase4_NB2_Hybrid_LORMA.ipynb": notebook(
        "CS781 Phase 4 - Notebook 2 (Kaggle account 2)",
        "Run B: Phase 1 **hybrid** model with **LORMA+** (frozen backbone, multiplicative low-rank adapters), hierarchical chunks.",
        [("Run B - hybrid / LORMA+ / hierarchical", {
            "model": "hybrid", "method": "lorma", "learning_rate": "5e-4", "rank": 16, "time_budget_min": 32,
            "run_name": "p4_B_hybrid_lorma", **COMMON})],
    ),
    "CS781_Phase4_NB3_EncDec_Partial.ipynb": notebook(
        "CS781 Phase 4 - Notebook 3 (Kaggle account 3)",
        "Run C: Phase 1 **encoder-decoder** model with **partial / layer-wise fine-tuning** (top 2 encoder + top 2 decoder "
        "layers, fusion and norm trainable; rest frozen), hierarchical chunks.",
        [("Run C - encoder-decoder / partial fine-tuning / hierarchical", {
            "model": "encoder_decoder", "method": "partial", "unfreeze_layers": 2, "learning_rate": "1e-4",
            "time_budget_min": 32, "run_name": "p4_C_encdec_partial", **COMMON})],
    ),
}
for name, book in books.items():
    (out / name).write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / name)
