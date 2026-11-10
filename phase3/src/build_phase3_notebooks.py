"""Build the three Kaggle notebooks (one per Kaggle account) for CS781 Phase 3."""

import json
from pathlib import Path

root = Path(__file__).resolve().parent
out = Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else root
SOURCES = ["phase1_pipeline.py", "phase1_foundation.py", "phase2_lner.py", "phase3_rr.py"]


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def run_cell(label, args):
    name = args["run_name"]
    flags = " ".join(f"--{k.replace('_', '-')} {v}" if v is not True else f"--{k.replace('_', '-')}" for k, v in args.items())
    return [
        md(f"## {label}\n"),
        code(f"!cd /kaggle/working && python phase3_rr.py {flags} --output-dir /kaggle/working/{name} 2>&1 | tee /kaggle/working/{name}.log\n"),
        code(
            "import json, shutil\nfrom pathlib import Path\nfrom IPython.display import FileLink, display\n"
            f"run = Path('/kaggle/working/{name}')\n"
            "zips = sorted(run.glob('submission_*.zip'))\n"
            "assert zips, 'No submission ZIP produced - read the log above'\n"
            "Path('/kaggle/working/SUBMIT').mkdir(exist_ok=True)\n"
            "shutil.copy(zips[0], '/kaggle/working/SUBMIT/')\n"
            "m = json.loads((run / 'manifest.json').read_text())\n"
            "print({k: m[k] for k in ('run', 'best_dev_macro_f1', 'dev_CL_macro_f1', 'dev_IT_macro_f1', 'trainable_parameters', 'parameters', 'train_minutes')})\n"
            "display(FileLink(str(zips[0].relative_to('/kaggle/working'))))\n"
            "print('UPLOAD THIS ZIP TO CODABENCH NOW:', zips[0])\n"
        ),
    ]


def notebook(title, intro, runs):
    cells = [
        md(f"# {title}\n\n{intro}\n\n"
           "**Setup:** GPU on (T4 or P100), Internet on, secret `HF` (Hugging Face read token with access to "
           "`Exploration-Lab/CS-781-Capstone`) enabled, and your private dataset with the Phase 1 artifact ZIPs attached.\n\n"
           "Use **Run All in the interactive editor** (not *Save Version*) so each ZIP can be downloaded as soon as its run finishes. "
           "Every finished ZIP is also copied to `/kaggle/working/SUBMIT/`.\n"),
        code(
            "!nvidia-smi\n"
            "!pip install -q tokenizers huggingface_hub\n"
            "import os\n"
            "from kaggle_secrets import UserSecretsClient\n"
            "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n"
            "print('HF token loaded; value intentionally hidden')\n"
            "from pathlib import Path\n"
            "found = []\n"
            "for folder, _, names in os.walk('/kaggle/input', followlinks=True):\n"
            "    for n in names:\n"
            "        if n.endswith(('.zip', '.pt', '.json')): found.append(os.path.join(folder, n)); print(found[-1])\n"
            "assert found, 'NOTHING attached under /kaggle/input - click Add Input in the right panel and attach your Phase 1 artifact dataset'\n"
        ),
    ]
    for name in SOURCES:
        cells.append(code(f"%%writefile /kaggle/working/{name}\n" + (root / name).read_text(encoding="utf-8")))
    for label, args in runs:
        cells += run_cell(label, args)
    cells.append(code("!ls -la /kaggle/working/SUBMIT\n!for z in /kaggle/working/SUBMIT/*.zip; do echo $z; unzip -l $z; done\n"))
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


COMMON = {"epochs": 4, "batch_size": 16, "max_length": 256, "evals_per_epoch": 2}

books = {
    "CS781_Phase3_NB1_Efficient_Full_then_Decoder_KD.ipynb": notebook(
        "CS781 Phase 3 - Notebook 1 (Kaggle account 1)",
        "Run A: Phase 1 **efficient** model, full fine-tuning, document context (also the KD teacher).\n"
        "Run D: Phase 1 **decoder** truncated to 8 of 16 layers, trained by **knowledge distillation** from Run A's soft labels.",
        [
            ("Run A - efficient / full fine-tuning / context window (teacher)", {
                "model": "efficient", "method": "full", "context": "window", "learning_rate": "3e-5",
                "time_budget_min": 40, "export_soft_labels": True, "run_name": "p3_A_efficient_full_ctx", **COMMON}),
            ("Run D - decoder student (8 layers) / knowledge distillation from Run A", {
                "model": "decoder", "method": "distill", "context": "window", "learning_rate": "5e-5",
                "teacher_soft_labels": "/kaggle/working/p3_A_efficient_full_ctx/teacher_soft_labels.pt",
                "student_layers": 8, "temperature": 2.0, "kd_alpha": 0.5,
                "time_budget_min": 30, "run_name": "p3_D_decoder_kd_ctx", **COMMON}),
        ],
    ),
    "CS781_Phase3_NB2_Hybrid_LORMA_then_Efficient_SentenceOnly.ipynb": notebook(
        "CS781 Phase 3 - Notebook 2 (Kaggle account 2)",
        "Run B: Phase 1 **hybrid** model with **LORMA+** (frozen backbone, multiplicative low-rank adapters).\n"
        "Run E: Phase 1 **efficient** model with LORMA+ and **no document context** (sentence-only ablation).",
        [
            ("Run B - hybrid / LORMA+ / context window", {
                "model": "hybrid", "method": "lorma", "context": "window", "learning_rate": "5e-4", "rank": 16,
                "time_budget_min": 40, "run_name": "p3_B_hybrid_lorma_ctx", **COMMON}),
            ("Run E - efficient / LORMA+ / sentence only (context ablation)", {
                "model": "efficient", "method": "lorma", "context": "none", "learning_rate": "5e-4", "rank": 16,
                "time_budget_min": 30, "run_name": "p3_E_efficient_lorma_sent", **COMMON}),
        ],
    ),
    "CS781_Phase3_NB3_EncDec_LoRA_then_Efficient_IA3.ipynb": notebook(
        "CS781 Phase 3 - Notebook 3 (Kaggle account 3)",
        "Run C: Phase 1 **encoder-decoder** model with **LoRA** on the feed-forward projections.\n"
        "Run F: Phase 1 **efficient** model with **IA3** (learned rescaling of keys, values and FFN activations).",
        [
            ("Run C - encoder-decoder / LoRA / context window", {
                "model": "encoder_decoder", "method": "lora", "context": "window", "learning_rate": "5e-4", "rank": 16,
                "time_budget_min": 40, "run_name": "p3_C_encdec_lora_ctx", **COMMON}),
            ("Run F - efficient / IA3 / context window", {
                "model": "efficient", "method": "ia3", "context": "window", "learning_rate": "3e-3",
                "time_budget_min": 30, "run_name": "p3_F_efficient_ia3_ctx", **COMMON}),
        ],
    ),
}
for name, book in books.items():
    (out / name).write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / name)


# ---- Notebook 4: document-level BiGRU tagger over frozen Phase 1 sentence features (fast, ~5-10 min per model)
SOURCES.append("phase3_doc.py")
doc_cells = notebook(
    "CS781 Phase 3 - Notebook 4: document-level tagger (all 4 Phase 1 models)",
    "Each run: frozen Phase 1 backbone encodes every sentence -> 2-layer BiGRU over the document's sentence sequence "
    "(+ position + domain) -> one label per sentence. 3-seed ensemble, selected on CL+IT dev Macro-F1.\n"
    "Attach all four Phase 1 artifact ZIPs. Each run takes roughly 5-10 min on a GPU.",
    [],
)["cells"]
doc_cells = [c for c in doc_cells if not "".join(c["source"]).startswith("!ls -la /kaggle/working/SUBMIT")]
for tag, model in (("G", "efficient"), ("H", "hybrid"), ("I", "encoder_decoder"), ("J", "decoder")):
    name = f"p3_{tag}_{model}_docgru"
    cells = run_cell(f"Run {tag} - {model} / frozen features + document BiGRU", {"model": model, "run_name": name})
    cells[1] = code(f"!cd /kaggle/working && python phase3_doc.py --model {model} --run-name {name} "
                    f"--output-dir /kaggle/working/{name} 2>&1 | tee /kaggle/working/{name}.log\n")
    cells[2]["source"] = "".join(cells[2]["source"]).replace(
        "'trainable_parameters', 'parameters', 'train_minutes'", "'tagger_parameters', 'phase1_parameters', 'minutes'").splitlines(True)
    doc_cells += cells
doc_cells.append(code("!ls -la /kaggle/working/SUBMIT\n"))
book = {"cells": doc_cells, "metadata": books[next(iter(books))]["metadata"], "nbformat": 4, "nbformat_minor": 5}
(out / "CS781_Phase3_NB4_DocLevel_All4.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
print(out / "CS781_Phase3_NB4_DocLevel_All4.ipynb")
