"""Build the self-contained Kaggle notebook for Phase 2."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
phase1_source = (root / "phase1_pipeline.py").read_text(encoding="utf-8")
phase2_source = (root / "phase2_lner.py").read_text(encoding="utf-8")


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


notebook = {
    "cells": [
        markdown(
            "# CS781 Phase 2 - Legal Named Entity Recognition\n\n"
            "Enable a GPU and Internet. Add a Kaggle secret named `HF`. Attach the four selected Phase 1 artifact ZIPs as a Kaggle dataset. "
            "Run setup, then run only the experiment assigned to this compute session.\n"
        ),
        code(
            "!nvidia-smi\n"
            "!pip install -q tokenizers huggingface_hub\n"
            "import os\n"
            "from kaggle_secrets import UserSecretsClient\n"
            "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n"
            "print('HF token loaded; value intentionally hidden')\n"
        ),
        code("%%writefile /kaggle/working/phase1_pipeline.py\n" + phase1_source),
        code("%%writefile /kaggle/working/phase2_lner.py\n" + phase2_source),
        code("!python /kaggle/working/phase2_lner.py --self-test\n"),
        code(
            "from pathlib import Path\n"
            "required = [\n"
            "    'phase1_improved_decoder_artifacts.zip',\n"
            "    'phase1_ed_baseline_artifacts.zip',\n"
            "    'phase1_hybrid_baseline_artifacts.zip',\n"
            "    'phase1_improved_efficient_artifacts.zip',\n"
            "]\n"
            "for name in required:\n"
            "    matches = list(Path('/kaggle/input').rglob(name))\n"
            "    print(name, '->', matches or 'MISSING')\n"
        ),
        markdown(
            "## Experiment A - encoder-decoder full fine-tuning\n\n"
            "Run this first. Its `best.pt` is the teacher required by Experiment D.\n"
        ),
        code(
            "!python /kaggle/working/phase2_lner.py --model encoder_decoder --method full "
            "--epochs 3 --batch-size 8 --grad-accum 2 "
            "--output-dir /kaggle/working/phase2_encoder_decoder_full\n"
        ),
        markdown("## Experiment B - hybrid partial/layer-wise fine-tuning\n"),
        code(
            "!python /kaggle/working/phase2_lner.py --model hybrid --method partial "
            "--epochs 3 --batch-size 8 --grad-accum 2 "
            "--output-dir /kaggle/working/phase2_hybrid_partial\n"
        ),
        markdown("## Experiment C - efficient model with compulsory LoRMA+\n"),
        code(
            "!python /kaggle/working/phase2_lner.py --model efficient --method lorma "
            "--rank 8 --epochs 3 --batch-size 8 --grad-accum 2 --learning-rate 0.0001 "
            "--output-dir /kaggle/working/phase2_efficient_lorma\n"
        ),
        markdown(
            "## Experiment D - compulsory knowledge distillation\n\n"
            "First download Experiment A's `best.pt`, create/attach a private Kaggle dataset containing it, and then run this cell.\n"
        ),
        code(
            "from pathlib import Path\n"
            "import subprocess, sys\n"
            "teachers = list(Path('/kaggle/input').rglob('best.pt'))\n"
            "if len(teachers) != 1:\n"
            "    raise RuntimeError(f'Attach exactly one teacher best.pt; found {teachers}')\n"
            "command = [\n"
            "    sys.executable, '/kaggle/working/phase2_lner.py',\n"
            "    '--model', 'decoder', '--method', 'distill',\n"
            "    '--teacher-checkpoint', str(teachers[0]),\n"
            "    '--student-layers', '8', '--epochs', '3',\n"
            "    '--batch-size', '8', '--grad-accum', '2',\n"
            "    '--output-dir', '/kaggle/working/phase2_decoder_distill',\n"
            "]\n"
            "subprocess.run(command, check=True)\n"
        ),
        markdown(
            "## Outputs\n\n"
            "Each experiment directory contains `best.pt`, metrics, a manifest, `prediction.json`, and a submission ZIP. "
            "Upload the submission ZIP to Codabench; do not upload the Phase 1 artifact ZIP.\n"
        ),
        code(
            "from pathlib import Path\n"
            "for path in Path('/kaggle/working').glob('phase2_*/submission_*.zip'):\n"
            "    print(path, path.stat().st_size)\n"
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(root / "CS781_Phase2_LNER.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(root / "CS781_Phase2_LNER.ipynb")
