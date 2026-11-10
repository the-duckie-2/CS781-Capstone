"""Build the self-contained Phase 1 flagship Kaggle notebook."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
pipeline_source = (root / "phase1_pipeline.py").read_text(encoding="utf-8")
foundation_source = (root / "phase1_foundation.py").read_text(encoding="utf-8")
evaluator_source = (root.parent / "evaluate_phase1.py").read_text(encoding="utf-8")


def markdown(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(lines),
    }


notebook = {
    "cells": [
        markdown(
            "# CS781 Phase 1 flagship foundation model\n",
            "Self-contained Kaggle notebook: it writes every required Python file itself. Enable a GPU and Internet, then run cells in order. No `.py` input files are required.\n",
        ),
        code(
            "!nvidia-smi\n",
            "!pip install -q ijson tokenizers huggingface_hub bert-score\n",
            "import os\n",
            "from kaggle_secrets import UserSecretsClient\n",
            "secret = None\n",
            "for secret_name in ('HF', 'HF_TOKEN'):\n",
            "    try:\n",
            "        secret = UserSecretsClient().get_secret(secret_name)\n",
            "        if secret:\n",
            "            print(f'Hugging Face token loaded from Kaggle secret {secret_name}')\n",
            "            break\n",
            "    except Exception:\n",
            "        pass\n",
            "assert secret, 'Create and enable a Kaggle secret named HF or HF_TOKEN'\n",
            "os.environ['HF_TOKEN'] = secret\n",
            "from huggingface_hub import login\n",
            "login(token=secret, add_to_git_credential=False)\n",
            "import torch\n",
            "assert torch.cuda.device_count() == 2, f'Select Kaggle GPU T4 x2; found {torch.cuda.device_count()} GPU(s)'\n",
            "os.environ['OMP_NUM_THREADS'] = '2'\n",
            "os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'\n",
        ),
        code("%%writefile /kaggle/working/phase1_pipeline.py\n", pipeline_source),
        code("%%writefile /kaggle/working/phase1_foundation.py\n", foundation_source),
        code("%%writefile /kaggle/working/evaluate_phase1.py\n", evaluator_source),
        markdown(
            "## Two-GPU infrastructure smoke test\n",
            "This performs two synchronized DDP optimizer steps across both T4s, writes a recovery checkpoint, and stops before expensive generation.\n",
        ),
        code(
            "!torchrun --standalone --nproc_per_node=2 /kaggle/working/phase1_foundation.py --steps 3 --run-steps 2 --train-chars 20000000 --token-pool 2000000 --validation-examples 8 --batch-size 1 --grad-accum 8 --output-dir /kaggle/working/foundation_ddp_smoke\n"
        ),
        markdown(
            "## Overnight dual-GPU training\n",
            "This produces the reusable model and a reliable local held-out score in one run. Both T4s are used while preserving the original global batch of eight sequences. The complete notebook-cell budget is approximately 11 hours: the trainer receives 10.5 hours, reserves its final hour for distributed generation across all 256 held-out examples, and then the official local evaluator computes BERTScore, ROUGE-L, and BLEU-4. Official test generation and Codabench packaging are skipped. The target is 40,000 optimizer steps; if time expires first, the best current checkpoint is finalized automatically.\n",
        ),
        code(
            "!torchrun --standalone --nproc_per_node=2 /kaggle/working/phase1_foundation.py --steps 40000 --max-hours 10.5 --generation-reserve-hours 1 --finalize-on-time-limit --validation-only --train-chars 500000000 --token-pool 100000000 --validation-examples 256 --batch-size 1 --grad-accum 8 --output-dir /kaggle/working/phase1_foundation_ddp\n",
            "import subprocess, sys\n",
            "from pathlib import Path\n",
            "score = subprocess.run([sys.executable, '/kaggle/working/evaluate_phase1.py', '--truth', '/kaggle/working/phase1_foundation_ddp/validation_truth.json', '--prediction', '/kaggle/working/phase1_foundation_ddp/validation_prediction_foundation.json'], text=True, capture_output=True)\n",
            "print(score.stdout, end='')\n",
            "if score.stderr:\n",
            "    print(score.stderr, file=sys.stderr, end='')\n",
            "Path('/kaggle/working/phase1_foundation_ddp/validation_metrics.txt').write_text(score.stdout + score.stderr, encoding='utf-8')\n",
            "score.check_returncode()\n",
        ),
        markdown(
            "## Recovery-only finalization fallback\n",
            "The overnight cell already finalizes automatically. Use these commands only if Kaggle interrupts the session before the held-out predictions and metrics are written. They generate only the 256 validation continuations and evaluate them; no official test data are loaded.\n",
        ),
        code(
            "# Remove the leading # tomorrow if the overnight run stopped at the time limit.\n",
            "# !torchrun --standalone --nproc_per_node=2 /kaggle/working/phase1_foundation.py --generate-only --validation-only --checkpoint /kaggle/working/phase1_foundation_ddp/foundation_recovery.pt --tokenizer /kaggle/working/phase1_foundation_ddp/legal_bpe_24k.json --output-dir /kaggle/working/phase1_foundation_ddp\n",
            "# !python /kaggle/working/evaluate_phase1.py --truth /kaggle/working/phase1_foundation_ddp/validation_truth.json --prediction /kaggle/working/phase1_foundation_ddp/validation_prediction_foundation.json | tee /kaggle/working/phase1_foundation_ddp/validation_metrics.txt\n",
        ),
        markdown(
            "## Download artifacts after the run\n",
            "Download `foundation.pt`, `legal_bpe_24k.json`, `manifest.json`, and `validation_metrics.txt`. Keep the resume bundle as a recovery copy. No Codabench submission is produced in validation-only mode.\n",
        ),
        code(
            "from pathlib import Path\n",
            "import zipfile\n",
            "from IPython.display import FileLink, display\n",
            "root = Path('/kaggle/working/phase1_foundation_ddp')\n",
            "names = ['foundation_recovery.pt', 'legal_bpe_24k.json', 'manifest_partial.json', 'foundation.pt', 'validation_prompts.json', 'validation_truth.json', 'validation_prediction_foundation.json', 'validation_metrics.txt', 'manifest.json']\n",
            "for name in names:\n",
            "    path = root / name\n",
            "    if path.exists():\n",
            "        display(FileLink(str(path)))\n",
            "resume_bundle = Path('/kaggle/working/phase1_resume_bundle.zip')\n",
            "with zipfile.ZipFile(resume_bundle, 'w', compression=zipfile.ZIP_STORED) as archive:\n",
            "    for name in ('foundation_recovery.pt', 'legal_bpe_24k.json', 'manifest_partial.json'):\n",
            "        path = root / name\n",
            "        if path.exists():\n",
            "            archive.write(path, arcname=name)\n",
            "display(FileLink(str(resume_bundle)))\n",
        ),
        markdown(
            "## Resume from a checkpoint attached in a later session\n",
            "Use this only after adding a saved checkpoint/tokenizer dataset. The scripts still come from this notebook; the only inputs are your training artifacts. Replace the two placeholder paths.\n",
        ),
        code(
            "# !torchrun --standalone --nproc_per_node=2 /kaggle/working/phase1_foundation.py --steps 40000 --max-hours 10.5 --generation-reserve-hours 1 --finalize-on-time-limit --validation-only --train-chars 500000000 --token-pool 100000000 --validation-examples 256 --batch-size 1 --grad-accum 8 --checkpoint /kaggle/input/YOUR-CHECKPOINT-DATASET/foundation_recovery.pt --tokenizer /kaggle/input/YOUR-CHECKPOINT-DATASET/legal_bpe_24k.json --output-dir /kaggle/working/phase1_foundation_ddp\n"
        ),
        markdown("## Local held-out evaluation\n"),
        code(
            "!python /kaggle/working/evaluate_phase1.py --truth /kaggle/working/phase1_foundation_ddp/validation_truth.json --prediction /kaggle/working/phase1_foundation_ddp/validation_prediction_foundation.json\n"
        ),
        markdown(
            "## Required architectural ablation\n",
            "After the flagship is secure, run a shorter controlled experiment with the same settings plus `--no-global-memory`. This tests the gated token/global source-memory contribution.\n",
        ),
    ],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = root / "CS781_Phase1_Foundation.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output)
