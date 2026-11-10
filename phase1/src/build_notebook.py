"""Build the self-contained Phase 1 Kaggle notebook."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
source = (root / "phase1_pipeline.py").read_text(encoding="utf-8")
evaluator_source = (root.parent / "evaluate_phase1.py").read_text(encoding="utf-8")
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CS781 Phase 1 training and submissions\n",
                "This notebook trains four distinct 50M+ parameter architectures from random initialization on the official legal corpus, validates them, and creates correctly structured submission archives. Enable a Kaggle GPU and Internet before running.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n",
                "!pip install -q ijson tokenizers huggingface_hub bert-score\n",
                "import os\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n",
                "    print('Hugging Face token loaded from Kaggle secret HF')\n",
                "except Exception as exc:\n",
                "    print('No Kaggle HF secret loaded:', exc)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%%writefile /kaggle/working/phase1_pipeline.py\n", source],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%%writefile /kaggle/working/evaluate_phase1.py\n", evaluator_source],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 1: four required architecture runs\n",
                "Run this first. On T4 x2 it launches two isolated workers, one per GPU; on a single GPU it automatically uses the sequential path.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, subprocess, sys, time\n",
                "import torch\n",
                "script = '/kaggle/working/phase1_pipeline.py'\n",
                "if torch.cuda.device_count() >= 2:\n",
                "    jobs = [\n",
                "        ('gpu0', '0', 'decoder,hybrid', '/kaggle/working/phase1_gpu0'),\n",
                "        ('gpu1', '1', 'encoder_decoder,efficient', '/kaggle/working/phase1_gpu1'),\n",
                "    ]\n",
                "    processes = []\n",
                "    for label, gpu, models, output in jobs:\n",
                "        log = open(f'/kaggle/working/{label}.log', 'w')\n",
                "        env = os.environ.copy()\n",
                "        env['CUDA_VISIBLE_DEVICES'] = gpu\n",
                "        command = [sys.executable, script, '--models', models, '--steps', '400', '--seq-len', '256', '--output-dir', output]\n",
                "        processes.append((label, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))\n",
                "    while any(process.poll() is None for _, process, _ in processes):\n",
                "        print(time.strftime('%H:%M:%S'), {label: process.poll() for label, process, _ in processes})\n",
                "        time.sleep(30)\n",
                "    for label, process, log in processes:\n",
                "        log.close()\n",
                "        print(f'\\n===== {label} exit={process.returncode} =====')\n",
                "        print(open(f'/kaggle/working/{label}.log').read()[-6000:])\n",
                "        if process.returncode:\n",
                "            raise RuntimeError(f'{label} failed; inspect /kaggle/working/{label}.log')\n",
                "else:\n",
                "    subprocess.run([sys.executable, script, '--model', 'all', '--steps', '400', '--seq-len', '256', '--output-dir', '/kaggle/working/phase1'], check=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "roots = [p for p in (Path('/kaggle/working/phase1'), Path('/kaggle/working/phase1_gpu0'), Path('/kaggle/working/phase1_gpu1')) if p.exists()]\n",
                "for root in roots:\n",
                "    print(f'\\n{root}')\n",
                "    for path in sorted(root.glob('submission_*.zip')):\n",
                "        print(path.name, path.stat().st_size)\n",
                "    manifest = root / 'manifest.json'\n",
                "    print(manifest.read_text() if manifest.exists() else 'manifest.json does not exist in this runtime yet')\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Local held-out evaluation\n",
                "This uses the supplied Phase 1 evaluator on eight complete documents excluded from tokenizer and model training. These scores are for model selection only; private Codabench references remain unavailable.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "from pathlib import Path\n",
                "parallel = Path('/kaggle/working/phase1_gpu0').exists()\n",
                "locations = {\n",
                "    'decoder': '/kaggle/working/phase1_gpu0' if parallel else '/kaggle/working/phase1',\n",
                "    'hybrid': '/kaggle/working/phase1_gpu0' if parallel else '/kaggle/working/phase1',\n",
                "    'encoder_decoder': '/kaggle/working/phase1_gpu1' if parallel else '/kaggle/working/phase1',\n",
                "    'efficient': '/kaggle/working/phase1_gpu1' if parallel else '/kaggle/working/phase1',\n",
                "}\n",
                "truth = f\"{locations['decoder']}/validation_truth.json\"\n",
                "for name in ('decoder', 'encoder_decoder', 'hybrid', 'efficient'):\n",
                "    result = subprocess.run([\n",
                "        'python', '/kaggle/working/evaluate_phase1.py',\n",
                "        '--truth', truth,\n",
                "        '--prediction', f\"{locations[name]}/validation_prediction_{name}.json\",\n",
                "    ], text=True, capture_output=True)\n",
                "    print(f'\\n{name}\\n{result.stdout or result.stderr}')\n",
                "    if result.returncode:\n",
                "        raise RuntimeError(f'Local evaluator failed for {name}')\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 2: score improvement\n",
                "After all four Stage 1 ZIPs have been downloaded and submitted, use remaining GPU time for a longer efficient-model run. This continues the saved model for 6,000 additional steps on a larger mixed official-data pool and creates a replacement candidate without affecting the four archived Stage 1 submissions.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import subprocess, sys\n",
                "base = Path('/kaggle/working/phase1_gpu1') if Path('/kaggle/working/phase1_gpu1').exists() else Path('/kaggle/working/phase1')\n",
                "subprocess.run([sys.executable, '/kaggle/working/phase1_pipeline.py', '--model', 'efficient', '--steps', '6000', '--seq-len', '512', '--batch-size', '1', '--grad-accum', '8', '--train-chars', '160000000', '--train-tokens', '40000000', '--tokenizer-path', str(base / 'phase1_bpe.json'), '--resume-checkpoint', str(base / 'efficient.pt'), '--validation-prompts', str(base / 'validation_prompts.json'), '--validation-truth', str(base / 'validation_truth.json'), '--output-dir', '/kaggle/working/phase1_improved'], check=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python /kaggle/working/evaluate_phase1.py --truth /kaggle/working/phase1_improved/validation_truth.json --prediction /kaggle/working/phase1_improved/validation_prediction_efficient.json\n",
            ],
        },
    ],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "CS781_Phase1_Training.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
)
