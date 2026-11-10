"""Build the three purpose-specific Kaggle notebooks for the five-hour Phase 2 plan."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
sources = {
    "phase1_pipeline.py": (root / "phase1_pipeline.py").read_text(encoding="utf-8"),
    "phase1_foundation.py": (root / "phase1_foundation.py").read_text(encoding="utf-8"),
    "phase2_lner.py": (root / "phase2_lner.py").read_text(encoding="utf-8"),
}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def common(title, description):
    cells = [
        markdown(f"# {title}\n\n{description}\n\nEnable GPU and Internet before running.\n"),
        code(
            "!nvidia-smi\n"
            "!pip install -q tokenizers huggingface_hub\n"
            "import os\n"
            "from kaggle_secrets import UserSecretsClient\n"
            "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n"
            "print('HF token loaded; value intentionally hidden')\n"
        ),
    ]
    for name, source in sources.items():
        cells.append(code(f"%%writefile /kaggle/working/{name}\n" + source))
    cells.append(code("!python /kaggle/working/phase2_lner.py --self-test\n"))
    return cells


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save(name, cells):
    path = root / name
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)


top1 = common(
    "CS781 Phase 2 TOP 1 - Foundation full encoder fine-tuning",
    "Highest-upside NER run. Attach one private Kaggle Dataset containing `foundation.pt` and `legal_bpe_24k.json` from the 17k-step run.",
)
top1 += [
    code(
        "from pathlib import Path\n"
        "for name in ('foundation.pt', 'legal_bpe_24k.json'):\n"
        "    matches = list(Path('/kaggle/input').rglob(name))\n"
        "    print(name, '->', matches)\n"
        "    if len(matches) != 1:\n"
        "        raise RuntimeError(f'Attach exactly one {name}')\n"
    ),
    code(
        "!python /kaggle/working/phase2_lner.py "
        "--model foundation --method full --epochs 2 --batch-size 1 --grad-accum 8 "
        "--learning-rate 0.00002 --max-length 512 --stride 128 "
        "--output-dir /kaggle/working/phase2_foundation_full\n"
    ),
    code(
        "from pathlib import Path\n"
        "root = Path('/kaggle/working/phase2_foundation_full')\n"
        "for name in ('validation_metrics.json', 'manifest.json', 'best.pt'):\n"
        "    path = root / name\n"
        "    print(name, path.stat().st_size if path.exists() else 'MISSING')\n"
        "print('CODABENCH:', list(root.glob('submission_*.zip')))\n"
    ),
]
save("CS781_Phase2_TOP1_Foundation.ipynb", top1)


top2 = common(
    "CS781 Phase 2 TOP 2 - Improved efficient full fine-tuning",
    "Second full-quality run. Attach the private artifact Dataset containing `phase1_improved_efficient_artifacts.zip`.",
)
top2 += [
    code(
        "from pathlib import Path\n"
        "matches = list(Path('/kaggle/input').rglob('phase1_improved_efficient_artifacts.zip'))\n"
        "checkpoints = list(Path('/kaggle/input').rglob('efficient.pt'))\n"
        "print('ZIP:', matches, 'unpacked checkpoint:', checkpoints)\n"
        "if len(matches) != 1 and len(checkpoints) != 1:\n"
        "    raise RuntimeError('Attach one efficient artifact dataset (ZIP or Kaggle-unpacked files)')\n"
    ),
    code(
        "!python /kaggle/working/phase2_lner.py "
        "--model efficient --method full --epochs 2 --batch-size 1 --grad-accum 8 "
        "--learning-rate 0.00002 --max-length 512 --stride 128 "
        "--output-dir /kaggle/working/phase2_efficient_full\n"
    ),
    markdown(
        "Download the whole output directory immediately. `best.pt` is also the teacher for the fast distillation run. "
        "Create a private Kaggle Dataset containing only this `best.pt` before starting Fast Run 4.\n"
    ),
    code(
        "from pathlib import Path\n"
        "root = Path('/kaggle/working/phase2_efficient_full')\n"
        "for name in ('validation_metrics.json', 'manifest.json', 'best.pt'):\n"
        "    path = root / name\n"
        "    print(name, path.stat().st_size if path.exists() else 'MISSING')\n"
        "print('CODABENCH:', list(root.glob('submission_*.zip')))\n"
    ),
]
save("CS781_Phase2_TOP2_Efficient.ipynb", top2)


fast = common(
    "CS781 Phase 2 FAST lane - Hybrid LoRMA then decoder distillation",
    "This lane produces the two compliance submissions sequentially. Attach the hybrid and improved-decoder artifact ZIPs. Run Hybrid first while the two top runs train.",
)
fast += [
    code(
        "from pathlib import Path\n"
        "for archive_name, checkpoint_name in (\n"
        "    ('phase1_hybrid_baseline_artifacts.zip', 'hybrid.pt'),\n"
        "    ('phase1_improved_decoder_artifacts.zip', 'decoder.pt'),\n"
        "):\n"
        "    archives = list(Path('/kaggle/input').rglob(archive_name))\n"
        "    checkpoints = list(Path('/kaggle/input').rglob(checkpoint_name))\n"
        "    print(archive_name, 'ZIP:', archives, 'unpacked:', checkpoints)\n"
        "    if len(archives) != 1 and len(checkpoints) != 1:\n"
        "        raise RuntimeError(f'Attach one dataset containing {archive_name} or {checkpoint_name}')\n"
    ),
    markdown("## Fast Run 3 - Hybrid with compulsory LoRMA+\n\nRun now. Download and submit its ZIP as soon as it finishes.\n"),
    code(
        "!python /kaggle/working/phase2_lner.py "
        "--model hybrid --method lorma --rank 8 --epochs 1 --batch-size 2 --grad-accum 4 "
        "--learning-rate 0.0001 --max-length 256 --stride 64 "
        "--output-dir /kaggle/working/phase2_hybrid_lorma\n"
    ),
    code(
        "from pathlib import Path\n"
        "print('SUBMIT NOW:', list(Path('/kaggle/working/phase2_hybrid_lorma').glob('submission_*.zip')))\n"
    ),
    markdown(
        "## Fast Run 4 - Decoder knowledge distillation\n\n"
        "Wait for TOP 2. Download `phase2_efficient_full/best.pt`, upload it as a private Kaggle Dataset, attach it here, and only then run this cell.\n"
    ),
    code(
        "from pathlib import Path\n"
        "import subprocess, sys\n"
        "teachers = list(Path('/kaggle/input').rglob('best.pt'))\n"
        "print('teacher candidates:', teachers)\n"
        "if len(teachers) != 1:\n"
        "    raise RuntimeError('Attach exactly one best.pt: the TOP 2 efficient teacher')\n"
        "command = [\n"
        "    sys.executable, '/kaggle/working/phase2_lner.py',\n"
        "    '--model', 'decoder', '--method', 'distill',\n"
        "    '--teacher-checkpoint', str(teachers[0]),\n"
        "    '--student-layers', '8', '--temperature', '2',\n"
        "    '--epochs', '1', '--batch-size', '1', '--grad-accum', '8',\n"
        "    '--learning-rate', '0.00005', '--max-length', '256', '--stride', '64',\n"
        "    '--output-dir', '/kaggle/working/phase2_decoder_distill',\n"
        "]\n"
        "subprocess.run(command, check=True)\n"
    ),
    code(
        "from pathlib import Path\n"
        "for directory in ('phase2_hybrid_lorma', 'phase2_decoder_distill'):\n"
        "    root = Path('/kaggle/working') / directory\n"
        "    print(directory, list(root.glob('submission_*.zip')))\n"
    ),
]
save("CS781_Phase2_FAST_Compliance.ipynb", fast)
