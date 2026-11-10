"""Build the two deadline-focused Phase 2 replacement notebooks."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
sources = {
    "phase1_pipeline.py": (root / "phase1_pipeline.py").read_text(encoding="utf-8"),
    "phase1_foundation.py": (root / "phase1_foundation.py").read_text(encoding="utf-8"),
    "phase2_lner.py": (root / "phase2_lner.py").read_text(encoding="utf-8"),
}


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": value.splitlines(True)}


def markdown(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)}


def make(title, description, inventory, command, output_dir):
    cells = [
        markdown(f"# {title}\n\n{description}\n\nEnable GPU and Internet, attach the stated private dataset, then Run All.\n"),
        code(
            "!nvidia-smi\n"
            "!pip install -q tokenizers huggingface_hub\n"
            "import os\n"
            "from kaggle_secrets import UserSecretsClient\n"
            "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n"
            "print('HF token loaded')\n"
        ),
    ]
    cells.extend(code(f"%%writefile /kaggle/working/{name}\n{source}") for name, source in sources.items())
    cells.extend([
        code("!python /kaggle/working/phase2_lner.py --self-test\n"),
        code(inventory),
        code(command),
        code(
            "from pathlib import Path\n"
            "import json\n"
            f"root = Path('{output_dir}')\n"
            "print(json.loads((root / 'validation_metrics.json').read_text()))\n"
            "print('UPLOAD:', list(root.glob('submission_*.zip')))\n"
        ),
    ])
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


foundation = make(
    "CS781 Phase 2 optimized foundation",
    "Primary replacement run: bidirectional 12-layer legal encoder, differential learning rates, stronger class balancing, and validation-selected O-tag bias.",
    "from pathlib import Path\n"
    "for name in ('foundation.pt', 'legal_bpe_24k.json'):\n"
    "    matches = list(Path('/kaggle/input').rglob(name))\n"
    "    print(name, matches)\n"
    "    assert len(matches) == 1, f'Attach exactly one {name}'\n",
    "!python /kaggle/working/phase2_lner.py "
    "--model foundation --method full --epochs 8 --batch-size 1 --grad-accum 2 "
    "--learning-rate 0.00002 --max-length 512 --stride 128 "
    "--output-dir /kaggle/working/phase2_foundation_optimized\n",
    "/kaggle/working/phase2_foundation_optimized",
)

efficient = make(
    "CS781 Phase 2 optimized bidirectional efficient model",
    "Secondary replacement run: the Phase 1 efficient weights with its 64-token local attention changed from causal-left to bidirectional-local for NER.",
    "from pathlib import Path\n"
    "checkpoints = list(Path('/kaggle/input').rglob('efficient.pt'))\n"
    "print('efficient checkpoints:', checkpoints)\n"
    "assert len(checkpoints) == 1, 'Attach exactly one unpacked improved-efficient dataset'\n"
    "tokenizers = list(checkpoints[0].parent.rglob('phase1_bpe.json'))\n"
    "print('tokenizers:', tokenizers)\n"
    "assert len(tokenizers) == 1\n",
    "!python /kaggle/working/phase2_lner.py "
    "--model efficient --method full --bidirectional-efficient "
    "--epochs 8 --batch-size 1 --grad-accum 2 "
    "--learning-rate 0.00003 --max-length 512 --stride 128 "
    "--output-dir /kaggle/working/phase2_efficient_bidirectional\n",
    "/kaggle/working/phase2_efficient_bidirectional",
)

for name, notebook in (
    ("CS781_Phase2_OPTIMIZED_Foundation.ipynb", foundation),
    ("CS781_Phase2_OPTIMIZED_Efficient_Bidirectional.ipynb", efficient),
):
    path = root / name
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)
