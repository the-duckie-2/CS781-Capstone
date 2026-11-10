"""Build the exact-character foundation NER Kaggle notebook."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": value.splitlines(True)}


def markdown(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)}


cells = [
    markdown(
        "# CS781 Phase 2 — Foundation exact-character NER\n\n"
        "This keeps the Phase 1 127.6M foundation model and replaces the BPE-level BIO head with an exact-character CNN head. "
        "Attach the private dataset containing `foundation.pt` and `legal_bpe_24k.json`, enable GPU and Internet, and enable the Kaggle secret `HF`.\n"
    ),
    code(
        "!nvidia-smi\n"
        "!pip install -q tokenizers huggingface_hub\n"
        "import os\n"
        "from kaggle_secrets import UserSecretsClient\n"
        "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n"
        "print('HF token loaded')\n"
    ),
    code(
        "from pathlib import Path\n"
        "for name in ('foundation.pt', 'legal_bpe_24k.json'):\n"
        "    matches = list(Path('/kaggle/input').rglob(name))\n"
        "    print(name, matches)\n"
        "    assert len(matches) == 1, f'Attach exactly one {name}'\n"
    ),
]

for name in ("phase1_pipeline.py", "phase1_foundation.py", "phase2_char_ner.py"):
    source = (root / name).read_text(encoding="utf-8")
    cells.append(code(f"%%writefile /kaggle/working/{name}\n{source}"))

cells.extend([
    code(
        "!python -m py_compile /kaggle/working/phase1_pipeline.py /kaggle/working/phase1_foundation.py /kaggle/working/phase2_char_ner.py\n"
        "!python /kaggle/working/phase2_char_ner.py --self-test\n"
    ),
    code(
        "!python /kaggle/working/phase2_char_ner.py "
        "--epochs 10 --max-tokens 384 --stride 96 --grad-accum 2 "
        "--backbone-lr 0.00002 --head-lr 0.001 "
        "--output-dir /kaggle/working/phase2_foundation_character\n"
    ),
    code(
        "from pathlib import Path\n"
        "import json\n"
        "root = Path('/kaggle/working/phase2_foundation_character')\n"
        "print(json.dumps(json.loads((root / 'validation_metrics.json').read_text()), indent=2))\n"
        "print('UPLOAD:', root / 'submission_phase2_foundation_character.zip')\n"
    ),
])

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = root / "CS781_Phase2_Foundation_Exact_Character.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)

# Reuse the same verified source for the efficient Phase 1 backbone.
efficient = json.loads(json.dumps(notebook))
efficient["cells"][0] = markdown(
    "# CS781 Phase 2 — Efficient exact-character NER\n\n"
    "This preserves the improved Phase 1 efficient weights, switches their local attention to bidirectional for NER, and predicts exact character boundaries. "
    "Attach the private unpacked improved-efficient dataset containing `efficient.pt` and `phase1_bpe.json`, enable GPU and Internet, and enable the Kaggle secret `HF`.\n"
)
efficient["cells"][2] = code(
    "from pathlib import Path\n"
    "for name in ('efficient.pt', 'phase1_bpe.json'):\n"
    "    matches = list(Path('/kaggle/input').rglob(name))\n"
    "    print(name, matches)\n"
    "    assert len(matches) == 1, f'Attach exactly one {name}'\n"
)
efficient["cells"][-2] = code(
    "!python /kaggle/working/phase2_char_ner.py "
    "--model efficient --epochs 10 --max-tokens 384 --stride 96 --grad-accum 2 "
    "--backbone-lr 0.00003 --head-lr 0.001 "
    "--output-dir /kaggle/working/phase2_efficient_character\n"
)
efficient["cells"][-1] = code(
    "from pathlib import Path\n"
    "import json\n"
    "root = Path('/kaggle/working/phase2_efficient_character')\n"
    "print(json.dumps(json.loads((root / 'validation_metrics.json').read_text()), indent=2))\n"
    "print('UPLOAD:', root / 'submission_phase2_efficient_character.zip')\n"
)
path = root / "CS781_Phase2_Efficient_Exact_Character.ipynb"
path.write_text(json.dumps(efficient, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
