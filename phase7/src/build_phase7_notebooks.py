"""Build the 3 Kaggle notebooks for CS781 Phase 7 (Prior Case Retrieval), one per Kaggle account.
Each notebook: shared lexical/window preparation -> two runs in parallel (one per T4 GPU) -> ensemble of its runs."""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
src = root / "src"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root
SOURCES = ["phase1_pipeline.py", "phase1_foundation.py", "phase2_lner.py", "phase3_rr.py", "phase4_cjpe.py", "phase7_pcr.py"]


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def flags(args):
    return " ".join(f"--{k.replace('_', '-')}" if v is True else f"--{k.replace('_', '-')} {v}"
                    for k, v in args.items() if v is not False)


SETUP = ("!nvidia-smi -L\n!pip install -q tokenizers huggingface_hub lightgbm\nimport os\n"
         "from kaggle_secrets import UserSecretsClient\n"
         "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF')\n"
         "print('HF token loaded; value intentionally hidden')\n"
         "found = [os.path.join(f, n) for f, _, ns in os.walk('/kaggle/input', followlinks=True) for n in ns "
         "if n.endswith(('.zip', '.pt'))]\nprint('\\n'.join(found))\n"
         "assert found, 'NOTHING attached under /kaggle/input - Add Input -> your Phase 1 artifact dataset'\n")

PREP = ("!cd /kaggle/working && python phase7_pcr.py --prepare-only --model efficient --output-dir /kaggle/working/p7_prep "
        "2>&1 | tee /kaggle/working/p7_prepare.log | grep -v '^\\s*$' | tail -n 40\n")

DATA_NOTE = ("Data: only the official Phase 7 files from `Exploration-Lab/CS-781-Capstone` (re-downloaded with "
             "`force_download`). Each split's queries are ranked only against that split's candidate pool. Train/dev queries "
             "duplicating a test query (by id or normalised text) are removed (`LEAKAGE CHECK`); the query itself and exact "
             "duplicates of it are never returned. The dense retriever trains on train queries only; dev is used for "
             "checkpoint selection and to fit the fusion ranker, which is scored by 5-fold cross-fitting on dev (the honest "
             "number). No pretrained weights other than the Phase 1 checkpoints are used.\n")

COMMON = {"grad_ckpt": True, "batch_queries": 8, "hard_negatives": 3, "train_q_windows": 4, "train_c_windows": 3,
          "eval_every_min": 15, "save_checkpoint": True}


def cmd(args, budget, log):
    name = args["run_name"]
    return (f"python phase7_pcr.py {flags(args)} --time-budget-min {budget} --output-dir /kaggle/working/{name} "
            f"> {log} 2>&1")


def parallel_cell(gpu0, gpu1, title):
    """gpu0/gpu1: list of (args, budget_minutes) run one after the other on that GPU."""
    def chain(runs, dev, scale):
        parts = []
        for args, budget in runs:
            parts.append(f"CUDA_VISIBLE_DEVICES={dev} " + cmd(args, round(budget * scale), f"{args['run_name']}.log"))
        return " ; ".join(parts)
    names = [a["run_name"] for a, _ in gpu0 + gpu1]
    greps = " ".join(f"{n}.log" for n in names)
    return code(
        "%%bash\nset +e\ncd /kaggle/working\nNGPU=$(nvidia-smi -L | wc -l); echo \"GPUs: $NGPU\"\n"
        "if [ \"$NGPU\" -ge 2 ]; then\n"
        f"  ( {chain(gpu0, 0, 1.0)} ) &\n  sleep 60\n  ( {chain(gpu1, 1, 1.0)} ) &\n  wait\n"
        "else\n"
        "  echo 'Only one GPU: running everything one after the other at ~45% budget'\n"
        f"  {chain(gpu0, 0, 0.45)} ; {chain(gpu1, 0, 0.45)}\n"
        "fi\n"
        f"for f in {greps}; do echo \"===== $f =====\"; "
        "grep -E 'LEAKAGE|DATA \\[|steps/s|best checkpoint|chosen|VALIDATED|SUBMISSION|Error|error|OOM' $f | tail -n 20; "
        "tail -n 4 $f; done\nexit 0\n")


COLLECT = (
    "import json, shutil, glob\nfrom pathlib import Path\nfrom IPython.display import FileLink, display\n"
    "Path('/kaggle/working/SUBMIT').mkdir(exist_ok=True)\nrows = []\n"
    "for m in sorted(glob.glob('/kaggle/working/p7_*/manifest.json')):\n"
    "    m = json.load(open(m))\n"
    "    z = Path(m.get('submission', '/nonexistent'))\n"
    "    if z.exists():\n"
    "        shutil.copy(z, '/kaggle/working/SUBMIT/')\n"
    "    rows.append((m['run'], m.get('method'), m.get('fusion_chosen'), m.get('dev_micro_f1@7_crossfit'),\n"
    "                 m.get('dense_rerank_dev_f1@7_eval_subset'), m.get('trainable_parameters')))\n"
    "print(f\"{'run':28s} {'method':8s} {'fusion':18s} {'dev F1@7 (crossfit)':>20s} {'dense-only':>11s} {'trainable':>12s}\")\n"
    "for r in sorted(rows, key=lambda r: -(r[3] or 0)):\n"
    "    print(f\"{r[0]:28s} {str(r[1]):8s} {str(r[2]):18s} {r[3] or 0:20.4f} {r[4] or 0:11.4f} {r[5] or 0:12,}\")\n"
    "for z in sorted(glob.glob('/kaggle/working/SUBMIT/*.zip')):\n"
    "    display(FileLink(z.replace('/kaggle/working/', '')))\n"
    "missing = [l for l in glob.glob('/kaggle/working/p7_*.log') if 'SUBMISSION ZIP' not in open(l).read()]\n"
    "print('runs WITHOUT a submission (read their log):', missing or 'none')\n")


def notebook(title, intro, gpu0, gpu1, members, attach, ens_name):
    cells = [md(f"# {title}\n\n{intro}\n\n"
                f"**Setup:** Accelerator **GPU T4 x2**, Internet **on**, secret `HF` enabled, Phase 1 artifact dataset attached "
                f"with **{attach}**. Then **Save Version -> Save & Run All (Commit)** (~2 h 45 min); when it finishes open the "
                "version -> Output -> `SUBMIT/`.\n\n" + DATA_NOTE),
             code(SETUP)]
    for name in SOURCES:
        cells.append(code(f"%%writefile /kaggle/working/{name}\n" + (src / name).read_text(encoding="utf-8")))
    cells += [md("## 1. Data + lexical stage + Phase 1 windows (CPU, shared by both runs)\n"
                 "Look for `Phase 7 files on HF`, `LEAKAGE CHECK`, `DATA [train]/[dev]/[test]`, the BM25 grid and "
                 "`lexical dev/train metrics` (the lexical baseline).\n"),
              code(PREP),
              md("## 2. Two runs in parallel (one per GPU)\n"),
              parallel_cell(gpu0, gpu1, title),
              md("## 3. Ensemble of this notebook's runs (dense scores of all members in one fusion ranker)\n"),
              code(f"!cd /kaggle/working && python phase7_pcr.py --fuse-runs {' '.join('/kaggle/working/' + m for m in members)} "
                   f"--output-dir /kaggle/working/{ens_name} --run-name {ens_name} 2>&1 | tee {ens_name}.log | tail -n 30\n"),
              md("## 4. Results + submission ZIPs\nUpload order is in the README. `dev F1@7 (crossfit)` is the honest dev "
                 "estimate; `dense-only` is the Phase 1 retriever alone (rerank of the lexical top-N).\n"),
              code(COLLECT),
              code("!cd /kaggle/working && zip -qr phase7_outputs.zip p7_* *.log -x '*/phase1/*' '*/best_fp16.pt' "
                   "'p7_cache/*' && ls -la phase7_outputs.zip SUBMIT\n!ls -la /kaggle/working/p7_*/best_fp16.pt\n")]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}


A = {"model": "efficient", "method": "full", "learning_rate": "3e-5", "export_teacher": True, "run_name": "p7_A_efficient_full", **COMMON}
D = {"model": "decoder", "method": "distill", "student_layers": 8, "teacher_scores": "/kaggle/working/p7_A_efficient_full/teacher_scores.pt",
     "learning_rate": "1e-4", "kd_alpha": 0.5, "kd_temperature": 2.0, "run_name": "p7_D_decoder8_kd",
     **{**COMMON, "grad_ckpt": False}}
E = {"model": "efficient", "method": "full", "pooling": "maxsim", "learning_rate": "3e-5", "seed": 7,
     "run_name": "p7_E_efficient_full_maxsim", **COMMON}
B = {"model": "efficient", "method": "lorma", "rank": 32, "learning_rate": "5e-4", "run_name": "p7_B_efficient_lorma", **COMMON}
Fr = {"model": "efficient", "method": "ia3", "learning_rate": "3e-3", "run_name": "p7_F_efficient_ia3", **COMMON}
C = {"model": "efficient", "method": "partial", "unfreeze_layers": 6, "pooling": "maxsim", "learning_rate": "5e-5",
     "run_name": "p7_C_efficient_partial_maxsim", **COMMON}
G = {"model": "decoder", "method": "full", "learning_rate": "5e-5", "run_name": "p7_G_decoder_full",
     **{**COMMON, "grad_ckpt": False, "batch_queries": 4}}

books = {
    "CS781_Phase7_NB1_Full_Teacher_KD_Student.ipynb": notebook(
        "CS781 Phase 7 (PCR) - Notebook 1 (Kaggle account 1)",
        "* **GPU 0 - Run A:** Phase 1 **efficient** model, **full fine-tuning**, attention pooling over windows (KD "
        "teacher; exports its scores over every train query's candidate list). Then **Run D:** Phase 1 **decoder cut "
        "from 16 to 8 layers**, **knowledge distillation** = 0.5 InfoNCE + 0.5 T^2 KL(teacher || student) over each "
        "query's [positive + hard negatives] list, T = 2.\n"
        "* **GPU 1 - Run E (extra):** efficient, full fine-tuning with **late interaction (maxsim)**, different seed "
        "(pooling ablation vs A + ensemble member).\n"
        "* **Ensemble A + D + E.**",
        [(A, 85), (D, 45)], [(E, 135)], ["p7_A_efficient_full", "p7_D_decoder8_kd", "p7_E_efficient_full_maxsim"],
        "phase1_improved_efficient_artifacts.zip + phase1_improved_decoder_artifacts.zip", "p7_ENS_NB1"),
    "CS781_Phase7_NB2_LORMA_IA3.ipynb": notebook(
        "CS781 Phase 7 (PCR) - Notebook 2 (Kaggle account 2)",
        "* **GPU 0 - Run B:** efficient + **LORMA+** (rank 32 on q / v / FFN-down, backbone frozen) - the compulsory "
        "LORMA run.\n* **GPU 1 - Run F (extra):** efficient + **IA3** (learned rescaling of k / v / FFN-up).\n"
        "* **Ensemble B + F.**",
        [(B, 135)], [(Fr, 135)], ["p7_B_efficient_lorma", "p7_F_efficient_ia3"],
        "phase1_improved_efficient_artifacts.zip", "p7_ENS_NB2"),
    "CS781_Phase7_NB3_Partial_Maxsim_DecoderFull.ipynb": notebook(
        "CS781 Phase 7 (PCR) - Notebook 3 (Kaggle account 3)",
        "* **GPU 0 - Run C:** efficient, **partial fine-tuning** (top 6 of the blocks + final norm; the rest frozen) with "
        "late-interaction (maxsim) scoring.\n* **GPU 1 - Run G (extra):** Phase 1 **decoder** (16 layers), full "
        "fine-tuning (architecture transfer comparison).\n* **Ensemble C + G.**",
        [(C, 135)], [(G, 135)], ["p7_C_efficient_partial_maxsim", "p7_G_decoder_full"],
        "phase1_improved_efficient_artifacts.zip + phase1_improved_decoder_artifacts.zip", "p7_ENS_NB3"),
}

for name, book in books.items():
    (out / name).write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / name)
