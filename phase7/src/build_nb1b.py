"""NB1b: account 1 re-run WITHOUT Run A. Uses A's teacher_scores.pt (+ dense_scores.npz for the ensemble)
from an attached Kaggle dataset made of the downloaded p7_A_efficient_full folder."""
import json
from pathlib import Path
import build_phase7_notebooks as b

D = {**b.D, "teacher_scores": "$TEACHER"}
E = b.E
setup_extra = (
    "import glob\n"
    "t = glob.glob('/kaggle/input/**/teacher_scores.pt', recursive=True)\n"
    "a = glob.glob('/kaggle/input/**/dense_scores.npz', recursive=True)\n"
    "print('teacher_scores.pt:', t)\nprint('A dense_scores.npz (for the ensemble):', a)\n"
    "assert t, 'teacher_scores.pt NOT FOUND: Add Input -> the dataset with your downloaded p7_A_efficient_full files'\n"
    "open('/kaggle/working/TEACHER_PATH', 'w').write(t[0])\n"
    "open('/kaggle/working/A_DIR', 'w').write(str(Path(a[0]).parent) if a else '')\n")
run_cell = b.code(
    "%%bash\nset +e\ncd /kaggle/working\nTEACHER=$(cat TEACHER_PATH); echo \"teacher: $TEACHER\"\n"
    "NGPU=$(nvidia-smi -L | wc -l); echo \"GPUs: $NGPU\"\n"
    "if [ \"$NGPU\" -ge 2 ]; then\n"
    f"  ( CUDA_VISIBLE_DEVICES=0 {b.cmd(D, 45, 'p7_D_decoder8_kd.log')} ) &\n  sleep 60\n"
    f"  ( CUDA_VISIBLE_DEVICES=1 {b.cmd(E, 75, 'p7_E_efficient_full_maxsim.log')} ) &\n  wait\n"
    "else\n"
    f"  CUDA_VISIBLE_DEVICES=0 {b.cmd(D, 40, 'p7_D_decoder8_kd.log')} ; CUDA_VISIBLE_DEVICES=0 {b.cmd(E, 45, 'p7_E_efficient_full_maxsim.log')}\n"
    "fi\n"
    "for f in p7_D_decoder8_kd.log p7_E_efficient_full_maxsim.log; do echo \"===== $f =====\"; "
    "grep -E 'LEAKAGE|steps/s|KD:|best checkpoint|chosen|VALIDATED|SUBMISSION|Error|error|OOM' $f | tail -n 20; tail -n 4 $f; done\nexit 0\n")
ens_cell = b.code(
    "%%bash\ncd /kaggle/working\nA=$(cat A_DIR)\n"
    "python phase7_pcr.py --fuse-runs $A /kaggle/working/p7_D_decoder8_kd /kaggle/working/p7_E_efficient_full_maxsim "
    "--output-dir /kaggle/working/p7_ENS_NB1 --run-name p7_ENS_NB1 2>&1 | tee p7_ENS_NB1.log | tail -n 30\n")
cells = [b.md("# CS781 Phase 7 (PCR) - Notebook 1b (Kaggle account 1, Run A already done)\n\n"
              "* **GPU 0 - Run D:** Phase 1 **decoder cut from 16 to 8 layers**, **knowledge distillation** from Run A "
              "(teacher scores = `teacher_scores.pt` from the earlier A run, attached as a dataset): 0.5 InfoNCE + 0.5 "
              "T^2 KL(teacher || student), T = 2.\n"
              "* **GPU 1 - Run E (extra):** efficient, full fine-tuning, maxsim.\n"
              "* **Ensemble A + D + E** (A's `dense_scores.npz` from the same dataset, if attached).\n\n"
              "**Setup:** GPU T4 x2, Internet on, secret `HF`, attach the Phase 1 dataset (efficient + decoder ZIPs) **and** "
              "a dataset with the downloaded `p7_A_efficient_full` files (`teacher_scores.pt`, `dense_scores.npz`). "
              "Then **Save Version -> Save & Run All (Commit)** (~1 h 45 min).\n\n" + b.DATA_NOTE),
         b.code(b.SETUP + "from pathlib import Path\n" + setup_extra)]
for name in b.SOURCES:
    cells.append(b.code(f"%%writefile /kaggle/working/{name}\n" + (b.src / name).read_text(encoding="utf-8")))
cells += [b.md("## 1. Data + lexical stage + Phase 1 windows (CPU)\n"), b.code(b.PREP),
          b.md("## 2. D (KD from A) and E in parallel\n"), run_cell,
          b.md("## 3. Ensemble A + D + E\n"), ens_cell,
          b.md("## 4. Results + submission ZIPs\n"), b.code(b.COLLECT),
          b.code("!cd /kaggle/working && zip -qr phase7_outputs.zip p7_* *.log -x '*/phase1/*' '*/best_fp16.pt' "
                 "'p7_cache/*' && ls -la phase7_outputs.zip SUBMIT\n")]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
Path("CS781_Phase7_NB1b_D_KD_and_E_without_A.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")
