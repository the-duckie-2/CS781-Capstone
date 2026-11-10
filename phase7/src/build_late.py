"""NB1c (account 1, 2 new maxsim members with 12 windows) + FUSE_ALL (CPU notebook: fuse every run's dense scores)."""
import json
from pathlib import Path
import build_phase7_notebooks as b

W12 = {"max_windows": 12, "train_q_windows": 6, "train_c_windows": 4}
E2 = {"model": "efficient", "method": "full", "pooling": "maxsim", "learning_rate": "3e-5", "seed": 11,
      "run_name": "p7_E2_efficient_full_maxsim_w12", **b.COMMON, **W12, "batch_queries": 6, "bagged_fusion": True}
E3 = {"model": "efficient", "method": "full", "pooling": "maxsim", "learning_rate": "4e-5", "seed": 23,
      "run_name": "p7_E3_efficient_full_maxsim_w12_hn5", **b.COMMON, **W12, "batch_queries": 4, "hard_negatives": 5,
      "bagged_fusion": True}


def book(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


def sources():
    return [b.code(f"%%writefile /kaggle/working/{n}\n" + (b.src / n).read_text(encoding="utf-8")) for n in b.SOURCES]


prep12 = b.PREP.replace("--model efficient", "--model efficient --max-windows 12")
run_cell = b.parallel_cell([(E2, 75)], [(E3, 75)], "")
nb1c = [b.md("# CS781 Phase 7 (PCR) - Notebook 1c (account 1): two extra late-interaction members\n\n"
             "Run E (efficient, full FT, **maxsim**) was the best single run, so two more maxsim members with **12 windows "
             "per document** (more of each long judgment covered; 6 query / 4 candidate windows sampled per step):\n"
             "* **GPU 0 - E2:** seed 11, 3 hard negatives.\n* **GPU 1 - E3:** seed 23, 5 hard negatives, lr 4e-5.\n"
             "Both use the bagged LambdaMART fusion option. Their `dense_scores.npz` go into the FUSE_ALL notebook.\n\n"
             "**Setup:** GPU T4 x2, Internet on, secret `HF`, Phase 1 dataset with `phase1_improved_efficient_artifacts.zip`. "
             "**Save Version -> Save & Run All (Commit)** (~1 h 45 min).\n\n" + b.DATA_NOTE),
        b.code(b.SETUP)] + sources() + [
        b.md("## 1. Data + lexical stage + 12-window Phase 1 windows\n"), b.code(prep12),
        b.md("## 2. E2 and E3 in parallel\n"), run_cell,
        b.md("## 3. Ensemble E2 + E3\n"),
        b.code("!cd /kaggle/working && python phase7_pcr.py --fuse-runs /kaggle/working/p7_E2_efficient_full_maxsim_w12 "
               "/kaggle/working/p7_E3_efficient_full_maxsim_w12_hn5 --bagged-fusion --output-dir /kaggle/working/p7_ENS_NB1c "
               "--run-name p7_ENS_NB1c 2>&1 | tee p7_ENS_NB1c.log | tail -n 30\n"),
        b.md("## 4. Results\n"), b.code(b.COLLECT),
        b.code("!cd /kaggle/working && zip -qr phase7_nb1c_outputs.zip p7_E* *.log -x '*/phase1/*' '*/best_fp16.pt' "
               "&& ls -la phase7_nb1c_outputs.zip SUBMIT\n")]
Path("CS781_Phase7_NB1c_Extra_Maxsim_W12.ipynb").write_text(json.dumps(book(nb1c), ensure_ascii=False, indent=1), encoding="utf-8")

fuse = [b.md("# CS781 Phase 7 (PCR) - FUSE_ALL: one fusion ranker over every run (CPU only)\n\n"
             "Fuses the dense scores of all finished runs (A, D, E, B, F, C, G, E2, E3, ...) with the lexical features. "
             "All runs share the same deterministic lexical candidate lists (checked array by array). The fusion method "
             "(LambdaMART, bagged LambdaMART, logistic regression) is chosen by 5-fold query-grouped cross-fitting on dev.\n\n"
             "**Setup:** Accelerator **None** (CPU is enough), Internet on, secret `HF`, attach the Phase 1 dataset "
             "(efficient ZIP, for the tokenizer) **and** your dataset of `*dense_scores*.npz` files. Run this **interactively**: "
             "run cells 1-2 now (data prep, ~15 min), add / update the npz dataset when the last runs finish, then run step 3.\n"),
        b.code(b.SETUP.replace("!nvidia-smi -L\n", ""))] + sources() + [
        b.md("## 1. Data + lexical stage (same cache as the runs)\n"), b.code(b.PREP),
        b.md("## 2. Which score files are attached?\nRe-run this after adding / updating the npz dataset "
             "(right panel -> the dataset -> check for updates).\n"),
        b.code("import os\nfor f, _, ns in os.walk('/kaggle/input', followlinks=True):\n    for n in ns:\n"
               "        if n.endswith('.npz'):\n            print(os.path.join(f, n))\n"),
        b.md("## 3. Fuse everything\n"),
        b.code("!cd /kaggle/working && python phase7_pcr.py --fuse-runs /kaggle/input --bagged-fusion "
               "--output-dir /kaggle/working/p7_ENS_ALL --run-name p7_ENS_ALL 2>&1 | tee p7_ENS_ALL.log | "
               "grep -vE '^\\s+\"(precision|recall)' | tail -n 60\n"),
        b.code("import json\nm = json.load(open('/kaggle/working/p7_ENS_ALL/manifest.json'))\n"
               "print('members:', *m['members'], sep='\\n  ')\nprint('chosen:', m['fusion_chosen'], ' dev F1@7 (crossfit):', "
               "m['dev_micro_f1@7_crossfit'])\nfrom IPython.display import FileLink, display\n"
               "display(FileLink('p7_ENS_ALL/submission_p7_ENS_ALL.zip'))\n")]
Path("CS781_Phase7_FUSE_ALL.ipynb").write_text(json.dumps(book(fuse), ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")
