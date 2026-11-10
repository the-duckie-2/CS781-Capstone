"""Build tiny fake Phase 1 artifacts + fake RR data and run every Phase 3 recipe on CPU."""
import json, random, subprocess, sys, zipfile
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import RunConfig, build_model
from phase3_rr import OFFICIAL_LABELS

root = Path("/tmp/p3smoke"); root.mkdir(exist_ok=True)
inputs = root / "inputs"; inputs.mkdir(exist_ok=True)
data = root / "data"; data.mkdir(exist_ok=True)
words = "court appeal petitioner respondent statute section act held ratio facts issue precedent".split()
sent = lambda: " ".join(random.choices(words, k=random.randint(3, 20))) + "."
tok = Tokenizer(models.BPE(unk_token="<unk>")); tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator([sent() for _ in range(500)], trainers.BpeTrainer(vocab_size=200, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]))
tok.save(str(root / "phase1_bpe.json"))
cfg = RunConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_heads=4)
names = {"decoder": "phase1_improved_decoder_artifacts.zip", "encoder_decoder": "phase1_ed_baseline_artifacts.zip",
         "hybrid": "phase1_hybrid_baseline_artifacts.zip", "efficient": "phase1_improved_efficient_artifacts.zip"}
for name, zname in names.items():
    m = build_model(name, cfg)
    torch.save({"model_name": name, "config": asdict(cfg), "model": m.state_dict(), "total_steps": 1}, root / f"{name}.pt")
    with zipfile.ZipFile(inputs / zname, "w") as z:
        z.write(root / f"{name}.pt", f"{name}.pt"); z.write(root / "phase1_bpe.json", "phase1_bpe.json")
def docs(n, labelled, prefix):
    out = []
    for i in range(n):
        k = random.randint(3, 15)
        d = {"id": f"{prefix}_{i}", "text": [sent() for _ in range(k)]}
        if labelled: d["labels"] = random.choices(OFFICIAL_LABELS, k=k)
        out.append(d)
    return out
for dom in ("CL", "IT"):
    json.dump(docs(6, True, dom), open(data / f"{dom}_train.json", "w"))
    json.dump({d["id"]: {"text": d["text"], "labels": d["labels"]} for d in docs(3, True, dom + "d")}, open(data / f"{dom}_dev.json", "w"))
    json.dump(docs(3, False, dom + "t"), open(data / f"{dom}_test.json", "w"))
json.dump(OFFICIAL_LABELS, open(data / "label_vocab.json", "w"))

def run(*a):
    cmd = [sys.executable, str(Path(__file__).parent / "phase3_rr.py"), "--data-dir", str(data), "--epochs", "1",
           "--batch-size", "4", "--evals-per-epoch", "2", *a]
    print(">>", " ".join(a)); r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(1)
    print(r.stdout.strip().splitlines()[-9:])
z = lambda n: ["--artifact-zip", str(inputs / names[n])]
out = lambda n: ["--output-dir", str(root / "out" / n)]
run("--model", "efficient", "--method", "full", "--export-soft-labels", "--save-checkpoint", *z("efficient"), *out("A"))
run("--model", "hybrid", "--method", "lorma", *z("hybrid"), *out("B"))
run("--model", "encoder_decoder", "--method", "lora", *z("encoder_decoder"), *out("C"))
run("--model", "decoder", "--method", "distill", "--student-layers", "4", "--teacher-soft-labels",
    str(root / "out/A/teacher_soft_labels.pt"), *z("decoder"), *out("D"))
run("--model", "efficient", "--method", "ia3", "--context", "none", *z("efficient"), *out("E"))
run("--model", "efficient", "--method", "lorma", *z("efficient"), *out("F"))
print("ALL OK")
