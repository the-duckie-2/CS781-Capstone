"""Tiny fake Phase 1 artifacts + fake LSI data; runs every Phase 5 recipe used in the notebooks on CPU."""
import json, random, subprocess, sys, zipfile
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import RunConfig, build_model

random.seed(0)
root = Path("/tmp/p5smoke"); root.mkdir(exist_ok=True)
inputs = root / "inputs"; inputs.mkdir(exist_ok=True)
data = root / "data"; data.mkdir(exist_ok=True)
words = "court appeal petitioner respondent statute section act held ratio facts issue precedent murder theft".split()
sent = lambda k: " ".join(random.choices(words, k=k))
tok = Tokenizer(models.BPE(unk_token="<unk>")); tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator([sent(20) for _ in range(500)], trainers.BpeTrainer(vocab_size=200, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]))
tok.save(str(root / "phase1_bpe.json"))
cfg = RunConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_heads=4)
names = {"decoder": "phase1_improved_decoder_artifacts.zip", "encoder_decoder": "phase1_ed_baseline_artifacts.zip",
         "hybrid": "phase1_hybrid_baseline_artifacts.zip", "efficient": "phase1_improved_efficient_artifacts.zip"}
for name, zname in names.items():
    m = build_model(name, cfg)
    torch.save({"model_name": name, "config": asdict(cfg), "model": m.state_dict(), "total_steps": 1}, root / f"{name}.pt")
    with zipfile.ZipFile(inputs / zname, "w") as z:
        z.write(root / f"{name}.pt", f"{name}.pt"); z.write(root / "phase1_bpe.json", "phase1_bpe.json")
STATUTES = [3, 7, 11, 23, 42, 81, 99]
def cases(n, labelled, prefix):
    out = []
    for i in range(n):
        d = {"id": f"{prefix}_{i}", "text": sent(random.choice([30, 200, 900, 2500]))}
        if labelled: d["labels"] = random.sample(STATUTES[:-1] if prefix == "tr" else STATUTES, k=random.randint(1, 3))
        out.append(d)
    return out
train, dev, test = cases(40, True, "tr"), cases(16, True, "dv"), cases(10, False, "te")
train.append({"id": "leak", "text": test[0]["text"], "labels": [3]})  # must be removed by the leakage check
json.dump(train, open(data / "train.json", "w")); json.dump(dev, open(data / "dev.json", "w")); json.dump(test, open(data / "test.json", "w"))

def run(*a):
    cmd = [sys.executable, str(Path(__file__).parent / "phase5_lsi.py"), "--data-dir", str(data), "--epochs", "2",
           "--batch-docs", "4", "--grad-accum", "1", "--chunk-len", "32", "--max-chunks", "4", "--overlap", "8",
           "--time-budget-min", "2", "--eval-docs", "8", *a]
    print(">>", " ".join(a), flush=True); r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-4000:], r.stderr[-4000:]); raise SystemExit(1)
    print("\n".join(r.stdout.strip().splitlines()[-14:]))
    if "LEAKAGE CHECK" in r.stdout: print([l for l in r.stdout.splitlines() if "LEAKAGE" in l][0])
z = lambda n: ["--artifact-zip", str(inputs / names[n])]
out = lambda n: ["--output-dir", str(root / "out" / n), "--run-name", n]
run("--model", "efficient", "--method", "full", "--export-soft-labels", "--save-checkpoint", *z("efficient"), *out("A"))
run("--model", "decoder", "--method", "distill", "--student-layers", "4", "--teacher-soft-labels",
    str(root / "out/A/teacher_soft_labels.pt"), *z("decoder"), *out("D"))
run("--model", "hybrid", "--method", "lorma", "--learning-rate", "5e-4", *z("hybrid"), *out("B"))
run("--model", "encoder_decoder", "--method", "partial", "--unfreeze-layers", "1", *z("encoder_decoder"), *out("C"))
run("--model", "efficient", "--method", "lorma", "--selection", "headtail", "--pooling", "attention", *z("efficient"), *out("F"))
run("--model", "efficient", "--method", "lora", "--pooling", "chunkmax", "--loss", "asl", *z("efficient"), *out("G"))
run("--model", "efficient", "--method", "ia3", "--pooling", "max", "--loss", "focal", "--selection", "head", *z("efficient"), *out("H"))
p = json.load(open(root / "out/A/prediction.json"))
assert len(p) == 10 and all(set(x["pred"]) <= set(STATUTES) for x in p), p
print(json.dumps(p[:3]))
v = json.load(open(root / "out/A/validation_metrics.json"))
print({k: v[k] for k in ("chosen_strategy", "frequency_buckets", "by_gold_cardinality")})
print("ALL OK")
