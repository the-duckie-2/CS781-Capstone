"""Tiny fake Phase 1 artifacts + fake Phase 7 PCR data; runs every Phase 7 command used in the notebooks on CPU."""
import json, random, subprocess, sys, zipfile
from dataclasses import asdict
from pathlib import Path
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import RunConfig, build_model
random.seed(0)
root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p7smoke"); root.mkdir(parents=True, exist_ok=True)
inputs = root / "inputs"; inputs.mkdir(exist_ok=True)
data = root / "data"; data.mkdir(exist_ok=True)
base = "the court held that appellant was guilty under the act and appeal is dismissed with costs petition".split()
topics = [[f"topic{t}word{i}" for i in range(8)] for t in range(12)]
def sent(t=None, cite=False):
    w = random.choices(base, k=random.randint(6, 14)) + (random.choices(topics[t], k=3) if t is not None else [])
    random.shuffle(w)
    s = " ".join(w) + (f" in {random.randint(1950, 2020)}" if random.random() < .2 else "") + (" section %d" % random.randint(1, 500) if random.random() < .3 else "")
    return s + (" <CITATION>" if cite else "") + "."
tok = Tokenizer(models.BPE(unk_token="<unk>")); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
tok.train_from_iterator([sent(random.randrange(12)) for _ in range(1500)], trainers.BpeTrainer(vocab_size=400, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]))
tok.save(str(root / "phase1_bpe.json"))
cfg = RunConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_heads=4)
names = {"decoder": "phase1_improved_decoder_artifacts.zip", "efficient": "phase1_improved_efficient_artifacts.zip"}
for name, zname in names.items():
    torch.save({"model_name": name, "config": asdict(cfg), "model": build_model(name, cfg).state_dict(), "total_steps": 1}, root / f"{name}.pt")
    with zipfile.ZipFile(inputs / zname, "w") as z:
        z.write(root / f"{name}.pt", f"{name}.pt"); z.write(root / "phase1_bpe.json", "phase1_bpe.json")
uid = iter(range(1000, 99999))
def split(name, nq, nc, labels=True, as_dict=False):
    cands = []
    for _ in range(nc):
        t = random.randrange(12)
        cands.append({"id": f"{next(uid):010d}", "t": t, "text": [sent(t if random.random() < .5 else None) for _ in range(random.choice([8, 40, 120]))]})
    qs = []
    for _ in range(nq):
        t = random.randrange(12)
        rel = [c["id"] for c in cands if c["t"] == t][:random.randint(1, 9)]
        q = {"id": f"{next(uid):010d}", "text": [sent(t if random.random() < .4 else None, cite=random.random() < .15) for _ in range(random.choice([20, 80]))]}
        if labels: q["relevant_candidates"] = rel
        qs.append(q)
    d = data / name.capitalize(); d.mkdir(exist_ok=True)
    json.dump(qs, open(d / f"{name}.json", "w"))
    cj = {c["id"]: c["text"] for c in cands} if as_dict else [{"id": c["id"], "text": c["text"]} for c in cands]
    json.dump(cj, open(d / f"{name}_candidates.json", "w"))
split("train", 60, 150); split("dev", 30, 100); split("test", 20, 90, labels=False, as_dict=True)
cache = root / "cache"
def run(*a):
    cmd = [sys.executable, str(Path(__file__).parent / "phase7_pcr.py"), "--data-dir", str(data), "--cache-dir", str(cache),
           "--time-budget-min", "0.4", "--eval-every-min", "0.2", "--chunk-len", "32", "--max-windows", "4", "--top-n", "30", "--batch-queries", "8",
           "--hn-pool", "20", "--cpu-jobs", "2", *a]
    print(">>", " ".join(a), flush=True); r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-6000:], r.stderr[-6000:]); raise SystemExit(1)
    lines = r.stdout.strip().splitlines()
    print("\n".join([l for l in lines if l.startswith(("LEAKAGE", "DATA", "-> chosen", "KD", "~", "VALID", "best"))] + lines[-6:]))
z = lambda n: ["--artifact-zip", str(inputs / names[n])]
out = lambda n: ["--output-dir", str(root / "out" / n), "--run-name", n]
run("--prepare-only", *z("efficient"), *out("prep"))
run("--model", "efficient", "--method", "full", "--grad-ckpt", "--export-teacher", "--save-checkpoint", *z("efficient"), *out("A"))
run("--model", "decoder", "--method", "distill", "--student-layers", "4", "--teacher-scores", str(root / "out/A/teacher_scores.pt"), *z("decoder"), *out("D"))
run("--model", "efficient", "--method", "lorma", "--rank", "32", "--learning-rate", "5e-4", "--grad-ckpt", *z("efficient"), *out("B"))
run("--model", "efficient", "--method", "partial", "--pooling", "maxsim", "--unfreeze-layers", "6", "--grad-ckpt", *z("efficient"), *out("C"))
run("--model", "efficient", "--method", "ia3", "--learning-rate", "3e-3", "--grad-ckpt", *z("efficient"), *out("F"))
run("--model", "efficient", "--method", "full", "--pooling", "maxsim", "--seed", "7", "--grad-ckpt", *z("efficient"), *out("E"))
run("--model", "decoder", "--method", "full", "--learning-rate", "5e-5", *z("decoder"), *out("G"))
run("--fuse-runs", str(root / "out/A"), str(root / "out/B"), str(root / "out/C"), "--output-dir", str(root / "out/ENS"), "--run-name", "ENS")
p = json.load(open(root / "out/A/prediction.json"))
test_ids = {x["id"] for x in json.load(open(data / "Test/test.json"))}; pool = set(json.load(open(data / "Test/test_candidates.json")))
assert set(p) == test_ids and all(len(v) == 7 == len(set(v)) and set(v) <= pool for v in p.values())
print(json.dumps(dict(list(p.items())[:1]))); print("ALL OK")
