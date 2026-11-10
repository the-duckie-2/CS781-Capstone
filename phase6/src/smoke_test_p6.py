"""Tiny fake Phase 1 artifacts + fake Phase 6 data; runs every Phase 6 recipe used in the notebooks on CPU."""
import json, random, subprocess, sys, zipfile
from dataclasses import asdict
from pathlib import Path
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import RunConfig, build_model
random.seed(0)
root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p6smoke"); root.mkdir(parents=True, exist_ok=True)
inputs = root / "inputs"; inputs.mkdir(exist_ok=True)
data = root / "data"; data.mkdir(exist_ok=True)
words = "The court held that the appellant was guilty under section 302 of the Act and the appeal is dismissed with costs".split()
sent = lambda: " ".join(random.choices(words, k=random.randint(6, 18))) + "."
tok = Tokenizer(models.BPE(unk_token="<unk>")); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
tok.train_from_iterator([sent() for _ in range(800)], trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]))
tok.save(str(root / "phase1_bpe.json"))
cfg = RunConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_heads=4)
names = {"decoder": "phase1_improved_decoder_artifacts.zip", "encoder_decoder": "phase1_ed_baseline_artifacts.zip",
         "hybrid": "phase1_hybrid_baseline_artifacts.zip", "efficient": "phase1_improved_efficient_artifacts.zip"}
for name, zname in names.items():
    torch.save({"model_name": name, "config": asdict(cfg), "model": build_model(name, cfg).state_dict(), "total_steps": 1}, root / f"{name}.pt")
    with zipfile.ZipFile(inputs / zname, "w") as z:
        z.write(root / f"{name}.pt", f"{name}.pt"); z.write(root / "phase1_bpe.json", "phase1_bpe.json")
doc = lambda: " ".join(sent() for _ in range(random.choice([5, 40, 150])))
train = [{"id": f"d{i}", "document": doc(), "summary": " ".join(sent() for _ in range(random.randint(2, 6)))} for i in range(60)]
test = [{"id": f"t{i}", "document": doc()} for i in range(9)]
train.append({"id": "leak", "document": test[0]["document"], "summary": "x."})
json.dump(train, open(data / "train.json", "w")); json.dump(test, open(data / "test.json", "w"))
def run(*a):
    cmd = [sys.executable, str(Path(__file__).parent / "phase6_summ.py"), "--data-dir", str(data), "--epochs", "1",
           "--batch-docs", "4", "--grad-accum", "1", "--src-tokens", "96", "--fid-chunks", "3", "--chunk-len", "32",
           "--tgt-tokens", "48", "--time-budget-min", "0.5", "--val-docs", "10", "--val-gen-docs", "6", "--gen-batch", "4",
           "--length-grid", "20,40", "--bertscore", "0", "--tapt-batch", "4", *a]
    print(">>", " ".join(a), flush=True); r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-5000:], r.stderr[-5000:]); raise SystemExit(1)
    lines = r.stdout.strip().splitlines()
    print("\n".join([l for l in lines if l.startswith(("LEAKAGE", "TAPT held", "chosen", "KD"))] + lines[-12:]))
z = lambda n: ["--artifact-zip", str(inputs / names[n])]
out = lambda n: ["--output-dir", str(root / "out" / n), "--run-name", n]
run("--model", "efficient", "--method", "adapter", "--adapter-dim", "8", *z("efficient"), *out("A"))
run("--model", "decoder", "--method", "distill", "--student-layers", "4", "--teacher-checkpoint", str(root / "out/A/best_fp16.pt"),
    "--teacher-artifact-zip", str(inputs / names["efficient"]), *z("decoder"), *out("D"))
run("--model", "hybrid", "--method", "lorma", "--tapt-minutes", "0.2", "--tapt-lr", "2e-4", *z("hybrid"), *out("B"))
run("--model", "encoder_decoder", "--method", "lora", "--tapt-minutes", "0.2", *z("encoder_decoder"), *out("C"))
run("--model", "efficient", "--method", "full", "--global-every", "2", "--tapt-minutes", "0.2", "--selection", "lead", *z("efficient"), *out("M1"))
run("--model", "decoder", "--method", "full", "--tapt-minutes", "0.2", *z("decoder"), *out("M2"))
p = json.load(open(root / "out/A/prediction.json")); assert len(p) == 9 and all(set(x) == {"id", "summary"} and x["summary"] for x in p)
print(json.dumps(p[:2])[:400]); print("ALL OK")
