"""CS781 Phase 5 - task-adaptive continued pretraining (TAPT) of a Phase 1 model on the LSI case texts.

Starts from the Phase 1 checkpoint and continues its own next-token objective on the TEXT of the official Phase 5
train + dev cases (no labels, no test text, no outside data).  Train/dev cases that duplicate a test case are
removed first (same leakage check as phase5_lsi.py).  The adapted weights are written as a Phase 1-style artifact
ZIP (<model>.pt + phase1_bpe.json) that phase5_lsi.py loads with --artifact-zip.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from phase2_lner import load_phase1
from phase3_rr import locate_phase1_files, seed_everything
from phase4_cjpe import leakage_report
from phase5_lsi import load_cases, obtain_data


def main(args):
    t0 = time.time()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = obtain_data(out / "data", args.data_dir)
    train = load_cases(files["train.json"], True)
    dev = load_cases(files["dev.json"], True)
    test = load_cases(files["test.json"], False)
    leak = leakage_report(train, dev, test)
    texts = [" ".join(c["text"].split()[:args.max_words]) for c in train + dev]

    checkpoint, tokenizer_path, artifact = locate_phase1_files(args.model, args.artifact_zip, out / "phase1")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos, bos = tokenizer.token_to_id("<eos>"), tokenizer.token_to_id("<bos>")
    pieces = []
    for s in range(0, len(texts), 1000):
        for enc in tokenizer.encode_batch(texts[s:s + 1000]):
            pieces.append(np.asarray(enc.ids + [eos], dtype=np.int32))
    stream = np.concatenate(pieces)
    del pieces
    block = args.block + 1
    n_blocks = len(stream) // block
    blocks = torch.from_numpy(stream[: n_blocks * block].astype(np.int64)).view(n_blocks, block)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_blocks, generator=g)
    n_val = max(64, n_blocks // 200)
    val, tr = blocks[perm[:n_val]], blocks[perm[n_val:]]
    print(f"TAPT corpus: {len(texts)} cases, {len(stream) / 1e6:.1f}M tokens, {len(tr)} train blocks, {n_val} val blocks",
          flush=True)

    model, cfg, saved = load_phase1(checkpoint, args.model)
    model.to(device)

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for s in range(0, min(len(val), 512), args.batch_size):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                losses.append(float(model.loss(val[s:s + args.batch_size].to(device), bos)))
        model.train()
        return sum(losses) / len(losses)

    before = val_loss()
    print(f"held-out LM loss before TAPT {before:.4f} (ppl {math.exp(before):.1f})", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    budget = args.time_budget_min * 60
    total = max(100, args.epochs * len(tr) // (args.batch_size * args.grad_accum))
    warmup = 100
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup,
                                              0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    start, step, opt_steps, history = time.time(), 0, 0, []
    order = torch.randperm(len(tr), generator=g)
    pos = 0
    best = before
    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    while opt_steps < total and time.time() - start < budget:
        for _ in range(args.grad_accum):
            if pos + args.batch_size > len(order):
                order, pos = torch.randperm(len(tr), generator=g), 0
            batch = tr[order[pos:pos + args.batch_size]].to(device)
            pos += args.batch_size
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model.loss(batch, bos) / args.grad_accum
            scaler.scale(loss).backward()
            step += 1
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        sched.step()
        opt_steps += 1
        if opt_steps == 50:
            rate = opt_steps / (time.time() - start)
            total = max(100, min(total, int(rate * budget * 0.95)))
            print(f"~{rate:.2f} steps/s -> cosine schedule over {total} steps "
                  f"({total * args.batch_size * args.grad_accum / len(tr):.2f} passes over the corpus)", flush=True)
        if opt_steps % args.eval_every == 0 or opt_steps == total:
            v = val_loss()
            history.append({"step": opt_steps, "val_loss": v, "minutes": round((time.time() - start) / 60, 1)})
            print(json.dumps(history[-1]), flush=True)
            if v < best:
                best = v
                best_state = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
    v = val_loss()
    if v < best:
        best, best_state = v, {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
    print(f"held-out LM loss after TAPT {best:.4f} (ppl {math.exp(best):.1f}); before {before:.4f}", flush=True)

    pt = out / f"{args.model}.pt"
    torch.save({**{k: v for k, v in saved.items() if k not in ("model", "optimizer")}, "model_name": args.model,
                "config": saved["config"], "model": best_state, "tapt": {"from": artifact, "steps": opt_steps}}, pt)
    archive = out / f"phase1_{args.model}_tapt_artifacts.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(pt, f"{args.model}.pt")
        bundle.write(tokenizer_path, "phase1_bpe.json")
    pt.unlink()
    (out / "tapt_manifest.json").write_text(json.dumps({
        "from_artifact": artifact, "leakage_check": leak, "cases": len(texts), "tokens": int(len(stream)),
        "val_loss_before": before, "val_loss_after": best, "opt_steps": opt_steps, "history": history,
        "minutes": (time.time() - t0) / 60, "config": vars(args)}, indent=2), encoding="utf-8")
    print("TAPT ARTIFACT:", archive, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficient", choices=("decoder", "efficient"))
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-words", type=int, default=12000)
    p.add_argument("--time-budget-min", type=float, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--artifact-zip")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir", default="/kaggle/working/p5_tapt")
    p.add_argument("--seed", type=int, default=230338)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
