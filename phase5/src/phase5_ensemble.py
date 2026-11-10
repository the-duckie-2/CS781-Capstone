"""Average the calibrated per-statute probabilities of several phase5_lsi.py runs, re-choose thresholds on dev
(2-fold cross-fitted strategy choice, refit on full dev) and write the ensemble submission."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import torch

from phase5_lsi import apply_thresholds, choose_thresholds, ml_scores, public


def main(args):
    runs = [Path(p) for p in args.runs]
    loaded = []
    for r in runs:
        f = r / "logits.pt"
        if not f.exists():
            print(f"skipping {r}: no logits.pt (run missing or crashed)")
            continue
        loaded.append(torch.load(f, map_location="cpu"))
    if not loaded:
        raise SystemExit("No member logits found")
    ref = loaded[0]
    for m in loaded[1:]:
        assert m["labels"] == ref["labels"] and m["dev_ids"] == ref["dev_ids"] and m["test_ids"] == ref["test_ids"]
    dev = json.loads(Path(args.dev_json).read_text()) if args.dev_json else None
    gold_by_id = {}
    for r in runs:  # gold labels from any member's dev_predictions.json
        f = r / "dev_predictions.json"
        if f.exists():
            gold_by_id = {d["id"]: d["gold"] for d in json.loads(f.read_text())}
            break
    labels = ref["labels"]
    index = {l: j for j, l in enumerate(labels)}
    Ydev = torch.zeros((len(ref["dev_ids"]), len(labels)), dtype=torch.bool)
    for i, cid in enumerate(ref["dev_ids"]):
        for l in gold_by_id[cid]:
            if l in index:
                Ydev[i, index[l]] = True

    class A:
        seed, min_support, threshold_shrink = args.seed, args.min_support, args.threshold_shrink

    report = {}
    dev_probs, test_probs = [], []
    for m in loaded:
        p = torch.sigmoid(m["dev_logits"] / m["temperature"])
        dev_probs.append(p)
        test_probs.append(torch.sigmoid(m["test_logits"] / m["temperature"]))
        s, thr, mo, _, rep = choose_thresholds(p, Ydev, A)
        report[m["run"]] = rep[s] | {"strategy": s}
    weights = torch.tensor(args.weights or [1.0] * len(loaded))[: len(loaded)]
    weights = weights / weights.sum()
    P = sum(w * p for w, p in zip(weights, dev_probs))
    PT = sum(w * p for w, p in zip(weights, test_probs))
    strategy, thr, min_one, t_g, rep = choose_thresholds(P, Ydev, A)
    report["ENSEMBLE"] = rep[strategy] | {"strategy": strategy, "members": [m["run"] for m in loaded],
                                           "weights": weights.tolist()}
    print(json.dumps(report, indent=1))
    pred = apply_thresholds(PT, thr, min_one)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = [{"id": cid, "pred": sorted({int(labels[j]) for j in row.nonzero().flatten().tolist()})}
                   for cid, row in zip(ref["test_ids"], pred)]
    assert len({p["id"] for p in predictions}) == len(predictions)
    (out / "prediction.json").write_text(json.dumps(predictions), encoding="utf-8")
    archive = out / "submission_p5_ENSEMBLE.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "prediction.json", arcname="prediction.json")
    full = public(ml_scores(Ydev, apply_thresholds(P, thr, min_one)))
    (out / "ensemble_report.json").write_text(json.dumps({"crossfit": report, "in_sample": full,
                                                          "thresholds": thr.tolist()}, indent=2), encoding="utf-8")
    print(f"ENSEMBLE dev macro-F1 cross-fitted {report['ENSEMBLE']['macro_f1']:.4f}; "
          f"test mean labels/case {float(pred.sum(1).float().mean()):.2f}")
    print("SUBMISSION ZIP:", archive)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    p.add_argument("--dev-json")
    p.add_argument("--output-dir", default="/kaggle/working/p5_ENSEMBLE")
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--threshold-shrink", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=230338)
    main(p.parse_args())
