"""Phase 6 max-score step: decode with an ensemble of fine-tuned prefix-LM summarizers (average of per-step token
log-probabilities; all Phase 1 models share one tokenizer), tune decoding/length on the same validation documents the
members were tuned on, and submit whichever of {member 1, member 2, ensemble} scores best on validation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

import torch

from phase6_summ import (BertScorer, Encoder, clean_text, generate, load_split, load_teacher, score_set, text_stats,
                         truncate_words, tune_decoding)


def main(cli):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in cli.runs if (Path(r) / "best_fp16.pt").exists() and (Path(r) / "manifest.json").exists()]
    print("finished members:", [r.name for r in runs], flush=True)
    if not runs:
        raise SystemExit("no finished member runs")
    manifests = {r.name: json.loads((r / "manifest.json").read_text()) for r in runs}
    for r in runs:
        print(r.name, "val", json.dumps(manifests[r.name]["val_scores"]), flush=True)
    best_member = max(runs, key=lambda r: manifests[r.name]["val_scores"]["selection_score"])
    choice = {"submission": best_member.name, "score": manifests[best_member.name]["val_scores"]["selection_score"]}

    if len(runs) >= 2:
        models, margs = [], []
        for r in runs:
            m, a = load_teacher(r / "best_fp16.pt", device, None, out / f"phase1_{r.name}")
            if device.type != "cuda":
                m = m.float()
            models.append(m)
            margs.append(a)
        keys = ("src_tokens", "tgt_tokens", "selection", "words_per_token", "seed", "val_docs", "val_gen_docs")
        for a in margs[1:]:
            for k in keys:
                assert getattr(a, k) == getattr(margs[0], k), f"members differ on {k}"
        assert len({m.kind for m in models}) == 1, "members must be the same kind (prefix)"
        args = margs[0]
        args.data_dir = cli.data_dir or args.data_dir
        args.gen_batch = cli.gen_batch
        train_recs, val, gen_val, test, _, _ = load_split(args)
        from tokenizers import Tokenizer
        from phase3_rr import locate_phase1_files
        _, tok_path, _ = locate_phase1_files(args.model, args.artifact_zip, out / "phase1_tokenizer")
        tokenizer = Tokenizer.from_file(str(tok_path))
        enc = Encoder(tokenizer, args, models[0].kind)
        args._train_tgt_lengths = enc.target_ids([r["summary"] for r in train_recs])
        scorer = BertScorer(cli.bertscore, device)
        gen_docs, gen_refs = [r["document"] for r in gen_val], [r["summary"] for r in gen_val]
        val_src = enc.source_ids(gen_docs)
        (score, cfg, max_words, texts, scores), grid = tune_decoding(models, enc, tokenizer, gen_docs, val_src, gen_refs,
                                                                     args, device, scorer)
        print("ENSEMBLE val", json.dumps(scores), flush=True)
        report = {"members": {k: v["val_scores"] for k, v in manifests.items()}, "ensemble": scores,
                  "ensemble_decoding": cfg, "ensemble_max_words": max_words, "grid": grid,
                  "ensemble_text_stats": text_stats([truncate_words(t, max_words) for t in texts], gen_refs, gen_docs)}
        if score > choice["score"] or cli.force_ensemble:
            t0 = time.time()
            ids = generate(models, enc, enc.source_ids([r["document"] for r in test]), device, cfg, cli.gen_batch)
            preds = []
            for r, g in zip(test, ids):
                text = truncate_words(clean_text(tokenizer.decode(g, skip_special_tokens=True)), max_words)
                preds.append({"id": r["id"], "summary": text or "The matter was decided by the court."})
            assert len({p["id"] for p in preds}) == len(test)
            (out / "prediction.json").write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
            with zipfile.ZipFile(out / "submission_p6_ENSEMBLE.zip", "w", zipfile.ZIP_DEFLATED) as z:
                z.write(out / "prediction.json", arcname="prediction.json")
            report["test_generation_minutes"] = (time.time() - t0) / 60
            choice = {"submission": "ENSEMBLE", "score": score}
        (out / "ensemble_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    best_zip = (out / "submission_p6_ENSEMBLE.zip") if choice["submission"] == "ENSEMBLE" else \
        sorted((Path([str(r) for r in runs if r.name == choice["submission"]][0])).glob("submission_*.zip"))[0]
    shutil.copy(best_zip, out / "BEST_submission.zip")
    print("CHOSEN:", json.dumps(choice), "->", out / "BEST_submission.zip", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data-dir")
    p.add_argument("--output-dir", default="/kaggle/working/p6_ENSEMBLE")
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--bertscore", type=int, default=1)
    p.add_argument("--force-ensemble", type=int, default=0)
    main(p.parse_args())
