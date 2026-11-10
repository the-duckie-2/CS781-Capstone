"""CS781 Phase 7 - Prior Case Retrieval (PCR) starting from Phase 1 legal language models.

For every query case, rank the candidate cases of the SAME split's pool and return exactly 7 unique ids.
Metric: Micro-F1@7 (micro precision/recall over the top 7 of every query vs the complete gold set).

Pipeline
  1. Lexical stage (no pretrained weights, built from the split's own pool):
       BM25 on word unigrams / bigrams / trigrams (hashed, so rare quoted phrases are kept) + TF-IDF cosine
       (uni, bi).  k1 / b are chosen on dev.  Candidate generation = reciprocal-rank fusion of these systems
       -> top-N per query (the query itself / exact duplicates of it are never returned).
     Extra legal signals: section / article number overlap, judgment-year difference (prior cases must be
     older), how often the candidate is cited by train queries, how central the candidate is in the pool.
  2. Dense retriever fine-tuned from a Phase 1 checkpoint (the required part of every experiment):
       * long documents -> up to K windows of 256 tokens (the Phase 1 context); queries use the windows
         around the masked citation markers (that is where a prior case is discussed) + the opening window;
         candidates use windows spread over the whole judgment;
       * window vector = masked mean of the Phase 1 hidden states -> projection (separate query / candidate
         heads); document = attention + mean pooling over windows (``attn``) or late interaction
         (``maxsim``: mean over query windows of the max cosine over candidate windows);
       * InfoNCE (learned temperature) over 1 positive + H BM25 hard negatives per query + in-batch
         candidates, with other gold candidates of the same query masked out (no false negatives);
       * adaptation: full FT, LORMA+, LoRA, IA3, partial (top layers), BitFit, knowledge distillation
         (student = decoder cut to 8 layers, listwise KL to the teacher's score distribution + InfoNCE).
  3. Fusion: per-query normalised lexical + dense + legal features -> LambdaMART (LightGBM) / logistic
     regression, trained on dev only (the dense model never saw dev).  The method is chosen by 5-fold
     query-grouped cross-fitting on dev (honest estimate) and only among methods that use the Phase 1
     retriever; the lexical-only number is logged for comparison.  Refit on all of dev -> test top 7.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ID = "Exploration-Lab/CS-781-Capstone"
SPLITS = ("train", "dev", "test")
TOP_K = 7
CITE_RES = [re.compile(r"<\s*/?\s*cit(?:ation|ed)?[^>]{0,20}>", re.I), re.compile(r"\[\s*citation\s*\]", re.I),
            re.compile(r"\bCITATION\b"), re.compile(r"\{\s*citation\s*\}", re.I)]
YEAR_RE = re.compile(r"\b(18[5-9]\d|19\d\d|20[0-2]\d)\b")
SECTION_RE = re.compile(r"\b(?:section|sec\.|sections|u/s|u/ss)\s*(\d+[a-z]?)", re.I)
ARTICLE_RE = re.compile(r"\b(?:article|art\.|articles)\s*(\d+[a-z]?)", re.I)


def log(*a):
    print(*a, flush=True)


# ============================================================================= data

def obtain_data(data_dir: str | None) -> dict:
    """{split: {"q": path, "c": path or None}} from --data-dir or the official HF dataset (re-downloaded)."""
    want = {s: (f"{s}.json", f"{s}_candidates.json") for s in SPLITS}
    if data_dir:
        files = [Path(f) / n for f, _, ns in os.walk(data_dir, followlinks=True) for n in ns]
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is missing: add the Kaggle secret HF and enable it")
        repo = list_repo_files(REPO_ID, repo_type="dataset", token=token)
        p7 = [f for f in repo if ("phase_7" in f.lower() or "pcr" in f.lower()) and f.lower().endswith((".json", ".jsonl"))]
        log("Phase 7 files on HF:", p7)
        if not p7:
            raise FileNotFoundError(f"No Phase 7 / PCR json files on HF yet. Repo has: {sorted({f.split('/')[0] for f in repo})}")
        files = []
        for f in p7:
            if Path(f).name.lower().replace(".jsonl", ".json") in {n for pair in want.values() for n in pair}:
                files.append(Path(hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token,
                                                  force_download=True)))
    found = {}
    for s, (qn, cn) in want.items():
        q = [f for f in files if f.name.lower().replace(".jsonl", ".json") == qn]
        c = [f for f in files if f.name.lower().replace(".jsonl", ".json") == cn]
        found[s] = {"q": q[0] if q else None, "c": c[0] if c else None}
    if not found["train"]["q"] or not found["train"]["c"] or not found["test"]["q"] or not found["test"]["c"]:
        raise FileNotFoundError(f"Missing Phase 7 files: {found}")
    return found


ID_KEYS = ("id", "query_id", "qid", "case_id", "doc_id", "candidate_id", "cid", "uid", "name", "file", "filename")
TEXT_KEYS = ("text", "sentences", "content", "document", "case_text", "raw_text", "doc", "body", "judgment")
REL_KEYS = ("relevant_candidates", "relevant", "relevant_ids", "relevant_cases", "positives", "positive",
            "labels", "label", "gold", "citations", "cited", "relevant_candidate_ids", "targets")


def canon(x) -> str:
    s = str(x).strip()
    for ext in (".txt", ".json"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    return (s.lstrip("0") or "0") if s.isdigit() else s


def to_sentences(t) -> list[str]:
    if t is None:
        return []
    if isinstance(t, str):
        parts = [p for p in re.split(r"\n+|(?<=[.?!])\s+(?=[A-Z(\"'])", t) if p.strip()]
        return parts or [t]
    if isinstance(t, dict):
        t = list(t.values())
    out = []
    for x in t:
        if isinstance(x, (list, tuple, dict)):
            out += to_sentences(x)
        elif x is not None and str(x).strip():
            out.append(str(x))
    return out


def _pick(rec, keys):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


def load_records(path: Path, with_rel: bool) -> list[dict]:
    content = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:  # JSON lines
        raw = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(raw, dict):
        for w in ("data", "queries", "candidates", "documents", "cases", "examples", "items"):
            if isinstance(raw.get(w), (list, dict)) and len(raw) <= 3:
                raw = raw[w]
                break
    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            if isinstance(v, dict):
                items.append({**v, "id": _pick(v, ID_KEYS) or k})
            else:
                items.append({"id": k, "text": v})
        raw = items
    out = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise ValueError(f"{Path(path).name} record #{i} is {type(r).__name__}, expected an object")
        rid = _pick(r, ID_KEYS)
        text = _pick(r, TEXT_KEYS)
        if rid is None or text is None:
            raise ValueError(f"{Path(path).name} record #{i}: cannot find id/text, keys={list(r)}")
        rel = None
        if with_rel:
            rel = _pick(r, REL_KEYS)
            if rel is None:
                raise ValueError(f"{Path(path).name}/{rid}: no relevance field among {REL_KEYS}; keys={list(r)}")
            if isinstance(rel, dict):
                rel = [k for k, v in rel.items() if v not in (0, False, None)]
            if not isinstance(rel, (list, tuple)):
                rel = [rel]
            rel = [(_pick(x, ID_KEYS) if isinstance(x, dict) else x) for x in rel]
        sents = to_sentences(text)
        out.append({"id": rid, "key": canon(rid), "sents": sents, "text": " ".join(sents),
                    "rel": None if rel is None else sorted({canon(x) for x in rel})})
    log(f"{Path(path).name}: {len(out)} records; keys of first record: {list(raw[0]) if raw else []}")
    return out


def text_key(text: str) -> str:
    return hashlib.md5(" ".join(text.lower().split()).encode()).hexdigest()


def load_all(data_dir, seed):
    files = obtain_data(data_dir)
    data = {}
    for s in SPLITS:
        f = files[s]
        if f["q"] is None:
            continue
        data[s] = {"q": load_records(f["q"], s != "test"), "c": load_records(f["c"], False)}
    if "dev" not in data:  # fallback: hold out 20% of the train queries (same train pool)
        rng = random.Random(seed)
        qs = data["train"]["q"][:]
        rng.shuffle(qs)
        cut = max(20, len(qs) // 5)
        data["dev"] = {"q": qs[:cut], "c": data["train"]["c"]}
        data["train"] = {"q": qs[cut:], "c": data["train"]["c"]}
        log(f"NO dev.json: held out {cut} train queries as dev (same candidate pool)")
    # leakage: never train on / select with a query that is also a test query
    test_keys = {q["key"] for q in data["test"]["q"]}
    test_text = {text_key(q["text"]) for q in data["test"]["q"]}
    leak = {}
    for s in ("train", "dev"):
        before = len(data[s]["q"])
        data[s]["q"] = [q for q in data[s]["q"] if q["key"] not in test_keys and text_key(q["text"]) not in test_text]
        leak[f"{s}_queries_removed_as_test_duplicates"] = before - len(data[s]["q"])
    dev_keys = {q["key"] for q in data["dev"]["q"]}
    before = len(data["train"]["q"])
    data["train"]["q"] = [q for q in data["train"]["q"] if q["key"] not in dev_keys]
    leak["train_queries_removed_as_dev_duplicates"] = before - len(data["train"]["q"])
    for s in SPLITS:
        pool = {c["key"] for c in data[s]["c"]}
        if len(pool) != len(data[s]["c"]):
            log(f"WARNING {s}: duplicate candidate ids in pool ({len(data[s]['c'])} records, {len(pool)} unique)")
            seen, keep = set(), []
            for c in data[s]["c"]:
                if c["key"] not in seen:
                    seen.add(c["key"])
                    keep.append(c)
            data[s]["c"] = keep
        leak[f"{s}_queries_inside_own_pool"] = sum(q["key"] in pool for q in data[s]["q"])
    pools = {s: {c["key"] for c in data[s]["c"]} for s in SPLITS}
    leak["train_test_pool_overlap"] = len(pools["train"] & pools["test"])
    leak["dev_test_pool_overlap"] = len(pools["dev"] & pools["test"])
    log("LEAKAGE CHECK:", json.dumps(leak))
    stats = {}
    for s in SPLITS:
        pool = pools[s]
        qs = data[s]["q"]
        words = sorted(len(q["text"].split()) for q in qs)
        cw = sorted(len(c["text"].split()) for c in data[s]["c"])
        st = {"queries": len(qs), "pool": len(pool),
              "query_words_p50_p90": [words[len(words) // 2], words[int(len(words) * .9)]] if words else None,
              "cand_words_p50_p90": [cw[len(cw) // 2], cw[int(len(cw) * .9)]] if cw else None,
              "queries_with_citation_markers": sum(any(r.search(q["text"]) for r in CITE_RES) for q in qs)}
        if s != "test":
            g = [len(q["rel"]) for q in qs]
            inpool = [sum(x in pool for x in q["rel"]) for q in qs]
            st.update({"gold_mean": round(sum(g) / max(1, len(g)), 3), "gold_hist": dict(sorted(Counter(min(x, 15) for x in g).items())),
                       "gold_in_pool_fraction": round(sum(inpool) / max(1, sum(g)), 4),
                       "queries_without_gold_in_pool": sum(x == 0 for x in inpool),
                       "oracle_micro_f1@7": round(oracle_f1(qs, pool), 4)})
        stats[s] = st
        log(f"DATA [{s}]", json.dumps(st))
    return data, leak, stats


def oracle_f1(queries, pool):
    tp = sum(min(TOP_K, sum(x in pool for x in q["rel"])) for q in queries)
    return micro_from(tp, TOP_K * len(queries), sum(len(q["rel"]) for q in queries))["f1"]


# ============================================================================= metrics

def micro_from(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r}


def micro_at_k(ranked: list[list[str]], gold: list[list[str]], k=TOP_K):
    tp = sum(len(set(r[:k]) & set(g)) for r, g in zip(ranked, gold))
    return micro_from(tp, k * len(ranked), sum(len(g) for g in gold))


# ============================================================================= lexical stage

def clean_for_lexical(text: str) -> str:
    for r in CITE_RES:
        text = r.sub(" ", text)
    return text


def hashed_tf(texts, n, n_features, n_jobs=4):
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer
    vec = HashingVectorizer(n_features=n_features, ngram_range=(n, n), alternate_sign=False, norm=None,
                            lowercase=True, stop_words=list(ENGLISH_STOP_WORDS), token_pattern=r"(?u)\b\w\w+\b",
                            dtype=np.float32)
    chunks = [texts[i:i + 200] for i in range(0, len(texts), 200)]
    mats = Parallel(n_jobs=n_jobs)(delayed(vec.transform)(c) for c in chunks)
    return sp.vstack(mats).tocsr()


def idf_of(D):
    N = D.shape[0]
    df = np.bincount(D.indices, minlength=D.shape[1]).astype(np.float32)
    return np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32), df


def bm25_scores(Qtf, Dtf, k1, b):
    idf, _ = idf_of(Dtf)
    D = Dtf.copy().astype(np.float32)
    dl = np.asarray(Dtf.sum(1)).ravel()
    avg = dl.mean() if dl.mean() > 0 else 1.0
    rows = np.repeat(np.arange(D.shape[0]), np.diff(D.indptr))
    denom = D.data + k1 * (1 - b + b * dl[rows] / avg)
    D.data = D.data * (k1 + 1) / denom * idf[D.indices]
    Q = Qtf.copy().astype(np.float32)
    Q.data = 1 + np.log(Q.data)
    return (Q @ D.T).toarray().astype(np.float32)


def tfidf_cos(Qtf, Dtf):
    idf, _ = idf_of(Dtf)

    def w(M):
        M = M.copy().astype(np.float32)
        M.data = (1 + np.log(M.data)) * idf[M.indices]
        norm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) + 1e-9
        return sp.diags(1 / norm) @ M
    return (w(Qtf) @ w(Dtf).T).toarray().astype(np.float32)


def rank_of(S):
    """rank (0 = best) of every column within each row."""
    order = np.argsort(-S, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(S.shape[0])[:, None], order] = np.arange(S.shape[1])[None, :]
    return ranks


def legal_sets(text):
    return ({m.lower() for m in SECTION_RE.findall(text)}, {m.lower() for m in ARTICLE_RE.findall(text)})


def doc_year(text):
    ys = [int(y) for y in YEAR_RE.findall(text)]
    ys = [y for y in ys if y <= 2026]
    return max(ys) if ys else -1


def exclusion_mask(queries, cands):
    ckeys = {c["key"]: j for j, c in enumerate(cands)}
    ctext = defaultdict(list)
    for j, c in enumerate(cands):
        ctext[text_key(c["text"])].append(j)
    mask = np.zeros((len(queries), len(cands)), dtype=bool)
    for i, q in enumerate(queries):
        if q["key"] in ckeys:
            mask[i, ckeys[q["key"]]] = True
        for j in ctext.get(text_key(q["text"]), []):
            mask[i, j] = True
    return mask


def lexical_split(split, queries, cands, args, params=None, train_popularity=None):
    """-> dict with candidate lists (top-N), feature tensor and the per-system score matrices' summaries."""
    t = time.time()
    qt = [clean_for_lexical(q["text"]) for q in queries]
    ct = [clean_for_lexical(c["text"]) for c in cands]
    excl = exclusion_mask(queries, cands)
    tfs = {}
    for n in ((1,) if split == "train" else (1, 2, 3)):
        both = hashed_tf(qt + ct, n, 2 ** (20 if n == 1 else 22), args.cpu_jobs)
        tfs[n] = (both[:len(qt)], both[len(qt):])
    log(f"[{split}] hashed n-gram tf in {time.time() - t:.0f}s")
    systems = {}
    gold = [q["rel"] for q in queries] if split != "test" else None
    tuned = {}
    for n in ((1,) if split == "train" else (1, 2, 3)):
        Q, D = tfs[n]
        if params is None and gold is not None and n in (1, 2):
            best = None
            for k1 in (0.9, 1.5, 3.0):
                for b in (0.5, 0.75, 1.0):
                    S = bm25_scores(Q, D, k1, b)
                    S[excl] = -1e9
                    f = micro_at_k(top_ids(S, cands, TOP_K), gold)["f1"]
                    if best is None or f > best[0]:
                        best = (f, k1, b)
            tuned[f"bm25_{n}"] = {"k1": best[1], "b": best[2], "dev_f1@7": round(best[0], 4)}
        p = (params or {}).get(f"bm25_{n}") or tuned.get(f"bm25_{n}") or {"k1": 1.5, "b": 0.75}
        systems[f"bm25_{n}"] = bm25_scores(Q, D, p["k1"], p["b"])
        if n in (1, 2) and split != "train":
            systems[f"tfidf_{n}"] = tfidf_cos(Q, D)
    for S in systems.values():
        S[excl] = -1e9
    if tuned:
        log(f"[{split}] BM25 grid (dev F1@7):", json.dumps(tuned))
    used_params = params or {k: tuned.get(k, {"k1": 1.5, "b": 0.75}) for k in ("bm25_1", "bm25_2", "bm25_3")}
    report = {}
    if gold is not None:
        for name, S in systems.items():
            report[name] = {k: round(v, 4) for k, v in micro_at_k(top_ids(S, cands, TOP_K), gold).items()}
    # RRF over the lexical systems -> candidate list
    rrf = np.zeros_like(next(iter(systems.values())))
    for S in systems.values():
        rrf += 1.0 / (60.0 + rank_of(S))
    rrf[excl] = -1e9
    N = min(args.top_n if split != "train" else args.hn_pool + 10, len(cands) - 1)
    order = np.argsort(-rrf, axis=1, kind="stable")
    cand_lists = order[:, :N]
    if gold is not None:
        report["rrf_lexical"] = {k: round(v, 4) for k, v in micro_at_k(top_ids(rrf, cands, TOP_K), gold).items()}
        tp = sum(len({cands[j]["key"] for j in row} & set(g)) for row, g in zip(cand_lists, gold))
        report[f"recall@{N}"] = round(tp / max(1, sum(len(g) for g in gold)), 4)
        log(f"[{split}] lexical dev/train metrics:", json.dumps(report))
    out = {"split": split, "query_ids": [q["id"] for q in queries], "query_keys": [q["key"] for q in queries],
           "cand_ids": [c["id"] for c in cands], "cand_keys": [c["key"] for c in cands], "lists": cand_lists,
           "gold": gold, "report": report, "params": used_params, "rrf_full_order": order[:, :max(N, 50)]}
    if split == "train":
        return out
    # ---------------- features for fusion (nq, N, F)
    names = list(systems)
    feats, fnames = [], []
    rows = np.arange(len(queries))[:, None]
    for name in names + ["rrf"]:
        S = systems[name] if name != "rrf" else rrf
        R = rank_of(S)
        v = S[rows, cand_lists]
        mu, sd = v.mean(1, keepdims=True), v.std(1, keepdims=True) + 1e-6
        top1 = v.max(1, keepdims=True)
        feats += [(v - mu) / sd, v / (np.abs(top1) + 1e-6), 1.0 / (1.0 + R[rows, cand_lists])]
        fnames += [f"{name}_z", f"{name}_rel_top", f"{name}_rr"]
    # legal features
    qsets = [legal_sets(q["text"]) for q in queries]
    csets = [legal_sets(c["text"]) for c in cands]
    qy = np.array([doc_year(q["text"]) for q in queries])
    cy = np.array([doc_year(c["text"]) for c in cands])
    clen = np.log1p(np.array([len(c["text"].split()) for c in cands], dtype=np.float32))
    pop = np.array([train_popularity.get(c["key"], 0) for c in cands], dtype=np.float32) if train_popularity else np.zeros(len(cands), np.float32)
    central = np.zeros(len(cands), np.float32)
    for row in order[:, :20]:
        central[row] += 1
    central /= max(1, len(queries))
    sec_j = np.zeros(cand_lists.shape, np.float32)
    sec_n = np.zeros(cand_lists.shape, np.float32)
    art_j = np.zeros(cand_lists.shape, np.float32)
    for i, row in enumerate(cand_lists):
        qs, qa = qsets[i]
        for r, j in enumerate(row):
            cs, ca = csets[j]
            inter = len(qs & cs)
            sec_n[i, r] = inter
            sec_j[i, r] = inter / max(1, len(qs | cs))
            art_j[i, r] = len(qa & ca) / max(1, len(qa | ca))
    ydiff = np.where((qy[:, None] > 0) & (cy[cand_lists] > 0), np.clip(qy[:, None] - cy[cand_lists], -30, 80), 0)
    feats += [sec_j, np.log1p(sec_n), art_j, ydiff.astype(np.float32),
              ((qy[:, None] > 0) & (cy[cand_lists] > qy[:, None])).astype(np.float32),
              ((qy[:, None] <= 0) | (cy[cand_lists] <= 0)).astype(np.float32),
              np.broadcast_to(clen[cand_lists], cand_lists.shape), np.log1p(pop[cand_lists]), central[cand_lists]]
    fnames += ["section_jaccard", "section_overlap_log", "article_jaccard", "year_diff", "cand_after_query",
               "year_missing", "cand_log_words", "train_cited_log", "pool_centrality"]
    out["features"] = np.stack(feats, -1).astype(np.float32)
    out["feature_names"] = fnames
    log(f"[{split}] lexical stage done in {time.time() - t:.0f}s; features {out['features'].shape}")
    return out


def top_ids(S, cands, k):
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return [[cands[j]["key"] for j in row] for row in order]


# ============================================================================= windows for the dense model

def select_spread(ids, L, K):
    n = len(ids)
    if n <= L:
        return [ids]
    starts = list(range(0, n - L + 1, L))
    if starts[-1] != n - L:
        starts.append(n - L)
    if len(starts) > K:
        picks = sorted({round(i * (len(starts) - 1) / (K - 1)) for i in range(K)}) if K > 1 else [0]
        starts = [starts[i] for i in picks]
    return [ids[s:s + L] for s in starts]


def citation_contexts(sents, before):
    out = []
    for i, s in enumerate(sents):
        if any(r.search(s) for r in CITE_RES):
            ctx = " ".join(sents[max(0, i - before): i + 1])
            out.append(clean_for_lexical(ctx))
    return out


def build_windows(docs, tokenizer, args, is_query):
    L, K = args.chunk_len, args.max_windows
    texts = [clean_for_lexical(" ".join(d["text"].split()[:args.max_words])) for d in docs]
    result = [None] * len(docs)
    marker_docs = 0
    ctx_jobs = []
    for s in range(0, len(docs), 256):
        for i, enc in zip(range(s, min(len(docs), s + 256)), tokenizer.encode_batch(texts[s:s + 256])):
            ids = np.asarray(enc.ids or [args.eos_id], dtype=np.int32)
            result[i] = select_spread(ids, L, K)
            if is_query and args.query_windows == "citation":
                ctx = citation_contexts(docs[i]["sents"], args.context_sentences)
                if ctx:
                    marker_docs += 1
                    keep = K - args.query_head_windows
                    if len(ctx) > keep:
                        picks = sorted({round(t * (len(ctx) - 1) / (keep - 1)) for t in range(keep)}) if keep > 1 else [0]
                        ctx = [ctx[p] for p in picks]
                    ctx_jobs.append((i, ids, ctx))
    for i, ids, ctx in ctx_jobs:
        encs = tokenizer.encode_batch(ctx)
        wins = [np.asarray(e.ids[-L:], dtype=np.int32) for e in encs if e.ids]
        head = [ids[h * L:(h + 1) * L] for h in range(args.query_head_windows) if len(ids) > h * L]
        result[i] = (head + wins)[:K] or result[i]
    if is_query:
        log(f"  {marker_docs}/{len(docs)} queries use citation-context windows")
    return result


def make_batch(window_lists, pad_id, device):
    flat = [w for ws in window_lists for w in ws]
    length = max(len(w) for w in flat)
    ids = torch.full((len(flat), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(flat), length), dtype=torch.bool)
    for r, w in enumerate(flat):
        ids[r, :len(w)] = torch.from_numpy(w.astype(np.int64))
        mask[r, :len(w)] = True
    K = max(len(ws) for ws in window_lists)
    valid = torch.zeros((len(window_lists), K), dtype=torch.bool)
    for i, ws in enumerate(window_lists):
        valid[i, :len(ws)] = True
    return ids.to(device), mask.to(device), valid.to(device)


# ============================================================================= model

class PCRModel(nn.Module):
    def __init__(self, backbone, model_name, proj_dim=256, pooling="attn", grad_ckpt=False):
        super().__init__()
        from phase2_lner import LNERModel
        self.encoder = LNERModel(backbone, model_name, 1)
        del self.encoder.classifier
        d = backbone.cfg.d_model
        self.model_name, self.pooling, self.grad_ckpt = model_name, pooling, grad_ckpt
        self.proj = nn.ModuleDict({s: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, proj_dim)) for s in ("q", "c")})
        self.attn = nn.ModuleDict({s: nn.Linear(proj_dim, 1) for s in ("q", "c")})
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def hidden(self, ids, mask):
        if self.model_name == "efficient" and self.grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            bb = self.encoder.backbone
            h = bb.embedding(ids)
            for block in bb.blocks:
                h = checkpoint(block, h, use_reentrant=False)
            return bb.norm(h)
        return self.encoder.hidden(ids, mask)

    def embed(self, ids, mask, valid, side):
        h = self.hidden(ids, mask)
        w = mask.unsqueeze(-1).to(h.dtype)
        v = (h * w).sum(1) / w.sum(1).clamp_min(1)
        v = self.proj[side](v).float()
        B, K = valid.shape
        chunks = v.new_zeros(B, K, v.size(-1))
        chunks[valid] = v
        if self.pooling == "maxsim":
            return F.normalize(chunks, dim=-1) * valid.unsqueeze(-1)
        s = self.attn[side](chunks).squeeze(-1).masked_fill(~valid, -1e4)
        a = s.softmax(-1)
        doc = (a.unsqueeze(-1) * chunks).sum(1) + (chunks * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(doc, dim=-1)

    def sim(self, q, c, qvalid=None, cvalid=None):
        if self.pooling != "maxsim":
            return q @ c.T
        s = torch.einsum("ikp,jlp->ijkl", q, c)
        s = s.masked_fill(~cvalid[None, :, None, :], -1e4).max(-1).values
        qv = qvalid[:, None, :].float()
        return (s * qv).sum(-1) / qv.sum(-1).clamp_min(1)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)


@torch.no_grad()
def encode_docs(model, window_lists, side, pad_id, device, batch_windows=192):
    """-> (doc embeddings on CPU (fp16), valid mask) ; attn: (N, p) ; maxsim: (N, K, p)."""
    model.eval()
    embs, valids = [None] * len(window_lists), [None] * len(window_lists)
    order = sorted(range(len(window_lists)), key=lambda i: -len(window_lists[i]))
    s = 0
    while s < len(order):
        rows, total = [], 0
        while s < len(order) and (not rows or total + len(window_lists[order[s]]) <= batch_windows):
            rows.append(order[s])
            total += len(window_lists[order[s]])
            s += 1
        ids, mask, valid = make_batch([window_lists[r] for r in rows], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            e = model.embed(ids, mask, valid, side)
        e = e.float().cpu()
        for j, r in enumerate(rows):
            embs[r], valids[r] = e[j], valid[j].cpu()
    if model.pooling == "maxsim":
        K = max(x.size(0) for x in embs)
        E = torch.zeros(len(embs), K, embs[0].size(-1))
        V = torch.zeros(len(embs), K, dtype=torch.bool)
        for i, (x, v) in enumerate(zip(embs, valids)):
            E[i, :x.size(0)] = x
            V[i, :v.size(0)] = v
        return E.half(), V
    return torch.stack(embs).half(), None


@torch.no_grad()
def dense_list_scores(model, q_windows, c_windows, lists, pad_id, device, q_rows=None):
    """Dense scores (cosine, before the temperature) for every (query, candidate in its list)."""
    q_rows = list(range(len(q_windows))) if q_rows is None else q_rows
    needed = sorted({int(j) for i in q_rows for j in lists[i]})
    Ec, Vc = encode_docs(model, [c_windows[j] for j in needed], "c", pad_id, device)
    Eq, Vq = encode_docs(model, [q_windows[i] for i in q_rows], "q", pad_id, device)
    pos = {j: r for r, j in enumerate(needed)}
    out = []
    for a, i in enumerate(q_rows):
        idx = torch.tensor([pos[int(j)] for j in lists[i]])
        q = Eq[a:a + 1].to(device).float()
        c = Ec[idx].to(device).float()
        if model.pooling == "maxsim":
            s = model.sim(q, c, Vq[a:a + 1].to(device), Vc[idx].to(device))
        else:
            s = model.sim(q, c)
        out.append(s[0].float().cpu().numpy())
    return np.stack(out)


# ============================================================================= training


def train(model, tr, dev, args, device, pad_id, teacher=None):
    """tr: {"queries": [{"key", "pos": [cand idx], "hn": [cand idx]}], "q_windows", "c_windows", "cand_keys"}
    dev: {"q_windows", "c_windows", "lists", "cand_keys", "gold"} (dense rerank of the lexical top-N)."""
    head_ids = {id(p) for p in model.head_parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": args.head_lr}] + ([{"params": body, "lr": args.learning_rate}] if body else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    budget = args.time_budget_min * 60
    est = [max(200, args.epochs * len(tr["queries"]) // args.batch_queries)]
    warm = 50
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warm, max(0.03, 1 - s / est[0])))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    trainable = [p for p in model.parameters() if p.requires_grad]
    rng = random.Random(args.seed)
    eval_rows = sorted(rng.sample(range(len(dev["q_windows"])), min(args.eval_queries, len(dev["q_windows"]))))
    start, last_eval = time.time(), time.time()
    best, best_state, history = {"f1": -1.0}, None, []
    step, ooms, kd_used = 0, 0, 0
    queries = tr["queries"]

    def evaluate(tag):
        nonlocal best, best_state, last_eval
        S = dense_list_scores(model, dev["q_windows"], dev["c_windows"], dev["lists"], pad_id, device, eval_rows)
        ranked = [[dev["cand_keys"][dev["lists"][i][j]] for j in np.argsort(-S[a], kind="stable")[:TOP_K]]
                  for a, i in enumerate(eval_rows)]
        sc = micro_at_k(ranked, [dev["gold"][i] for i in eval_rows])
        row = {"tag": tag, "opt_steps": step, "minutes": round((time.time() - start) / 60, 2),
               "dense_rerank_f1@7": round(sc["f1"], 4), "logit_scale": round(float(model.scale()), 2),
               "loss": round(float(np.mean(losses[-100:])), 4) if losses else None,
               "kd": round(float(np.mean(kd_losses[-100:])), 4) if kd_losses else None}
        history.append(row)
        log(json.dumps(row))
        if sc["f1"] > best["f1"]:
            best = {**sc, "tag": tag}
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        model.train()
        last_eval = time.time()

    losses, kd_losses = [], []
    evaluate("start (no training)")
    epoch, done = 0, False
    while not done:
        epoch += 1
        order = list(range(len(queries)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        for s in range(0, len(order) - args.batch_queries + 1, args.batch_queries):
            rows = order[s:s + args.batch_queries]
            qw, cw, col_cand, targets, own, tlogits = [], [], [], [], [], []
            for i in rows:
                q = queries[i]
                ws = tr["q_windows"][i]
                pick = sorted(rng.sample(range(len(ws)), min(args.train_q_windows, len(ws))))
                qw.append([ws[t] for t in pick])
                pos = rng.choice(q["pos"])
                pool = [j for j in q["hn"][:args.hn_pool] if j not in q["pos_set"]]
                negs = rng.sample(pool, min(args.hard_negatives, len(pool)))
                while len(negs) < args.hard_negatives:
                    r = rng.randrange(len(tr["c_windows"]))
                    if r not in q["pos_set"]:
                        negs.append(r)
                lst = [pos] + negs
                targets.append(len(col_cand))
                own.append(list(range(len(col_cand), len(col_cand) + len(lst))))
                col_cand += lst
                for j in lst:
                    ws = tr["c_windows"][j]
                    pick = sorted(rng.sample(range(len(ws)), min(args.train_c_windows, len(ws))))
                    cw.append([ws[t] for t in pick])
                if teacher is not None:
                    tq = teacher.get(q["key"])
                    vals = None if tq is None else [tq.get(tr["cand_keys"][j]) for j in lst]
                    tlogits.append(vals if vals is not None and all(v is not None for v in vals) else None)
            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    ids, mask, qvalid = make_batch(qw, pad_id, device)
                    Q = model.embed(ids, mask, qvalid, "q")
                    ids, mask, cvalid = make_batch(cw, pad_id, device)
                    C = model.embed(ids, mask, cvalid, "c")
                logits = model.sim(Q.float(), C.float(), qvalid, cvalid) * model.scale()
                false_neg = torch.zeros_like(logits, dtype=torch.bool)
                for r, i in enumerate(rows):
                    ps = queries[i]["pos_set"]
                    for col, j in enumerate(col_cand):
                        if col != targets[r] and j in ps:
                            false_neg[r, col] = True
                logits = logits.masked_fill(false_neg, -1e4)
                loss = F.cross_entropy(logits, torch.tensor(targets, device=device))
                if teacher is not None:
                    kd_terms = []
                    for r, tl in enumerate(tlogits):
                        if tl is None:
                            continue
                        st = logits[r, own[r]] / args.kd_temperature
                        tt = torch.tensor(tl, device=device, dtype=torch.float32) / args.kd_temperature
                        kd_terms.append(F.kl_div(F.log_softmax(st, -1), F.softmax(tt, -1), reduction="sum"))
                    if kd_terms:
                        kd = torch.stack(kd_terms).mean() * args.kd_temperature ** 2
                        kd_losses.append(float(kd))
                        kd_used += 1
                        loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
                scaler.scale(loss / args.grad_accum).backward()
            except torch.cuda.OutOfMemoryError:
                ooms += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log(f"OOM #{ooms}: batch skipped")
                if ooms > 15:
                    raise RuntimeError("Too many OOMs: append --batch-queries 2 --train-c-windows 2 to the command")
                continue
            losses.append(float(loss))
            step += 1
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                sched.step()
            if step == 60:
                rate = step / (time.time() - start)
                est[0] = max(100, min(est[0] * 10, int(rate * budget * 0.95 / args.grad_accum)))
                log(f"~{rate:.2f} steps/s -> LR schedule over {est[0]} optimizer steps "
                    f"(~{est[0] * args.grad_accum * args.batch_queries / max(1, len(queries)):.1f} epochs over train queries)")
            if step % 100 == 0:
                log(f"epoch {epoch} step {step} loss {np.mean(losses[-100:]):.4f}"
                    + (f" kd {np.mean(kd_losses[-100:]):.4f}" if kd_losses else "")
                    + f" elapsed {(time.time() - start) / 60:.1f} min")
            if time.time() - start > budget:
                log("Time budget reached.")
                done = True
                break
            if time.time() - last_eval > args.eval_every_min * 60:
                evaluate(f"epoch{epoch}")
        if epoch >= args.epochs:
            done = True
    evaluate("final")
    model.load_state_dict(best_state)
    log(f"best checkpoint: {best.get('tag')} dense rerank F1@7 {best['f1']:.4f} (KD batches used: {kd_used})")
    return best, history, time.time() - start


# ============================================================================= fusion

def per_query_transform(S):
    mu, sd = S.mean(1, keepdims=True), S.std(1, keepdims=True) + 1e-6
    R = rank_of(S)
    return [(S - mu) / sd, S - S.max(1, keepdims=True), 1.0 / (1.0 + R)]


def fusion_matrix(lex, dense_list):
    """lex: (nq, N, F) ; dense_list: [(nq, N)] -> (nq, N, F') and feature names suffix."""
    parts = [lex]
    for S in dense_list:
        parts += [x[..., None] for x in per_query_transform(S)]
    if len(dense_list) > 1:
        Z = np.mean([per_query_transform(S)[0] for S in dense_list], 0)
        parts.append(Z[..., None])
    return np.concatenate(parts, -1).astype(np.float32)


def labels_for(split_lex):
    gold = split_lex["gold"]
    keys = split_lex["cand_keys"]
    return np.array([[keys[j] in set(g) for j in row] for row, g in zip(split_lex["lists"], gold)], dtype=np.int32)


def make_learner(kind, seed):
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=15,
                              min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, random_state=seed, verbose=-1, lambdarank_truncation_level=20)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    raise ValueError(kind)


def fit_predict(kind, X, y, Xte, seed):
    nq, N, Fd = X.shape
    if kind == "lgbm_bag":  # 5 LambdaMART rankers with different seeds / row+column subsamples, z-averaged per query
        preds, last = [], None
        for b in range(5):
            p, last = fit_predict("lgbm", X, y, Xte, seed + 101 * b)
            preds.append((p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-9))
        return np.mean(preds, 0), last
    m = make_learner(kind, seed)
    if kind == "lgbm":
        m.fit(X.reshape(-1, Fd), y.reshape(-1), group=[N] * nq)
        return m.predict(Xte.reshape(-1, Fd)).reshape(Xte.shape[0], N), m
    m.fit(X.reshape(-1, Fd), y.reshape(-1))
    return m.predict_proba(Xte.reshape(-1, Fd))[:, 1].reshape(Xte.shape[0], N), m


def lgbm_available():
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def ranked_from(scores, split_lex, rows=None):
    rows = range(scores.shape[0]) if rows is None else rows
    keys = split_lex["cand_keys"]
    out = []
    for a, i in enumerate(rows):
        order = np.argsort(-scores[a], kind="stable")
        out.append([keys[split_lex["lists"][i][j]] for j in order[:TOP_K]])
    return out


def choose_fusion(dev_lex, test_lex, dense_dev, dense_test, args, n_dense_cols):
    """5-fold query-grouped cross-fit on dev -> pick a method among those that use the Phase 1 retriever."""
    y = labels_for(dev_lex)
    Xdev = fusion_matrix(dev_lex["features"], dense_dev)
    Xte = fusion_matrix(test_lex["features"], dense_test)
    nlex = dev_lex["features"].shape[-1]
    nq = Xdev.shape[0]
    gold = dev_lex["gold"]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(nq)
    folds = [perm[f::5] for f in range(5)]
    kinds = (["lgbm"] if lgbm_available() else ["hgb"]) + ["lr"]
    methods = {}
    for k in kinds:
        methods[f"{k}_all"] = (k, slice(None))
        methods[f"{k}_lexical_only"] = (k, slice(0, nlex))
    if lgbm_available() and getattr(args, "bagged_fusion", False):
        methods["lgbm_bag_all"] = ("lgbm_bag", slice(None))
    report = {}
    oof = {}
    for name, (kind, cols) in methods.items():
        pred = np.zeros(y.shape, np.float32)
        for f in range(5):
            te = folds[f]
            tr_ = np.concatenate([folds[g] for g in range(5) if g != f])
            if y[tr_].sum() == 0:
                continue
            p, _ = fit_predict(kind, Xdev[tr_][..., cols], y[tr_], Xdev[te][..., cols], args.seed)
            pred[te] = p
        oof[name] = pred
        report[name] = {k: round(v, 4) for k, v in micro_at_k(ranked_from(pred, dev_lex), gold).items()}
    # simple no-learning baselines
    zd = np.mean([per_query_transform(S)[0] for S in dense_dev], 0)
    rrf_col = dev_lex["feature_names"].index("rrf_rr")
    report["dense_only"] = {k: round(v, 4) for k, v in micro_at_k(ranked_from(zd, dev_lex), gold).items()}
    rrf_dense = dev_lex["features"][..., rrf_col] + np.mean([per_query_transform(S)[2] for S in dense_dev], 0)
    report["rrf_lexical+dense"] = {k: round(v, 4) for k, v in micro_at_k(ranked_from(rrf_dense, dev_lex), gold).items()}
    report["rrf_lexical_only"] = {k: round(v, 4) for k, v in micro_at_k(
        ranked_from(dev_lex["features"][..., rrf_col], dev_lex), gold).items()}
    candidates = [m for m in report if m not in ("rrf_lexical_only",) and not m.endswith("lexical_only")]
    chosen = max(candidates, key=lambda m: report[m]["f1"])
    log("FUSION (5-fold cross-fitted dev micro F1/P/R@7):", json.dumps(report, indent=1))
    log(f"-> chosen: {chosen} (dev F1@7 {report[chosen]['f1']:.4f}; lexical-only best "
        f"{max(report[m]['f1'] for m in report if 'lexical' in m and 'dense' not in m):.4f})")
    importances = None
    if chosen in methods:
        kind, cols = methods[chosen]
        test_scores, model = fit_predict(kind, Xdev[..., cols], y, Xte[..., cols], args.seed)
        names = dev_lex["feature_names"] + [f"dense{d}_{t}" for d in range(len(dense_dev)) for t in ("z", "minus_top", "rr")] \
            + (["dense_mean_z"] if len(dense_dev) > 1 else [])
        try:
            if kind == "lr":
                coef = model[-1].coef_[0]
            elif kind == "lgbm_bag":
                coef = getattr(model, "feature_importances_", None)
            else:
                coef = getattr(model, "feature_importances_", None)
            if coef is not None:
                importances = sorted(zip(names, map(float, coef)), key=lambda t: -abs(t[1]))[:25]
        except Exception:  # noqa: BLE001
            pass
        dev_scores = oof[chosen]
    elif chosen == "dense_only":
        test_scores = np.mean([per_query_transform(S)[0] for S in dense_test], 0)
        dev_scores = zd
    else:
        test_scores = test_lex["features"][..., rrf_col] + np.mean([per_query_transform(S)[2] for S in dense_test], 0)
        dev_scores = rrf_dense
    return chosen, report, test_scores, dev_scores, importances


def write_submission(test_lex, test_scores, test_queries_ids, out, name):
    keys = test_lex["cand_keys"]
    key_to_id = dict(zip(keys, test_lex["cand_ids"]))
    pool = set(keys)
    pred = {}
    for i, qid in enumerate(test_queries_ids):
        order = np.argsort(-test_scores[i], kind="stable")
        picked = []
        for j in order:
            k = keys[test_lex["lists"][i][j]]
            if k not in picked:
                picked.append(k)
            if len(picked) == TOP_K:
                break
        for j in test_lex["rrf_full_order"][i]:  # pad (never needed when top-N >= 7)
            if len(picked) == TOP_K:
                break
            if keys[j] not in picked and keys[j] != canon(qid):
                picked.append(keys[j])
        pred[str(qid)] = [str(key_to_id[k]) for k in picked]
    assert len(pred) == len(test_queries_ids)
    assert all(len(v) == TOP_K and len(set(v)) == TOP_K for v in pred.values())
    assert all(canon(c) in pool for v in pred.values() for c in v)
    path = out / "prediction.json"
    path.write_text(json.dumps(pred, indent=1), encoding="utf-8")
    archive = out / f"submission_{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ["prediction.json"]
        assert len(json.loads(z.read("prediction.json"))) == len(test_queries_ids)
    log(f"VALIDATED prediction.json: {len(pred)} test queries x {TOP_K} unique candidates from the test pool")
    return archive


def analysis(dev_lex, dev_scores, dev_queries, report):
    gold = dev_lex["gold"]
    ranked = ranked_from(dev_scores, dev_lex)
    per = []
    for q, r, g in zip(dev_queries, ranked, gold):
        per.append({"id": q["id"], "gold": g, "top7": r, "hits": len(set(r) & set(g)), "words": q["words"],
                    "has_citation_markers": q["has_markers"]})
    words = np.array([p["words"] for p in per])
    cuts = np.quantile(words, [0.33, 0.66]).tolist() if len(words) > 3 else [0, 0]
    by_len = {}
    for name, sel in (("short", words <= cuts[0]), ("medium", (words > cuts[0]) & (words <= cuts[1])), ("long", words > cuts[1])):
        idx = np.nonzero(sel)[0]
        if len(idx):
            by_len[name] = {"queries": int(len(idx)), **{k: round(v, 4) for k, v in micro_at_k(
                [ranked[i] for i in idx], [gold[i] for i in idx]).items()}}
    by_gold = {}
    for b in sorted({min(len(g), 10) for g in gold}):
        idx = [i for i, g in enumerate(gold) if min(len(g), 10) == b]
        by_gold[f"{b}{'+' if b == 10 else ''}"] = {"queries": len(idx), **{k: round(v, 4) for k, v in micro_at_k(
            [ranked[i] for i in idx], [gold[i] for i in idx]).items()}}
    hits = Counter(p["hits"] for p in per)
    return per, {"f1_by_query_length": by_len, "length_cut_words": cuts, "f1_by_gold_count": by_gold,
                 "hits_in_top7_hist": dict(sorted(hits.items())), "queries_with_zero_hits": hits.get(0, 0)}


# ============================================================================= preparation (shared cache)

def prepare(args, tokenizer_path):
    key = hashlib.md5(json.dumps({"v": 3, "top_n": args.top_n, "hn_pool": args.hn_pool, "L": args.chunk_len,
                                  "K": args.max_windows, "qw": args.query_windows, "ctx": args.context_sentences,
                                  "head": args.query_head_windows, "words": args.max_words, "data": args.data_dir,
                                  "tok": hashlib.md5(Path(tokenizer_path).read_bytes()).hexdigest(),
                                  "limit": args.limit_queries}).encode()).hexdigest()[:10]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"p7_cache_{key}.pkl"
    lock = cache_dir / f"p7_cache_{key}.lock"
    for _ in range(240):  # another run is building the same cache -> wait for it
        if path.exists() or not lock.exists():
            break
        time.sleep(15)
    if path.exists():
        log(f"Using cached lexical stage + windows: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    lock.write_text(str(os.getpid()))
    try:
        t = time.time()
        data, leak, stats = load_all(args.data_dir, args.seed)
        if args.limit_queries:
            for s in ("train", "dev"):
                data[s]["q"] = data[s]["q"][:args.limit_queries]
        popularity = Counter(k for q in data["train"]["q"] for k in q["rel"])
        lex = {}
        lex["dev"] = lexical_split("dev", data["dev"]["q"], data["dev"]["c"], args, None, popularity)
        lex["test"] = lexical_split("test", data["test"]["q"], data["test"]["c"], args, lex["dev"]["params"], popularity)
        lex["train"] = lexical_split("train", data["train"]["q"], data["train"]["c"], args, lex["dev"]["params"])
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        args.eos_id = tok.token_to_id("<eos>")
        windows = {}
        for s in SPLITS:
            tw = time.time()
            windows[s] = {"q": build_windows(data[s]["q"], tok, args, True), "c": build_windows(data[s]["c"], tok, args, False)}
            log(f"[{s}] windows built in {time.time() - tw:.0f}s; windows/query "
                f"{np.mean([len(w) for w in windows[s]['q']]):.1f}, windows/candidate {np.mean([len(w) for w in windows[s]['c']]):.1f}")
        slim = {s: {"q": [{"id": q["id"], "key": q["key"], "rel": q["rel"], "words": len(q["text"].split()),
                           "has_markers": any(r.search(q["text"]) for r in CITE_RES)} for q in data[s]["q"]],
                    "c": [{"id": c["id"], "key": c["key"]} for c in data[s]["c"]]} for s in SPLITS}
        bundle = {"lex": lex, "windows": windows, "data": slim, "leak": leak, "stats": stats, "key": key,
                  "minutes": (time.time() - t) / 60}
        with open(str(path) + ".tmp", "wb") as f:
            pickle.dump(bundle, f, protocol=4)
        os.replace(str(path) + ".tmp", path)
        log(f"Prepared cache {path} in {(time.time() - t) / 60:.1f} min")
        return bundle
    finally:
        if lock.exists():
            lock.unlink()


# ============================================================================= main

def run(args):
    t0 = time.time()
    from phase3_rr import locate_phase1_files, seed_everything
    seed_everything(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"p7_{args.model}_{args.method}"
    out = Path(args.output_dir or f"/kaggle/working/{name}")
    out.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    checkpoint, tokenizer_path, artifact = locate_phase1_files(args.model, args.artifact_zip, out / "phase1")
    bundle = prepare(args, tokenizer_path)
    if args.prepare_only:
        log("prepare-only: done")
        return
    lex, win, data = bundle["lex"], bundle["windows"], bundle["data"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    pad_id, args.eos_id = tok.token_to_id("<pad>"), tok.token_to_id("<eos>")

    # ---------------- train structures
    tkeys = lex["train"]["cand_keys"]
    tindex = {k: j for j, k in enumerate(tkeys)}
    tr_queries, tr_qwin = [], []
    for i, q in enumerate(data["train"]["q"]):
        pos = [tindex[k] for k in q["rel"] if k in tindex and k != q["key"]]
        if not pos:
            continue
        ps = set(pos)
        hn = [int(j) for j in lex["train"]["lists"][i] if int(j) not in ps]
        tr_queries.append({"key": q["key"], "pos": pos, "pos_set": ps, "hn": hn})
        tr_qwin.append(win["train"]["q"][i])
    tr = {"queries": tr_queries, "q_windows": tr_qwin, "c_windows": win["train"]["c"], "cand_keys": tkeys}
    dev = {"q_windows": win["dev"]["q"], "c_windows": win["dev"]["c"], "lists": lex["dev"]["lists"],
           "cand_keys": lex["dev"]["cand_keys"], "gold": lex["dev"]["gold"]}
    log(f"train queries with >=1 positive in the train pool: {len(tr_queries)}; positives "
        f"{sum(len(q['pos']) for q in tr_queries)}; dev queries {len(dev['q_windows'])}")

    teacher = None
    if args.method == "distill":
        teacher = torch.load(args.teacher_scores, map_location="cpu", weights_only=False)
        log(f"KD: teacher {teacher.get('_teacher')} scores for {len(teacher) - 1} train queries")
        teacher = {k: v for k, v in teacher.items() if not k.startswith("_")}

    # ---------------- model
    from phase2_lner import load_phase1, make_student
    from phase4_cjpe import apply_method
    backbone, phase1_cfg, phase1_saved = load_phase1(checkpoint, args.model)
    phase1_params = sum(p.numel() for p in backbone.parameters())
    if args.method == "distill":
        if args.model != "decoder":
            raise ValueError("The distillation student recipe uses the decoder Phase 1 model (cut to fewer layers)")
        backbone = make_student(backbone, layers=args.student_layers)
    model = PCRModel(backbone, args.model, args.proj_dim, args.pooling, args.grad_ckpt)
    apply_method(model, args.model, args.method, args.rank, args.unfreeze_layers)
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {"run": name, "model": args.model, "method": args.method, "pooling": args.pooling,
               "phase1_artifact": artifact, "phase1_parameters": phase1_params, "parameters": total,
               "trainable_parameters": trainable, "trainable_fraction": trainable / total,
               "leakage_check": bundle["leak"], "data_stats": bundle["stats"], "lexical_dev": lex["dev"]["report"],
               "bm25_params": lex["dev"]["params"], "device": str(device), "config": vars(args)}
    log(json.dumps({k: v for k, v in summary.items() if k not in ("data_stats", "config")}, indent=1, default=str))

    best, history, seconds = train(model, tr, dev, args, device, pad_id, teacher)
    (out / "training_log.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    # ---------------- dense scores for the full dev / test candidate lists
    inf_t = time.time()
    dense_dev = dense_list_scores(model, win["dev"]["q"], win["dev"]["c"], lex["dev"]["lists"], pad_id, device)
    dev_sec = time.time() - inf_t
    inf_t = time.time()
    dense_test = dense_list_scores(model, win["test"]["q"], win["test"]["c"], lex["test"]["lists"], pad_id, device)
    test_sec = time.time() - inf_t
    np.savez_compressed(out / "dense_scores.npz", dev=dense_dev, test=dense_test, cache_key=bundle["key"],
                        dev_lists=lex["dev"]["lists"], test_lists=lex["test"]["lists"])

    if args.export_teacher:
        ex = time.time()
        lists = []
        for q in tr["queries"]:
            lists.append(list(dict.fromkeys(q["pos"] + q["hn"][:args.hn_pool])))
        maxlen = max(len(x) for x in lists)
        padded = [x + [x[0]] * (maxlen - len(x)) for x in lists]
        S = dense_list_scores(model, tr["q_windows"], tr["c_windows"], padded, pad_id, device) * float(model.scale())
        scores = {"_teacher": name}
        for q, row, x in zip(tr["queries"], S, lists):
            scores[q["key"]] = {tkeys[j]: float(v) for j, v in zip(x, row[:len(x)])}
        torch.save(scores, out / "teacher_scores.pt")
        log(f"Saved teacher scores for {len(lists)} train queries in {(time.time() - ex) / 60:.1f} min")

    chosen, report, test_scores, dev_scores, importances = choose_fusion(lex["dev"], lex["test"], [dense_dev], [dense_test], args, 1)
    archive = write_submission(lex["test"], test_scores, [q["id"] for q in data["test"]["q"]], out, name)
    try:
        per, details = analysis(lex["dev"], dev_scores, data["dev"]["q"], report)
        (out / "dev_predictions.json").write_text(json.dumps(per), encoding="utf-8")
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        details = {"analysis_error": repr(error)}
    if args.save_checkpoint:
        torch.save({"model": {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                    "phase1_config": asdict(phase1_cfg), "run_config": vars(args)}, out / "best_fp16.pt")
    manifest = {**summary, "selected_checkpoint": best.get("tag"), "dense_rerank_dev_f1@7_eval_subset": best["f1"],
                "fusion_chosen": chosen, "dev_micro_f1@7_crossfit": report[chosen]["f1"],
                "dev_micro_precision@7_crossfit": report[chosen]["precision"],
                "dev_micro_recall@7_crossfit": report[chosen]["recall"], "fusion_report": report,
                "fusion_top_features": importances, "analysis": details, "train_minutes": seconds / 60,
                "dev_inference_seconds": dev_sec, "test_inference_seconds": test_sec,
                "test_queries_per_second": len(data["test"]["q"]) / max(1e-6, test_sec),
                "total_minutes": (time.time() - t0) / 60,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
                "phase1_steps": phase1_saved.get("total_steps"), "submission": str(archive)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(json.dumps({k: manifest[k] for k in ("run", "method", "trainable_parameters", "fusion_chosen",
                                             "dev_micro_f1@7_crossfit", "dense_rerank_dev_f1@7_eval_subset",
                                             "total_minutes")}, indent=1))
    log("SUBMISSION ZIP:", archive)


def fuse_runs(args):
    """Ensemble: lexical features + the dense scores of several finished runs -> one fusion model.
    --fuse-runs takes run folders, *.npz files, or folders searched recursively for *dense_scores.npz.
    Runs may come from different notebooks/accounts and window settings: they only need the same lexical
    candidate lists (checked array by array) - the lexical stage is deterministic."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for r in args.fuse_runs:
        r = Path(r)
        if r.is_file() and r.suffix == ".npz":
            files.append(r)
        elif (r / "dense_scores.npz").exists():
            files.append(r / "dense_scores.npz")
        elif r.is_dir():
            files += sorted(Path(f) / n for f, _, ns in os.walk(r, followlinks=True) for n in ns
                            if n.endswith(".npz") and "dense_scores" in n)
    seen, uniq = set(), []
    for f in files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append(f)
    files = uniq
    if len(files) < 2:
        raise RuntimeError(f"Need >= 2 dense_scores.npz files, found {files}")
    loaded = [np.load(f) for f in files]
    caches = sorted(Path(f) / n for f, _, ns in os.walk(args.cache_dir, followlinks=True) for n in ns
                    if n.startswith("p7_cache_") and n.endswith(".pkl"))
    if not caches:
        raise FileNotFoundError(f"no p7_cache_*.pkl under {args.cache_dir} - run the prepare cell first")
    bundle = None
    for c in caches:
        with open(c, "rb") as fh:
            b = pickle.load(fh)
        if np.array_equal(b["lex"]["dev"]["lists"], loaded[0]["dev_lists"]) and \
                np.array_equal(b["lex"]["test"]["lists"], loaded[0]["test_lists"]):
            bundle = b
            break
    if bundle is None:
        raise RuntimeError("no cache has the same lexical candidate lists as the runs")
    lex, data = bundle["lex"], bundle["data"]
    keep, names = [], []
    for f, d in zip(files, loaded):
        if np.array_equal(d["dev_lists"], lex["dev"]["lists"]) and np.array_equal(d["test_lists"], lex["test"]["lists"]):
            keep.append(d)
            names.append(str(f))
        else:
            log(f"SKIPPING {f}: different candidate lists")
    log(f"fusing {len(keep)} runs:", json.dumps(names, indent=1))
    chosen, report, test_scores, dev_scores, imp = choose_fusion(lex["dev"], lex["test"], [d["dev"] for d in keep],
                                                                 [d["test"] for d in keep], args, len(keep))
    name = args.run_name or "p7_ENSEMBLE"
    archive = write_submission(lex["test"], test_scores, [q["id"] for q in data["test"]["q"]], out, name)
    (out / "manifest.json").write_text(json.dumps({"run": name, "method": "ensemble", "submission": str(archive),
                                                   "members": names, "fusion_chosen": chosen,
                                                   "dev_micro_f1@7_crossfit": report[chosen]["f1"], "fusion_report": report,
                                                   "fusion_top_features": imp}, indent=1, default=str), encoding="utf-8")
    log("SUBMISSION ZIP:", archive)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficient", choices=("decoder", "encoder_decoder", "hybrid", "efficient"))
    p.add_argument("--method", default="full", choices=("full", "lorma", "lora", "ia3", "partial", "bitfit", "distill"))
    p.add_argument("--pooling", default="attn", choices=("attn", "maxsim"))
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--max-words", type=int, default=15000)
    p.add_argument("--query-windows", default="citation", choices=("citation", "spread"))
    p.add_argument("--context-sentences", type=int, default=2)
    p.add_argument("--query-head-windows", type=int, default=1)
    p.add_argument("--train-q-windows", type=int, default=4)
    p.add_argument("--train-c-windows", type=int, default=3)
    p.add_argument("--batch-queries", type=int, default=4)
    p.add_argument("--hard-negatives", type=int, default=3)
    p.add_argument("--hn-pool", type=int, default=50)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--unfreeze-layers", type=int, default=4)
    p.add_argument("--student-layers", type=int, default=8)
    p.add_argument("--teacher-scores")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--export-teacher", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--time-budget-min", type=float, default=60.0)
    p.add_argument("--eval-every-min", type=float, default=15.0)
    p.add_argument("--eval-queries", type=int, default=250)
    p.add_argument("--limit-queries", type=int, default=0)
    p.add_argument("--cpu-jobs", type=int, default=4)
    p.add_argument("--cache-dir", default="/kaggle/working/p7_cache")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--fuse-runs", nargs="*")
    p.add_argument("--bagged-fusion", action="store_true")
    p.add_argument("--artifact-zip")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--run-name")
    p.add_argument("--seed", type=int, default=230338)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.fuse_runs:
        fuse_runs(a)
    else:
        run(a)
