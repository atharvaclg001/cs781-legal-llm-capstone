"""
lsi_utils.py  -  shared code for Phase 5 (Legal Statute Identification, LSI). No PyTorch needed.

Every Phase-5 experiment uses these functions, so all of them see exactly the same data, cleaning, scorer,
thresholds and submission writer. Only the model / training differ.

What P5_01 (data exploration) found, and what this file does about it:
  * 100 statute IDs (0..99), 3.7 per case on average, every case has >= 1   -> multi-hot label matrix (n, 100);
                                                                               a prediction always keeps >= 1 ID
  * statute mentions are masked (<SECTION>, <ACT>), names are <ENTITY>      -> placeholders kept as they are
  * train contains identical texts; 101 groups with DIFFERENT label sets    -> train: keep one copy of each text,
                                                                               drop groups with conflicting labels;
                                                                               dev / test untouched
  * text is cased (like the Phase-1 data), a few HTML tags, signature stamps -> Phase-1 cleaning + HTML tags out,
                                                                               NO lower-casing (unlike Phase 4)
  * 64-69% of cases fit in 1536 tokens, no single part of a case dominates  -> select_tokens (cjpe_utils):
                                                                               first 768 + last 768 by default
  * metric = macro-F1 over the 100 statutes                                 -> thresholds tuned on dev
                                                                               (one global, or one per statute)
"""

import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ner_utils import CHAR_MAP, find_project_root  # noqa: F401  (same cleaning table + root finder as Phases 2-4)
from cjpe_utils import select_tokens, plan_chunks   # noqa: F401  (re-used: which part of a case the model reads)

CLEAN_VERSION = "v1"      # part of the cache file names: change it if clean_lsi changes
N_LABELS = 100            # statute IDs 0..99 (checked in P5_01: all 100 occur in train and dev)


# ----------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------
def lsi_dir(root):
    return Path(root) / "data" / "raw" / "Phase_5_LSI"


def text_key(t):
    """Fingerprint of a case: case- and whitespace-insensitive (same as P5_01)."""
    return hashlib.md5(re.sub(r"\s+", " ", t).strip().lower().encode()).hexdigest()


def load_lsi(root):
    """Returns (data, info). data = {"train": [...], "dev": [...], "test": [...]}, each a list of
    {"id": str, "text": str, "labels": sorted list of int (None for test)}.
    train: one copy per identical text; texts whose copies have DIFFERENT label sets are dropped completely.
    dev / test: kept exactly as given (they are what we are scored on). Test labels (if any) are removed."""
    files = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}
    data, info = {}, {}
    for split, rel in files.items():
        with open(lsi_dir(root) / rel, encoding="utf-8") as f:
            raw = json.load(f)
        docs = [{"id": str(d["id"]), "text": d["text"],
                 "labels": None if split == "test" else sorted({int(x) for x in d["labels"]})} for d in raw]
        rec = {"records": len(docs)}
        if split == "train":
            groups = {}
            for d in docs:
                groups.setdefault(text_key(d["text"]), []).append(d)
            kept, n_dup, n_conf = [], 0, 0
            for g in groups.values():
                if len({tuple(d["labels"]) for d in g}) > 1:
                    n_conf += 1                      # same text, different statutes -> contradictory, drop all copies
                    continue
                kept.append(g[0])
                n_dup += len(g) - 1
            order = {d["id"]: i for i, d in enumerate(docs)}
            docs = sorted(kept, key=lambda d: order[d["id"]])      # keep the original file order
            rec.update(removed_duplicate_copies=n_dup, dropped_conflicting_groups=n_conf)
        ids = [d["id"] for d in docs]
        assert len(ids) == len(set(ids)), f"{split}: duplicate ids"
        if split != "test":
            assert all(d["labels"] for d in docs), f"{split}: a case without labels"
            assert all(0 <= l < N_LABELS for d in docs for l in d["labels"]), f"{split}: label outside 0..{N_LABELS - 1}"
            rec["labels_per_case"] = round(float(np.mean([len(d["labels"]) for d in docs])), 3)
        rec["kept"] = len(docs)
        data[split], info[split] = docs, rec
    return data, info


def label_matrix(docs, n_labels=N_LABELS):
    """Multi-hot matrix: Y[i, l] = 1 if statute l applies to case i."""
    Y = np.zeros((len(docs), n_labels), dtype=np.float32)
    for i, d in enumerate(docs):
        Y[i, d["labels"]] = 1.0
    return Y


# ----------------------------------------------------------------------------------------
# 2. Cleaning + tokenisation (cached)
# ----------------------------------------------------------------------------------------
SIG_RE = re.compile(r"Signature Not Verified.{0,300}?Reason:", re.S)
HTML_RE = re.compile(r"</?(?:span|font|p|br|div|b|i|u|sup|sub|strong|em)\b[^<>]{0,200}>", re.I)


def clean_lsi(t):
    """Phase-1 cleaning (cased, as in Phase 1):
    1. remove PDF signature stamps and a few HTML tags (</span>, </font>, <p> ...);
       the placeholders <ENTITY>, <SECTION>, <ACT> are NOT tags of this list, so they stay
    2. special characters -> ASCII (CHAR_MAP), other non-ASCII dropped, all whitespace -> one space"""
    t = SIG_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)
    t = "".join(CHAR_MAP.get(c, c) for c in t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip()


def encode_split(root, split, docs, tokenizer, batch=256):
    """Token ids of every case (cleaned, with a leading space as in running text), as a list of uint16 arrays.
    Cached in data/processed/phase5/<split>_<CLEAN_VERSION>.npz (one long array + start offsets)."""
    cache = Path(root) / "data" / "processed" / "phase5" / f"{split}_{CLEAN_VERSION}.npz"
    ids_order = [d["id"] for d in docs]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["doc_ids"]) == ids_order:
            flat, starts = z["ids"], z["starts"]
            return [flat[starts[i]:starts[i + 1]] for i in range(len(docs))]
        print(f"[cache] {cache.name} belongs to different documents -> re-tokenising")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for i in tqdm(range(0, len(docs), batch), desc=f"cleaning + tokenising {split}"):
        texts = [" " + clean_lsi(d["text"]) for d in docs[i:i + batch]]
        out += [np.asarray(e.ids, dtype=np.uint16) for e in tokenizer.encode_batch(texts, add_special_tokens=False)]
    starts = np.zeros(len(out) + 1, dtype=np.int64)
    np.cumsum([len(x) for x in out], out=starts[1:])
    np.savez(cache, ids=np.concatenate(out), starts=starts, doc_ids=np.array(ids_order))
    return out


# ----------------------------------------------------------------------------------------
# 3. Scoring (same definitions as sklearn f1_score on the multi-hot matrix; checked in P5_01)
# ----------------------------------------------------------------------------------------
def per_label_counts(Y, P):
    """Y, P: (n, L) 0/1 arrays (gold, predicted) -> TP, FP, FN per statute, each (L,)."""
    Y, P = Y.astype(bool), P.astype(bool)
    return (Y & P).sum(0), (~Y & P).sum(0), (Y & ~P).sum(0)


def ml_scores(Y, P):
    """macro-F1 (mean over statutes that occur in gold), micro-F1 / P / R, subset accuracy, labels per case.
    F1 of one statute = 2 TP / (2 TP + FP + FN)."""
    tp, fp, fn = per_label_counts(Y, P)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    present = Y.sum(0) > 0
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    return {"macro_f1": round(float(f1[present].mean()), 4),
            "micro_f1": round(float(2 * TP / max(2 * TP + FP + FN, 1)), 4),
            "micro_p": round(float(TP / max(TP + FP, 1)), 4), "micro_r": round(float(TP / max(TP + FN, 1)), 4),
            "macro_p": round(float(np.mean(np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)[present])), 4),
            "macro_r": round(float(np.mean((tp / np.maximum(tp + fn, 1))[present])), 4),
            "subset_acc": round(float((Y.astype(bool) == P.astype(bool)).all(1).mean()), 4),
            "labels_per_case": round(float(P.sum(1).mean()), 3)}


def per_label_f1(Y, P):
    tp, fp, fn = per_label_counts(Y, P)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


# ----------------------------------------------------------------------------------------
# 4. From probabilities to statute sets: thresholds
# ----------------------------------------------------------------------------------------
GRID = np.round(np.arange(0.02, 0.951, 0.01), 2)


def decide(prob, thr, at_least_one=True):
    """prob (n, L) sigmoid probabilities, thr = one number or (L,) per-statute thresholds -> 0/1 matrix.
    at_least_one: every case has >= 1 statute (true for all train/dev cases) -> if nothing passes the
    threshold, keep the single most probable statute (an empty prediction can only lose)."""
    P = (prob >= np.asarray(thr)).astype(np.uint8)
    if at_least_one:
        empty = P.sum(1) == 0
        P[empty, prob[empty].argmax(1)] = 1
    return P


def tune_global(Y, prob, grid=GRID):
    """One threshold for all statutes, chosen for the best macro-F1 on (Y, prob)."""
    best = max((ml_scores(Y, decide(prob, t))["macro_f1"], t) for t in grid)
    return float(best[1]), best[0]


def tune_per_label(Y, prob, grid=GRID, min_pos=1, fallback=None):
    """One threshold per statute: for statute l, the threshold with the best F1 of that statute alone.
    Statutes with fewer than `min_pos` positives keep `fallback` (default: the tuned global threshold).
    Vectorised: for each grid value, TP/FP/FN of all statutes at once."""
    Yb = Y.astype(bool)
    f1 = np.zeros((len(grid), Y.shape[1]))
    for k, t in enumerate(grid):
        P = prob >= t
        tp, fp, fn = (Yb & P).sum(0), (~Yb & P).sum(0), (Yb & ~P).sum(0)
        f1[k] = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    thr = grid[f1.argmax(0)].astype(float)
    if fallback is None:
        fallback = tune_global(Y, prob, grid)[0]
    thr[Y.sum(0) < min_pos] = fallback
    return thr


def cross_fitted_scores(Y, prob, folds=2, seed=0):
    """Honest dev estimate of both threshold strategies: tune on one half of dev, score on the other half
    (and the other way round). Tuning 100 thresholds and scoring on the SAME cases would be optimistic.
    Returns {'global': scores, 'per_label': scores} on the out-of-fold decisions."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(Y)) % folds
    P_glob = np.zeros_like(Y, dtype=np.uint8)
    P_lab = np.zeros_like(Y, dtype=np.uint8)
    for f in range(folds):
        tr, te = fold != f, fold == f
        t_g, _ = tune_global(Y[tr], prob[tr])
        t_l = tune_per_label(Y[tr], prob[tr], min_pos=5, fallback=t_g)
        P_glob[te] = decide(prob[te], t_g)
        P_lab[te] = decide(prob[te], t_l)
    return {"global": ml_scores(Y, P_glob), "per_label": ml_scores(Y, P_lab)}


# ----------------------------------------------------------------------------------------
# 5. Analysis helpers
# ----------------------------------------------------------------------------------------
BUCKETS = [(1000, np.inf, ">=1000 train cases"), (300, 1000, "300-999"), (0, 300, "<300")]


def f1_by_frequency(Y, P, train_counts):
    """Mean F1 of the statutes in each training-frequency bucket (rare-statute analysis)."""
    f1 = per_label_f1(Y, P)
    out = []
    for lo, hi, name in BUCKETS:
        m = (train_counts >= lo) & (train_counts < hi)
        if m.any():
            out.append({"bucket": name, "statutes": int(m.sum()), "mean_f1": round(float(f1[m].mean()), 4)})
    return out


def top_confusions(Y, P, k=10):
    """Wrongly predicted statute B in cases whose gold set contains A: count (A, B) pairs over all false
    positives -> which statutes are mistaken for which. Normalised by the number of cases with A."""
    Yb, Pb = Y.astype(bool), P.astype(bool)
    FPm = (~Yb & Pb).astype(np.float32)
    pair = Yb.astype(np.float32).T @ FPm            # (L, L): [A, B] = cases with gold A and false positive B
    np.fill_diagonal(pair, 0)
    flat = np.argsort(-pair, axis=None)[:k]
    n_a = Yb.sum(0)
    return [{"gold": int(a), "wrongly_added": int(b), "cases": int(pair[a, b]),
             "share_of_gold_cases": round(float(pair[a, b] / max(n_a[a], 1)), 3)}
            for a, b in zip(*np.unravel_index(flat, pair.shape))]


# ----------------------------------------------------------------------------------------
# 6. Submission
# ----------------------------------------------------------------------------------------
def write_submission(ids, P, out_dir):
    """prediction.json = [{"id": ..., "pred": [statute IDs]}, ...] at the root of submission.zip.
    P: (n, 100) 0/1 matrix -> sorted unique integer IDs per case."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"id": str(i), "pred": [int(l) for l in np.flatnonzero(p)]} for i, p in zip(ids, P)]
    (out_dir / "prediction.json").write_text(json.dumps(rows))
    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", arcname="prediction.json")
    return zip_path


def check_submission(zip_path, test_ids, verbose=True):
    """Re-open the zip and check: prediction.json at the root, exactly one entry per test id,
    pred = non-empty list of unique ints in 0..99."""
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        rows = json.loads(z.read("prediction.json"))
    assert isinstance(rows, list)
    got = [r["id"] for r in rows]
    assert len(got) == len(set(got)), "duplicate ids"
    assert set(got) == set(map(str, test_ids)), "ids differ from test.json"
    for r in rows:
        p = r["pred"]
        assert isinstance(p, list) and len(p) > 0, f"empty / non-list pred for {r['id']}"
        assert all(type(x) is int and 0 <= x < N_LABELS for x in p), f"invalid ID in {r['id']}"
        assert len(p) == len(set(p)), f"duplicate IDs in {r['id']}"
    if verbose:
        k = np.mean([len(r["pred"]) for r in rows])
        print(f"submission OK: {len(rows)} predictions, {k:.2f} statutes per case -> {zip_path}")
    return True


# ----------------------------------------------------------------------------------------
# 7. Experiment log
# ----------------------------------------------------------------------------------------
def log_path(root):
    return Path(root) / "experiments" / "phase5" / "experiments.jsonl"


def log_experiment(root, record):
    p = log_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_experiments(root):
    p = log_path(root)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
