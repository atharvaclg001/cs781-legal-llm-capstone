"""
cjpe_utils.py  -  shared code for Phase 4 (Court Judgment Prediction, CJPE). No PyTorch needed.

Every Phase-4 notebook uses these functions, so all experiments see exactly the same data, the same
cleaning, the same scorer and the same submission writer. Only the model / training differ.

What P4_01 (data exploration) found, and what this file does about it:
  * train has 5,735 exact duplicate texts (ILDC_single is a subset of ILDC_multi), dev lists every case
    twice                                   -> keep the first copy of every text (train 31,652, dev 994)
  * test.json contains labels               -> dropped on load; never used
  * train is cased modern text without line breaks; dev/test are lower-case old cases with hard line
    wraps ("peti-\\ntion")                  -> lower-case EVERY split and repair the wraps, so train and
                                               test look the same to the model
  * cases are long (median ~2.6k-3.5k tokens) and the end of a case carries most of the signal
                                            -> `select_tokens` cuts a case to the part the model reads
"""

import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ner_utils import CHAR_MAP, find_project_root  # noqa: F401  (same cleaning table + root finder as Phases 2-3)

CLEAN_VERSION = "lower_v1"      # part of the cache file names: change it if clean_case changes


# ----------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------
def cjpe_dir(root):
    return Path(root) / "data" / "raw" / "Phase_4_CJPE"


def text_key(t):
    """Fingerprint of a case: case- and whitespace-insensitive (same as P4_01)."""
    return hashlib.md5(re.sub(r"\s+", " ", t).strip().lower().encode()).hexdigest()


def load_cjpe(root):
    """Returns (data, info). data = {"train": [...], "dev": [...], "test": [...]}, each a list of
    {"id": str, "text": str, "label": 0/1 or None}. Exact duplicate texts are removed (first copy kept);
    test labels are removed."""
    files = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}
    data, info = {}, {}
    for split, rel in files.items():
        with open(cjpe_dir(root) / rel, encoding="utf-8") as f:
            raw = json.load(f)
        seen, docs = set(), []
        for d in raw:
            h = text_key(d["text"])
            if split != "test" and h in seen:
                continue                     # exact duplicate text (always with the same label, checked in P4_01)
            seen.add(h)
            lab = None if split == "test" else int(d["label"])
            docs.append({"id": str(d["id"]), "text": d["text"], "label": lab})
        ids = [d["id"] for d in docs]
        assert len(ids) == len(set(ids)), f"{split}: duplicate ids after removing duplicate texts"
        if split != "test":
            assert set(d["label"] for d in docs) <= {0, 1}
        data[split] = docs
        info[split] = {"records": len(raw), "kept": len(docs), "removed_duplicates": len(raw) - len(docs),
                       "label_1_rate": None if split == "test" else round(float(np.mean([d["label"] for d in docs])), 4)}
    return data, info


# ----------------------------------------------------------------------------------------
# 2. Cleaning + tokenisation (cached)
# ----------------------------------------------------------------------------------------
# PDF signature stamps in train look like "Signature Not Verified ... Digitally signed by NAME ... Date 2020.02.03
# 165132 IST Reason" and are mixed into the running text by the PDF extraction, so we delete the fixed pieces only.
SIG_RE = re.compile(r"signature not verified|digitally signed by|date \d{4}\.\d{2}\.\d{2} \d{6} ist reason:?")
WRAP_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")     # "peti-\ntion" -> "petition"


def clean_case(t):
    """Lower-case + Phase-1 cleaning.
    1. lower-case (dev/test are lower-case; train is cased -> make train look like test)
    2. remove the fixed pieces of PDF signature stamps (only in train, ~5% of cases)
    3. repair words broken at a line end: letter + '-' + newline + letter -> joined
    4. special characters -> ASCII (CHAR_MAP), other non-ASCII dropped, all whitespace -> one space"""
    t = t.lower()
    t = SIG_RE.sub(" ", t)
    t = WRAP_RE.sub("", t)
    t = "".join(CHAR_MAP.get(c, c) for c in t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip()


def encode_split(root, split, docs, tokenizer, batch=256):
    """Token ids of every case (cleaned, with a leading space as in running text), as a list of uint16 arrays.
    Cached in data/processed/phase4/<split>_<CLEAN_VERSION>.npz (one long array + start offsets)."""
    cache = Path(root) / "data" / "processed" / "phase4" / f"{split}_{CLEAN_VERSION}.npz"
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
        texts = [" " + clean_case(d["text"]) for d in docs[i:i + batch]]
        out += [np.asarray(e.ids, dtype=np.uint16) for e in tokenizer.encode_batch(texts, add_special_tokens=False)]
    starts = np.zeros(len(out) + 1, dtype=np.int64)
    np.cumsum([len(x) for x in out], out=starts[1:])
    np.savez(cache, ids=np.concatenate(out), starts=starts, doc_ids=np.array(ids_order))
    return out


# ----------------------------------------------------------------------------------------
# 3. Which part of a case the model reads
# ----------------------------------------------------------------------------------------
def select_tokens(ids, mode="last", max_len=1536):
    """Cut one case to at most max_len tokens.
    'last'      : the last max_len tokens (the end holds the court's reasoning; best section in P4_01)
    'first'     : the first max_len tokens (facts, parties, lower-court history)
    'first_last': first half + last half of the budget (both ends)"""
    if len(ids) <= max_len:
        return ids
    if mode == "last":
        return ids[-max_len:]
    if mode == "first":
        return ids[:max_len]
    if mode == "first_last":
        h = max_len // 2
        return np.concatenate([ids[:h], ids[-(max_len - h):]])
    raise ValueError(mode)


def plan_chunks(n, max_len=1536, stride=1024):
    """Overlapping windows covering a whole case of n tokens: list of (start, end); last one aligned to the end."""
    if n <= max_len:
        return [(0, n)]
    starts = list(range(0, n - max_len, stride)) + [n - max_len]
    return [(s, s + max_len) for s in starts]


# ----------------------------------------------------------------------------------------
# 4. Scoring
# ----------------------------------------------------------------------------------------
def f1_scores(gold, pred):
    """Macro-F1 / macro precision / macro recall over the classes 0 and 1 (= sklearn average='macro')."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    per = {}
    for c in (0, 1):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[c] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    return {"macro_f1": round((per[0][2] + per[1][2]) / 2, 4), "macro_p": round((per[0][0] + per[1][0]) / 2, 4),
            "macro_r": round((per[0][1] + per[1][1]) / 2, 4), "f1_0": round(per[0][2], 4), "f1_1": round(per[1][2], 4),
            "acc": round(float((gold == pred).mean()), 4), "pred_1_rate": round(float(pred.mean()), 4)}


def confusion(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    return {"TN": int(((gold == 0) & (pred == 0)).sum()), "FP": int(((gold == 0) & (pred == 1)).sum()),
            "FN": int(((gold == 1) & (pred == 0)).sum()), "TP": int(((gold == 1) & (pred == 1)).sum())}


def best_threshold(gold, p1, grid=np.arange(0.05, 0.951, 0.01)):
    """Decision threshold on P(label 1) with the best macro-F1 (chosen on dev; one number -> little overfitting).
    Needed because train has 41% label 1 but dev/test have 50%: a model trained on train is biased towards 0."""
    best = max(((f1_scores(gold, (p1 >= t).astype(int))["macro_f1"], round(float(t), 2)) for t in grid))
    return best[1], best[0]


# ----------------------------------------------------------------------------------------
# 5. Submission
# ----------------------------------------------------------------------------------------
def write_submission(ids, preds, out_dir):
    """prediction.json = [{"id": ..., "pred": 0/1}, ...] at the root of submission.zip."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"id": str(i), "pred": int(p)} for i, p in zip(ids, preds)]
    (out_dir / "prediction.json").write_text(json.dumps(rows))
    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", arcname="prediction.json")
    return zip_path


def check_submission(zip_path, test_ids, verbose=True):
    """Re-open the zip and check: prediction.json at the root, one entry per test id, pred is int 0/1."""
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        rows = json.loads(z.read("prediction.json"))
    assert isinstance(rows, list)
    got = [r["id"] for r in rows]
    assert len(got) == len(set(got)), "duplicate ids"
    assert set(got) == set(map(str, test_ids)), "ids differ from test.json"
    assert all(type(r["pred"]) is int and r["pred"] in (0, 1) for r in rows), "pred must be int 0/1"
    if verbose:
        print(f"submission OK: {len(rows)} predictions, {sum(r['pred'] for r in rows)} x label 1 -> {zip_path}")
    return True


# ----------------------------------------------------------------------------------------
# 6. Experiment log
# ----------------------------------------------------------------------------------------
def log_path(root):
    return Path(root) / "experiments" / "phase4" / "experiments.jsonl"


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
