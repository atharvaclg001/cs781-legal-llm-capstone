"""
pcr_utils.py  -  shared code for Phase 7 (Prior Case Retrieval, PCR). No PyTorch needed.

Task: for every query case, rank the cases of the SAME split's candidate pool and submit the top 7.
Metric: Micro-F1@7 (hits are counted over all queries together, see `micro_prf_at_k`).

Every Phase-7 notebook uses these functions, so data loading, metric, lexical baseline and submission writer are
identical for all experiments. Only the neural model / training differ.

Contents
  1. data:      pcr_dir, find_files, load_pcr (robust to the file format; ids kept as STRINGS), text_key
  2. cleaning:  clean_pcr (Phase-1 cleaning, cased, fast str.translate version), placeholders
  3. metric:    micro_prf_at_k, oracle_micro_f1, recall_at_k, gold_ranks, top_k
  4. BM25:      bm25_tokens, BM25 (own inverted index in numpy; k1 / b can be changed without rebuilding)
  5. years:     doc_year (latest plausible year written in a case = rough judgment date)
  6. output:    write_submission + check_submission (dict {query_id: [7 candidate ids]} in prediction.json)
"""

import collections
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ner_utils import CHAR_MAP, find_project_root  # noqa: F401  (same cleaning table + root finder as Phases 2-6)

CLEAN_VERSION = "v1"      # part of cache file names: change it if clean_pcr changes
TOP_K = 7                 # the doc: exactly 7 unique candidates per query, best first
SPLITS = ("train", "dev", "test")

ID_KEYS = ["id", "query_id", "qid", "case_id", "candidate_id", "doc_id"]
TEXT_KEYS = ["text", "sentences", "document", "content", "case_text", "judgment", "judgement"]
REL_KEYS = ["relevant_candidates", "relevant", "relevant_cases", "gold", "labels", "positives", "cited_cases"]


# ----------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------
def pcr_dir(root):
    return Path(root) / "data" / "raw" / "Phase_7_PCR"


def text_key(t):
    """Fingerprint of a text: case- and whitespace-insensitive (same idea as Phases 4-6)."""
    return hashlib.md5(re.sub(r"\s+", " ", t).strip().lower().encode()).hexdigest()


def find_files(root):
    """{"train": .../train.json, "train_candidates": .../train_candidates.json, "dev": ..., ...}.
    The doc says Train/train.json + Train/train_candidates.json etc.; we search in case names differ."""
    base = pcr_dir(root)
    found = {}
    for split in SPLITS:
        files = sorted(base.rglob(f"{split}*.json"))
        q = [p for p in files if "cand" not in p.name.lower()]
        c = [p for p in files if "cand" in p.name.lower()]
        if q:
            found[split] = q[0]
        if c:
            found[f"{split}_candidates"] = c[0]
    return found


def _norm_id(x):
    """IDs are IndianKanoon numbers written with leading zeros ("0000021652"). They must stay STRINGS:
    int("0000021652") = 21652 would no longer match the evaluator's ids."""
    return str(x).strip()


def _as_sentences(x):
    """The doc: 'the text field contains the case document divided into sentences' -> a list of strings.
    Also accepts one string (-> one piece) or a list of lists (paragraphs of sentences -> flattened)."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    out = []
    for s in x:
        if isinstance(s, (list, tuple)):
            out.extend(str(t) for t in s)
        else:
            out.append(str(s))
    return out


def _records(raw):
    """A file can be a list of records or a dict {id: record} / {id: text} / {id: [sentences]}."""
    if isinstance(raw, list):
        return [dict(r) if isinstance(r, dict) else {"text": r} for r in raw]
    if isinstance(raw, dict):
        # a wrapper like {"queries": [...]} or {"candidates": [...]}: one key holding the whole list
        if len(raw) == 1 and isinstance(next(iter(raw.values())), list) and \
                all(isinstance(v, dict) for v in next(iter(raw.values()))[:5]):
            return [dict(r) for r in next(iter(raw.values()))]
        out = []
        for k, v in raw.items():
            r = dict(v) if isinstance(v, dict) else {"text": v}
            r.setdefault("id", k)
            out.append(r)
        return out
    raise TypeError(f"unexpected JSON type {type(raw)}")


def _load_one(path, with_labels):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    recs = _records(raw)
    first = recs[0]
    ikey = next((k for k in ID_KEYS if k in first), None)
    tkey = next((k for k in TEXT_KEYS if k in first), None)
    rkey = next((k for k in REL_KEYS if k in first), None) if with_labels else None
    assert ikey and tkey, f"{Path(path).name}: no id / text field among {list(first.keys())}"
    docs = []
    for r in recs:
        rel = r.get(rkey) if rkey else None
        if isinstance(rel, (str, int)):
            rel = [rel]
        docs.append({"id": _norm_id(r[ikey]), "sents": _as_sentences(r.get(tkey)),
                     "rel": [_norm_id(c) for c in rel] if rel is not None else None})
    info = {"file": Path(path).name, "file_type": type(raw).__name__, "records": len(recs),
            "keys": list(first.keys()), "id_key": ikey, "text_key": tkey, "rel_key": rkey,
            "id_type_in_file": type(first[ikey]).__name__, "text_type": type(first[tkey]).__name__,
            "rel_type": type(first[rkey]).__name__ if rkey else None}
    return docs, info


def load_pcr(root=None, files=None):
    """Returns (data, info).
    data[split] = {"queries": [{"id", "sents", "rel"}], "pool": [{"id", "sents", "rel": None}]}
    Test labels (if the file has any) are REMOVED here and never used (Phase-4 lesson).
    If gold ids do not match pool ids only because of missing leading zeros, all ids of that split are
    zero-padded to the same width (reported in info)."""
    files = files or find_files(root)
    data, info = {}, {}
    for split in SPLITS:
        if split not in files or f"{split}_candidates" not in files:
            continue
        queries, qi = _load_one(files[split], with_labels=True)
        pool, pi = _load_one(files[f"{split}_candidates"], with_labels=False)
        rec = {"queries": qi, "pool": pi}
        if split == "test" and any(q["rel"] for q in queries):
            rec["test_had_labels"] = True          # dropped: must not be used
            for q in queries:
                q["rel"] = None
        # id format check: do gold ids exist in the pool as written?
        pool_ids = {d["id"] for d in pool}
        gold = [c for q in queries for c in (q["rel"] or [])]
        if gold:
            miss = sum(c not in pool_ids for c in gold) / len(gold)
            rec["gold_missing_from_pool_share"] = round(miss, 4)
            if miss > 0.5:
                width = max(len(i) for i in pool_ids | set(gold) | {q["id"] for q in queries})
                padded = {i.zfill(width) for i in pool_ids}
                miss_pad = sum(c.zfill(width) not in padded for c in gold) / len(gold)
                if miss_pad < miss:
                    for d in pool + queries:
                        d["id"] = d["id"].zfill(width)
                    for q in queries:
                        q["rel"] = [c.zfill(width) for c in q["rel"]] if q["rel"] is not None else None
                    rec["ids_zero_padded_to"] = width
                    rec["gold_missing_from_pool_share"] = round(miss_pad, 4)
        data[split], info[split] = {"queries": queries, "pool": pool}, rec
    return data, info


# ----------------------------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------------------------
SIG_RE = re.compile(r"Signature Not Verified.{0,300}?Reason:", re.S)
HTML_RE = re.compile(r"</?(?:span|font|p|br|div|b|i|u|sup|sub|strong|em)\b[^<>]{0,200}>", re.I)
PLACEHOLDER_RE = re.compile(r"<[A-Za-z_]{2,25}>|\[[A-Z_]{3,25}\]")
_TRANSLATE = str.maketrans(CHAR_MAP)      # one C-level pass instead of a Python loop over characters


def clean_pcr(t):
    """Phase-1 cleaning (cased), same result as Phase 6's clean_summ but faster:
    signature stamps + HTML tags out, special characters -> ASCII (CHAR_MAP), other non-ASCII dropped,
    all whitespace -> one space."""
    t = SIG_RE.sub(" ", t or "")
    t = HTML_RE.sub(" ", t)
    t = t.translate(_TRANSLATE)
    if not t.isascii():
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip()


def doc_text(d):
    """All sentences of a case as one string (raw, not cleaned)."""
    return " ".join(d["sents"])


# ----------------------------------------------------------------------------------------
# 3. Metric (Micro-F1@7) and ranking helpers
# ----------------------------------------------------------------------------------------
def micro_prf_at_k(pred, gold, k=TOP_K):
    """pred = {qid: ranked candidate ids}, gold = {qid: relevant candidate ids}.
    Only the first k predictions count. MICRO = add up over all queries first, then divide:
        hits = sum_q |top_k(q) & gold(q)|,  P = hits / (number of predictions),  R = hits / (number of gold ids),
        F1 = 2PR / (P + R).
    With exactly 7 predictions per query, P = hits / (7 * queries)."""
    hits = n_pred = n_gold = 0
    for q, g in gold.items():
        p = list(dict.fromkeys(pred.get(q, [])))[:k]        # unique, first k
        g = set(g)
        hits += len(set(p) & g)
        n_pred += len(p)
        n_gold += len(g)
    P = hits / n_pred if n_pred else 0.0
    R = hits / n_gold if n_gold else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"micro_f1": F, "micro_p": P, "micro_r": R, "hits": hits, "n_pred": n_pred, "n_gold": n_gold}


def oracle_micro_f1(gold, k=TOP_K):
    """Best possible score when we must submit exactly k ids: put all gold first (min(k, |gold|) hits each)."""
    hits = sum(min(k, len(set(g))) for g in gold.values())
    n_pred, n_gold = k * len(gold), sum(len(set(g)) for g in gold.values())
    P, R = hits / n_pred, hits / n_gold
    return {"micro_f1": 2 * P * R / (P + R), "micro_p": P, "micro_r": R}


def top_k(scores, cand_ids, k=TOP_K, exclude=None):
    """Indices -> ids of the k best candidates; `exclude` = ids that must never be returned (the query itself)."""
    order = np.argsort(-scores, kind="stable")
    out = []
    for j in order:
        c = cand_ids[j]
        if exclude and c in exclude:
            continue
        out.append(c)
        if len(out) == k:
            break
    return out


def gold_ranks(scores, cand_ids, gold_ids, exclude=None):
    """1-based rank of every gold candidate in the full ranking (np.inf if the gold id is not in the pool).
    Rank = 1 + number of (non-excluded) candidates with a strictly higher score."""
    pos = {c: j for j, c in enumerate(cand_ids)}
    s = scores.astype(np.float64).copy()
    if exclude:
        for c in exclude:
            if c in pos:
                s[pos[c]] = -np.inf
    out = []
    for g in gold_ids:
        if g not in pos:
            out.append(np.inf)
        else:
            out.append(1 + int((s > s[pos[g]]).sum()))
    return out


def recall_at_k(all_ranks, ks=(7, 20, 50, 100, 200, 500)):
    """Share of ALL gold pairs found within the top K (micro) - tells how deep a re-ranker must look."""
    r = np.asarray(all_ranks, dtype=float)
    return {f"R@{k}": round(float((r <= k).mean()), 4) for k in ks}


# ----------------------------------------------------------------------------------------
# 4. BM25 (classical lexical ranking, Robertson & Zaragoza 2009) - no learned weights, no pretrained model
# ----------------------------------------------------------------------------------------
STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its
this that these those he she they them his her their we our you your i not no so than then there here which who whom
whose what when where why how all any both each few more most other some such only own same too very can will shall
should would could may might must do does did done has have had having into out over under again further once also
upon said such per vide ie viz etc""".split())
_WORD_RE = re.compile(r"[a-z0-9]+")


def bm25_tokens(text):
    """Lower-case words and numbers, stop-words and 1-character tokens removed.
    Numbers are KEPT: section numbers ("302"), years and article numbers are strong legal signals."""
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS]


class BM25:
    """Inverted index stored as numpy arrays (no scipy needed).

    For every term t: the documents that contain it (post_doc) and how often (post_tf), stored term after term;
    term_ptr[t] .. term_ptr[t+1] is the slice of term t.  Score of document d for a query:
        score(d) = sum over query terms t in d of  idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avg_len))
        idf(t)   = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))           (always >= 0)
    k1 = how fast repeated words stop adding score (saturation); b = how much long documents are penalised.
    `set_params(k1, b)` only recomputes the per-posting weights (fast), the index stays."""

    def __init__(self, docs_tokens, k1=1.2, b=0.75, desc="BM25 index"):
        vocab, tdoc, tterm, ttf, lens = {}, [], [], [], []
        for d, toks in enumerate(tqdm(docs_tokens, desc=desc, leave=False)):
            cnt = collections.Counter(toks)
            lens.append(len(toks))
            ids = [vocab.setdefault(w, len(vocab)) for w in cnt]
            tterm.append(np.asarray(ids, dtype=np.int32))
            ttf.append(np.fromiter(cnt.values(), dtype=np.float32, count=len(cnt)))
            tdoc.append(np.full(len(cnt), d, dtype=np.int32))
        term = np.concatenate(tterm) if tterm else np.zeros(0, np.int32)
        order = np.argsort(term, kind="stable")
        self.vocab = vocab
        self.N = len(lens)
        self.doc_len = np.asarray(lens, dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.N else 0.0
        self.post_doc = np.concatenate(tdoc)[order]
        self.post_tf = np.concatenate(ttf)[order]
        df = np.bincount(term, minlength=len(vocab))
        self.term_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(df, out=self.term_ptr[1:])
        self.df = df
        self.idf = np.log1p((self.N - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.set_params(k1, b)

    def set_params(self, k1, b):
        self.k1, self.b = k1, b
        dl = self.doc_len[self.post_doc] / max(self.avg_len, 1e-9)
        tf = self.post_tf
        idf_post = np.repeat(self.idf, np.diff(self.term_ptr))
        self.post_w = (idf_post * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl))).astype(np.float32)

    def score(self, query_tokens, query_tf=False):
        """Scores of all N documents for one query (array of length N).
        query_tf=False: every distinct query word counts once (usual for long queries = whole cases);
        query_tf=True : a word counts as often as it occurs in the query."""
        cnt = collections.Counter(query_tokens)
        tids = [(self.vocab[w], c) for w, c in cnt.items() if w in self.vocab]
        if not tids:
            return np.zeros(self.N, dtype=np.float32)
        idx = np.concatenate([np.arange(self.term_ptr[t], self.term_ptr[t + 1]) for t, _ in tids])
        w = self.post_w[idx]
        if query_tf:
            mult = np.concatenate([np.full(self.term_ptr[t + 1] - self.term_ptr[t], c, np.float32) for t, c in tids])
            w = w * mult
        return np.bincount(self.post_doc[idx], weights=w, minlength=self.N).astype(np.float32)

    def score_many(self, queries_tokens, query_tf=False, desc="BM25 scoring", total=None):
        """queries_tokens may be a generator (memory: only one query's words exist at a time) -> (Q, N) array."""
        return np.stack([self.score(q, query_tf) for q in tqdm(queries_tokens, desc=desc, leave=False, total=total)])


# ----------------------------------------------------------------------------------------
# 5. Years ("prior" case = decided before the query)
# ----------------------------------------------------------------------------------------
YEAR_RE = re.compile(r"\b(18[5-9]\d|19\d\d|20[0-2]\d)\b")
DATE_OF_JUDGMENT_RE = re.compile(r"DATE OF JUDGMENT[:\s]*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})")
CITATOR_RE = re.compile(r"CITATOR INFO\s*:.*?(?=\bACT\s*:|HEADNOTE\s*:|$)", re.S)


def doc_year(text, max_year=2026, return_source=False):
    """Rough judgment year of a case.
    1. Supreme-Court-style header 'DATE OF JUDGMENT: 08/12/1965' (about half of the cases) -> that year.
    2. otherwise the latest plausible year (1850..max_year) in the text: a judgment mentions older events and
       cites older cases, so its own date is usually the newest year in it. The header block 'CITATOR INFO'
       (lists LATER cases that cite this one) is removed first - it would make old cases look new.
    None if no year is found."""
    m = DATE_OF_JUDGMENT_RE.search(text[:3000])
    if m and 1850 <= int(m.group(3)) <= max_year:
        y, src = int(m.group(3)), "header"
    else:
        head = CITATOR_RE.sub(" ", text[:6000])
        ys = [int(v) for v in YEAR_RE.findall(head + text[6000:]) if int(v) <= max_year]
        y, src = (max(ys) if ys else None), "latest year"
    return (y, src) if return_source else y


# ----------------------------------------------------------------------------------------
# 5b. Which part of a case is read (same rules as P7_01, shared by BM25 and the neural models)
# ----------------------------------------------------------------------------------------
CIT = "<CITATION>"


def first_words(d, n):
    """The first n words of a case (n None = whole case)."""
    t = doc_text(d)
    return t if n is None else " ".join(t.split()[:n])


def cit_text(d, window=1):
    """Query sentences that contain the citation marker, plus `window` sentences on each side
    (P7_01: the best BM25 query). Falls back to the whole case if it has no marker."""
    keep = set()
    for i, s in enumerate(d["sents"]):
        if CIT in s:
            keep.update(range(max(0, i - window), min(len(d["sents"]), i + window + 1)))
    return " ".join(d["sents"][i] for i in sorted(keep)) or doc_text(d)


def query_snippet(d, before=50, after=10, fallback_words=300):
    """Short query text for the neural models: for every <CITATION> marker the `before` words in front of it and
    the `after` words behind it (the words that say WHY a case is cited), overlapping windows merged, windows
    joined with ' ... '. P7_01: 'sentences' are long paragraphs (~120 words), so whole citation sentences would not
    fit into the model's window. No marker -> the first `fallback_words` words."""
    w = doc_text(d).split()
    marks = [i for i, x in enumerate(w) if CIT in x]
    if not marks:
        return " ".join(w[:fallback_words])
    spans = []
    for i in marks:
        s, e = max(0, i - before), min(len(w), i + after + 1)
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return " ... ".join(" ".join(w[s:e]) for s, e in spans)


# ----------------------------------------------------------------------------------------
# 6. Submission
# ----------------------------------------------------------------------------------------
def write_submission(out_dir, preds):
    """preds = {query_id: [7 candidate ids, best first]} -> prediction.json (the doc's dict format) at the root
    of submission.zip. IDs are written as strings exactly as in the data files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = {str(q): [str(c) for c in cs] for q, cs in preds.items()}
    (out_dir / "prediction.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", arcname="prediction.json")
    return out_dir / "submission.zip"


def check_submission(zip_path, query_ids, pool_ids, k=TOP_K):
    """All the doc's rules (13.3.5): one entry per test query and nothing extra, exactly k ids, unique, every id
    from the test candidate pool, never the query itself; prediction.json at the zip root."""
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        obj = json.loads(z.read("prediction.json").decode("utf-8"))
    assert isinstance(obj, dict), "prediction.json must be a dict {query_id: [ids]}"
    assert set(obj) == {str(q) for q in query_ids}, "query ids differ from test.json"
    pool = {str(c) for c in pool_ids}
    for q, cs in obj.items():
        assert isinstance(cs, list) and len(cs) == k, f"{q}: {len(cs)} ids instead of {k}"
        assert all(isinstance(c, str) for c in cs), f"{q}: ids must be strings"
        assert len(set(cs)) == k, f"{q}: duplicate ids"
        assert all(c in pool for c in cs), f"{q}: id not in the candidate pool"
        assert q not in cs, f"{q}: query predicted as its own prior case"
    return {"queries": len(obj), "ids_per_query": k, "distinct_ids_used": len({c for cs in obj.values() for c in cs})}
