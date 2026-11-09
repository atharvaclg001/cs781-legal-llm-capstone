"""
summ_utils.py  -  shared code for Phase 6 (Abstractive Summarization). No PyTorch needed.

Every Phase-6 notebook uses these functions, so the data loading, cleaning, validation split, metrics and
submission writer are identical for all experiments. Only the model / training / decoding differ.

Contents
  1. data:     summ_dir, load_summ (robust to the file format), text_key (fingerprint for duplicates)
  2. cleaning: clean_summ (Phase-1 cleaning, cased), split_sentences (legal-aware sentence splitter)
  3. split:    make_val_split (hash-based held-out validation set + small local-eval subset; no public dev set)
  4. metrics:  rouge_n_f1 / rouge_l_f1 (same whitespace tokens as the course's Phase-1 evaluator),
               bertscore (wrapper around bert_score with the evaluator's settings), score_summaries
  5. analysis: novel_ngram_share, fragments (extractive coverage / density, Grusky et al. 2018)
  6. output:   write_submission + check_submission (list format [{"id", "summary"}], zip with prediction.json)
"""

import collections
import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np

from ner_utils import CHAR_MAP, find_project_root  # noqa: F401  (same cleaning table + root finder as Phases 2-5)

CLEAN_VERSION = "v1"   # part of cache file names: change it if clean_summ changes

DOC_KEYS = ["document", "text", "judgment", "judgement", "doc", "source", "article", "case_text", "content"]
SUM_KEYS = ["summary", "summaries", "abstract", "headnote", "target", "reference", "ref_summary"]


# ----------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------
def summ_dir(root):
    return Path(root) / "data" / "raw" / "Phase_6_Summ"


def text_key(t):
    """Fingerprint of a text: case- and whitespace-insensitive (same idea as Phases 4-5)."""
    return hashlib.md5(re.sub(r"\s+", " ", t).strip().lower().encode()).hexdigest()


def _as_text(x):
    """A field can be one string or a list of sentences / paragraphs -> one string (pieces joined by a space)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return " ".join(str(p) for p in x)
    return str(x)


def _records(raw):
    """A file can be a list of records or a dict {id: record} (or {id: text})."""
    if isinstance(raw, list):
        return [dict(r) for r in raw]
    if isinstance(raw, dict):
        out = []
        for k, v in raw.items():
            r = dict(v) if isinstance(v, dict) else {"document": v}
            r.setdefault("id", k)
            out.append(r)
        return out
    raise TypeError(f"unexpected JSON type {type(raw)}")


def find_files(root):
    """train / test json files inside Phase_6_Summ (the doc says Train/train.json, Test/test.json;
    we search, in case names differ)."""
    base = summ_dir(root)
    found = {}
    for split in ("train", "test"):
        cands = sorted(base.rglob(f"{split}*.json")) or sorted((base / split.capitalize()).glob("*.json"))
        if cands:
            found[split] = cands[0]
    return found


def load_summ(root=None, files=None):
    """Returns (data, info).
    data = {"train": [...], "test": [...]}; each item {"id": str, "document": str, "summary": str or None}.
    info = per split: file type, keys, which key was used for document / summary, piece counts.
    Test summaries (if the file has any) are REMOVED here and never used (Phase-4 lesson)."""
    files = files or find_files(root)
    data, info = {}, {}
    for split, path in files.items():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        recs = _records(raw)
        first = recs[0]
        dkey = next((k for k in DOC_KEYS if k in first), None)
        skey = next((k for k in SUM_KEYS if k in first), None)
        assert dkey, f"{split}: no document field among {list(first.keys())}"
        docs = []
        for r in recs:
            docs.append({"id": str(r.get("id", r.get("doc_id", len(docs)))),
                         "document": _as_text(r.get(dkey)) or "",
                         "summary": _as_text(r.get(skey)) if skey else None})
        rec = {"file": str(Path(path).name), "file_type": type(raw).__name__, "records": len(recs),
               "keys": list(first.keys()), "document_key": dkey, "summary_key": skey,
               "document_type": type(first[dkey]).__name__,
               "summary_type": type(first[skey]).__name__ if skey else None}
        if split == "test" and skey and any(d["summary"] for d in docs):
            rec["test_had_summaries"] = True       # dropped: must not be used
            for d in docs:
                d["summary"] = None
        data[split], info[split] = docs, rec
    return data, info


# ----------------------------------------------------------------------------------------
# 2. Cleaning + sentences
# ----------------------------------------------------------------------------------------
SIG_RE = re.compile(r"Signature Not Verified.{0,300}?Reason:", re.S)
HTML_RE = re.compile(r"</?(?:span|font|p|br|div|b|i|u|sup|sub|strong|em)\b[^<>]{0,200}>", re.I)


def clean_summ(t):
    """Phase-1 cleaning (cased): signature stamps + a few HTML tags out, special characters -> ASCII (CHAR_MAP),
    other non-ASCII dropped, all whitespace (also line breaks) -> one space. Used for documents AND summaries."""
    t = SIG_RE.sub(" ", t or "")
    t = HTML_RE.sub(" ", t)
    t = "".join(CHAR_MAP.get(c, c) for c in t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip()


# words that end with a full stop but do NOT end a sentence in Indian legal text
ABBREV = {"no", "nos", "vs", "v", "mr", "mrs", "ms", "dr", "sr", "jr", "ltd", "pvt", "co", "corpn", "govt", "art",
          "arts", "sec", "secs", "s", "ss", "cl", "cr", "crl", "cri", "civ", "etc", "ie", "eg", "viz", "ors", "anr",
          "rs", "p", "pp", "para", "paras", "j", "jj", "cj", "st", "smt", "shri", "sri", "kum", "ibid", "supra",
          "hon'ble", "honble", "dt", "dtd", "ref", "sl", "u", "r", "w", "o", "ch", "vol", "ed", "approx", "misc",
          "spl", "addl", "asst", "dy", "jt", "inspr", "insp", "const", "hc", "sc", "ex", "exh", "ext", "fig"}
_SENT_END = re.compile(r"(?<=[.?!])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(t):
    """Simple rule-based splitter: a sentence ends at . ? ! followed by a space and a capital letter / digit,
    unless the word before the full stop is a known abbreviation ("No.", "vs.", "Ors.") or a single initial
    ("A. K. Sen"). Good enough for statistics (we never need perfect sentences)."""
    t = t.strip()
    if not t:
        return []
    out, start = [], 0
    for m in _SENT_END.finditer(t):
        before = t[start:m.start()]
        last = before.rsplit(" ", 1)[-1].rstrip(".?!").lower().replace(".", "")
        if before.endswith(".") and (last in ABBREV or (len(last) == 1 and last.isalpha())):
            continue
        out.append(t[start:m.start()].strip())
        start = m.end()
    out.append(t[start:].strip())
    return [s for s in out if s]


# ----------------------------------------------------------------------------------------
# 3. Validation split (no public dev set in Phase 6)
# ----------------------------------------------------------------------------------------
def make_val_split(train_docs, val_per_mille=50, n_local=150, path=None):
    """Hash-based held-out split, reproducible and independent of the file order:
      fingerprint = md5(document text, case/whitespace-insensitive) -> identical documents always land on the
      SAME side (no leakage from duplicates);  val if int(fingerprint, 16) % 1000 < val_per_mille.
    local_eval = the n_local validation docs with the smallest fingerprints (a fixed random-looking subset, used
    for slow generation + BERTScore; the whole val set is used for the cheap validation loss).
    Cached as JSON at `path` (if given): re-running loads it so the split never changes silently."""
    if path is not None and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    fp = {d["id"]: text_key(d["document"]) for d in train_docs}
    val = [i for i, h in fp.items() if int(h, 16) % 1000 < val_per_mille]
    train = [i for i in fp if i not in set(val)]
    local = sorted(val, key=lambda i: fp[i])[:n_local]
    split = {"train": train, "val": val, "local_eval": local,
             "rule": f"md5(document) % 1000 < {val_per_mille}; local_eval = {n_local} smallest fingerprints"}
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f)
    return split


# ----------------------------------------------------------------------------------------
# 4. Metrics
# ----------------------------------------------------------------------------------------
def _words(t):
    """Tokens for ROUGE = plain whitespace split, case-sensitive: the same tokens the course's Phase-1 evaluator
    (scripts/evaluate_phase1.py) used. The Phase-6 evaluator is not public, so ROUGE here is a close estimate."""
    return t.split()


def _ngrams(toks, n):
    return collections.Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def rouge_n_f1(pred, ref, n=1):
    """ROUGE-N F1: overlap of n-gram counts (clipped), P = overlap / pred n-grams, R = overlap / ref n-grams."""
    p, r = _ngrams(_words(pred), n), _ngrams(_words(ref), n)
    ov = sum((p & r).values())
    if not ov:
        return 0.0
    prec, rec = ov / sum(p.values()), ov / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def lcs_length(a, b):
    """Length of the longest common subsequence, bit-parallel (copied from the course's Phase-1 evaluator):
    fast enough for 1,000-word summaries."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks = {}
    for j, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << j)
    state = 0
    for t in b:
        m = masks.get(t, 0)
        x = m | state
        y = (state << 1) | 1
        state = x & ~(x - y)
    return bin(state).count("1")


def rouge_l_f1(pred, ref):
    """ROUGE-L F1 = F1 of the longest common subsequence of words (order matters, gaps allowed)."""
    p, r = _words(pred), _words(ref)
    l = lcs_length(p, r)
    if not l:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def bertscore(preds, refs, batch_size=8, model_type=None, device=None):
    """BERTScore with the settings of the course evaluator: bert_score.score(..., lang='en') -> roberta-large
    (layer 17), no idf, no baseline rescaling. Returns (P, R, F1) as numpy arrays (one value per pair).
    NOTE: roberta-large reads at most 512 tokens -> longer summaries are CUT for the metric (see P6_01)."""
    import bert_score
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    kw = {"model_type": model_type} if model_type else {"lang": "en"}
    P, R, F = bert_score.score([p.strip() for p in preds], [r.strip() for r in refs], batch_size=batch_size,
                               verbose=False, device=device, **kw)
    return P.numpy(), R.numpy(), F.numpy()


def score_summaries(preds, refs, with_bertscore=True, **bs_kw):
    """All metrics for lists of predicted and reference summaries (same order). Means over documents."""
    out = {"n": len(refs),
           "rouge1_f1": float(np.mean([rouge_n_f1(p, r, 1) for p, r in zip(preds, refs)])),
           "rouge2_f1": float(np.mean([rouge_n_f1(p, r, 2) for p, r in zip(preds, refs)])),
           "rougeL_f1": float(np.mean([rouge_l_f1(p, r) for p, r in zip(preds, refs)])),
           "pred_words": float(np.mean([len(p.split()) for p in preds]))}
    if with_bertscore:
        P, R, F = bertscore(preds, refs, **bs_kw)
        out.update(bertscore_p=float(P.mean()), bertscore_r=float(R.mean()), bertscore_f1=float(F.mean()))
    return out


# ----------------------------------------------------------------------------------------
# 5. Analysis helpers (how abstractive are the summaries?)
# ----------------------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")


def norm_words(t):
    """Lower-case alphanumeric words (punctuation dropped): used only for the overlap statistics."""
    return _WORD_RE.findall(t.lower())


def novel_ngram_share(summary_words, doc_words, n):
    """Share of the summary's n-gram TYPES that never occur in the document (0 = fully extractive)."""
    s = set(zip(*[summary_words[i:] for i in range(n)]))
    if not s:
        return np.nan
    d = set(zip(*[doc_words[i:] for i in range(n)]))
    return len(s - d) / len(s)


def fragments(summary_words, doc_words):
    """Extractive fragments (Grusky, Naaman, Artzi 2018, "Newsroom"): walk through the summary and greedily take the
    longest piece that is copied word-for-word from the document.
      coverage = share of summary words inside a copied piece (0..1)
      density  = mean squared piece length per summary word (high = long copied passages)
    Returns (coverage, density, list of piece lengths)."""
    S, D = summary_words, doc_words
    if not S:
        return np.nan, np.nan, []
    pos = collections.defaultdict(list)          # bigram -> start positions in the document
    for j in range(len(D) - 1):
        pos[(D[j], D[j + 1])].append(j)
    dset = set(D)
    frags, i = [], 0
    while i < len(S):
        best = 0
        if i + 1 < len(S):
            for j in pos.get((S[i], S[i + 1]), ()):
                k = 2
                while i + k < len(S) and j + k < len(D) and S[i + k] == D[j + k]:
                    k += 1
                best = max(best, k)
        if best == 0 and S[i] in dset:
            best = 1
        if best:
            frags.append(best)
            i += best
        else:
            i += 1
    n = len(S)
    return sum(frags) / n, sum(f * f for f in frags) / n, frags


# ----------------------------------------------------------------------------------------
# 5b. Model inputs: token ids of documents and target summaries (cached)
# ----------------------------------------------------------------------------------------
DOC_KEEP = 3072        # tokens kept per document (the longest input any experiment reads: 3 chunks of 1024)
SUM_MAX = 620          # target summary tokens (P6_01: BERTScore sees only ~426 words ~ 510 tokens; + margin)
EOS_ID = 0


def clean_target(t):
    """Reference summary -> training target: Phase-1 cleaning + remove the stray leading '% ' (180 summaries)."""
    return re.sub(r"^%\s*", "", clean_summ(t))


def encode_inputs(root, split, docs, tokenizer, batch=128):
    """Token ids for every document (first DOC_KEEP tokens) and, for train, every target summary:
    the first SUM_MAX tokens, followed by EOS only if the summary ENDS within SUM_MAX tokens
    (a cut summary has no EOS: the model must not learn to stop there).
    Cached in data/processed/phase6/<split>_<CLEAN_VERSION>.npz -> {id: (doc_ids, sum_ids or None)}."""
    cache = Path(root) / "data" / "processed" / "phase6" / f"{split}_{CLEAN_VERSION}.npz"
    ids = [d["id"] for d in docs]
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            # read every array ONCE: each z["..."] access re-reads the whole array from the file
            arr = {k: z[k] for k in z.files}
        if list(arr["ids"]) == ids:
            doc, doc_off = arr["doc"], arr["doc_off"]
            summ, sum_off = arr.get("sum"), arr.get("sum_off")
            out = {}
            for i, k in enumerate(ids):
                d = doc[doc_off[i]:doc_off[i + 1]]
                s = summ[sum_off[i]:sum_off[i + 1]] if summ is not None else None
                out[k] = (d, s if s is not None and len(s) else None)
            return out
    from tqdm.auto import tqdm
    doc_tok, sum_tok = [], []
    for i in tqdm(range(0, len(docs), batch), desc=f"tokenising {split}"):
        part = docs[i:i + batch]
        enc = tokenizer.encode_batch([" " + clean_summ(d["document"]) for d in part], add_special_tokens=False)
        doc_tok += [np.array(e.ids[:DOC_KEEP], dtype=np.uint16) for e in enc]
        if part[0]["summary"] is not None or any(d["summary"] for d in part):
            enc = tokenizer.encode_batch([" " + clean_target(d["summary"] or "") for d in part], add_special_tokens=False)
            for e in enc:
                t = e.ids[:SUM_MAX] + ([EOS_ID] if len(e.ids) <= SUM_MAX else [])
                sum_tok.append(np.array(t if len(e.ids) else [], dtype=np.uint16))
    cache.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"ids": np.array(ids), "doc": np.concatenate(doc_tok),
              "doc_off": np.concatenate([[0], np.cumsum([len(x) for x in doc_tok])])}
    if sum_tok:
        arrays.update(sum=np.concatenate(sum_tok), sum_off=np.concatenate([[0], np.cumsum([len(x) for x in sum_tok])]))
    np.savez(cache, **arrays)
    return {k: (doc_tok[i], sum_tok[i] if sum_tok and len(sum_tok[i]) else None) for i, k in enumerate(ids)}


def length_rule(doc_words, rule, lo=80, hi=500):
    """Target summary length in words from the document length (P6_01 fit: 4.62 * words^0.612), capped."""
    return int(np.clip(np.exp(rule["a"]) * max(doc_words, 1) ** rule["b"], lo, hi))


def repetition_rate(text, n=4):
    """Share of the summary's word n-grams that are repeats of an earlier n-gram in the same summary."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


# ----------------------------------------------------------------------------------------
# 6. Submission
# ----------------------------------------------------------------------------------------
def write_submission(out_dir, ids, summaries):
    """prediction.json in the doc's list format [{"id": ..., "summary": ...}] at the root of submission.zip."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"id": str(i), "summary": str(s)} for i, s in zip(ids, summaries)]
    (out_dir / "prediction.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", arcname="prediction.json")
    return out_dir / "submission.zip"


def check_submission(zip_path, test_ids):
    """Exactly one non-empty summary per test id, nothing extra, prediction.json at the zip root."""
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        rows = json.loads(z.read("prediction.json").decode("utf-8"))
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert set(ids) == {str(i) for i in test_ids}, "ids differ from test"
    assert all(isinstance(r["summary"], str) and r["summary"].strip() for r in rows), "empty summary"
    words = [len(r["summary"].split()) for r in rows]
    return {"docs": len(rows), "mean_words": round(float(np.mean(words)), 1), "min_words": int(min(words)),
            "max_words": int(max(words))}
