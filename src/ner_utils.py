"""
ner_utils.py  -  shared code for Phase 2 (Legal NER)

Every Phase-2 notebook imports this file, so all experiments use exactly the same
data loading, tokenisation, BIO tags, windows, decoding, scoring and submission code.
That makes the comparison between experiments fair: only the model / fine-tuning differs.
Nothing in this file knows about a particular model. A notebook gives `predict_spans`
a function that turns token ids into tag scores.

Pipeline for one document:
  (1) original text --offset-preserving normalisation--> normalised text + position maps
  (2) bpe32k tokenizer -> token ids + the character range of every token
  (3) training only: gold spans -> one BIO tag per token
  (4) cut the token sequence into overlapping windows of <= 1536 tokens (the model's context)
  (5) the model scores every token in every window; for each token we keep the scores
      from the window in which it is most central (most context on both sides)
  (6) Viterbi: the best *valid* BIO sequence (no I-X directly after O or after another label)
  (7) runs of B-X I-X I-X -> character spans in the normalised text -> ORIGINAL offsets
  (8) strict scorer (exact start, end and label) and the submission writer
"""

import json
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


# ----------------------------------------------------------------------------------------
# 1. Paths and data
# ----------------------------------------------------------------------------------------
def find_project_root(start=None):
    """Walk up from `start` (default: current folder) until a folder with .gitignore is found."""
    p = Path(start or Path.cwd()).resolve()
    while not (p / ".gitignore").exists():
        if p.parent == p:
            raise FileNotFoundError("Project root with .gitignore not found")
        p = p.parent
    return p


def load_ner_data(root):
    """Load labels, the two training folds (text + gold) and the test texts.

    Gold spans are returned as tuples (start, end, label); the 4th field of the file
    (entity text) is dropped because the scorer ignores it.
    """
    base = Path(root) / "data" / "raw" / "Phase_2_LNER"

    def load(rel):
        with open(base / rel, encoding="utf-8") as f:
            return json.load(f)

    labels = [l.strip() for l in open(base / "Train" / "labels.txt", encoding="utf-8") if l.strip()]
    texts = {f: load(f"Train/{f}-text.json") for f in ("fold1", "fold2")}
    golds = {}
    for f in ("fold1", "fold2"):
        raw = load(f"Train/{f}-gold.json")
        golds[f] = {doc_id: [(int(e[0]), int(e[1]), str(e[2])) for e in ents] for doc_id, ents in raw.items()}
    test = load("Test/fold3-text.json")
    return {"labels": labels, "texts": texts, "golds": golds, "test": test}


# ----------------------------------------------------------------------------------------
# 2. BIO tags
# ----------------------------------------------------------------------------------------
def make_tags(labels):
    """'O' + a B- and an I- tag per label: 1 + 2 x 12 = 25 tags.
    B-X = first token of an X entity, I-X = a later token of the same entity, O = outside."""
    tags = ["O"] + [f"{p}-{lab}" for lab in labels for p in ("B", "I")]
    return tags, {t: i for i, t in enumerate(tags)}


# ----------------------------------------------------------------------------------------
# 3. Offset-preserving normalisation
#    Phase 1 trained the models on cleaned ASCII text. Phase 2 needs the ORIGINAL offsets.
#    We clean character by character and remember, for every cleaned character, which
#    original character it came from, so every position can be mapped back exactly.
# ----------------------------------------------------------------------------------------
CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",   # curly single quotes, prime
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',   # curly double quotes
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",   # hyphens and dashes
    "―": "-", "−": "-",
    "…": "...",                                                           # ellipsis: 1 char -> 3 chars
    "­": "", "​": "", "‌": "", "‍": "", "﻿": "",      # soft hyphen, zero-width chars: deleted
    " ": " ", "\t": " ", "\n": " ", "\r": " ",                            # non-breaking space, tab, newlines
    "₹": "Rs.", "§": "S.", "•": "-",                            # rupee sign, section sign, bullet
}


@lru_cache(maxsize=None)
def char_replacement(ch):
    """What one original character becomes in the normalised text ('' = deleted)."""
    if ch in CHAR_MAP:
        return CHAR_MAP[ch]
    if ord(ch) < 128:
        return ch
    ascii_part = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()   # e.g. 'é' -> 'e'
    # Characters with no ASCII form (e.g. Devanagari) are KEPT (Phase 1 dropped them):
    # deleting them would make any entity written in them impossible to predict.
    return ascii_part if ascii_part else ch


def normalize_with_map(text):
    """Return (norm, src, fwd).

    norm     : the cleaned text that the tokenizer sees
    src[j]   : original position of normalised character j      (normalised -> original)
    fwd[i]   : normalised position where original character i starts, fwd[len(text)] = len(norm)
                                                                 (original -> normalised)
    Original span [s, e)   -> normalised span [fwd[s], fwd[e])
    Normalised span [a, b) -> original span   [src[a], src[b-1] + 1)
    """
    pieces = [char_replacement(ch) for ch in text]
    lengths = np.fromiter((len(p) for p in pieces), dtype=np.int64, count=len(pieces))
    norm = "".join(pieces)
    fwd = np.zeros(len(text) + 1, dtype=np.int64)
    np.cumsum(lengths, out=fwd[1:])
    src = np.repeat(np.arange(len(text), dtype=np.int64), lengths)
    return norm, src, fwd


# ----------------------------------------------------------------------------------------
# 4. Tokenisation with character offsets
# ----------------------------------------------------------------------------------------
@dataclass
class EncodedDoc:
    doc_id: str
    text: str              # original text: all offsets we OUTPUT refer to this
    norm: str              # text the tokenizer actually saw
    src: np.ndarray        # normalised position -> original position
    fwd: np.ndarray        # original position -> normalised position
    ids: np.ndarray        # token ids, shape (n_tokens,)
    tok_start: np.ndarray  # first character of each token in `norm` (its leading space skipped)
    tok_end: np.ndarray    # one past the last character of each token in `norm`


def encode_doc(doc_id, text, tokenizer, normalize=True):
    if normalize:
        norm, src, fwd = normalize_with_map(text)
    else:
        norm, src, fwd = text, np.arange(len(text), dtype=np.int64), np.arange(len(text) + 1, dtype=np.int64)
    enc = tokenizer.encode(norm, add_special_tokens=False)
    offs = np.array(enc.offsets, dtype=np.int64).reshape(-1, 2)
    start, end = offs[:, 0].copy(), offs[:, 1].copy()
    # A byte-level BPE token such as 'ĠState' covers ' State': the space belongs to the token.
    # Entities never start with a space, so we move the token start past leading whitespace.
    for k in range(len(start)):
        while start[k] < end[k] and norm[start[k]].isspace():
            start[k] += 1
    return EncodedDoc(doc_id, text, norm, src, fwd, np.array(enc.ids, dtype=np.int64), start, end)


def encode_docs(texts, tokenizer, normalize=True, desc="tokenising"):
    """texts: {doc_id: text} -> list of EncodedDoc (same order)."""
    return [encode_doc(d, t, tokenizer, normalize) for d, t in tqdm(texts.items(), desc=desc, leave=False)]


# ----------------------------------------------------------------------------------------
# 5. Spans <-> BIO tags
# ----------------------------------------------------------------------------------------
def spans_to_tags(doc, spans, tag2id):
    """Gold character spans (original offsets) -> one tag id per token.
    A token belongs to an entity if it shares at least one (non-space) character with it."""
    tags = np.zeros(len(doc.ids), dtype=np.int64)          # 0 = 'O'
    real = doc.tok_end > doc.tok_start                     # whitespace-only tokens have an empty range
    for s, e, lab in sorted(spans):
        a, b = doc.fwd[s], doc.fwd[e]                      # entity in normalised coordinates
        k = np.flatnonzero(real & (doc.tok_start < b) & (doc.tok_end > a))
        if len(k) == 0:
            continue
        tags[k[0]] = tag2id["B-" + lab]
        tags[k[0] + 1: k[-1] + 1] = tag2id["I-" + lab]
    return tags


TRIM_CHARS = ",;:[("


def trim_span(text, s, e):
    """Punctuation-trim rule (span decoding). Our tokenizer glues punctuation together
    ('Ors.,' -> 'Ors' + '.,'), so a predicted entity often ends with one character too many.
    Drop trailing , ; : [ (  and a '.' that comes right after ) or ].
    Measured on gold (Step 2): ceiling 0.9865 -> 0.9932, PREC 0.932 -> 0.992; changes only 2 gold spans."""
    while e > s:
        c = text[e - 1]
        if c in TRIM_CHARS or (c == "." and e - 2 >= s and text[e - 2] in ")]"):
            e -= 1
        else:
            break
    return s, e


def tags_to_spans(doc, tags, id2tag, trim=True):
    """Tag ids -> list of (start, end, label) in ORIGINAL character offsets.
    A run 'B-X I-X I-X' (or a stray 'I-X' run) becomes one entity.
    trim=True applies the punctuation-trim rule above."""
    names = [id2tag[t] for t in tags]
    spans, i, n = [], 0, len(names)
    while i < n:
        if names[i] == "O":
            i += 1
            continue
        lab = names[i][2:]
        j = i + 1
        while j < n and names[j] == "I-" + lab:
            j += 1
        a, b = int(doc.tok_start[i]), int(doc.tok_end[j - 1])   # tokens i .. j-1, normalised coords
        while a < b and doc.norm[a].isspace():
            a += 1
        while b > a and doc.norm[b - 1].isspace():
            b -= 1
        if a < b:
            s, e = int(doc.src[a]), int(doc.src[b - 1]) + 1
            if trim:
                s, e = trim_span(doc.text, s, e)
            if s < e:
                spans.append((s, e, lab))
        i = j
    return spans


# ----------------------------------------------------------------------------------------
# 6. Windows: documents are longer than the model's context (1536 tokens)
# ----------------------------------------------------------------------------------------
def make_windows(n_tokens, max_len=1536, stride=1024):
    """(start, end) token ranges of overlapping windows covering the whole document.
    Consecutive windows overlap by max_len - stride tokens. The last window is aligned to the
    end of the document, so every window is full length (no padding needed)."""
    if n_tokens == 0:
        return []
    if n_tokens <= max_len:
        return [(0, n_tokens)]
    starts = list(range(0, n_tokens - max_len, stride)) + [n_tokens - max_len]
    return [(s, s + max_len) for s in starts]


def plan_windows(n_tokens, max_len=1536, stride=1024, header=0):
    """Windows as (a, b, h): the model input is  ids[0:h] + ids[a:b]  (h = 0: no header).

    header = 0 : plain windows from make_windows.
    header = H : "header memory" (our own idea). The first window is the normal ids[0:max_len]; it
                 already contains the header. Every later window gets the document's first H tokens
                 (title, parties, counsel) in front of its body of max_len - H tokens, so the model can
                 still see "X ... Appellant Versus Y ... Respondent" deep inside a long judgment.
                 The overlap between bodies stays max_len - stride tokens."""
    if header == 0 or n_tokens <= max_len:
        return [(a, b, 0) for a, b in make_windows(n_tokens, max_len, stride)]
    body = max_len - header
    overlap = max_len - stride
    step = body - overlap
    wins = [(0, max_len, 0)]
    a = max_len - overlap
    while a + body < n_tokens:
        wins.append((a, a + body, header))
        a += step
    wins.append((n_tokens - body, n_tokens, header))
    return wins


def window_input(ids, win):
    """Token ids the model sees for window (a, b, h)."""
    a, b, h = win
    return ids[a:b] if h == 0 else np.concatenate([ids[:h], ids[a:b]])


def merge_windows(n_tokens, windows, window_scores):
    """Combine per-window scores into one (n_tokens, n_tags) array.
    Each token takes its scores from the window in which it is farthest from an edge,
    i.e. where the model saw the most context on both sides.
    windows: (a, b) or (a, b, h); for a header window the first h score rows belong to the header and are skipped."""
    out = np.zeros((n_tokens, window_scores[0].shape[1]), dtype=np.float32)
    best = np.full(n_tokens, -1, dtype=np.int64)
    for win, sc in zip(windows, window_scores):
        a, b = win[0], win[1]
        h = win[2] if len(win) == 3 else 0
        sc = sc[h:]
        pos = np.arange(a, b)
        centre = np.minimum(pos - a, b - 1 - pos)          # distance to the nearer window edge
        better = centre > best[a:b]
        out[a:b][better] = sc[better]                      # out[a:b] is a view: this writes into out
        best[a:b][better] = centre[better]
    return out


IGNORE = -100   # label for positions that must not count in the loss (header copies)


def make_training_windows(docs, tags_list, max_len=1536, stride=1024, header=0, rng=None):
    """Cut every training document (ids + gold tags) into windows. Returns two lists of arrays.
    Header tokens copied in front of later windows get the label IGNORE (they are trained in window 1).
    rng (augmentation): every document's windows start at a random offset (0 .. stride-1), so each epoch
    the model sees the text cut at different places instead of memorising fixed windows."""
    X, Y = [], []
    for doc, tags in zip(docs, tags_list):
        n = len(doc.ids)
        off = int(rng.integers(0, stride)) if (rng is not None and n > max_len) else 0
        off = min(off, n - max_len) if n > max_len else 0
        for win in plan_windows(n - off, max_len, stride, header):
            win = (win[0] + off, win[1] + off, win[2])
            a, b, h = win
            X.append(window_input(doc.ids, win))
            Y.append(tags[a:b] if h == 0 else np.concatenate([np.full(h, IGNORE, dtype=np.int64), tags[a:b]]))
    return X, Y


# ----------------------------------------------------------------------------------------
# 7. Constrained Viterbi decoding
# ----------------------------------------------------------------------------------------
def bio_constraints(tags):
    """allowed[i, j]: may tag j follow tag i?  start_ok[j]: may a document start with tag j?
    The only forbidden moves: I-X after anything except B-X or I-X (and I-X at the very start)."""
    C = len(tags)
    allowed = np.ones((C, C), dtype=bool)
    start_ok = np.ones(C, dtype=bool)
    for j, cur in enumerate(tags):
        if cur.startswith("I-"):
            start_ok[j] = False
            for i, prev in enumerate(tags):
                allowed[i, j] = prev in ("B-" + cur[2:], cur)
    return allowed, start_ok


def viterbi(log_probs, allowed, start_ok):
    """Most probable tag sequence (sum of per-token log-probabilities) among VALID BIO sequences.
    Dynamic programming: score[j] = best total score of any valid path ending in tag j."""
    T, C = log_probs.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    lp = log_probs.astype(np.float64)
    trans = np.where(allowed, 0.0, -1e9)                   # -1e9 = forbidden
    score = lp[0] + np.where(start_ok, 0.0, -1e9)
    back = np.zeros((T, C), dtype=np.int32)                # back[t, j] = best previous tag
    cols = np.arange(C)
    for t in range(1, T):
        cand = score[:, None] + trans                      # cand[i, j]: come from i, go to j
        back[t] = cand.argmax(axis=0)
        score = cand[back[t], cols] + lp[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):                          # follow the pointers backwards
        path[t - 1] = back[t, path[t]]
    return path


def doc_log_probs(docs, score_windows, max_len=1536, stride=1024, header=0):
    """Run the model over every window of every document (one call, so the model can batch),
    then merge the windows back per document.
    score_windows(list_of_id_arrays) -> list of (window_len, n_tags) LOG-probability arrays.
    Returns {doc_id: (n_tokens, n_tags) array}; empty documents get a (0, n_tags)-shaped None."""
    plans = {d.doc_id: plan_windows(len(d.ids), max_len, stride, header) for d in docs}
    inputs = [window_input(d.ids, w) for d in docs for w in plans[d.doc_id]]
    scores = score_windows(inputs) if inputs else []
    out, k = {}, 0
    for d in docs:
        wins = plans[d.doc_id]
        out[d.doc_id] = merge_windows(len(d.ids), wins, scores[k:k + len(wins)]) if wins else None
        k += len(wins)
    return out


def decode_docs(docs, log_probs, tags, decode="viterbi", trim=True):
    """Merged log-probabilities -> {doc_id: [(start, end, label), ...]} in original offsets."""
    allowed, start_ok = bio_constraints(tags)
    out = {}
    for d in docs:
        lp = log_probs[d.doc_id]
        if lp is None:
            out[d.doc_id] = []                              # empty document
            continue
        path = viterbi(lp, allowed, start_ok) if decode == "viterbi" else lp.argmax(axis=1)
        out[d.doc_id] = tags_to_spans(d, path, tags, trim=trim)
    return out


def predict_spans(docs, score_windows, tags, max_len=1536, stride=1024, header=0, decode="viterbi", trim=True):
    """Model-agnostic prediction: windows -> model -> merge -> Viterbi -> spans (original offsets)."""
    lp = doc_log_probs(docs, score_windows, max_len, stride, header)
    return decode_docs(docs, lp, tags, decode, trim)


# ----------------------------------------------------------------------------------------
# 8. Strict scorer and error analysis
# ----------------------------------------------------------------------------------------
def _span_sets(d):
    return {doc: {(int(x[0]), int(x[1]), str(x[2])) for x in spans} for doc, spans in d.items()}


def strict_scores(gold, pred, labels):
    """Strict matching: a prediction counts only if start, end AND label are all identical.
    Per label: precision, recall, F1 over all documents. Macro = plain average over the 12 labels
    (every label counts the same, however rare). Micro = pooled over all entities.
    `macro_f1` is the mean of per-label F1; `macro_f1_of_means` = 2PR/(P+R) of macro P and R
    (we don't know which one Codabench uses; they are usually very close)."""
    G, P = _span_sets(gold), _span_sets(pred)
    tp, fp, fn = Counter(), Counter(), Counter()
    for doc in set(G) | set(P):
        g, p = G.get(doc, set()), P.get(doc, set())
        for x in g & p:
            tp[x[2]] += 1
        for x in p - g:
            fp[x[2]] += 1
        for x in g - p:
            fn[x[2]] += 1
    per = {}
    for lab in labels:
        pr = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        rc = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[lab] = {"p": pr, "r": rc, "f1": f1, "tp": tp[lab], "fp": fp[lab], "fn": fn[lab], "support": tp[lab] + fn[lab]}
    mp = float(np.mean([v["p"] for v in per.values()]))
    mr = float(np.mean([v["r"] for v in per.values()]))
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = TP / (TP + FP) if TP + FP else 0.0
    micro_r = TP / (TP + FN) if TP + FN else 0.0
    return {
        "macro_f1": float(np.mean([v["f1"] for v in per.values()])),
        "macro_p": mp,
        "macro_r": mr,
        "macro_f1_of_means": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
        "micro_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0,
        "unknown_labels": sorted((set(tp) | set(fp) | set(fn)) - set(labels)),
        "per_label": per,
    }


def print_scores(sc, title=""):
    if title:
        print(title)
    print(f"  macro-F1 {sc['macro_f1']:.4f} | macro-P {sc['macro_p']:.4f} | macro-R {sc['macro_r']:.4f}"
          f" | F1(of macro P,R) {sc['macro_f1_of_means']:.4f} | micro-F1 {sc['micro_f1']:.4f}")
    print(f"  {'label':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'gold':>5s}")
    for lab, v in sc["per_label"].items():
        print(f"  {lab:10s} {v['p']:6.3f} {v['r']:6.3f} {v['f1']:6.3f} {v['tp']:5d} {v['fp']:5d} {v['fn']:5d} {v['support']:5d}")


def error_breakdown(gold, pred, texts=None, n_examples=0):
    """Sort every mistake into a category (the 'domain-specific error analysis' the task asks for).
    Wrong predictions (false positives):
      'right span, wrong label'        exact boundaries, but e.g. APP predicted where gold says RESP
      'wrong boundary, right label'    overlaps a gold entity of the same label, but start/end differ
      'wrong boundary and label'       overlaps a gold entity of another label
      'spurious'                       nothing in the gold there at all
    Missed gold entities (false negatives) with no overlapping prediction at all: 'missed'.
    Also returns a confusion Counter {(gold_label, predicted_label): count} for exact-span label errors."""
    G, P = _span_sets(gold), _span_sets(pred)
    cats, confusion, examples = Counter(), Counter(), defaultdict(list)
    for doc in set(G) | set(P):
        g, p = G.get(doc, set()), P.get(doc, set())
        for (s, e, lab) in sorted(p - g):
            same = [gl for (gs, ge, gl) in g if gs == s and ge == e]
            over = [gl for (gs, ge, gl) in g if gs < e and s < ge]
            if same:
                cat = "right span, wrong label"
                confusion[(same[0], lab)] += 1
            elif lab in over:
                cat = "wrong boundary, right label"
            elif over:
                cat = "wrong boundary and label"
            else:
                cat = "spurious"
            cats[cat] += 1
            if texts is not None and len(examples[cat]) < n_examples:
                examples[cat].append((doc, s, e, lab, texts[doc][max(0, s - 30):e + 30]))
        for (s, e, lab) in sorted(g - p):
            if not any(ps < e and s < pe for (ps, pe, _) in p):
                cats["missed"] += 1
                if texts is not None and len(examples["missed"]) < n_examples:
                    examples["missed"].append((doc, s, e, lab, texts[doc][max(0, s - 30):e + 30]))
    return cats, confusion, examples


# ----------------------------------------------------------------------------------------
# 9. Submission
# ----------------------------------------------------------------------------------------
def write_submission(pred, test_texts, labels, out_dir):
    """Write prediction.json (every test document, also empty ones) and submission.zip with
    prediction.json at the ZIP ROOT. Checks every span before writing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = set(pred) - set(test_texts)
    assert not extra, f"predictions for unknown documents: {list(extra)[:5]}"
    clean = {}
    for doc_id, text in test_texts.items():
        spans = sorted({(int(x[0]), int(x[1]), str(x[2])) for x in pred.get(doc_id, [])})
        for s, e, lab in spans:
            assert 0 <= s < e <= len(text), f"bad offsets {doc_id} {s} {e}"
            assert lab in labels, f"unknown label {lab}"
        clean[doc_id] = [[s, e, lab] for s, e, lab in spans]
    json_path = out_dir / "prediction.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False)
    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="prediction.json")
    check_submission(zip_path, test_texts, labels)
    return zip_path


def check_submission(zip_path, test_texts, labels, verbose=True):
    """Re-open the zip exactly as the grader would and validate it. Returns the predictions."""
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        pred = json.loads(z.read("prediction.json").decode("utf-8"))
    assert set(pred) == set(test_texts), "document ids differ from the test set"
    n = 0
    for doc_id, spans in pred.items():
        for item in spans:
            s, e, lab = item[0], item[1], item[2]
            assert isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(test_texts[doc_id])
            assert lab in labels
            n += 1
    if verbose:
        print(f"submission OK: {zip_path} | {len(pred)} documents | {n:,} entities")
    return pred


# ----------------------------------------------------------------------------------------
# 10. Experiment log (one JSON line per run, like Phase 1)
# ----------------------------------------------------------------------------------------
def log_experiment(root, record):
    path = Path(root) / "experiments" / "phase2" / "experiments.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_experiments(root):
    path = Path(root) / "experiments" / "phase2" / "experiments.jsonl"
    rows = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                                    # tolerate a broken/blank line
    return rows
