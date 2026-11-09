"""
rr_utils.py  -  shared code for Phase 3 (Rhetorical Role prediction, RR)

Every Phase-3 notebook imports this file, so all six experiments use exactly the same data
loading, tokenisation, windows, class weights, scorer and submission writer. Only the model
and the way it is trained differ between experiments -> a fair comparison.
Nothing in this file needs PyTorch.

The task: every SENTENCE of a legal document gets one of 13 roles (Fact, Issue, ...).

Pipeline for one document (the "hierarchical" design):
  (1) every sentence -> bpe32k token ids (with a leading space, as inside running text);
      very long sentences are cut to `max_sent_tokens` tokens
  (2) all sentences are glued into one token sequence; `bounds[i] = (start, end)` remembers
      which tokens belong to sentence i
  (3) the sequence is cut into windows of WHOLE sentences (<= 1536 tokens, overlapping),
      because the Phase-1 backbone can only read 1536 tokens at a time
  (4) the model turns every window into one vector per sentence (average of its tokens);
      each sentence keeps the vector from the window in which it is most central
  (5) a BiLSTM reads ALL sentence vectors of the document -> one role per sentence
      (this connects the windows: sentence 1,500 can still use information from sentence 1)
  (6) scorer: macro-F1 over the labels (+ macro P/R, per label, per domain CL/IT)
  (7) submission writer: prediction.json = {doc_id: [label, ...]}, one label per sentence,
      zipped at the root of submission.zip, checked against the test files
"""

import json
import re
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ner_utils import find_project_root  # same "walk up to .gitignore" as Phase 2  # noqa: F401

DOMAINS = ("CL", "IT")


# ----------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------
def rr_dir(root):
    return Path(root) / "data" / "raw" / "Phase_3_RR"


def load_label_vocab(root):
    """label_vocab.json is {label: id}. Returns the 13 labels ordered by id (Fact = 0 ... None = 12)."""
    with open(rr_dir(root) / "Train" / "label_vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    return [lab for lab, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def _to_docs(data, domain, split):
    """Train/dev files are a LIST of {id, text, labels}; test files are a DICT {doc_id: [sentences]}.
    Both become a list of {id, domain, split, text, labels} (labels = None for test)."""
    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            v = dict(v) if isinstance(v, dict) else {"text": v}
            v["id"] = str(v.get("id", k))
            items.append(v)
    else:
        items = data
    return [{"id": str(d["id"]), "domain": domain, "split": split,
             "text": list(d["text"]), "labels": list(d["labels"]) if "labels" in d else None}
            for d in items]


def load_rr_data(root):
    """Returns (labels, data) with data = {"train": [...], "dev": [...], "test": [...]};
    each list holds the CL documents followed by the IT documents. Checks the data while loading."""
    labels = load_label_vocab(root)
    base = rr_dir(root)
    files = {"train": "Train/{d}_train.json", "dev": "Train/{d}_dev.json", "test": "Test/{d}_test.json"}
    data = {}
    for split, pattern in files.items():
        docs = []
        for dom in DOMAINS:
            with open(base / pattern.format(d=dom), encoding="utf-8") as f:
                docs += _to_docs(json.load(f), dom, split)
        for d in docs:
            if d["labels"] is not None:
                assert len(d["labels"]) == len(d["text"]), f"{d['id']}: {len(d['labels'])} labels for {len(d['text'])} sentences"
                bad = set(d["labels"]) - set(labels)
                assert not bad, f"{d['id']}: unknown labels {bad}"
        ids = [d["id"] for d in docs]
        assert len(ids) == len(set(ids)), f"duplicate document ids in {split}"
        data[split] = docs
    return labels, data


# ----------------------------------------------------------------------------------------
# 2. Tokenisation: sentences -> one token sequence + sentence boundaries
# ----------------------------------------------------------------------------------------
def clean_sentence(s):
    """Same cleaning idea as Phase 1: ASCII only, single spaces. Never returns an empty string
    (an empty sentence would have no tokens and therefore no vector)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "."


@dataclass
class RRDoc:
    id: str
    domain: str
    ids: np.ndarray          # all token ids of the document, shape (n_tokens,)
    bounds: np.ndarray       # (n_sent, 2): tokens of sentence i are ids[bounds[i,0]:bounds[i,1]]
    y: np.ndarray            # label id per sentence (n_sent,), or None for test documents
    n_truncated: int         # how many sentences were cut to max_sent_tokens

    @property
    def n_sent(self):
        return len(self.bounds)

    def rel_pos(self):
        """Relative position of every sentence: 0.0 = first, 1.0 = last (input feature for the BiLSTM)."""
        n = self.n_sent
        return np.zeros(1, np.float32) if n == 1 else (np.arange(n) / (n - 1)).astype(np.float32)


def encode_rr_doc(doc, tokenizer, label2id, max_sent_tokens=256):
    encs = tokenizer.encode_batch([" " + clean_sentence(s) for s in doc["text"]], add_special_tokens=False)
    pieces, bounds, pos, n_cut = [], [], 0, 0
    for e in encs:
        ids = e.ids[:max_sent_tokens]
        n_cut += len(e.ids) > max_sent_tokens
        pieces.append(ids)
        bounds.append((pos, pos + len(ids)))
        pos += len(ids)
    y = None if doc["labels"] is None else np.array([label2id[l] for l in doc["labels"]], dtype=np.int64)
    return RRDoc(doc["id"], doc["domain"], np.array([t for p in pieces for t in p], dtype=np.int64),
                 np.array(bounds, dtype=np.int64).reshape(-1, 2), y, n_cut)


def encode_rr_docs(docs, tokenizer, labels, max_sent_tokens=256, desc="tokenising"):
    label2id = {l: i for i, l in enumerate(labels)}
    return [encode_rr_doc(d, tokenizer, label2id, max_sent_tokens) for d in tqdm(docs, desc=desc, leave=False)]


# ----------------------------------------------------------------------------------------
# 3. Windows of whole sentences
# ----------------------------------------------------------------------------------------
def plan_sentence_windows(bounds, max_len=1536, stride=1024, mode="context"):
    """Cut a document into windows of whole sentences. Returns a list of (s0, s1): sentences s0..s1-1.

    mode="context": each window holds as many consecutive sentences as fit into max_len tokens.
        The next window starts at the first sentence that begins >= `stride` tokens after the start
        of the current one, so consecutive windows overlap by about max_len - stride tokens.
        No sentence is ever split between windows, and no sentence is skipped.
    mode="sentence": every sentence is its own window (experiment 6: no document context at all)."""
    n = len(bounds)
    if mode == "sentence":
        return [(i, i + 1) for i in range(n)]
    starts, ends = bounds[:, 0], bounds[:, 1]
    assert (ends - starts).max() <= max_len, "a sentence is longer than a window: lower max_sent_tokens"
    wins, s0 = [], 0
    while True:
        s1 = int(np.searchsorted(ends, starts[s0] + max_len, side="right"))   # all sentences that still fit
        wins.append((s0, s1))
        if s1 >= n:
            return wins
        nxt = int(np.searchsorted(starts, starts[s0] + stride, side="left"))
        s0 = min(max(nxt, s0 + 1), s1)          # always move forward, never leave a gap


def window_owners(wins, n_sent):
    """For every sentence, the index of the window where it is most central (most sentences on
    both sides), i.e. where the backbone saw the most context around it. Same idea as Phase 2's merge_windows."""
    owner = np.full(n_sent, -1, dtype=np.int64)
    best = np.full(n_sent, -1, dtype=np.int64)
    for w, (s0, s1) in enumerate(wins):
        idx = np.arange(s0, s1)
        centre = np.minimum(idx - s0, s1 - 1 - idx)
        better = centre > best[s0:s1]
        owner[s0:s1][better] = w
        best[s0:s1][better] = centre[better]
    assert (owner >= 0).all(), "a sentence is not covered by any window"
    return owner


def window_input(doc, win):
    """Token ids of window (s0, s1) and, for every token, the index (0..s1-s0-1) of its sentence
    inside the window. The model averages the token vectors of each sentence with this index."""
    s0, s1 = win
    a, b = doc.bounds[s0, 0], doc.bounds[s1 - 1, 1]
    seg = np.repeat(np.arange(s1 - s0), doc.bounds[s0:s1, 1] - doc.bounds[s0:s1, 0])
    return doc.ids[a:b], seg


# ----------------------------------------------------------------------------------------
# 4. Class weights (rare roles matter as much as Fact under macro-F1)
# ----------------------------------------------------------------------------------------
def class_weights(docs, n_labels, power=0.5, max_weight=20.0):
    """weight(label) = (count of most frequent label / count of label) ** power.
    power = 0.5 ("square-root weights"): Issue (107 training sentences) gets ~6x the weight of
    RatioOfTheDecision (4,301) instead of ~40x with full inverse weights (power = 1), which would make
    the model guess rare labels everywhere. Labels never seen in training get max_weight (they cannot be learned anyway).
    The weights are rescaled so that the average weight over the training sentences is 1
    (the loss keeps the same size as unweighted cross-entropy)."""
    counts = np.bincount(np.concatenate([d.y for d in docs]), minlength=n_labels).astype(np.float64)
    w = np.where(counts > 0, (counts.max() / np.maximum(counts, 1)) ** power, max_weight)
    w = np.minimum(w, max_weight)
    w = w / ((w * counts).sum() / counts.sum())
    return w.astype(np.float32), counts.astype(np.int64)


# ----------------------------------------------------------------------------------------
# 5. Scoring
# ----------------------------------------------------------------------------------------
def rr_scores(gold, pred, labels):
    """gold, pred: {doc_id: [label, ...]} (pred must have the same length per document).
    Per label: precision, recall, F1 over all sentences.
    macro_f1      = average F1 over the labels that occur in gold OR pred (what sklearn's
                    f1_score(average="macro") does; most likely what Codabench uses).
    macro_f1_all  = average over all 13 labels (absent labels count as 0) - shown for reference."""
    g_all, p_all = [], []
    for doc_id, g in gold.items():
        p = pred[doc_id]
        assert len(p) == len(g), f"{doc_id}: {len(p)} predictions for {len(g)} sentences"
        g_all += list(g)
        p_all += list(p)
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(g_all, p_all):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    present = [l for l in labels if tp[l] + fp[l] + fn[l] > 0]
    per = {}
    for lab in labels:
        pr = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        rc = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[lab] = {"p": pr, "r": rc, "f1": f1, "support": tp[lab] + fn[lab], "predicted": tp[lab] + fp[lab]}
    mean = lambda key, labs: float(np.mean([per[l][key] for l in labs])) if labs else 0.0
    return {
        "macro_f1": mean("f1", present), "macro_p": mean("p", present), "macro_r": mean("r", present),
        "macro_f1_all": mean("f1", labels), "accuracy": sum(tp.values()) / max(len(g_all), 1),
        "n_sentences": len(g_all), "labels_scored": len(present), "per_label": per,
    }


def score_by_domain(docs, pred, labels):
    """docs: list of RRDoc or raw docs (with labels). Returns {"all": ..., "CL": ..., "IT": ...}."""
    id2lab = {l: i for i, l in enumerate(labels)}
    out = {}
    for name in ("all",) + DOMAINS:
        sub = [d for d in docs if name == "all" or _domain(d) == name]
        if sub:
            gold = {_id(d): _gold_labels(d, labels) for d in sub}
            out[name] = rr_scores(gold, {k: pred[k] for k in gold}, labels)
    return out


def _id(d):
    return d.id if isinstance(d, RRDoc) else d["id"]


def _domain(d):
    return d.domain if isinstance(d, RRDoc) else d["domain"]


def _gold_labels(d, labels):
    return [labels[i] for i in d.y] if isinstance(d, RRDoc) else d["labels"]


def print_rr_scores(by_dom, title="", per_label=True):
    if title:
        print(title)
    for name, sc in by_dom.items():
        print(f"  {name:3s}: macro-F1 {sc['macro_f1']:.4f} | macro-P {sc['macro_p']:.4f} | macro-R {sc['macro_r']:.4f}"
              f" | accuracy {sc['accuracy']:.4f} | macro-F1 over all 13 {sc['macro_f1_all']:.4f}"
              f" | {sc['labels_scored']} labels, {sc['n_sentences']} sentences")
    if per_label:
        doms = list(by_dom)
        print(f"  {'label':24s}" + "".join(f" {d + ' F1':>8s}" for d in doms) + "".join(f" {d + ' n':>7s}" for d in doms))
        for lab in by_dom[doms[0]]["per_label"]:
            print(f"  {lab:24s}" + "".join(f" {by_dom[d]['per_label'][lab]['f1']:8.3f}" for d in doms)
                  + "".join(f" {by_dom[d]['per_label'][lab]['support']:7d}" for d in doms))


def confusion_matrix(docs, pred, labels):
    """Rows = gold label, columns = predicted label (counts). For the error analysis of close roles."""
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for d in docs:
        for g, p in zip(_gold_labels(d, labels), pred[_id(d)]):
            m[idx[g], idx[p]] += 1
    return m


def top_confusions(m, labels, k=10):
    pairs = [(m[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j and m[i, j]]
    return [(f"{g} -> {p}", int(c)) for c, g, p in sorted(pairs, reverse=True)[:k]]


# ----------------------------------------------------------------------------------------
# 6. Simple baselines (reference points for the report and the viva; NOT submissions)
# ----------------------------------------------------------------------------------------
def baseline_majority(train_docs, docs, labels):
    """Every sentence gets the most frequent training label of its domain."""
    best = {dom: Counter(l for d in train_docs if d["domain"] == dom for l in d["labels"]).most_common(1)[0][0]
            for dom in DOMAINS}
    return {d["id"]: [best[d["domain"]]] * len(d["text"]) for d in docs}


def baseline_position(train_docs, docs, labels, n_bins=20):
    """Only the sentence's POSITION is used: cut every document into n_bins equal parts; each part
    gets the most frequent training label at that relative position (per domain)."""
    def bins(n):
        return np.minimum((np.arange(n) / max(n, 1) * n_bins).astype(int), n_bins - 1)
    table = {}
    for dom in DOMAINS:
        cnt = [Counter() for _ in range(n_bins)]
        for d in train_docs:
            if d["domain"] == dom:
                for b, l in zip(bins(len(d["labels"])), d["labels"]):
                    cnt[b][l] += 1
        table[dom] = [c.most_common(1)[0][0] if c else "Fact" for c in cnt]
    return {d["id"]: [table[d["domain"]][b] for b in bins(len(d["text"]))] for d in docs}


# ----------------------------------------------------------------------------------------
# 7. Submission
# ----------------------------------------------------------------------------------------
def write_submission(pred, test_docs, labels, out_dir):
    """pred: {doc_id: [label, ...]}. Writes prediction.json (CL and IT test documents together,
    original sentence order) and submission.zip with prediction.json at the ZIP ROOT, then re-checks it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {}
    for d in test_docs:
        p = [str(l) for l in pred[d["id"]]]
        assert len(p) == len(d["text"]), f"{d['id']}: {len(p)} predictions for {len(d['text'])} sentences"
        clean[d["id"]] = p
    extra = set(pred) - set(clean)
    assert not extra, f"predictions for unknown documents: {list(extra)[:3]}"
    json_path = out_dir / "prediction.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean, f)
    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="prediction.json")
    check_submission(zip_path, test_docs, labels)
    return zip_path


def check_submission(zip_path, test_docs, labels, verbose=True):
    """Re-open the zip exactly as the grader would: right file at the root, every official test id
    exactly once, one valid label per sentence."""
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        pred = json.loads(z.read("prediction.json").decode("utf-8"))
    want = {d["id"]: len(d["text"]) for d in test_docs}
    assert set(pred) == set(want), "document ids differ from the official test ids"
    for doc_id, labs in pred.items():
        assert isinstance(labs, list) and len(labs) == want[doc_id], f"{doc_id}: wrong number of labels"
        assert all(l in labels for l in labs), f"{doc_id}: label not in label_vocab.json"
    if verbose:
        counts = Counter(l for labs in pred.values() for l in labs)
        print(f"submission OK: {zip_path} | {len(pred)} documents | {sum(want.values()):,} sentences | "
              f"label counts {dict(counts.most_common())}")
    return pred


# ----------------------------------------------------------------------------------------
# 8. Experiment log (one JSON line per run)
# ----------------------------------------------------------------------------------------
def log_experiment(root, record):
    path = Path(root) / "experiments" / "phase3" / "experiments.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_experiments(root):
    path = Path(root) / "experiments" / "phase3" / "experiments.jsonl"
    rows = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows
