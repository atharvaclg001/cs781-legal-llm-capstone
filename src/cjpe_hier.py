"""
cjpe_hier.py  -  Phase-4 hierarchical (whole-case) model: chunk vectors -> chunk-level BiGRU + attention -> outcome

Why: experiments 1-5 read only the LAST 1536 tokens. Dev cases have 3,400 tokens on average and the longest
quarter (5,400-52,000 tokens) is the hardest (A2: 0.755 vs 0.867). A hierarchical model reads the WHOLE case:

    case (n tokens) --cut into chunks of 1536 tokens (from the end backwards; at most the last 16 chunks)-->
    every chunk --FROZEN experiment-1 model (A2 fine-tuned)--> one 640-d chunk vector (its attention pooling)
    chunk vectors + "distance from the end" embedding --> BiGRU over the chunks --> attention over the chunks
    --> Linear -> outcome scores

Technique: LAYER FREEZING / two-stage training. Stage 1 = experiment 1 (encoder fine-tuned on last-1536 inputs);
stage 2 = the encoder is completely frozen, only the small chunk-level head (~0.3M parameters) is trained.

Which cases train the head?  The encoder was fine-tuned on the TRAINING cases, so its vectors for those cases are
"too good" (it has memorised their labels) - a head trained on them would learn to trust the encoder blindly and
fail on new cases (classic stacking leakage). The encoder has NEVER seen the dev cases, so we train the head on
dev with 5-fold cross-validation: every dev case gets a prediction from a head that did not see it
(out-of-fold), and the 5 fold heads are averaged for test. (Variant 'train': head trained on the train
vectors, evaluated on dev -> measures the leakage effect.)

The same chunk vectors give the SECTION ABLATION almost for free: the same head on
  'last' (only the last chunk = what experiment 1 sees), 'first' (only the first chunk),
  'first+last', 'all' (whole case).
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm.auto import tqdm

import cjpe_utils as cu
import cjpe_models as cm


# ----------------------------------------------------------------------------------------
# 1. Chunk vectors from the frozen fine-tuned encoder (cached)
# ----------------------------------------------------------------------------------------
def chunk_bounds(n, L=1536, max_chunks=16):
    """Non-overlapping chunks covering a case of n tokens, built from the END backwards (the end matters most);
    the first chunk may be shorter. At most the last `max_chunks` chunks are kept (99% of cases fit in 16)."""
    ends = list(range(n, 0, -L))[:max_chunks]
    return [(max(0, e - L), e) for e in reversed(ends)]


@torch.no_grad()
def encode_chunks(model, ids_list, device, amp_dtype, cache_path, L=1536, max_chunks=16, batch_size=8, desc="chunks"):
    """Returns (vecs, logits, starts): vecs[starts[i]:starts[i+1]] = chunk vectors of case i (first -> last chunk),
    logits = the encoder's own 0/1 scores for every chunk. Cached as .npz."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        z = np.load(cache_path)
        return z["vecs"], z["logits"], z["starts"]
    model.eval()
    chunks, owner = [], []
    for i, ids in enumerate(ids_list):
        for a, b in chunk_bounds(len(ids), L, max_chunks):
            chunks.append(ids[a:b])
            owner.append(i)
    order = np.argsort([len(c) for c in chunks])               # similar lengths in a batch -> little padding
    d = model.out.in_features
    vecs = np.zeros((len(chunks), d), dtype=np.float16)
    logits = np.zeros((len(chunks), 2), dtype=np.float32)
    for k in tqdm(range(0, len(chunks), batch_size), desc=desc):
        idx = order[k:k + batch_size]
        x, m = cm.collate([chunks[j] for j in idx])
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            v = model.encode(x.to(device), m.to(device)).float()
            lg = model.out(v).float()
        vecs[idx] = v.cpu().numpy().astype(np.float16)
        logits[idx] = lg.cpu().numpy()
    starts = np.zeros(len(ids_list) + 1, dtype=np.int64)
    np.cumsum(np.bincount(owner, minlength=len(ids_list)), out=starts[1:])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, vecs=vecs, logits=logits, starts=starts)
    return vecs, logits, starts


def select_chunks(vecs, starts, which="all"):
    """List (one per case) of chunk-vector arrays, keeping only the requested section of each case."""
    out = []
    for i in range(len(starts) - 1):
        v = vecs[starts[i]:starts[i + 1]]
        if which == "last":
            v = v[-1:]
        elif which == "first":
            v = v[:1]
        elif which == "first+last":
            v = v[[0, -1]] if len(v) > 1 else v
        elif which != "all":
            raise ValueError(which)
        out.append(np.asarray(v, dtype=np.float32))
    return out


# ----------------------------------------------------------------------------------------
# 2. The chunk-level head
# ----------------------------------------------------------------------------------------
class HierHead(nn.Module):
    """chunk vectors (n_chunks, 640) -> scores (2,).
    LayerNorm + Linear 640 -> 128, + embedding of the chunk's distance from the end (0 = last chunk, 15 = 16th from
    the end): tells the head WHERE in the case a chunk is. BiGRU over the chunks (each chunk sees its neighbours),
    attention pooling over chunks (learns which chunks matter), dropout, Linear(256 -> 2)."""
    def __init__(self, d=640, hidden=128, max_chunks=16, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, hidden)
        self.pos = nn.Embedding(max_chunks, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, x, lengths, return_attn=False):
        # x: (B, C, d) padded at the end, lengths: (B,) number of real chunks
        B, C, _ = x.shape
        ar = torch.arange(C, device=x.device)
        mask = ar.unsqueeze(0) < lengths.unsqueeze(1)                          # (B, C) real chunks
        dist = (lengths.unsqueeze(1) - 1 - ar.unsqueeze(0)).clamp(min=0)      # distance from the last real chunk
        h = self.drop(F.gelu(self.proj(self.norm(x)))) + self.pos(dist)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=C)
        a = torch.softmax(self.att(h).squeeze(-1).masked_fill(~mask, float("-inf")), dim=1)
        v = torch.einsum("bc,bcd->bd", a, h)
        logits = self.out(self.drop(v))
        return (logits, a) if return_attn else logits


def _batch(seqs, idx, device):
    C = max(len(seqs[i]) for i in idx)
    d = seqs[idx[0]].shape[1]
    x = np.zeros((len(idx), C, d), dtype=np.float32)
    for r, i in enumerate(idx):
        x[r, :len(seqs[i])] = seqs[i]
    lengths = torch.tensor([len(seqs[i]) for i in idx])
    return torch.from_numpy(x).to(device), lengths.to(device)


def train_head(seqs, y, device, epochs=30, lr=1e-3, weight_decay=1e-2, batch_size=32, class_weights=None, seed=0,
               hidden=128, dropout=0.2):
    """Train one HierHead (small: trains in seconds). Fixed number of epochs (no early stopping: inside
    cross-validation there is no extra data to stop on)."""
    torch.manual_seed(seed)
    head = HierHead(seqs[0].shape[1], hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    w = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    steps = epochs * math.ceil(len(seqs) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    head.train()
    for _ in range(epochs):
        perm = rng.permutation(len(seqs))
        for k in range(0, len(perm), batch_size):
            idx = perm[k:k + batch_size]
            x, ln = _batch(seqs, idx, device)
            loss = F.cross_entropy(head(x, ln), yt[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    head.eval()
    return head


@torch.no_grad()
def predict_head(heads, seqs, device, batch_size=128, return_attn=False):
    """Average P(label 1) of one or several heads (an ensemble of the cross-validation fold heads)."""
    heads = heads if isinstance(heads, list) else [heads]
    p = np.zeros(len(seqs))
    attn = [None] * len(seqs)
    for k in range(0, len(seqs), batch_size):
        idx = list(range(k, min(k + batch_size, len(seqs))))
        x, ln = _batch(seqs, idx, device)
        for h in heads:
            lg, a = h(x, ln, return_attn=True)
            p[idx] += torch.softmax(lg, -1)[:, 1].cpu().numpy() / len(heads)
            if return_attn:
                for r, i in enumerate(idx):
                    a_i = a[r, :len(seqs[i])].cpu().numpy() / len(heads)
                    attn[i] = a_i if attn[i] is None else attn[i] + a_i
    return (p, attn) if return_attn else p


def cross_validate(seqs, y, device, folds=5, seed=0, **kw):
    """Stratified k-fold on dev. Returns (out-of-fold P(1) for every case, list of the k fold heads)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(y), dtype=int)
    for c in (0, 1):                                   # stratified: same label balance in every fold
        idx = rng.permutation(np.flatnonzero(y == c))
        fold_of[idx] = np.arange(len(idx)) % folds
    oof, heads = np.zeros(len(y)), []
    for f in range(folds):
        tr, te = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
        n1 = y[tr].sum()
        cw = [len(tr) / (2 * (len(tr) - n1)), len(tr) / (2 * n1)]
        h = train_head([seqs[i] for i in tr], y[tr], device, class_weights=cw, seed=seed + f, **kw)
        oof[te] = predict_head(h, [seqs[i] for i in te], device)
        heads.append(h)
    return oof, heads


def scores_with_threshold(y, p):
    thr, _ = cu.best_threshold(y, p)
    return {"at_0.5": cu.f1_scores(y, (p >= 0.5).astype(int)), "threshold": thr,
            "tuned": cu.f1_scores(y, (p >= thr).astype(int))}
