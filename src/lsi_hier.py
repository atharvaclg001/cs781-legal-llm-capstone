"""
lsi_hier.py  -  Phase-5 experiment #6: hierarchical WHOLE-CASE model on top of the frozen experiment-1 model
(same two-stage idea as Phase 4's cjpe_hier.py, which gave the best Phase-4 test score)

Why: experiments 1-5 read only the first 768 + last 768 tokens. 31-36% of the cases are longer than that, and a
statute can be used anywhere in a case (P5_01: no single part of a case dominates). This model reads everything:

    case (n tokens) --cut into chunks of 1536 tokens (at most 16; longer cases: first 8 + last 8 chunks)-->
    every chunk --FROZEN experiment-1 model (A2 fully fine-tuned)--> chunk vector (640, its attention pooling)
                                                                  + the model's own 100 statute logits for that chunk
    chunk vectors + chunk-position embedding --> BiGRU over the chunks (each chunk sees its neighbours)
    --> LABEL-WISE attention: every statute has its OWN attention over the chunks
        (statute 302 may be decided by the chunk with the injuries, statute 498A by the chunk with the dowry demands)
    --> one score per statute  +  skip connection: max over chunks of the frozen model's logit for that statute

The skip connection makes the head START from a training-free rule ("a statute applies if ANY chunk says so",
= max pooling of the chunk logits) and only learn corrections on top of it.

Technique: layer freezing / two-stage training (encoder frozen, only the small head is trained).

Which cases train the head?  The encoder was fine-tuned on the TRAINING cases, so its outputs on those cases are
too confident (it memorised them) -> a head trained on them would trust the encoder blindly (stacking leakage).
The encoder never saw dev: the head is trained on dev with 5-fold cross-validation x 3 seeds; every dev case gets an
out-of-fold prediction (honest dev score); test = average of the 15 fold heads.

Same chunk outputs give the SECTION ABLATION: first chunk / last chunk / first + last / all chunks.
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

import lsi_utils as lu
import lsi_models as lm


# ----------------------------------------------------------------------------------------
# 1. Chunk outputs of the frozen fine-tuned encoder (cached)
# ----------------------------------------------------------------------------------------
def chunk_bounds(n, L=1536, max_chunks=16):
    """Non-overlapping chunks covering a case of n tokens, from the start; the last chunk may be shorter.
    More than max_chunks -> keep the first max_chunks/2 and the last max_chunks/2 chunks."""
    b = [(s, min(s + L, n)) for s in range(0, n, L)]
    if len(b) > max_chunks:
        h = max_chunks // 2
        b = b[:h] + b[-h:]
    return b


@torch.no_grad()
def encode_chunks(model, ids_list, device, amp_dtype, cache_path, L=1536, max_chunks=16, token_budget=12288,
                  desc="chunks"):
    """Returns (vecs, logits, starts): vecs[starts[i]:starts[i+1]] = chunk vectors of case i (in reading order),
    logits = the frozen model's 100 statute logits for every chunk. Cached as .npz."""
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
    lengths = np.array([len(c) for c in chunks])
    d, nl = model.out.in_features, model.out.out_features
    vecs = np.zeros((len(chunks), d), dtype=np.float16)
    logits = np.zeros((len(chunks), nl), dtype=np.float32)
    for b in tqdm(lm.micro_batches(np.argsort(lengths), lengths, token_budget), desc=desc):
        x, m = lm.collate([chunks[j] for j in b])
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            v = model.encode(x.to(device), m.to(device)).float()
            lg = model.out(v).float()
        vecs[b] = v.cpu().numpy().astype(np.float16)
        logits[b] = lg.cpu().numpy()
    starts = np.zeros(len(ids_list) + 1, dtype=np.int64)
    np.cumsum(np.bincount(owner, minlength=len(ids_list)), out=starts[1:])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, vecs=vecs, logits=logits, starts=starts)
    return vecs, logits, starts


def select_chunks(arr, starts, which="all"):
    """List (one per case) of per-chunk arrays, keeping only the requested section of each case."""
    out = []
    for i in range(len(starts) - 1):
        v = arr[starts[i]:starts[i + 1]]
        if which == "first":
            v = v[:1]
        elif which == "last":
            v = v[-1:]
        elif which == "first+last":
            v = v[[0, -1]] if len(v) > 1 else v
        elif which != "all":
            raise ValueError(which)
        out.append(np.asarray(v, dtype=np.float32))
    return out


# ----------------------------------------------------------------------------------------
# 2. The chunk-level head
# ----------------------------------------------------------------------------------------
class HierHeadML(nn.Module):
    """(n_chunks, 640) chunk vectors + (n_chunks, 100) chunk logits -> (100,) statute scores.
    LayerNorm + Linear 640 -> hidden, + embedding of the chunk position (0 = first chunk ... 15),
    BiGRU over the chunks, then for every statute l:
        a_l = softmax_c( q_l . h_c )          its own attention over the chunks   (label_attention=True)
        s_l = w_l . (sum_c a_l,c h_c) + b_l
    label_attention=False: ONE attention shared by all statutes (ablation).
    Final score_l = s_l + g_l * max_c(chunk_logit_c,l), g_l starts at 1 and s_l starts at 0
    -> at step 0 the head equals the training-free "max over chunks" rule."""
    def __init__(self, d=640, n_labels=100, hidden=128, max_chunks=16, dropout=0.2, label_attention=True):
        super().__init__()
        self.label_attention = label_attention
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, hidden)
        self.pos = nn.Embedding(max_chunks, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        h2 = 2 * hidden
        self.query = nn.Parameter(torch.zeros(n_labels if label_attention else 1, h2))   # 0 -> uniform attention
        self.w = nn.Parameter(torch.zeros(n_labels, h2))                                 # 0 -> s_l = 0 at start
        self.b = nn.Parameter(torch.zeros(n_labels))
        self.gate = nn.Parameter(torch.ones(n_labels))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, z, lengths, return_attn=False):
        # x: (B, C, d) chunk vectors, z: (B, C, L) chunk logits, padded at the end; lengths: (B,)
        B, C, _ = x.shape
        mask = torch.arange(C, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)          # (B, C)
        pos = torch.arange(C, device=x.device).clamp(max=self.pos.num_embeddings - 1)
        h = self.drop(F.gelu(self.proj(self.norm(x)))) + self.pos(pos).unsqueeze(0)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=C)  # (B, C, 2H)
        h = self.drop(h)
        att = torch.einsum("bch,lh->blc", h, self.query).masked_fill(~mask.unsqueeze(1), float("-inf"))
        a = torch.softmax(att, dim=-1)                                                      # (B, L or 1, C)
        v = torch.einsum("blc,bch->blh", a, h)                                              # (B, L or 1, 2H)
        s = (v * self.w.unsqueeze(0)).sum(-1) + self.b                                      # (B, L)
        zmax = z.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values              # (B, L)
        out = s + self.gate * zmax
        return (out, a) if return_attn else out


def _batch(V, Z, idx, device):
    C = max(len(V[i]) for i in idx)
    d, nl = V[idx[0]].shape[1], Z[idx[0]].shape[1]
    x = np.zeros((len(idx), C, d), dtype=np.float32)
    z = np.zeros((len(idx), C, nl), dtype=np.float32)
    for r, i in enumerate(idx):
        x[r, :len(V[i])] = V[i]
        z[r, :len(Z[i])] = Z[i]
    lengths = torch.tensor([len(V[i]) for i in idx])
    return torch.from_numpy(x).to(device), torch.from_numpy(z).to(device), lengths.to(device)


def train_head(V, Z, Y, device, epochs=20, lr=1e-3, weight_decay=1e-2, batch_size=64, pos_weight_power=0.5,
               seed=0, label_attention=True, hidden=128, dropout=0.2):
    """Train one head (small: seconds to a minute). Fixed number of epochs (inside cross-validation there is
    no extra data to choose an epoch on)."""
    torch.manual_seed(seed)
    head = HierHeadML(V[0].shape[1], Y.shape[1], hidden, dropout=dropout, label_attention=label_attention).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor(lm.pos_weights(Y, pos_weight_power), dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    steps = epochs * math.ceil(len(V) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    head.train()
    for _ in range(epochs):
        perm = rng.permutation(len(V))
        for k in range(0, len(perm), batch_size):
            idx = perm[k:k + batch_size]
            x, z, ln = _batch(V, Z, idx, device)
            loss = F.binary_cross_entropy_with_logits(head(x, z, ln), Yt[idx], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    head.eval()
    return head


@torch.no_grad()
def predict_head(heads, V, Z, device, batch_size=256):
    """Average sigmoid probabilities of one or several heads (ensemble of the fold heads)."""
    heads = heads if isinstance(heads, list) else [heads]
    P = np.zeros((len(V), heads[0].w.shape[0]), dtype=np.float32)
    for k in range(0, len(V), batch_size):
        idx = list(range(k, min(k + batch_size, len(V))))
        x, z, ln = _batch(V, Z, idx, device)
        for h in heads:
            P[idx] += torch.sigmoid(h(x, z, ln)).cpu().numpy() / len(heads)
    return P


@torch.no_grad()
def attention_by_position(heads, V, Z, device, min_chunks=4):
    """Mean attention weight per chunk position (first ... last), averaged over statutes, for cases with
    >= min_chunks chunks: which parts of long cases does the head use?"""
    rows = []
    for i in range(len(V)):
        if len(V[i]) < min_chunks:
            continue
        x, z, ln = _batch(V, Z, [i], device)
        a = np.mean([h(x, z, ln, return_attn=True)[1][0].mean(0).cpu().numpy() for h in heads], axis=0)
        rows.append(a)
    if not rows:
        return {}
    n = min(len(r) for r in rows)
    first = np.mean([r[:n // 2] for r in rows], axis=0) if n >= 2 else []
    return {"cases": len(rows), "first_chunks": [round(float(x), 3) for x in first],
            "last_chunk": round(float(np.mean([r[-1] for r in rows])), 3),
            "uniform_would_be": round(float(np.mean([1 / len(r) for r in rows])), 3)}


def cross_validate(V, Z, Y, device, folds=5, seed=0, **kw):
    """k-fold on dev (random folds; with 10k cases every statute appears in every fold).
    Returns (out-of-fold probabilities (n, 100), list of the k fold heads)."""
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(Y)) % folds
    oof, heads = np.zeros(Y.shape, dtype=np.float32), []
    for f in range(folds):
        tr, te = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
        h = train_head([V[i] for i in tr], [Z[i] for i in tr], Y[tr], device, seed=seed + f, **kw)
        oof[te] = predict_head(h, [V[i] for i in te], [Z[i] for i in te], device)
        heads.append(h)
    return oof, heads


def threshold_report(Y, prob):
    """Same threshold analysis as the flat experiments (cross-fitted global vs per statute)."""
    t_g, _ = lu.tune_global(Y, prob)
    t_l = lu.tune_per_label(Y, prob, min_pos=5, fallback=t_g)
    cf = lu.cross_fitted_scores(Y, prob)
    strategy = "per_label" if cf["per_label"]["macro_f1"] > cf["global"]["macro_f1"] else "global"
    return {"global_threshold": t_g, "per_label_thresholds": [float(t) for t in t_l], "cross_fitted": cf,
            "strategy": strategy, "best_cross_fitted_macro_f1": cf[strategy]["macro_f1"]}


# ----------------------------------------------------------------------------------------
# 3. The whole experiment (cached step by step)
# ----------------------------------------------------------------------------------------
def run_hier(root, encoder_exp, seeds=(0, 1, 2), folds=5, out_name="A2_hier_frozen", ablations=True, **head_kw):
    """encoder_exp = the experiment-1 Experiment object (its trained model is loaded from its run folder).
    Writes everything into checkpoints/phase5/<out_name>/ and returns the report dict."""
    root = Path(root)
    out = root / "checkpoints" / "phase5" / out_name
    out.mkdir(parents=True, exist_ok=True)
    rep_path = out / "report.json"
    rep = json.load(open(rep_path)) if rep_path.exists() else {}
    test_ids = [d["id"] for d in encoder_exp.data["test"]]
    Y = encoder_exp.Y["dev"]
    dev_npz, test_npz = out / "chunks_dev.npz", out / "chunks_test.npz"
    if not (dev_npz.exists() and test_npz.exists()):
        model = encoder_exp._model()
        enc = lambda split, p: encode_chunks(model, encoder_exp.ids[split], encoder_exp.device, encoder_exp.amp, p,
                                             desc=f"encoding {split} chunks")
        enc("dev", dev_npz)
        enc("test", test_npz)
    Vd, Zd, Sd = encode_chunks(None, None, None, None, dev_npz)
    Vt, Zt, St = encode_chunks(None, None, None, None, test_npz)
    n_chunks = np.diff(Sd)
    print(f"dev: {len(Y)} cases, {Sd[-1]} chunks (mean {n_chunks.mean():.2f}, max {n_chunks.max()}) | test: {St[-1]} chunks")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V_all, Z_all = select_chunks(Vd, Sd), select_chunks(Zd, Sd)

    # --- training-free baselines from the chunk logits ---
    if "baselines" not in rep:
        sig = lm.sigmoid
        base = {"first chunk only (frozen #1)": sig(np.stack([z[0] for z in Z_all])),
                "max over chunks": sig(np.stack([z.max(0) for z in Z_all])),
                "mean over chunks": sig(np.stack([z.mean(0) for z in Z_all]))}
        rep["baselines"] = {k: threshold_report(Y, p)["best_cross_fitted_macro_f1"] for k, p in base.items()}
        json.dump(rep, open(rep_path, "w"), indent=1)
    print("training-free baselines (dev, cross-fitted macro-F1):", rep["baselines"])

    # --- section ablation + attention ablation (one seed each, cheap) ---
    if ablations and "ablations" not in rep:
        rep["ablations"] = {}
        for which in tqdm(["first", "last", "first+last", "all"], desc="section ablation"):
            V, Z = select_chunks(Vd, Sd, which), select_chunks(Zd, Sd, which)
            oof, _ = cross_validate(V, Z, Y, device, folds, seed=0, **head_kw)
            rep["ablations"][f"chunks: {which}"] = threshold_report(Y, oof)["best_cross_fitted_macro_f1"]
        oof, _ = cross_validate(V_all, Z_all, Y, device, folds, seed=0, label_attention=False,
                                **{k: v for k, v in head_kw.items() if k != "label_attention"})
        rep["ablations"]["all chunks, ONE shared attention"] = threshold_report(Y, oof)["best_cross_fitted_macro_f1"]
        json.dump(rep, open(rep_path, "w"), indent=1)
    if ablations:
        print("ablations (dev, cross-fitted macro-F1, seed 0):", rep["ablations"])

    # --- final model: all chunks, label-wise attention, folds x seeds ---
    oof_path, test_path = out / "dev_oof_probs.npy", out / "test_probs.npy"
    if not (oof_path.exists() and test_path.exists()):
        oofs, heads = [], []
        for s in tqdm(seeds, desc="final heads (seeds)"):
            o, h = cross_validate(V_all, Z_all, Y, device, folds, seed=100 * s, **head_kw)
            oofs.append(o)
            heads += h
        oof = np.mean(oofs, axis=0)
        test_p = predict_head(heads, select_chunks(Vt, St), select_chunks(Zt, St), device)
        np.save(oof_path, oof)
        np.save(test_path, test_p)
        rep["per_seed_cross_fitted"] = [threshold_report(Y, o)["best_cross_fitted_macro_f1"] for o in oofs]
        rep["attention_by_position"] = attention_by_position(heads[:folds], V_all, Z_all, device)
        rep["trainable_params"] = int(sum(p.numel() for p in heads[0].parameters()))
        json.dump(rep, open(rep_path, "w"), indent=1)
    oof, test_p = np.load(oof_path), np.load(test_path)

    if "final" not in rep:
        tr = threshold_report(Y, oof)
        thr = np.asarray(tr["per_label_thresholds"]) if tr["strategy"] == "per_label" else tr["global_threshold"]
        P = lu.decide(oof, thr)
        n_tok = np.array([len(x) for x in encoder_exp.ids["dev"]])
        r1 = encoder_exp.report()                       # experiment #1's own dev decisions, for comparison
        thr1 = np.asarray(r1["per_label_thresholds"]) if r1["strategy"] == "per_label" else r1["global_threshold"]
        P1 = lu.decide(lm.sigmoid(encoder_exp._logits("dev").astype(np.float32)), thr1)
        by_len = []
        for lo, hi, name in [(0, 1537, "<= 1536 tokens (fits #1 completely)"), (1537, 6145, "1537-6144"),
                             (6145, 10 ** 9, "> 6144")]:
            sel = (n_tok >= lo) & (n_tok < hi)
            if sel.sum():
                by_len.append({"length": name, "cases": int(sel.sum()),
                               "exp1_macro_f1": lu.ml_scores(Y[sel], P1[sel])["macro_f1"],
                               "hier_macro_f1": lu.ml_scores(Y[sel], P[sel])["macro_f1"]})
        rep["final"] = {**tr, "chosen": lu.ml_scores(Y, P),
                        "f1_by_frequency": lu.f1_by_frequency(Y, P, encoder_exp.train_counts),
                        "by_length_vs_exp1": by_len, "top_confusions": lu.top_confusions(Y, P)}
        json.dump(rep, open(rep_path, "w"), indent=1)
    f = rep["final"]
    print(f"final (dev, out-of-fold): global {f['cross_fitted']['global']['macro_f1']} | per statute "
          f"{f['cross_fitted']['per_label']['macro_f1']} -> {f['strategy']} | per-seed {rep.get('per_seed_cross_fitted')}")
    print("by length (exp #1 vs hierarchical):", [(r["length"], r["exp1_macro_f1"], r["hier_macro_f1"]) for r in f["by_length_vs_exp1"]])
    print("attention by chunk position (cases with >= 4 chunks):", rep.get("attention_by_position"))

    zip_path = out / "submission.zip"
    if not zip_path.exists():
        thr = np.asarray(f["per_label_thresholds"]) if f["strategy"] == "per_label" else f["global_threshold"]
        lu.write_submission(test_ids, lu.decide(test_p, thr), out)
    lu.check_submission(zip_path, test_ids)

    if not any(r.get("run") == out_name for r in lu.read_experiments(root)):
        lu.log_experiment(root, {"run": out_name, "phase1": "A2", "method": "frozen #1 + hierarchical head (dev CV)",
                                 "input": "whole case (<= 16 x 1536)", "trainable_M_train": round(rep["trainable_params"] / 1e6, 3),
                                 "dev_macro_f1_global": f["cross_fitted"]["global"]["macro_f1"],
                                 "dev_macro_f1_per_label": f["cross_fitted"]["per_label"]["macro_f1"],
                                 "strategy": f["strategy"], "dev_micro_f1": f["chosen"]["micro_f1"],
                                 "dev_subset_acc": f["chosen"]["subset_acc"]})
    return rep
