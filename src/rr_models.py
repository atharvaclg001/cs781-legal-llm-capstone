"""
rr_models.py  -  the Phase-3 model (hierarchical sentence tagger) + shared training loop

    document --(windows of whole sentences, rr_utils)--> Phase-1 backbone (one window at a time)
             --> one vector per sentence = mean of its token vectors (from the window where it is most central)
             --> [sentence vector, relative position, domain vector]
             --> BiLSTM over ALL sentences of the document (connects the windows)
             --> Linear -> 13 role scores per sentence

Re-used from Phase 2 (src/ner_models.py): the backbone (same blocks as Phase 1), the strict Phase-1 weight
loader, LoRA / LoRMA adapters and the parameter counter. New here: sentence pooling, the sentence-level head,
gradient checkpointing per window (so a 71,000-token document fits in 6 GB) and the training loop.

Why gradient checkpointing?  The BiLSTM needs the vectors of ALL sentences before it can compute the loss,
so all windows of a document must be in one backward pass. Normally PyTorch keeps every layer's activations
of every window in memory for the backward pass (47 windows x 10 layers for the longest document: too much).
With checkpointing it keeps only each window's small OUTPUT (one 640-vector per sentence) and recomputes
the window's activations during backward, one window at a time. Cost: ~1 extra forward pass (~30% slower).
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

from ner_models import load_phase1_backbone, add_adapters, count_params   # noqa: F401  (re-exported)
import rr_utils as ru


# ----------------------------------------------------------------------------------------
# 1. The model
# ----------------------------------------------------------------------------------------
class SentenceHead(nn.Module):
    """Sentence features -> 13 role scores.
    'bilstm': BiLSTM over the sentences of the document, then Linear (context across the whole document).
    'mlp'   : every sentence on its own: Linear -> GELU -> Linear (used for the sentence-only ablation)."""
    def __init__(self, in_dim, n_labels, kind="bilstm", hidden=256, dropout=0.2):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        if kind == "bilstm":
            self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * hidden, n_labels)
        elif kind == "mlp":
            self.mlp = nn.Sequential(nn.Linear(in_dim, 2 * hidden), nn.GELU())
            self.out = nn.Linear(2 * hidden, n_labels)
        else:
            raise ValueError(kind)

    def forward(self, feats):                      # feats: (n_sent, in_dim), float32
        x = self.drop(feats)
        if self.kind == "bilstm":
            x, _ = self.lstm(x.unsqueeze(0))       # one document = a batch of 1 sequence
            x = x.squeeze(0)
        else:
            x = self.mlp(x)
        return self.out(self.drop(x))


class RRModel(nn.Module):
    """backbone (Phase-1 weights) -> mean pooling per sentence -> SentenceHead.
    use_pos: append the relative position (0..1) of the sentence; use_domain: add a learned CL/IT vector."""
    def __init__(self, backbone, n_labels, causal=True, head="bilstm", hidden=256, dropout=0.2,
                 use_pos=True, use_domain=True, domain_dim=16):
        super().__init__()
        self.backbone, self.causal = backbone, causal
        self.use_pos, self.use_domain = use_pos, use_domain
        d = backbone.tok_emb.weight.shape[1]
        self.d = d
        self.domain_emb = nn.Embedding(len(ru.DOMAINS), domain_dim) if use_domain else None
        in_dim = d + (1 if use_pos else 0) + (domain_dim if use_domain else 0)
        self.head = SentenceHead(in_dim, n_labels, head, hidden, dropout)

    def head_parameters(self):
        """Everything that is NOT the Phase-1 backbone (new, randomly initialised -> higher learning rate)."""
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    # -- backbone part: token ids -> sentence vectors --------------------------------------
    def _pool(self, x, slot, n_slots):
        """x: (B, T) token ids, slot: (B*T,) which output row each token is added to (n_slots = 'throw away').
        Returns (n_slots, d) SUMS of token vectors (float32). Only this small tensor leaves the checkpoint."""
        h = self.backbone(x, causal=self.causal)
        h = h.reshape(-1, h.shape[-1]).float()
        out = torch.zeros(n_slots + 1, h.shape[-1], device=h.device, dtype=torch.float32)
        return out.index_add_(0, slot, h)[:n_slots]

    def sentence_vectors(self, doc, wins, owner, device, amp_dtype=torch.float32, win_batch=2, use_ckpt=False):
        """Mean token vector of every sentence of `doc`, each taken from its owner window.
        Windows are processed `win_batch` at a time; for causal backbones shorter windows are padded at the
        END (a causal token never looks right, so padding cannot change the real tokens). Bidirectional
        backbones (the encoder) would see the padding, so they get one window at a time."""
        bs = win_batch if self.causal else 1
        n_tok = torch.from_numpy(doc.bounds[:, 1] - doc.bounds[:, 0]).to(device).float()
        parts, order = [], []
        for b0 in range(0, len(wins), bs):
            group = list(range(b0, min(b0 + bs, len(wins))))
            inputs = [ru.window_input(doc, wins[w]) for w in group]
            T = max(len(ids) for ids, _ in inputs)
            owned = [np.nonzero(owner[wins[w][0]:wins[w][1]] == w)[0] + wins[w][0] for w in group]
            n_slots = sum(len(o) for o in owned)
            if n_slots == 0:
                continue                                   # this group owns no sentence: skip it
            x = np.zeros((len(group), T), dtype=np.int64)
            slot = np.full((len(group), T), n_slots, dtype=np.int64)
            base = 0
            for r, (w, (ids, seg)) in enumerate(zip(group, inputs)):
                x[r, :len(ids)] = ids
                local = np.full(wins[w][1] - wins[w][0], n_slots, dtype=np.int64)   # sentence in window -> slot
                o = owned[r] - wins[w][0]
                local[o] = base + np.arange(len(o))
                slot[r, :len(ids)] = local[seg]
                base += len(o)
            xt = torch.from_numpy(x).to(device)
            st = torch.from_numpy(slot.reshape(-1)).to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                if use_ckpt:
                    part = checkpoint(self._pool, xt, st, n_slots, use_reentrant=False)
                else:
                    part = self._pool(xt, st, n_slots)
            parts.append(part)
            order.append(np.concatenate(owned))
        sums = torch.cat(parts)
        order = np.concatenate(order)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))                 # row of sentence i in `sums`
        return sums[torch.from_numpy(inv).to(device)] / n_tok.unsqueeze(1)

    # -- full model: document -> role scores ------------------------------------------------
    def forward_doc(self, doc, wins, owner, device, amp_dtype=torch.float32, win_batch=2, use_ckpt=False):
        v = self.sentence_vectors(doc, wins, owner, device, amp_dtype, win_batch, use_ckpt)
        feats = [v]
        if self.use_pos:
            feats.append(torch.from_numpy(doc.rel_pos()).to(device).unsqueeze(1))
        if self.use_domain:
            dom = torch.full((doc.n_sent,), ru.DOMAINS.index(doc.domain), device=device, dtype=torch.long)
            feats.append(self.domain_emb(dom))
        return self.head(torch.cat(feats, dim=1))          # head runs in float32 (outside autocast)


# ----------------------------------------------------------------------------------------
# 1b. More parameter-efficient methods: rsLoRA and IA^3 (LoRA / LoRMA come from ner_models.add_adapters)
# ----------------------------------------------------------------------------------------
class IA3Scale(nn.Module):
    """IA^3 (Liu et al. 2022, "Few-shot PEFT is better and cheaper than in-context learning"):
    no new matrices at all - only a learned VECTOR that multiplies activations element-wise.
    where='out': y = W x, then y[..., start:] *= l    (used on qkv: scales the KEYS and VALUES, not the queries)
    where='in' : y = W (l * x)                         (used on w2: scales the FFN's inner activations)
    l starts at 1, so at step 0 the model is exactly the Phase-1 model."""
    def __init__(self, base, where, start=0):
        super().__init__()
        self.base, self.where, self.start = base, where, start
        base.weight.requires_grad_(False)
        n = (base.out_features - start) if where == "out" else base.in_features
        self.l = nn.Parameter(torch.ones(n))

    def forward(self, x):
        if self.where == "in":
            return self.base(x * self.l)
        y = self.base(x)
        return torch.cat([y[..., :self.start], y[..., self.start:] * self.l], dim=-1)


def add_peft(backbone, method, r=8, alpha=16, targets=("qkv", "proj", "w1", "w2", "w3")):
    """Freeze the backbone and add the trainable parts of one PEFT method. Returns the number of new parameters.
    lora / lorma : low-rank matrices (Phase-2 code), scale alpha / r.
    rslora       : LoRA with scale alpha / sqrt(r) (Kalajdzievski 2023, rank-stabilised LoRA). With alpha / r the
                   update shrinks as the rank grows, so high ranks learn slowly; alpha / sqrt(r) keeps the size of
                   the update stable, which makes a larger rank (e.g. 32) actually useful.
    ia3          : vectors on keys, values (inside qkv) and the FFN inner activations (input of w2)."""
    if method in ("lora", "lorma"):
        return add_adapters(backbone, method, r, alpha, targets)
    if method == "rslora":
        n = add_adapters(backbone, "lora", r, alpha, targets)
        for m in backbone.modules():
            if hasattr(m, "A") and hasattr(m, "scale"):
                m.scale = alpha / math.sqrt(r)
        return n
    if method == "ia3":
        for p in backbone.parameters():
            p.requires_grad_(False)
        n = 0
        for blk in backbone.blocks:
            d = blk.attn.qkv.in_features
            blk.attn.qkv = IA3Scale(blk.attn.qkv, "out", start=d)     # outputs [q | k | v]: scale k and v
            blk.mlp.w2 = IA3Scale(blk.mlp.w2, "in")                   # w2(l * (silu(w1 x) * w3 x))
            n += blk.attn.qkv.l.numel() + blk.mlp.w2.l.numel()
        return n
    raise ValueError(method)


# ----------------------------------------------------------------------------------------
# 2. Prediction
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def predict_logprobs(model, docs, plans, device, amp_dtype=torch.float32, win_batch=2, desc="predicting"):
    """{doc_id: (n_sent, 13) log-probabilities}. plans[doc_id] = (windows, owner)."""
    model.eval()
    out = {}
    for d in tqdm(docs, desc=desc, leave=False):
        wins, owner = plans[d.id]
        out[d.id] = F.log_softmax(model.forward_doc(d, wins, owner, device, amp_dtype, win_batch).float(), -1).cpu().numpy()
    return out


def to_labels(logprobs, labels):
    return {k: [labels[i] for i in v.argmax(1)] for k, v in logprobs.items()}


# ----------------------------------------------------------------------------------------
# 3. Training loop (shared by all Phase-3 experiments)
# ----------------------------------------------------------------------------------------
def kd_loss(student_logits, teacher_logprobs, T):
    """Knowledge distillation (Hinton et al. 2015): T^2 * KL(teacher_T || student_T), averaged over sentences.
    softmax(scores / T) with T > 1 softens both distributions, so the student also learns the teacher's
    'second choices' (e.g. 'this Ratio sentence looks a bit like PrecedentReliedUpon')."""
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logprobs / T, dim=-1)
    return (T ** 2) * (t_log.exp() * (t_log - s_log)).sum(-1).mean()


def train_rr(model, docs, plans, run_dir, cfg, device, tag, eval_fn=None, amp_dtype=torch.float32, teacher=None):
    """Train on whole documents. One optimiser step = `docs_per_step` documents; the loss of a step is the
    (class-weighted) cross-entropy averaged over all sentences of those documents.
    teacher: optional {doc_id: (n_sent, 13) log-probs} -> loss = (1-a) CE + a * KD  (a = cfg['kd_alpha']).
    eval_fn(): returns a dict with 'macro_f1' (dev) -> early stopping and best-epoch selection.

    Files in run_dir (tag = 'train' for Stage A, 'full' for Stage B):
      ckpt_{tag}.pt       after every epoch: model + optimiser + progress -> automatic resume after a crash
      best_{tag}.pt       weights of the best epoch (best dev macro-F1; last epoch if there is no eval_fn)
      history_{tag}.json  per-epoch log; 'done': true -> training finished, next time just load best_{tag}.pt"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_path, best_path, ckpt_path = run_dir / f"history_{tag}.json", run_dir / f"best_{tag}.pt", run_dir / f"ckpt_{tag}.pt"
    if hist_path.exists():
        state = json.load(open(hist_path))
        if state.get("done") and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"[cache] training '{tag}' already done -> loaded {best_path.name} (best epoch {state['best_epoch']})")
            return state

    head_ids = {id(p) for p in model.head_parameters()}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [
        {"params": [p for p in body if p.ndim >= 2], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
        {"params": [p for p in body if p.ndim < 2], "lr": cfg["lr"], "weight_decay": 0.0},
        {"params": [p for p in model.head_parameters() if p.requires_grad], "lr": cfg["head_lr"], "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))

    per_step = cfg["docs_per_step"]
    steps_per_epoch = math.ceil(len(docs) / per_step)
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))

    def lr_scale(step):                     # linear warm-up, then linear decay to 0
        return (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))

    weight = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device) if cfg.get("class_weights") else None
    state = {"epoch": 0, "step": 0, "history": [], "best": -1.0, "best_epoch": 0, "bad": 0, "done": False,
             "steps_per_epoch": steps_per_epoch, "n_docs": len(docs)}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        state = ck["state"]
        print(f"Resuming '{tag}' after epoch {state['epoch']}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(state["epoch"], cfg["epochs"]):
        model.train()
        t0 = time.time()
        perm = np.random.default_rng(cfg["seed"] + epoch).permutation(len(docs))
        losses = []
        bar = tqdm(range(steps_per_epoch), desc=f"{tag} epoch {epoch + 1}/{cfg['epochs']}", leave=False)
        for s in bar:
            batch = [docs[i] for i in perm[s * per_step:(s + 1) * per_step]]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(state["step"])
            n_step = sum(d.n_sent for d in batch)
            step_loss = 0.0
            for d in batch:                                 # one backward per document (frees its graph)
                wins, owner = plans[d.id]
                logits = model.forward_doc(d, wins, owner, device, amp_dtype, cfg["win_batch"], use_ckpt=True)
                y = torch.from_numpy(d.y).to(device)
                ce = F.cross_entropy(logits, y, weight=weight)
                loss = ce
                if teacher is not None:
                    t = torch.from_numpy(teacher[d.id]).float().to(device)
                    loss = (1 - cfg["kd_alpha"]) * ce + cfg["kd_alpha"] * kd_loss(logits, t, cfg["kd_T"])
                (loss * d.n_sent / n_step).backward()       # step loss = sentence-weighted mean over its documents
                step_loss += loss.item() * d.n_sent / n_step
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            losses.append(step_loss)
            bar.set_postfix(loss=f"{np.mean(losses[-10:]):.4f}")

        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "train_sec": round(time.time() - t0, 1)}
        if eval_fn is not None:
            t1 = time.time()
            sc = eval_fn()
            row.update(val_macro_f1=sc["macro_f1"], val_cl=sc.get("CL"), val_it=sc.get("IT"),
                       val_acc=sc.get("accuracy"), eval_sec=round(time.time() - t1, 1))
            improved = sc["macro_f1"] > state["best"]
        else:
            improved = True                                   # no validation: keep the last epoch
        if improved:
            state["best"] = row.get("val_macro_f1", -1.0)
            state["best_epoch"] = epoch + 1
            state["bad"] = 0
            torch.save(model.state_dict(), best_path)
        else:
            state["bad"] += 1
        state["history"].append(row)
        state["epoch"] = epoch + 1
        msg = f"epoch {epoch + 1:2d} | train loss {row['train_loss']:.4f} | {row['train_sec']:.0f}s"
        if "val_macro_f1" in row:
            msg += (f" | dev macro-F1 {row['val_macro_f1']:.4f} (CL {row['val_cl']:.4f}, IT {row['val_it']:.4f})"
                    f" acc {row['val_acc']:.4f}" + ("  <- best" if improved else ""))
        print(msg)
        stop = eval_fn is not None and state["bad"] >= cfg["patience"]
        if stop or state["epoch"] == cfg["epochs"]:
            state["done"] = True
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "state": state}, ckpt_path)
        json.dump(state, open(hist_path, "w"), indent=1)
        if stop:
            print(f"early stop: no improvement for {cfg['patience']} epochs")
            break

    state["done"] = True
    json.dump(state, open(hist_path, "w"), indent=1)
    if ckpt_path.exists():
        ckpt_path.unlink()          # resume file (with optimiser state, ~0.8 GB) no longer needed
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"training '{tag}' finished -> loaded best epoch {state['best_epoch']}")
    return state
