"""
pcr_models.py  -  Phase 7 (Prior Case Retrieval): a Phase-1 model as a CROSS-ENCODER RE-RANKER.

Two-stage retrieval (standard in legal IR, e.g. COLIEE systems):
  stage 1  BM25 (best P7_01 setting) ranks the whole pool -> top K = 100 candidates per query  (recall@100 = 0.73)
  stage 2  the Phase-1 model reads ONE (candidate, query) pair at a time and outputs ONE relevance score:

      [candidate: first tokens of the case] [EOS] [query: words around its <CITATION> markers] [EOS]
            -> Phase-1 backbone (same blocks + weights as Phase 1, strict loading)
            -> attention pooling over the QUERY part (causal models: only these tokens have read BOTH texts;
               bidirectional D encoder: over all tokens)
            -> dropout -> Linear(640 -> 1) = score

  Why a cross-encoder and not two separate case vectors (bi-encoder)? In a cross-encoder every query token can
  attend to every candidate token, so the model can check "does THIS candidate discuss the point made next to
  this <CITATION>?". A bi-encoder squeezes a 5,000-token case into one vector before it ever sees the query.
  The price: one forward pass per pair (dev: 118 x 100 pairs), which is why we only re-rank BM25's top 100.

Training objective (listwise / contrastive, "InfoNCE"): one group = 1 relevant candidate + n_hard BM25 hard
negatives (high BM25 rank, not relevant) + n_rand random negatives from the pool. Loss = softmax cross-entropy
over the group's scores with the relevant one as the "correct class":
      loss = -log( exp(s_pos) / sum_j exp(s_j) )
Hard negatives teach the difference between "same words" and "actually cited".

Knowledge distillation (Hinton et al. 2015, listwise version): the teacher (a fully fine-tuned 10-layer model)
scores the SAME group; the student (4-layer D encoder) matches the teacher's distribution over the group:
      loss = (1 - a) * CE(gold) + a * T^2 * KL( softmax(t / T) || softmax(s / T) )
i.e. the student learns from the teacher's similarity scores how much MORE relevant one candidate is than another.

Re-used: load_phase1_backbone + LoRMA (Phase 2, ner_models), IA3 (Phase 3, rr_models), attention Pooling +
partial fine-tuning `set_unfrozen` (Phase 4, cjpe_models), checkpoint/resume pattern of Phases 4-6.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ner_models import load_phase1_backbone, count_params          # noqa: F401  (re-exported)
from rr_models import add_peft                                     # noqa: F401
from cjpe_models import Pooling, set_unfrozen                      # noqa: F401

EOS_ID, PAD_ID = 0, 1


# ----------------------------------------------------------------------------------------
# 1. Inputs
# ----------------------------------------------------------------------------------------
def make_pair(cand, query, max_len, q_max):
    """(candidate tokens, query tokens) -> (ids, query_start).  ids = cand[:c] + [EOS] + query[:q] + [EOS]
    The query keeps up to q_max tokens; the candidate gets the rest of the max_len budget."""
    q = query[:q_max]
    c = cand[:max_len - len(q) - 2]
    ids = np.concatenate([c, [EOS_ID], q, [EOS_ID]]).astype(np.int64)
    return ids, len(c) + 1


def collate_pairs(pairs, causal):
    """List of (ids, qstart) -> ids (B, T), mask (B, T) real tokens, pool_mask (B, T) tokens that are pooled."""
    T = max(len(p[0]) for p in pairs)
    x = np.full((len(pairs), T), PAD_ID, dtype=np.int64)
    m = np.zeros((len(pairs), T), dtype=bool)
    pm = np.zeros((len(pairs), T), dtype=bool)
    for i, (ids, qs) in enumerate(pairs):
        x[i, :len(ids)] = ids
        m[i, :len(ids)] = True
        if causal:
            pm[i, qs:len(ids)] = True          # query part + final EOS: they have read the candidate AND the query
        else:
            pm[i, :len(ids)] = True
    return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(pm)


# ----------------------------------------------------------------------------------------
# 2. The model
# ----------------------------------------------------------------------------------------
class PairScorer(nn.Module):
    """backbone (Phase-1 weights) -> attention pooling -> dropout -> Linear(d -> 1)."""
    def __init__(self, backbone, causal=True, dropout=0.1):
        super().__init__()
        self.backbone, self.causal = backbone, causal
        d = backbone.tok_emb.weight.shape[1]
        self.pool = Pooling(d, "attn")          # starts as a plain mean (score weights = 0), then learns what to read
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 1)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def forward(self, x, mask, pool_mask):
        if not self.causal and not bool(mask.all()):
            # bidirectional backbone: padding would be visible -> padded rows run one by one without padding
            vecs = []
            for r in range(x.shape[0]):
                n = int(mask[r].sum())
                h = self.backbone(x[r:r + 1, :n], causal=False).float()
                vecs.append(self.pool(h, pool_mask[r:r + 1, :n]))
            v = torch.cat(vecs)
        else:
            h = self.backbone(x, causal=self.causal).float()
            v = self.pool(h, pool_mask)
        return self.out(self.drop(v)).squeeze(-1)


def build_scorer(phase1_dir, method, causal, rank=8, alpha=16, top_blocks=None, dropout=0.1):
    """Load a Phase-1 backbone (strict) and add the trainable parts of one technique.
    full    : everything trainable
    partial : only the top `top_blocks` blocks + final norm + head (layer-wise fine-tuning)
    lorma   : LoRMA r / alpha on qkv, proj, w1, w2, w3 (Phase-1 weights frozen)
    ia3     : IA3 vectors on keys, values and FFN activations (Phase-1 weights frozen)"""
    backbone, _, info = load_phase1_backbone(phase1_dir)
    if method in ("lorma", "lora", "ia3", "rslora"):
        add_peft(backbone, method, rank, alpha)
    elif method not in ("full", "partial"):
        raise ValueError(f"method {method} not implemented")
    model = PairScorer(backbone, causal, dropout)
    if method == "partial":
        set_unfrozen(model, top_blocks)
    return model, info


# ----------------------------------------------------------------------------------------
# 3. Training groups
# ----------------------------------------------------------------------------------------
def epoch_groups(train_q, seed, max_pos=8, n_hard=6, n_rand=1, hard_depth=100):
    """One epoch = a list of groups (qid, [pos, neg, neg, ...]), position 0 = the relevant candidate.
    train_q: list of dicts {"id", "gold": [...], "top": BM25 ranking (self excluded), "pool": all pool ids}
    max_pos: at most this many relevant candidates per query per epoch (a query with 52 gold would otherwise
             dominate an epoch); a new random subset every epoch.
    Deterministic for a given seed -> after a resume the epoch has exactly the same groups."""
    rng = np.random.default_rng(seed)
    groups = []
    for q in train_q:
        gold = list(q["gold"])
        if not gold:
            continue
        gset = set(gold) | {q["id"]}
        hard = [c for c in q["top"][:hard_depth] if c not in gset]
        pos = gold if len(gold) <= max_pos else list(rng.choice(gold, size=max_pos, replace=False))
        for p in pos:
            hn = list(rng.choice(hard, size=min(n_hard, len(hard)), replace=False)) if hard else []
            rn = []
            while len(rn) < n_rand:
                c = q["pool"][rng.integers(len(q["pool"]))]
                if c not in gset and c not in hn and c not in rn:
                    rn.append(c)
            groups.append((q["id"], [p] + hn + rn))
    order = rng.permutation(len(groups))
    return [groups[i] for i in order]


def kd_listwise(s, t, T):
    """T^2 * KL( softmax(t/T) || softmax(s/T) ) over the candidates of one group."""
    s_log = F.log_softmax(s / T, dim=-1)
    t_log = F.log_softmax(t / T, dim=-1)
    return (T ** 2) * (t_log.exp() * (t_log - s_log)).sum(-1)


# ----------------------------------------------------------------------------------------
# 4. Training loop (checkpoint every `ckpt_every` optimiser steps + resume, best epoch kept)
# ----------------------------------------------------------------------------------------
def train_ranker(model, train_q, get_pair, run_dir, cfg, device, amp_dtype=torch.float32, eval_fn=None,
                 teacher=None, tag="train"):
    """get_pair(qid, cid) -> (ids, qstart).  eval_fn() -> dict with 'f1' (dev Micro-F1@7 used to pick the epoch).
    Files in run_dir: ckpt_{tag}.pt (model + optimiser + position), best_{tag}.pt (best epoch weights),
    history_{tag}.json ('done': true -> next time only best_{tag}.pt is loaded)."""
    run_dir = Path(run_dir)
    hist_path, best_path, ckpt_path = run_dir / f"history_{tag}.json", run_dir / f"best_{tag}.pt", run_dir / f"ckpt_{tag}.pt"
    if hist_path.exists():
        state = json.load(open(hist_path))
        if state.get("done") and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"[cache] training already done -> loaded {best_path.name} (best epoch {state['best_epoch']})")
            return state

    head_ids = {id(p) for p in model.head_parameters()}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": [p for p in body if p.ndim >= 2], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
              {"params": [p for p in body if p.ndim < 2], "lr": cfg["lr"], "weight_decay": 0.0},
              {"params": [p for p in model.head_parameters() if p.requires_grad], "lr": cfg["head_lr"], "weight_decay": 0.0}]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    trainable = [p for p in model.parameters() if p.requires_grad]

    accum = cfg["groups_per_step"]
    n_groups = len(epoch_groups(train_q, cfg["seed"], cfg["max_pos"], cfg["n_hard"], cfg["n_rand"]))
    steps_per_epoch = math.ceil(n_groups / accum)
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))

    def lr_scale(step):                     # linear warm-up, then linear decay to 0
        return (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))

    state = {"epoch": 0, "step_in_epoch": 0, "step": 0, "history": [], "best": -1.0, "best_epoch": 0, "bad": 0,
             "done": False, "steps_per_epoch": steps_per_epoch, "groups_per_epoch": n_groups, "losses": []}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        state = ck["state"]
        print(f"Resuming: epoch {state['epoch'] + 1}, step {state['step_in_epoch']}/{steps_per_epoch}")

    def save_ckpt():
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "state": state}, ckpt_path)

    def scores_of(m, cands_pairs):
        x, mk, pm = collate_pairs(cands_pairs, m.causal)
        x, mk, pm = x.to(device), mk.to(device), pm.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            return m(x, mk, pm).float()

    if teacher is not None:
        teacher.eval()
    for epoch in range(state["epoch"], cfg["epochs"]):
        model.train()
        t0 = time.time()
        plan = epoch_groups(train_q, cfg["seed"] + epoch, cfg["max_pos"], cfg["n_hard"], cfg["n_rand"])
        bar = tqdm(range(state["step_in_epoch"], steps_per_epoch), desc=f"epoch {epoch + 1}/{cfg['epochs']}",
                   initial=state["step_in_epoch"], total=steps_per_epoch)
        for s in bar:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(state["step"])
            part = plan[s * accum:(s + 1) * accum]
            step_loss = 0.0
            for qid, cands in part:
                pairs = [get_pair(qid, c) for c in cands]
                sc = scores_of(model, pairs)
                loss = F.cross_entropy(sc.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
                if teacher is not None:
                    with torch.no_grad():
                        tsc = scores_of(teacher, pairs)
                    loss = (1 - cfg["kd_alpha"]) * loss + cfg["kd_alpha"] * kd_listwise(sc, tsc, cfg["kd_T"])
                (loss / len(part)).backward()
                step_loss += loss.item() / len(part)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            state["step_in_epoch"] = s + 1
            state["losses"].append(round(step_loss, 5))
            bar.set_postfix(loss=f"{np.mean(state['losses'][-50:]):.4f}")
            if cfg["ckpt_every"] and state["step_in_epoch"] % cfg["ckpt_every"] == 0 and s + 1 < steps_per_epoch:
                save_ckpt()
        bar.close()

        ep_losses = state["losses"][-steps_per_epoch:]
        row = {"epoch": epoch + 1, "train_loss": round(float(np.mean(ep_losses)), 4),
               "train_min": round((time.time() - t0) / 60, 1)}
        improved = True
        if eval_fn is not None:
            t1 = time.time()
            ev = eval_fn(model)
            row.update({f"dev_{k}": v for k, v in ev.items()}, eval_min=round((time.time() - t1) / 60, 1))
            improved = ev["f1"] > state["best"]
        if improved:
            state["best"] = row.get("dev_f1", -1.0)
            state["best_epoch"] = epoch + 1
            state["bad"] = 0
            torch.save(model.state_dict(), best_path)
        else:
            state["bad"] += 1
        state["history"].append(row)
        state["epoch"] = epoch + 1
        state["step_in_epoch"] = 0
        print(f"epoch {epoch + 1} | " + " | ".join(f"{k} {v}" for k, v in row.items() if k != "epoch")
              + ("  <- best" if improved else ""))
        stop = eval_fn is not None and state["bad"] >= cfg["patience"]
        state["done"] = bool(stop or state["epoch"] == cfg["epochs"])
        save_ckpt()
        json.dump(state, open(hist_path, "w"), indent=1)
        if stop:
            print(f"early stop: no improvement for {cfg['patience']} epoch(s)")
            break

    state["done"] = True
    json.dump(state, open(hist_path, "w"), indent=1)
    if ckpt_path.exists():
        ckpt_path.unlink()              # resume file (with optimiser state) no longer needed
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"training finished -> loaded best epoch {state['best_epoch']}")
    return state


# ----------------------------------------------------------------------------------------
# 5. Scoring the BM25 top-K of every query
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def score_topk(model, queries, get_pair, depth, device, amp_dtype=torch.float32, rows=16, desc="scoring"):
    """queries: list of (qid, [K candidate ids]) -> float32 array (Q, depth) (NaN where a query has < depth)."""
    model.eval()
    out = np.full((len(queries), depth), np.nan, dtype=np.float32)
    jobs = [(i, j, qid, c) for i, (qid, cands) in enumerate(queries) for j, c in enumerate(cands[:depth])]
    lens = [len(get_pair(q, c)[0]) for _, _, q, c in jobs]
    jobs = [jobs[k] for k in np.argsort(lens, kind="stable")[::-1]]   # similar lengths together -> little padding
    for s in tqdm(range(0, len(jobs), rows), desc=desc, leave=False):
        part = jobs[s:s + rows]
        x, mk, pm = collate_pairs([get_pair(q, c) for _, _, q, c in part], model.causal)
        x, mk, pm = x.to(device), mk.to(device), pm.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            sc = model(x, mk, pm).float().cpu().numpy()
        for (i, j, _, _), v in zip(part, sc):
            out[i, j] = v
    return out
