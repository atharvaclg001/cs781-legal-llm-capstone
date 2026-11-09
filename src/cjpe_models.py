"""
cjpe_models.py  -  Phase-4 model (case -> outcome 0/1) + shared training loop

    token ids of (part of) a case  -> Phase-1 backbone (same blocks + weights as Phase 1)
                                   -> one vector per token (640 numbers)
                                   -> POOLING: many token vectors -> one case vector
                                   -> dropout -> Linear(640 -> 2) -> scores for outcome 0 / 1

Pooling options
  'attn' : attention pooling (learned): score_t = w . h_t,  a = softmax(score) over real tokens,  v = sum_t a_t h_t.
           The model learns WHICH tokens matter (e.g. "set aside", "dismissed", "we are of the opinion").
  'mean' : average of all real token vectors.
  'last' : vector of the last real token. With a CAUSAL backbone the last token has read the whole input.
Padding: cases shorter than the window are padded at the END with <pad> (id 1). A causal token never looks
to the right, so padding cannot change real tokens; pooling ignores padded positions via `mask`.
A bidirectional backbone (D's encoder) WOULD see padding, so its padded rows are run one at a time.

Re-used: Phase-1 backbone loader, LoRA/LoRMA (ner_models), IA3/rsLoRA (rr_models).
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

from ner_models import load_phase1_backbone, add_adapters, count_params   # noqa: F401  (re-exported)
from rr_models import add_peft                                           # noqa: F401  (re-exported)

PAD_ID = 1     # <pad> in bpe32k (EOS = 0)


# ----------------------------------------------------------------------------------------
# 1. The model
# ----------------------------------------------------------------------------------------
class Pooling(nn.Module):
    def __init__(self, d, kind="attn"):
        super().__init__()
        self.kind = kind
        if kind == "attn":
            self.score = nn.Linear(d, 1)
            nn.init.zeros_(self.score.weight)       # start = plain mean pooling, then learns what to look at
            nn.init.zeros_(self.score.bias)
        elif kind not in ("mean", "last"):
            raise ValueError(kind)

    def forward(self, h, mask):                     # h: (B, T, d) float32, mask: (B, T) bool (True = real token)
        if self.kind == "last":
            last = mask.long().sum(1) - 1           # index of the last real token per row
            return h[torch.arange(h.shape[0], device=h.device), last]
        if self.kind == "mean":
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)
        return torch.einsum("bt,btd->bd", a, h)


class CJPEModel(nn.Module):
    """backbone (Phase-1 weights) -> pooling -> dropout -> Linear(d -> 2)."""
    def __init__(self, backbone, causal=True, pooling="attn", dropout=0.1, n_classes=2):
        super().__init__()
        self.backbone, self.causal = backbone, causal
        d = backbone.tok_emb.weight.shape[1]
        self.pool = Pooling(d, pooling)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, n_classes)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def head_parameters(self):
        """Everything that is NOT the Phase-1 backbone (new, randomly initialised -> higher learning rate)."""
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def encode(self, x, mask):
        """(B, T) ids -> (B, d) case vectors."""
        if not self.causal and not bool(mask.all()):
            # bidirectional backbone: padding would be visible -> run padded rows one by one, without padding
            vecs = []
            for r in range(x.shape[0]):
                n = int(mask[r].sum())
                h = self.backbone(x[r:r + 1, :n], causal=False).float()
                vecs.append(self.pool(h, mask[r:r + 1, :n]))
            return torch.cat(vecs)
        h = self.backbone(x, causal=self.causal).float()
        return self.pool(h, mask)

    def forward(self, x, mask):
        return self.out(self.drop(self.encode(x, mask)))


def collate(seqs):
    """List of 1-D token arrays -> (ids, mask) padded at the end to the longest one."""
    T = max(len(s) for s in seqs)
    x = np.full((len(seqs), T), PAD_ID, dtype=np.int64)
    m = np.zeros((len(seqs), T), dtype=bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        m[i, :len(s)] = True
    return torch.from_numpy(x), torch.from_numpy(m)


# ----------------------------------------------------------------------------------------
# 1b. Bottleneck adapters (Houlsby et al. 2019) and gradual unfreezing (Howard & Ruder 2018, ULMFiT)
# ----------------------------------------------------------------------------------------
class Adapter(nn.Module):
    """Wraps one sub-layer (attention or feed-forward) of a frozen block:
        y = sublayer(x);   out = y + W_up( GELU( W_down(y) ) )
    W_down: 640 -> 64 (squeeze), W_up: 64 -> 640 (expand back). W_up starts at ZERO, so at step 0 the adapter
    adds nothing and the model is exactly the Phase-1 model. Only the adapters (and the head) are trained.
    Difference to LoRA/LoRMA: those change existing weight matrices (W0 + BA or (I + BA) W0); an adapter is a
    NEW small layer with a non-linearity (GELU) inserted after the sub-layer."""
    def __init__(self, sublayer, d, bottleneck=64):
        super().__init__()
        self.sublayer = sublayer
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, *args, **kwargs):
        y = self.sublayer(*args, **kwargs)
        return y + self.up(F.gelu(self.down(y)))


def add_bottleneck_adapters(backbone, bottleneck=64):
    """Freeze the backbone; put one adapter after the attention and one after the feed-forward of every block
    (Houlsby placement: 2 per block). Returns the number of new parameters."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    d = backbone.tok_emb.weight.shape[1]
    n = 0
    for blk in backbone.blocks:
        blk.attn = Adapter(blk.attn, d, bottleneck)
        blk.mlp = Adapter(blk.mlp, d, bottleneck)
        n += sum(p.numel() for a in (blk.attn, blk.mlp) for p in (*a.down.parameters(), *a.up.parameters()))
    return n


class AddBias(nn.Module):
    """y = W0 x + b. Our Phase-1 linear layers have NO bias (bias=False, as in LLaMA), so BitFit cannot simply
    'unfreeze the biases': we add a bias vector b (starts at 0 -> model = Phase-1 model at step 0) and train only b."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        base.weight.requires_grad_(False)
        self.bias = nn.Parameter(torch.zeros(base.out_features))

    def forward(self, x):
        return self.base(x) + self.bias


def add_bitfit(backbone, targets=("qkv", "proj", "w1", "w2", "w3")):
    """BitFit (Ben Zaken et al. 2022): train only the bias terms (+ here the RMSNorm gains, the only other
    vectors in our blocks); every weight matrix and the embeddings stay frozen. ~0.1% of the parameters."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    n = 0
    for blk in backbone.blocks:
        for name in targets:
            parent = blk.attn if name in ("qkv", "proj") else blk.mlp
            wrapped = AddBias(getattr(parent, name))
            setattr(parent, name, wrapped)
            n += wrapped.bias.numel()
        for norm in (blk.norm1, blk.norm2):
            norm.weight.requires_grad_(True)
            n += norm.weight.numel()
    backbone.norm_f.weight.requires_grad_(True)
    return n + backbone.norm_f.weight.numel()


def set_unfrozen(model, top_blocks):
    """Gradual unfreezing: only the top `top_blocks` blocks (+ final norm + head) are trainable;
    top_blocks = 'all' also unfreezes every block and the token embeddings. Lower layers keep their Phase-1
    weights while the new head and the upper layers adapt first ("don't destroy general knowledge early")."""
    bb = model.backbone
    n_blocks = len(bb.blocks)
    k = n_blocks if top_blocks == "all" else int(top_blocks)
    for p in bb.parameters():
        p.requires_grad_(False)
    for blk in bb.blocks[n_blocks - k:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    for p in bb.norm_f.parameters():
        p.requires_grad_(True)
    if top_blocks == "all":
        for p in bb.tok_emb.parameters():
            p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------
# 2. Prediction
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, X, device, amp_dtype=torch.float32, batch_size=8, desc="predicting"):
    """(n, 2) logits for the token arrays X. Cases are sorted by length so a batch needs little padding."""
    model.eval()
    order = np.argsort([len(s) for s in X])
    out = np.zeros((len(X), 2), dtype=np.float32)
    for i in tqdm(range(0, len(X), batch_size), desc=desc, leave=False):
        idx = order[i:i + batch_size]
        x, m = collate([X[j] for j in idx])
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            logits = model(x.to(device), m.to(device))
        out[idx] = logits.float().cpu().numpy()
    return out


def probs_1(logits):
    """P(label 1) from (n, 2) logits."""
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(1)


# ----------------------------------------------------------------------------------------
# 3. Training loop (shared by all Phase-4 experiments)
# ----------------------------------------------------------------------------------------
def kd_loss(student_logits, teacher_logits, T):
    """Knowledge distillation (Hinton et al. 2015): T^2 * KL(teacher_T || student_T), averaged over cases.
    softmax(logits / T) with T > 1 softens the teacher's 0/1 probabilities, so the student also learns HOW SURE
    the teacher is (a 0.55 case is different from a 0.99 case), not only its decision."""
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logits / T, dim=-1)
    return (T ** 2) * (t_log.exp() * (t_log - s_log)).sum(-1).mean()


def train_cls(model, X, y, run_dir, cfg, device, tag="train", eval_fn=None, amp_dtype=torch.float32, teacher=None):
    """Train on cases X (list of token arrays) with labels y (0/1).
    One optimiser step = micro_batch x grad_accum cases. Loss = class-weighted cross-entropy
    (+ knowledge distillation if `teacher` = (n, 2) teacher logits is given: (1-a) CE + a KD).
    eval_fn(): dict with 'macro_f1' (threshold 0.5) and 'tuned_f1' (dev-tuned threshold) -> best epoch (by tuned_f1)
               + early stopping.

    Files in run_dir:
      ckpt_{tag}.pt       every `ckpt_every` steps AND after every epoch: model + optimiser + position in the epoch
                          -> after a crash / restart, training continues from the last saved step
      best_{tag}.pt       weights of the best epoch (best dev macro-F1)
      history_{tag}.json  per-epoch log; 'done': true -> next time just load best_{tag}.pt"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_path, best_path, ckpt_path = run_dir / f"history_{tag}.json", run_dir / f"best_{tag}.pt", run_dir / f"ckpt_{tag}.pt"
    if hist_path.exists():
        state = json.load(open(hist_path))
        if state.get("done") and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"[cache] training '{tag}' already done -> loaded {best_path.name} (best epoch {state['best_epoch']})")
            return state

    schedule = cfg.get("unfreeze_schedule")          # e.g. [3, 6, "all"]: trainable top blocks per epoch
    if schedule:
        for p in model.backbone.parameters():         # the optimiser must know every weight that will be unfrozen;
            p.requires_grad_(True)                    # frozen ones get no gradient -> AdamW skips them
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

    mb, acc = cfg["micro_batch"], cfg["grad_accum"]
    per_step = mb * acc
    steps_per_epoch = math.ceil(len(X) / per_step)
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))

    def lr_scale(step):                     # linear warm-up, then linear decay to 0
        return (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))

    weight = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device) if cfg.get("class_weights") else None
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    t_t = torch.as_tensor(teacher, dtype=torch.float32) if teacher is not None else None
    state = {"epoch": 0, "step_in_epoch": 0, "step": 0, "history": [], "best": -1.0, "best_epoch": 0, "bad": 0,
             "done": False, "steps_per_epoch": steps_per_epoch, "n_train": len(X), "losses": []}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        state = ck["state"]
        print(f"Resuming '{tag}': epoch {state['epoch'] + 1}, step {state['step_in_epoch']}/{steps_per_epoch}")

    def save_ckpt():
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "state": state}, ckpt_path)

    for epoch in range(state["epoch"], cfg["epochs"]):
        if schedule:
            stage = schedule[min(epoch, len(schedule) - 1)]
            n_tr = set_unfrozen(model, stage)
            print(f"epoch {epoch + 1}: gradual unfreezing -> top {stage} blocks trainable ({n_tr / 1e6:.2f}M parameters)")
        trainable = [p for p in model.parameters() if p.requires_grad]
        model.train()
        t0 = time.time()
        perm = np.random.default_rng(cfg["seed"] + epoch).permutation(len(X))   # same order after a resume
        bar = tqdm(range(state["step_in_epoch"], steps_per_epoch), desc=f"{tag} epoch {epoch + 1}/{cfg['epochs']}",
                   initial=state["step_in_epoch"], total=steps_per_epoch)
        for s in bar:
            idx = perm[s * per_step:(s + 1) * per_step]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(state["step"])
            step_loss = 0.0
            for j in range(0, len(idx), mb):
                part = idx[j:j + mb]
                x, m = collate([X[i] for i in part])
                x, m, yy = x.to(device), m.to(device), y_t[part].to(device)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    logits = model(x, m).float()
                loss = F.cross_entropy(logits, yy, weight=weight)
                if t_t is not None:
                    loss = (1 - cfg["kd_alpha"]) * loss + cfg["kd_alpha"] * kd_loss(logits, t_t[part].to(device), cfg["kd_T"])
                (loss * len(part) / len(idx)).backward()
                step_loss += loss.item() * len(part) / len(idx)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            state["step_in_epoch"] = s + 1
            state["losses"].append(round(step_loss, 4))
            bar.set_postfix(loss=f"{np.mean(state['losses'][-50:]):.4f}")
            if cfg.get("ckpt_every") and state["step_in_epoch"] % cfg["ckpt_every"] == 0 and s + 1 < steps_per_epoch:
                save_ckpt()
        bar.close()

        ep_losses = state["losses"][-steps_per_epoch:]
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(ep_losses)), "train_min": round((time.time() - t0) / 60, 1)}
        if eval_fn is not None:
            t1 = time.time()
            sc = eval_fn()
            row.update(dev_macro_f1=sc["macro_f1"], dev_acc=sc["acc"], dev_pred_1_rate=sc["pred_1_rate"],
                       dev_tuned_f1=sc.get("tuned_f1"), eval_sec=round(time.time() - t1, 1))
            # best epoch = best dev macro-F1 at the dev-tuned threshold (the rule used for the submission);
            # at a fixed 0.5 the score jumps around while the model is still biased towards one class
            improved = sc.get("tuned_f1", sc["macro_f1"]) > state["best"]
        else:
            improved = True
        if improved:
            state["best"] = row.get("dev_tuned_f1") or row.get("dev_macro_f1", -1.0)
            state["best_epoch"] = epoch + 1
            state["bad"] = 0
            torch.save(model.state_dict(), best_path)
        else:
            state["bad"] += 1
        state["history"].append(row)
        state["epoch"] = epoch + 1
        state["step_in_epoch"] = 0
        msg = f"epoch {epoch + 1} | train loss {row['train_loss']:.4f} | {row['train_min']:.0f} min"
        if "dev_macro_f1" in row:
            msg += (f" | dev macro-F1 {row['dev_macro_f1']:.4f} (threshold 0.5; tuned {row['dev_tuned_f1']}) "
                    f"acc {row['dev_acc']:.4f} pred-1 rate {row['dev_pred_1_rate']:.2f}" + ("  <- best" if improved else ""))
        print(msg)
        stop = eval_fn is not None and state["bad"] >= cfg["patience"]
        if stop or state["epoch"] == cfg["epochs"]:
            state["done"] = True
        save_ckpt()
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
