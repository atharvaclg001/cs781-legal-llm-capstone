"""
lsi_models.py  -  Phase-5 multi-label model (case -> set of statute IDs) + training loop

    token ids of (part of) a case -> Phase-1 backbone (same blocks + weights as Phase 1)
                                  -> one vector per token (640 numbers)
                                  -> POOLING (attention, as in Phase 4): many token vectors -> one case vector
                                  -> dropout -> Linear(640 -> 100) -> 100 scores ("logits"), one per statute
    P(statute l applies) = sigmoid(logit_l)     <- 100 INDEPENDENT yes/no decisions ("binary relevance")

Why sigmoid and not softmax?  softmax makes the 100 probabilities compete and sum to 1 (one class wins: right for
Phase 4's 0/1 outcome). Here a case has 3.7 statutes on average, so every statute gets its own yes/no probability.

Loss: binary cross-entropy (BCE) per statute, averaged:
    BCE = -[ y log p + (1 - y) log(1 - p) ]
Positive/negative imbalance: each statute is "yes" in only 0.6%-27% of the cases, so the "no" answers dominate the
loss and the model learns to say "no" too often. pos_weight multiplies the loss of the "yes" cases of statute l by
    w_l = (negatives_l / positives_l) ** power        power 0 = no weighting, 1 = fully balanced, 0.5 = in between
(power 1 gives weights up to 156 for the rarest statute -> very noisy; 0.5 gives at most ~12.5).

Re-used from Phase 4 (cjpe_models): Phase-1 backbone loader, CJPEModel (backbone -> pooling -> head) with
n_classes=100, LoRA / LoRMA / IA3 (add_peft), bottleneck adapters, BitFit, set_unfrozen (layer freezing).

New here:
  * output bias starts at log(prior / (1 - prior)) of each statute: at step 0 the model already predicts each
    statute with its training frequency instead of 50% (otherwise the first hundreds of steps only learn priors)
  * length-bucketed batches with a TOKEN budget: P5_01 showed that most cases are short (median ~650 tokens).
    Cases of similar length go into the same optimiser step, and a micro-batch holds as many cases as fit in
    `token_budget` tokens (3072 = the same memory as Phase 4's 2 x 1536). Short cases -> up to 16 per micro-batch,
    long cases -> 2. Far less padding = faster epochs; the optimiser step still has 16 cases.
  * knowledge distillation for SIGMOID outputs (kd_loss_ml)
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from cjpe_models import (CJPEModel, collate, load_phase1_backbone, add_peft, add_bottleneck_adapters,  # noqa: F401
                         add_bitfit, set_unfrozen, count_params)


# ----------------------------------------------------------------------------------------
# 1. The model
# ----------------------------------------------------------------------------------------
def build_lsi_model(backbone, n_labels, prior, causal=True, pooling="attn", dropout=0.1):
    """CJPEModel with 100 outputs; output bias = log-odds of each statute's training frequency."""
    model = CJPEModel(backbone, causal=causal, pooling=pooling, dropout=dropout, n_classes=n_labels)
    p = np.clip(np.asarray(prior, dtype=np.float64), 1e-4, 1 - 1e-4)
    with torch.no_grad():
        model.out.bias.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
    return model


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------------------
# 2. Batching by length with a token budget
# ----------------------------------------------------------------------------------------
def make_steps(lengths, per_step, seed):
    """Groups of `per_step` case indices of similar length, in a random group order.
    Sort by length (random tie-breaking), cut into consecutive groups, shuffle the groups.
    Same seed -> same groups (needed to resume in the middle of an epoch)."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(lengths)), np.asarray(lengths)))   # by length, ties broken randomly
    groups = [order[i:i + per_step] for i in range(0, len(order), per_step)]
    return [groups[i] for i in rng.permutation(len(groups))]


def micro_batches(idx, lengths, token_budget):
    """Split one group into micro-batches with (number of cases) x (longest case) <= token_budget tokens."""
    idx = sorted(idx, key=lambda i: lengths[i])
    out, cur = [], []
    for i in idx:
        if cur and (len(cur) + 1) * lengths[i] > token_budget:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


# ----------------------------------------------------------------------------------------
# 3. Prediction
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, X, device, amp_dtype=torch.float32, token_budget=12288, desc="predicting"):
    """(n, 100) logits for the token arrays X; cases sorted by length, batches limited by a token budget."""
    model.eval()
    lengths = np.array([len(s) for s in X])
    order = np.argsort(lengths)
    out = np.zeros((len(X), model.out.out_features), dtype=np.float32)
    batches = micro_batches(order, lengths, token_budget)
    for b in tqdm(batches, desc=desc, leave=False):
        x, m = collate([X[j] for j in b])
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            logits = model(x.to(device), m.to(device))
        out[b] = logits.float().cpu().numpy()
    return out


# ----------------------------------------------------------------------------------------
# 4. Losses
# ----------------------------------------------------------------------------------------
def pos_weights(Y, power):
    """w_l = (negatives_l / positives_l) ** power for every statute l."""
    pos = Y.sum(0)
    neg = len(Y) - pos
    return (neg / np.maximum(pos, 1)) ** power


def kd_loss_ml(student_logits, teacher_logits, T):
    """Knowledge distillation for multi-label (sigmoid) outputs: for every statute, the teacher's softened
    probability q = sigmoid(teacher_logit / T) is the target of a BCE on sigmoid(student_logit / T), times T^2
    (Hinton et al. 2015 scaling, keeps gradients comparable across T). The student learns HOW SURE the teacher
    is about each of the 100 statutes, not only the 0/1 gold labels."""
    q = torch.sigmoid(teacher_logits / T)
    return (T ** 2) * F.binary_cross_entropy_with_logits(student_logits / T, q)


# ----------------------------------------------------------------------------------------
# 5. Training loop (shared by all Phase-5 experiments; same checkpoint/resume logic as Phase 4)
# ----------------------------------------------------------------------------------------
def train_ml(model, X, Y, run_dir, cfg, device, tag="train", eval_fn=None, amp_dtype=torch.float32, teacher=None):
    """Train on cases X (list of token arrays) with multi-hot labels Y (n, 100).
    One optimiser step = `per_step` cases of similar length, split into micro-batches by `token_budget`.
    Loss = BCE with pos_weight (+ KD if `teacher` = (n, 100) teacher logits: (1 - a) BCE + a KD).
    eval_fn(): dict with 'macro_f1' (threshold 0.5) and 'tuned_f1' (best global threshold on dev) -> best epoch
               (by tuned_f1) + early stopping.

    Files in run_dir:
      ckpt_{tag}.pt       every `ckpt_every` steps AND after every epoch: model + optimiser + position in the epoch
      best_{tag}.pt       weights of the best epoch
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

    if cfg.get("top_blocks"):                       # partial / layer-wise fine-tuning: only the top k blocks
        n_tr = set_unfrozen(model, cfg["top_blocks"])
        print(f"partial fine-tuning: top {cfg['top_blocks']} blocks + final norm + head trainable ({n_tr / 1e6:.2f}M)")
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
    trainable = [p for p in model.parameters() if p.requires_grad]

    per_step, budget = cfg["per_step"], cfg["token_budget"]
    lengths = np.array([len(s) for s in X])
    steps_per_epoch = math.ceil(len(X) / per_step)
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))

    def lr_scale(step):                     # linear warm-up, then linear decay to 0
        return (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))

    pw = torch.tensor(pos_weights(Y, cfg["pos_weight_power"]), dtype=torch.float32, device=device)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)
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
        model.train()
        t0 = time.time()
        steps = make_steps(lengths, per_step, cfg["seed"] + epoch)          # same groups after a resume
        bar = tqdm(range(state["step_in_epoch"], steps_per_epoch), desc=f"{tag} epoch {epoch + 1}/{cfg['epochs']}",
                   initial=state["step_in_epoch"], total=steps_per_epoch)
        for s in bar:
            idx = steps[s]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(state["step"])
            step_loss = 0.0
            for part in micro_batches(idx, lengths, budget):
                x, m = collate([X[i] for i in part])
                x, m, yy = x.to(device), m.to(device), Y_t[part].to(device)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    logits = model(x, m).float()
                loss = F.binary_cross_entropy_with_logits(logits, yy, pos_weight=pw)
                if t_t is not None:
                    loss = (1 - cfg["kd_alpha"]) * loss + cfg["kd_alpha"] * kd_loss_ml(logits, t_t[part].to(device), cfg["kd_T"])
                (loss * len(part) / len(idx)).backward()
                step_loss += loss.item() * len(part) / len(idx)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            state["step_in_epoch"] = s + 1
            state["losses"].append(round(step_loss, 5))
            bar.set_postfix(loss=f"{np.mean(state['losses'][-50:]):.4f}")
            if cfg.get("ckpt_every") and state["step_in_epoch"] % cfg["ckpt_every"] == 0 and s + 1 < steps_per_epoch:
                save_ckpt()
        bar.close()

        ep_losses = state["losses"][-steps_per_epoch:]
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(ep_losses)), "train_min": round((time.time() - t0) / 60, 1)}
        if eval_fn is not None:
            t1 = time.time()
            sc = eval_fn()
            row.update(dev_macro_f1_05=sc["macro_f1"], dev_tuned_f1=sc["tuned_f1"], dev_threshold=sc["threshold"],
                       dev_micro_f1=sc["micro_f1"], dev_labels_per_case=sc["labels_per_case"],
                       eval_min=round((time.time() - t1) / 60, 1))
            improved = sc["tuned_f1"] > state["best"]
        else:
            improved = True
        if improved:
            state["best"] = row.get("dev_tuned_f1", -1.0)
            state["best_epoch"] = epoch + 1
            state["bad"] = 0
            torch.save(model.state_dict(), best_path)
        else:
            state["bad"] += 1
        state["history"].append(row)
        state["epoch"] = epoch + 1
        state["step_in_epoch"] = 0
        msg = f"epoch {epoch + 1} | train loss {row['train_loss']:.4f} | {row['train_min']:.0f} min"
        if "dev_tuned_f1" in row:
            msg += (f" | dev macro-F1 {row['dev_macro_f1_05']:.4f} @0.5, {row['dev_tuned_f1']:.4f} @{row['dev_threshold']}"
                    f" | micro-F1 {row['dev_micro_f1']:.4f} | {row['dev_labels_per_case']:.2f} statutes/case"
                    + ("  <- best" if improved else ""))
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
        ckpt_path.unlink()          # resume file (with optimiser state) no longer needed
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"training '{tag}' finished -> loaded best epoch {state['best_epoch']}")
    return state
