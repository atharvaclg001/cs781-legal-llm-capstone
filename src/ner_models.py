"""
ner_models.py  -  Phase-1 backbones + NER tagging head + shared training loop (Phase 2)

All four Phase-1 models are built from the same pieces (RoPE, RMSNorm, SwiGLU, attention):
  A  (plain decoder), A2 (decoder + copy), B (prefix-LM): tok_emb -> 10 blocks -> norm_f -> lm_head
  D  (encoder-decoder + copy): tok_emb -> 4 encoder blocks -> enc_norm  (+ a decoder we do not need)
For NER we only need the part that turns tokens into hidden vectors (the "backbone").
The LM head, copy mechanism and D's decoder are left out: they predict the NEXT TOKEN, not a tag.

    NERTagger = backbone (Phase-1 weights) -> dropout -> Linear(640 -> 25 tags)   (the new "head")

Attention can be causal (each token sees only the left) or bidirectional (sees both sides).
Phase-1 decoders were trained causally; for NER we switch to bidirectional because the label of
a name often depends on words AFTER it ("Jage Ram ... Appellant").
"""

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

IGNORE = -100   # label value the loss skips (header copies in "header memory" windows)


# ----------------------------------------------------------------------------------------
# 1. Building blocks (identical maths to the Phase-1 notebooks, so the weights fit exactly)
# ----------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim, max_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention. causal=True: token t sees tokens <= t. causal=False: sees all."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, causal=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_hidden):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d_model), Attention(d_model, n_heads)
        self.norm2, self.mlp = RMSNorm(d_model), SwiGLU(d_model, mlp_hidden)

    def forward(self, x, cos, sin, causal=False):
        x = x + self.attn(self.norm1(x), cos, sin, causal)
        return x + self.mlp(self.norm2(x))


class Backbone(nn.Module):
    """Token ids (B, T) -> hidden vectors (B, T, d_model)."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_hidden, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, mlp_hidden) for _ in range(n_layers))
        self.norm_f = RMSNorm(d_model)
        cos, sin = rope_tables(d_model // n_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids, causal=False):
        T = ids.shape[1]
        assert T <= self.max_seq_len, f"window of {T} tokens > {self.max_seq_len} (positions never trained)"
        x = self.tok_emb(ids)
        for blk in self.blocks:
            x = blk(x, self.cos[:T], self.sin[:T], causal)
        return self.norm_f(x)


# ----------------------------------------------------------------------------------------
# 2. Loading a Phase-1 model's weights into a Backbone
# ----------------------------------------------------------------------------------------
def load_phase1_backbone(run_dir):
    """Read config.json + model_final.pt of a Phase-1 run and return (backbone, phase1_cfg, info).
    Decoder models (A, A2, B): keep tok_emb.*, blocks.*, norm_f.*  -> same names as here.
    Encoder-decoder (D): keep tok_emb.*, enc.* and enc_norm.*, renamed to blocks.* / norm_f
    (D called its norms n1/n2, we call them norm1/norm2). Everything else is dropped.
    Loading uses strict=True: every backbone weight MUST come from Phase 1 (no silent random init)."""
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    sd = torch.load(run_dir / "model_final.pt", map_location="cpu")
    new, dropped = {}, []
    if cfg["arch"] == "D_encdec":
        n_layers = cfg["n_enc_layers"]
        for k, v in sd.items():
            m = re.match(r"^enc\.(\d+)\.(n1|n2|attn|mlp)\.(.*)$", k)
            if k.startswith("tok_emb."):
                new[k] = v
            elif m:
                part = {"n1": "norm1", "n2": "norm2"}.get(m.group(2), m.group(2))
                new[f"blocks.{m.group(1)}.{part}.{m.group(3)}"] = v
            elif k.startswith("enc_norm."):
                new["norm_f." + k[len("enc_norm."):]] = v
            else:
                dropped.append(k)
    else:
        n_layers = cfg["n_layers"]
        for k, v in sd.items():
            if k.startswith(("tok_emb.", "blocks.", "norm_f.")):
                new[k] = v
            else:
                dropped.append(k)
    bb = Backbone(cfg["vocab_size"], cfg["d_model"], n_layers, cfg["n_heads"], cfg["mlp_hidden"], cfg["max_seq_len"])
    bb.load_state_dict(new, strict=True)
    groups = sorted({k.split(".")[0] for k in dropped})
    info = {"arch": cfg["arch"], "layers": n_layers, "loaded_tensors": len(new),
            "dropped_tensors": len(dropped), "dropped_groups": groups}
    return bb, cfg, info


# ----------------------------------------------------------------------------------------
# 3. The NER model: backbone + tagging head
# ----------------------------------------------------------------------------------------
class TagHead(nn.Module):
    """Turns each token's hidden vector into 25 tag scores. Three kinds:
    'linear'    : Linear(640 -> 25) on the token's own vector.
    'lookahead' : (our idea for CAUSAL backbones) concatenate the vectors of tokens t, t+1, ..., t+k.
                  With causal attention, the vector of token t+j has read everything up to t+j, so the head of
                  token t gets k tokens of RIGHT context while the backbone keeps its pretrained left-to-right reading.
    'bilstm'    : a small bidirectional LSTM (reads the vectors left->right and right->left) then Linear.
                  Gives full right context on top of a causal backbone (classic BiLSTM tagger idea)."""
    def __init__(self, d, n_tags, kind="linear", lookahead=0, lstm_hidden=256, dropout=0.1):
        super().__init__()
        self.kind, self.k = kind, lookahead
        self.drop = nn.Dropout(dropout)
        if kind == "linear":
            in_dim = d
        elif kind == "lookahead":
            assert lookahead >= 1
            in_dim = d * (lookahead + 1)
        elif kind == "bilstm":
            self.lstm = nn.LSTM(d, lstm_hidden, batch_first=True, bidirectional=True)
            in_dim = 2 * lstm_hidden
        else:
            raise ValueError(kind)
        self.out = nn.Linear(in_dim, n_tags)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, h):
        h = self.drop(h)
        if self.kind == "lookahead":
            # h[:, j:] shifted left by j positions, zero-padded at the end -> feature of token t+j at position t
            h = torch.cat([h] + [F.pad(h[:, j:], (0, 0, 0, j)) for j in range(1, self.k + 1)], dim=-1)
        elif self.kind == "bilstm":
            with torch.autocast(device_type=h.device.type, enabled=False):   # LSTM in float32 (bf16 LSTM is not supported everywhere)
                h, _ = self.lstm(h.float())
            h = self.drop(h)
        return self.out(h)


class NERTagger(nn.Module):
    """backbone (Phase-1 weights) -> TagHead (new, randomly initialised)."""
    def __init__(self, backbone, n_tags, dropout=0.1, causal=False, head="linear", lookahead=0, lstm_hidden=256):
        super().__init__()
        self.backbone = backbone
        self.causal = causal
        d = backbone.tok_emb.weight.shape[1]
        self.head = TagHead(d, n_tags, head, lookahead, lstm_hidden, dropout)

    def forward(self, ids):
        return self.head(self.backbone(ids, causal=self.causal))


def load_weights(model, path, device):
    """Load saved NERTagger weights. Also accepts files from the first version (head.weight -> head.out.weight)."""
    sd = torch.load(path, map_location=device)
    if "head.weight" in sd:
        sd["head.out.weight"] = sd.pop("head.weight")
        sd["head.out.bias"] = sd.pop("head.bias")
    model.load_state_dict(sd)


# ----------------------------------------------------------------------------------------
# 3b. Parameter-efficient fine-tuning: LoRA (additive) and LoRMA (multiplicative)
# ----------------------------------------------------------------------------------------
class LowRankLinear(nn.Module):
    """Wraps a frozen Phase-1 nn.Linear (weight W0, shape d_out x d_in) with a small trainable low-rank update.

    LoRA  (Hu et al. 2021):            h = W0 x + (alpha/r) * B A x          A: r x d_in,  B: d_out x r
    LoRMA (Bihany, Patel, Modi 2025):  h = (I + (alpha/r) * B A) W0 x       A: r x d_out, B: d_out x r
          = W0 x + (alpha/r) * B (A (W0 x))   <- the LoRMA+ variant ("additive rank inflation":
          the identity I keeps the update full-rank). We multiply x first, so it costs about the same as LoRA.
    In both, B starts at 0, so at step 0 the model is exactly the Phase-1 model; A starts random.
    Only A and B are trained (2 x r x d numbers instead of d_out x d_in)."""
    def __init__(self, base, kind="lora", r=8, alpha=16):
        super().__init__()
        assert kind in ("lora", "lorma")
        self.base, self.kind, self.scale = base, kind, alpha / r
        base.weight.requires_grad_(False)
        inner = base.in_features if kind == "lora" else base.out_features
        self.A = nn.Parameter(torch.empty(r, inner))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))     # same init as the LoRA paper
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        y = self.base(x)                                     # W0 x  (frozen)
        z = x if self.kind == "lora" else y                  # LoRA adapts the input, LoRMA the output of W0
        return y + self.scale * F.linear(F.linear(z, self.A), self.B)


def add_adapters(backbone, kind, r=8, alpha=16, targets=("qkv", "proj", "w1", "w2", "w3")):
    """Freeze the whole backbone, then wrap the chosen linear layers of every block with LowRankLinear.
    qkv/proj live in block.attn, w1/w2/w3 in block.mlp. Returns the number of adapter parameters."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    n = 0
    for blk in backbone.blocks:
        for name in targets:
            parent = blk.attn if name in ("qkv", "proj") else blk.mlp
            wrapped = LowRankLinear(getattr(parent, name), kind, r, alpha)
            setattr(parent, name, wrapped)
            n += wrapped.A.numel() + wrapped.B.numel()
    return n


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------------------------------
# 4. Scoring windows (inference) - plugs into ner_utils.predict_spans / doc_log_probs
# ----------------------------------------------------------------------------------------
def load_trained(root, run_name, tag, device):
    """Rebuild a finished Phase-2 model (e.g. the KD teacher) from its run folder: config.json + best_{tag}.pt."""
    run_dir = Path(root) / "checkpoints" / "phase2" / run_name
    c = json.load(open(run_dir / "config.json"))
    backbone, _, _ = load_phase1_backbone(Path(root) / "checkpoints" / "phase1" / c["phase1_run"])
    model = NERTagger(backbone, 25, c["dropout"], c["causal"], c["head"], c.get("lookahead", 0), c.get("lstm_hidden", 256))
    if c["method"] in ("lora", "lorma"):
        add_adapters(model.backbone, c["method"], c["rank"], c["alpha"], c["adapter_targets"])
    model = model.to(device)
    load_weights(model, run_dir / f"best_{tag}.pt", device)
    model.eval()
    return model, c


def make_scorer(model, device, amp_dtype=torch.float32, batch_size=8, desc="scoring windows"):
    """Returns score(list_of_id_arrays) -> list of (len, n_tags) log-probability arrays.
    Windows of equal length are batched together (no padding needed)."""
    @torch.no_grad()
    def score(windows):
        model.eval()
        out = [None] * len(windows)
        by_len = {}
        for i, w in enumerate(windows):
            by_len.setdefault(len(w), []).append(i)
        batches = [idx[j:j + batch_size] for idx in by_len.values() for j in range(0, len(idx), batch_size)]
        for idx in tqdm(batches, desc=desc, leave=False):
            x = torch.from_numpy(np.stack([windows[i] for i in idx])).long().to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logits = model(x)
            lp = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
            for row, i in enumerate(idx):
                out[i] = lp[row]
        return out
    return score


# ----------------------------------------------------------------------------------------
# 5. Shared training loop (used by every Phase-2 experiment, so they are trained the same way)
# ----------------------------------------------------------------------------------------
def kd_loss(student_logits, teacher_logprobs, y, ce, alpha, T):
    """Knowledge distillation (Hinton et al. 2015):
       loss = (1 - alpha) * CE(student, gold tags) + alpha * T^2 * KL(teacher_T || student_T)
    teacher_T / student_T = softmax(scores / T). T > 1 softens both distributions, so the student also learns
    the teacher's "second choices" (e.g. APP vs RESP probabilities), not only its top tag. T^2 keeps the
    gradient size comparable to the CE term. Positions labelled IGNORE are skipped."""
    mask = (y != IGNORE).float()
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logprobs / T, dim=-1)        # log-probs differ from logits by a constant per token
    kl = (t_log.exp() * (t_log - s_log)).sum(-1)               # KL per token
    kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    return (1 - alpha) * ce + alpha * (T ** 2) * kl


def train_tagger(model, X, Y, run_dir, cfg, device, tag, eval_fn=None, amp_dtype=torch.float32):
    """Train `model` on windows X (token ids) with labels Y (tag ids, IGNORE = skip).
    X may also be a function epoch -> (X, Y) or (X, Y, T): new windows every epoch (augmentation); Y is then ignored.
    T (optional) = teacher log-probabilities per window -> knowledge-distillation loss (kd_alpha, kd_T in cfg).

    Only parameters with requires_grad=True are trained (full fine-tuning: all of them;
    LoRA/LoRMA later: only the small adapter matrices + head).
    cfg keys: epochs, micro_batch, grad_accum, lr, head_lr, weight_decay, warmup_frac,
              grad_clip, seed, patience, class_weights (None or list of n_tags floats)
    eval_fn(): returns a scores dict with 'macro_f1' (validation) or None.

    Files in run_dir (tag = e.g. 'fold1' or 'full'):
      ckpt_{tag}.pt     after every epoch: model + optimizer + progress  -> automatic resume
      best_{tag}.pt     weights of the best epoch (by validation macro-F1; last epoch if no eval_fn)
      history_{tag}.json  per-epoch log; 'done': true means training finished -> just load best_{tag}.pt
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_path, best_path, ckpt_path = run_dir / f"history_{tag}.json", run_dir / f"best_{tag}.pt", run_dir / f"ckpt_{tag}.pt"
    if hist_path.exists():
        state = json.load(open(hist_path))
        if state.get("done") and best_path.exists():
            load_weights(model, best_path, device)
            print(f"[cache] training '{tag}' already done -> loaded {best_path.name} (best epoch {state['best_epoch']})")
            return state

    # optimiser: AdamW; weight decay only on weight matrices of the body; the new head gets a higher lr
    head_ids = {id(p) for p in model.head.parameters()}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [
        {"params": [p for p in body if p.ndim >= 2], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
        {"params": [p for p in body if p.ndim < 2], "lr": cfg["lr"], "weight_decay": 0.0},
        {"params": list(model.head.parameters()), "lr": cfg["head_lr"], "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=device.type == "cuda")

    data_fn = X if callable(X) else (lambda epoch: (X, Y))
    n, mb, acc = len(data_fn(0)[0]), cfg["micro_batch"], cfg["grad_accum"]
    per_step = mb * acc
    steps_per_epoch = math.ceil(n / per_step)
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))

    def lr_scale(step):                     # linear warm-up, then linear decay to 0
        return (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))

    weight = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device) if cfg.get("class_weights") else None
    state = {"epoch": 0, "step": 0, "history": [], "best": -1.0, "best_epoch": 0, "bad": 0, "done": False,
             "steps_per_epoch": steps_per_epoch, "n_windows": n}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        state = ck["state"]
        print(f"Resuming '{tag}' after epoch {state['epoch']}")

    for epoch in range(state["epoch"], cfg["epochs"]):
        model.train()
        t0 = time.time()
        batch_data = data_fn(epoch)
        Xe, Ye = batch_data[0], batch_data[1]
        Te = batch_data[2] if len(batch_data) == 3 else None     # teacher log-probs (knowledge distillation)
        perm = np.random.default_rng(cfg["seed"] + epoch).permutation(len(Xe))[:steps_per_epoch * per_step]
        losses = []
        bar = tqdm(range(steps_per_epoch), desc=f"{tag} epoch {epoch + 1}/{cfg['epochs']}", leave=False)
        for s in bar:
            idx = perm[s * per_step:(s + 1) * per_step]
            if len(idx) == 0:
                break
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(state["step"])
            for j in range(0, len(idx), mb):
                chunk = idx[j:j + mb]
                # windows of different length cannot be stacked; group by length (training windows are all 1536)
                by_len = {}
                for i in chunk:
                    by_len.setdefault(len(Xe[i]), []).append(i)
                for same in by_len.values():
                    x = torch.from_numpy(np.stack([Xe[i] for i in same])).long().to(device)
                    y = torch.from_numpy(np.stack([Ye[i] for i in same])).long().to(device)
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                        logits = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                           weight=weight, ignore_index=IGNORE)
                    if Te is not None:
                        t = torch.from_numpy(np.stack([Te[i] for i in same])).float().to(device)
                        loss = kd_loss(logits.float(), t, y, loss, cfg["kd_alpha"], cfg["kd_T"])
                    (loss * len(same) / len(idx)).backward()
                    losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "train_sec": round(time.time() - t0, 1)}
        if eval_fn is not None:
            t1 = time.time()
            sc = eval_fn()
            row.update(val_macro_f1=sc["macro_f1"], val_micro_f1=sc["micro_f1"], eval_sec=round(time.time() - t1, 1))
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
            msg += f" | val macro-F1 {row['val_macro_f1']:.4f} micro-F1 {row['val_micro_f1']:.4f}" + ("  <- best" if improved else "")
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
        ckpt_path.unlink()          # resume file (with optimizer, ~0.8 GB) no longer needed; best_{tag}.pt is kept
    load_weights(model, best_path, device)
    print(f"training '{tag}' finished -> loaded best epoch {state['best_epoch']}")
    return state
