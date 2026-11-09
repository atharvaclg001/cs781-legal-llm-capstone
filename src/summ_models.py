"""
summ_models.py  -  Phase 6 (abstractive summarization): the four Phase-1 language models as summarizers.

Phases 2-5 only used the Phase-1 "backbone" (tokens -> vectors) plus a new classification head. Summarization is
GENERATION, so here we keep the WHOLE Phase-1 language model: embeddings, blocks, final norm, LM head (tied to
the embeddings) and, for A2 / D, the copy mechanism. Nothing new is added except the PEFT parameters.

Two layouts
  decoder-only (A2, A, B):   [document (first 900 tokens)] [EOS] [summary ...] [EOS]
        the model continues the document with its summary; loss only on the summary tokens (+ final EOS).
        EOS = <|endoftext|> (id 0), the Phase-1 document separator: "a new text starts here".
        A2, A: causal attention everywhere (as pretrained).
        B    : prefix-LM, as pretrained: document + EOS are read BIDIRECTIONALLY, the summary causally.
  encoder-decoder (D):       encoder reads the document (1024 tokens, or C chunks of 1024 = "fusion-in-decoder"),
                             decoder writes [EOS] summary -> summary [EOS]; cross-attention over all encoder tokens.

Copy mechanism (A2, D; See et al. 2017, as in Phase 1):
    p(y) = g * p_vocab(y) + (1 - g) * sum_j a_j [x_j = y]
    g = sigmoid(w . h + b) (gate), a = attention over the source tokens (A2: all earlier tokens; D: the document).
P6_01: 94% of summary tokens occur in the document -> copying is directly useful.

RoPE positions: rotary embeddings make attention scores depend only on the DISTANCE between tokens, so the D decoder
can start at position 0 (Phase 1 continued after the prompt; same scores, checked in the CPU test).

PEFT (only these parameters train; Phase-1 weights frozen):
  LoRA  (Hu et al. 2021)          h = W0 x + (alpha/r) B A x
  LoRMA (Bihany et al. 2025)      h = (I + (alpha/r) B A) W0 x       (multiplicative, compulsory)
  Houlsby adapters (2019)         y = sublayer(x);  y + W_up GELU(W_down y)   after attention and feed-forward
  -> re-used from Phases 2-5 (ner_models.LowRankLinear; Adapter copied from cjpe_models). With PEFT the copy layers
     (copy_q, copy_k, copy_gate: 0.16M) are also trained: they decide WHAT to copy from a document.

Knowledge distillation (word-level, Kim & Rush 2016): at every summary position the student matches the
teacher's full next-token distribution (incl. the teacher's copy probabilities):
    loss = (1 - alpha) * NLL(gold token) + alpha * KL(p_teacher || p_student)
Both models use the same bpe32k tokens, so their distributions are over the same 32,000 ids even though the
teacher is a decoder-only model and the student an encoder-decoder.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ner_models import RMSNorm, rope_tables, apply_rope, SwiGLU, LowRankLinear, count_params  # noqa: F401


class Adapter(nn.Module):
    """Houlsby bottleneck adapter around one sub-layer (same as Phases 4-5, cjpe_models.Adapter):
        y = sublayer(x);   out = y + W_up( GELU( W_down(y) ) ),   W_down: 640 -> 64, W_up: 64 -> 640.
    W_up starts at ZERO -> at step 0 the model is exactly the Phase-1 model."""
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

EOS_ID, PAD_ID, IGNORE = 0, 1, -100


# ----------------------------------------------------------------------------------------
# 1. Attention (same maths as Phase 1, plus padding / prefix masks)
# ----------------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """mask: bool (B, 1, T, T) or (B, 1, 1, T), True = may attend. Without a mask: plain causal / full attention.
    cache: dict for generation (keys / values of earlier tokens)."""
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, mask=None, causal=True, cache=None):
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in self.qkv(x).split(D, dim=2))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past = 0
        if cache is not None:
            if "k" in cache:
                past = cache["k"].shape[2]
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        if mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal and past == 0 and T > 1)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    """Decoder token -> all encoder tokens (no RoPE, as in Phase 1). mem_mask (B, M): True = real token."""
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def memory(self, enc):
        B, S, D = enc.shape
        k, v = self.kv(enc).split(D, dim=2)
        return k.view(B, S, self.h, self.hd).transpose(1, 2), v.view(B, S, self.h, self.hd).transpose(1, 2)

    def forward(self, x, mem, mem_mask=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        m = None if mem_mask is None else mem_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, mem[0], mem[1], attn_mask=m)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):                       # decoder-only block (A, A2, B): names as in Phase 1
    def __init__(self, d, h, hidden):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d), SelfAttention(d, h)
        self.norm2, self.mlp = RMSNorm(d), SwiGLU(d, hidden)

    def forward(self, x, cos, sin, mask=None, causal=True, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask, causal, cache)
        return x + self.mlp(self.norm2(x))


class EncBlock(nn.Module):                    # D encoder block (names n1/n2 as in Phase 1)
    def __init__(self, d, h, hidden):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), SelfAttention(d, h)
        self.n2, self.mlp = RMSNorm(d), SwiGLU(d, hidden)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.n1(x), cos, sin, mask, causal=False)
        return x + self.mlp(self.n2(x))


class DecBlock(nn.Module):                    # D decoder block: causal self-attn + cross-attn + feed-forward
    def __init__(self, d, h, hidden):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), SelfAttention(d, h)
        self.n2, self.cross = RMSNorm(d), CrossAttention(d, h)
        self.n3, self.mlp = RMSNorm(d), SwiGLU(d, hidden)

    def forward(self, x, cos, sin, mem, mem_mask, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, None, True, cache)
        x = x + self.cross(self.n2(x), mem, mem_mask)
        return x + self.mlp(self.n3(x))


# ----------------------------------------------------------------------------------------
# 2. The two model types
# ----------------------------------------------------------------------------------------
def copy_mix(model, logits, hs, ys, scores, key_ids, key_mask, full):
    """Copy mechanism for N summary positions at once.
    logits (N, V) vocabulary scores, hs (N, d) hidden vectors, ys (N,) gold tokens,
    scores (N, K) copy-attention scores over K source tokens, key_ids (N, K) their token ids, key_mask (N, K).
      p(y) = g * softmax(logits)[y] + (1 - g) * sum_j a_j [key_id_j = y]
    full=False: only the gold token's probability (cheap, for normal training).
    full=True : the whole (N, V) distribution, needed for knowledge distillation."""
    a = torch.softmax(scores.float().masked_fill(~key_mask, float("-inf")), dim=-1)
    g = torch.sigmoid(model.copy_gate(hs).float())                     # (N, 1) = P(generate from vocabulary)
    lf = logits.float()
    out = {"ys": ys, "copy": (1 - g).squeeze(-1)}
    if full:
        p = g * torch.softmax(lf, dim=-1)
        p = p.scatter_add(1, key_ids, (1 - g) * a)
        logp = torch.log(p.clamp_min(1e-12))
        out.update(logp=logp, nll=-logp.gather(1, ys[:, None]).squeeze(1))
    else:
        p_gen = torch.exp(lf.gather(1, ys[:, None]).squeeze(1) - torch.logsumexp(lf, dim=-1))
        p_copy = (a * (key_ids == ys[:, None])).sum(-1)
        p = g.squeeze(-1) * p_gen + (1 - g.squeeze(-1)) * p_copy
        out["nll"] = -torch.log(p.clamp_min(1e-9))
    return out


def no_copy(logits, ys, full):
    logp = F.log_softmax(logits.float(), dim=-1)
    out = {"ys": ys, "copy": None, "nll": -logp.gather(1, ys[:, None]).squeeze(1)}
    if full:
        out["logp"] = logp
    return out


class DecoderOnlyLM(nn.Module):
    """A2 (copy), A (no copy), B (no copy, prefix-LM attention)."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, h = cfg["d_model"], cfg["n_heads"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.blocks = nn.ModuleList(Block(d, h, cfg["mlp_hidden"]) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight
        if cfg.get("use_copy"):
            self.copy_q = nn.Linear(d, cfg["copy_dim"], bias=False)
            self.copy_k = nn.Linear(d, cfg["copy_dim"], bias=False)
            self.copy_gate = nn.Linear(d, 1)
        cos, sin = rope_tables(d // h, cfg["max_seq_len"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def hidden(self, ids, mask=None, cache=None, start_pos=0):
        T = ids.shape[1]
        assert start_pos + T <= self.cfg["max_seq_len"], "sequence longer than the trained context"
        cos, sin = self.cos[start_pos:start_pos + T], self.sin[start_pos:start_pos + T]
        x = self.tok_emb(ids)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, mask, True, None if cache is None else cache[i])
        return self.norm_f(x)

    @staticmethod
    def prefix_mask(prefix_len, T, device):
        """B (prefix-LM): key j visible to query i if j is inside the prefix (document + EOS) OR j <= i."""
        i = torch.arange(T, device=device)
        causal = i[None, :] <= i[:, None]
        in_prefix = i.view(1, 1, T) < prefix_len.view(-1, 1, 1)
        return (causal.unsqueeze(0) | in_prefix).unsqueeze(1)

    def summ_forward(self, batch, full=False):
        """batch: x (B, T) ids, y (B, T) targets (IGNORE outside the summary), prefix_len (B,) (used by B).
        Returns dict: nll (N,) per summary token, ys (N,), copy (N,) copy weight or None, logp (N, V) if full."""
        x, y = batch["x"], batch["y"]
        T = x.shape[1]
        mask = self.prefix_mask(batch["prefix_len"], T, x.device) if self.cfg.get("prefix_lm") else None
        h = self.hidden(x, mask)
        sel = y != IGNORE
        hs, ys = h[sel], y[sel]
        logits = self.lm_head(hs)
        if not self.cfg.get("use_copy"):
            return no_copy(logits, ys, full)
        q, k = self.copy_q(h).float(), self.copy_k(h).float()
        scores = (q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]))[sel]           # (N, T): one row per summary token
        pos = sel.nonzero()[:, 1]
        key_mask = torch.arange(T, device=x.device)[None, :] <= pos[:, None]      # copy from j <= t (as Phase 1)
        key_ids = x.unsqueeze(1).expand(-1, T, -1)[sel]                            # (N, T) token id of every key
        return copy_mix(self, logits, hs, ys, scores, key_ids, key_mask, full)


class EncDecLM(nn.Module):
    """D: encoder (bidirectional) + decoder (causal + cross-attention) + copy from the document.
    src (B, C, S): C chunks of the document, each encoded on its own (positions 0..S-1), then all chunk vectors
    are joined into one memory of C*S tokens for cross-attention and copying ("fusion-in-decoder",
    Izacard & Grave 2021). C = 1 is the normal encoder-decoder."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, h, hid = cfg["d_model"], cfg["n_heads"], cfg["mlp_hidden"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.enc = nn.ModuleList(EncBlock(d, h, hid) for _ in range(cfg["n_enc_layers"]))
        self.enc_norm = RMSNorm(d)
        self.dec = nn.ModuleList(DecBlock(d, h, hid) for _ in range(cfg["n_dec_layers"]))
        self.dec_norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight
        if cfg.get("use_copy"):
            self.copy_q = nn.Linear(d, cfg["copy_dim"], bias=False)
            self.copy_k = nn.Linear(d, cfg["copy_dim"], bias=False)
            self.copy_gate = nn.Linear(d, 1)
        cos, sin = rope_tables(d // h, cfg["max_seq_len"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def encode(self, src, src_mask):
        """src, src_mask (B, C, S) -> memory (B, C*S, d), memory mask (B, C*S)."""
        B, C, S = src.shape
        s, m = src.reshape(B * C, S), src_mask.reshape(B * C, S)
        m_self = m.clone()
        m_self[:, 0] = True                         # an all-padding chunk must still attend to something (ignored later)
        x = self.tok_emb(s)
        cos, sin = self.cos[:S], self.sin[:S]
        for blk in self.enc:
            x = blk(x, cos, sin, m_self[:, None, None, :])
        x = self.enc_norm(x)
        return x.reshape(B, C * S, -1), m.reshape(B, C * S)

    def decode(self, dec_in, mems, mem_mask, start_pos=0, caches=None):
        T = dec_in.shape[1]
        cos, sin = self.cos[start_pos:start_pos + T], self.sin[start_pos:start_pos + T]
        x = self.tok_emb(dec_in)
        for i, blk in enumerate(self.dec):
            x = blk(x, cos, sin, mems[i], mem_mask, None if caches is None else caches[i])
        return self.dec_norm(x)

    def summ_forward(self, batch, full=False):
        """batch: src, src_mask (B, C, S); dec_in, y (B, T) (y = IGNORE on padding)."""
        enc, mem_mask = self.encode(batch["src"], batch["src_mask"])
        mems = [blk.cross.memory(enc) for blk in self.dec]
        h = self.decode(batch["dec_in"], mems, mem_mask)
        y = batch["y"]
        sel = y != IGNORE
        hs, ys = h[sel], y[sel]
        logits = self.lm_head(hs)
        if not self.cfg.get("use_copy"):
            return no_copy(logits, ys, full)
        B, M = mem_mask.shape
        q, k = self.copy_q(h).float(), self.copy_k(enc).float()
        scores = (q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]))[sel]           # (N, M)
        T = h.shape[1]
        key_mask = mem_mask.unsqueeze(1).expand(-1, T, -1)[sel]
        key_ids = batch["src"].reshape(B, M).unsqueeze(1).expand(-1, T, -1)[sel]
        return copy_mix(self, logits, hs, ys, scores, key_ids, key_mask, full)


# ----------------------------------------------------------------------------------------
# 3. Loading Phase-1 weights (strict: every weight must come from Phase 1)
# ----------------------------------------------------------------------------------------
def load_phase1_lm(run_dir, keep_dec_layers=None):
    """Whole Phase-1 language model from config.json + model_final.pt.
    A2/A/B -> DecoderOnlyLM (B gets prefix_lm=True); D -> EncDecLM. MTP weights (if any) are dropped.
    keep_dec_layers (D only), e.g. [0, 2, 4]: a SMALLER D that keeps only these decoder layers (KD student,
    initialised from every other Phase-1 decoder layer, as DistilBERT does)."""
    run_dir = Path(run_dir)
    pcfg = json.load(open(run_dir / "config.json"))
    sd = torch.load(run_dir / "model_final.pt", map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith(("mtp_block.", "norm_mtp."))}
    cfg = {k: pcfg[k] for k in ("vocab_size", "d_model", "n_heads", "mlp_hidden", "max_seq_len")}
    cfg.update(use_copy=bool(pcfg.get("use_copy")), copy_dim=pcfg.get("copy_dim", 128), arch=pcfg["arch"])
    if pcfg["arch"] == "D_encdec":
        keep = list(range(pcfg["n_dec_layers"])) if keep_dec_layers is None else list(keep_dec_layers)
        cfg.update(n_enc_layers=pcfg["n_enc_layers"], n_dec_layers=len(keep), kind="encdec")
        new = {}
        for k, v in sd.items():
            m = re.match(r"^dec\.(\d+)\.(.*)$", k)
            if m:
                if int(m.group(1)) in keep:
                    new[f"dec.{keep.index(int(m.group(1)))}.{m.group(2)}"] = v
            else:
                new[k] = v
        model = EncDecLM(cfg)
    else:
        cfg.update(n_layers=pcfg["n_layers"], prefix_lm=pcfg["arch"].startswith("B"), kind="decoder")
        new = sd
        model = DecoderOnlyLM(cfg)
    model.load_state_dict(new, strict=True)
    return model, cfg


# ----------------------------------------------------------------------------------------
# 4. PEFT
# ----------------------------------------------------------------------------------------
LINEAR_TARGETS = ("qkv", "proj", "w1", "w2", "w3", "q", "kv")      # every weight matrix inside the blocks
COPY_PREFIX = ("copy_q", "copy_k", "copy_gate")


def _block_lists(model):
    return [model.blocks] if hasattr(model, "blocks") else [model.enc, model.dec]


def freeze_all_but_copy(model):
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(COPY_PREFIX))


def add_low_rank(model, kind, r=8, alpha=16):
    """LoRA or LoRMA on every linear layer of every block (attention, cross-attention, feed-forward)."""
    freeze_all_but_copy(model)
    n = 0
    for blocks in _block_lists(model):
        for blk in blocks:
            for sub in [m for m in (getattr(blk, "attn", None), getattr(blk, "cross", None), blk.mlp) if m is not None]:
                for name in LINEAR_TARGETS:
                    if isinstance(getattr(sub, name, None), nn.Linear):
                        w = LowRankLinear(getattr(sub, name), kind, r, alpha)
                        setattr(sub, name, w)
                        n += w.A.numel() + w.B.numel()
    return n


def add_houlsby(model, bottleneck=64):
    """Houlsby adapters: one after the self-attention and one after the feed-forward of every block."""
    freeze_all_but_copy(model)
    d = model.cfg["d_model"]
    n = 0
    for blocks in _block_lists(model):
        for blk in blocks:
            blk.attn = Adapter(blk.attn, d, bottleneck)
            blk.mlp = Adapter(blk.mlp, d, bottleneck)
            n += sum(p.numel() for a in (blk.attn, blk.mlp) for p in (*a.down.parameters(), *a.up.parameters()))
    return n


def kd_loss(student_logp, teacher_logp):
    """KL(p_teacher || p_student) per summary token (word-level KD, Kim & Rush 2016), averaged."""
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()


# ----------------------------------------------------------------------------------------
# 5. Generation (greedy, KV cache, copy mechanism, length control, no-repeat n-gram)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, src_ids, tokenizer, device, amp_dtype, max_words=500, min_words=80, no_repeat=8,
             max_new_tokens=None, chunk_len=1024, n_chunks=1, rep_penalty=1.0):
    """One summary. src_ids = document tokens (already cut to the model's input budget).
    decoder-only: prompt = document + EOS (B: read bidirectionally), then the summary token by token.
    encoder-decoder: document -> encoder (n_chunks chunks of chunk_len), decoder starts from EOS.
    Rules: greedy (argmax); EOS (= end of summary) is forbidden before `min_words` words; stop at EOS or when
    `max_words` words are reached (then cut back to the last full sentence if it is close); no 8-token sequence is repeated
    inside the SUMMARY (the document itself may be copied freely, unlike the Phase-1 repetition penalty).
    Returns (text, info) with info = number of tokens, mean copy weight, stop reason."""
    model.eval()
    src_ids = [int(t) for t in src_ids]
    ac = dict(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32)
    has_copy = model.cfg.get("use_copy", False)
    out_ids, copy_w, ngrams = [], [], {}
    V = model.cfg["vocab_size"]

    if model.cfg["kind"] == "decoder":
        prompt = list(src_ids) + [EOS_ID]
        max_new = max_new_tokens or (model.cfg["max_seq_len"] - len(prompt))
        cache = [dict() for _ in model.blocks]
        x = torch.tensor([prompt], device=device)
        mask = None
        if model.cfg.get("prefix_lm"):
            mask = torch.ones(1, 1, len(prompt), len(prompt), dtype=torch.bool, device=device)   # prompt: bidirectional
        with torch.autocast(**ac):
            h = model.hidden(x, mask, cache=cache)
        h_last = h[0, -1]
        keys = model.copy_k(h[0]).float() if has_copy else None
        key_ids = torch.tensor(prompt, device=device)
        pos = len(prompt)
    else:
        S = chunk_len
        chunks = [src_ids[i * S:(i + 1) * S] for i in range(n_chunks)]
        chunks = [c for c in chunks if len(c)] or [src_ids[:1] or [EOS_ID]]
        src = np.full((1, len(chunks), S), PAD_ID, dtype=np.int64)
        msk = np.zeros((1, len(chunks), S), dtype=bool)
        for i, c in enumerate(chunks):
            src[0, i, :len(c)], msk[0, i, :len(c)] = c, True
        src, msk = torch.from_numpy(src).to(device), torch.from_numpy(msk).to(device)
        with torch.autocast(**ac):
            enc, mem_mask = model.encode(src, msk)
            mems = [blk.cross.memory(enc) for blk in model.dec]
        keys = model.copy_k(enc[0][mem_mask[0]]).float() if has_copy else None
        key_ids = src.reshape(1, -1)[0][mem_mask[0]]
        caches = [dict() for _ in model.dec]
        max_new = max_new_tokens or (model.cfg["max_seq_len"] - 1)
        nxt, pos = EOS_ID, 0
        with torch.autocast(**ac):
            h_last = model.decode(torch.tensor([[nxt]], device=device), mems, mem_mask, 0, caches)[0, -1]
        pos = 1

    stop, words = "max_tokens", 0
    for t in range(max_new):
        logits = model.lm_head(h_last).float()
        if has_copy:
            q = model.copy_q(h_last).float()
            a = torch.softmax(keys @ q / math.sqrt(q.shape[-1]), dim=0)
            g = torch.sigmoid(model.copy_gate(h_last).float())
            p = g * torch.softmax(logits, dim=-1)
            p.scatter_add_(0, key_ids, (1 - g) * a)
            logits = torch.log(p + 1e-12)
            copy_w.append(float(1 - g))
        logits[PAD_ID] = -float("inf")
        if rep_penalty != 1.0 and out_ids:
            # repetition penalty (Keskar et al. 2019) on tokens ALREADY WRITTEN IN THE SUMMARY only (copying a document
            # word for the first time stays free). logits here are log-probabilities (< 0): multiplying by 1.3 makes
            # an already-used token 1.3x less likely in log space.
            used = torch.tensor(sorted(set(out_ids)), device=logits.device)
            logits[used] = torch.where(logits[used] > 0, logits[used] / rep_penalty, logits[used] * rep_penalty)
        if words < min_words:
            logits[EOS_ID] = -float("inf")
        if no_repeat and len(out_ids) >= no_repeat - 1:
            banned = ngrams.get(tuple(out_ids[len(out_ids) - no_repeat + 1:]))
            if banned:
                logits[list(banned)] = -float("inf")
        nxt = int(torch.argmax(logits))
        if nxt == EOS_ID:
            stop = "eos"
            break
        out_ids.append(nxt)
        if no_repeat and len(out_ids) >= no_repeat:
            ngrams.setdefault(tuple(out_ids[-no_repeat:-1]), set()).add(nxt)
        if (t + 1) % 8 == 0:
            words = len(tokenizer.decode(out_ids).split())
            if words >= max_words:
                stop = "max_words"
                break
        # next step
        if model.cfg["kind"] == "decoder":
            if pos >= model.cfg["max_seq_len"]:
                stop = "context_full"
                break
            with torch.autocast(**ac):
                h = model.hidden(torch.tensor([[nxt]], device=device), None, cache=cache, start_pos=pos)
            h_last = h[0, -1]
            if has_copy:
                keys = torch.cat([keys, model.copy_k(h_last).float()[None]])
                key_ids = torch.cat([key_ids, torch.tensor([nxt], device=device)])
        else:
            if pos >= model.cfg["max_seq_len"]:
                stop = "context_full"
                break
            with torch.autocast(**ac):
                h_last = model.decode(torch.tensor([[nxt]], device=device), mems, mem_mask, pos, caches)[0, -1]
        pos += 1

    text = tokenizer.decode(out_ids).strip()
    ws = text.split()
    if len(ws) > max_words:
        ws = ws[:max_words]
        cut = " ".join(ws)
        last = cut.rfind(". ")
        text = cut[:last + 1] if last > 0.8 * len(cut) else cut       # end on a full sentence if one is close
    return text, {"tokens": len(out_ids), "copy_weight": float(np.mean(copy_w)) if copy_w else None, "stop": stop}
