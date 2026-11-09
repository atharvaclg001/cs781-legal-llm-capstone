"""
summ_experiment.py  -  one Phase-6 experiment = one config dict -> train, validate, generate, score, submit, log
(same structure as Phases 4-5: cjpe_experiment.py / lsi_experiment.py)

    exp = Experiment(ROOT, {"phase1": "A2", "method": "full"})
    exp.memory_check()    # one forward + backward on the largest micro-batch: does it fit in 6 GB? (cached)
    exp.train()           # train on the train part, validation loss after every epoch, keep the best epoch (cached,
                          # checkpoint every 250 optimiser steps -> an interrupted run resumes where it stopped)
    exp.evaluate_local()  # generate summaries for the 150 local-eval documents, BERTScore + ROUGE + length,
                          # repetition, copying (generation cached every 10 documents)
    exp.predict_test()    # 100 test summaries -> submission.zip (checked) (cached)
    exp.log()             # one line in experiments/phase6/experiments.jsonl

Data (decided in P6_01, PROJECT_NOTES 13.3): own split train 6,661 / val 367 (local-eval = 150 of val);
document = first 900 tokens (decoder-only) or 1024 per encoder chunk (D); target = first 620 summary tokens
(+ EOS if the summary ends there); generation length = length rule from the document length, capped at 500 words.
Run folder: checkpoints/phase6/<run name>; the name is built from the config.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm.auto import tqdm

import summ_utils as su
import summ_models as sm

PHASE1_RUNS = {
    "A2": "A2_copy1_mtp0_d640_L10_sc70tr30_200M",     # decoder + copy (best Phase-1 model)
    "A": "A_decoder_d640_L10_sc70tr30_200M",          # plain decoder
    "B": "B_prefixlm_d640_L10_sc70tr30_200M",         # prefix-LM
    "D": "D_encdec_copy1_e4d5_640_sc70tr30_200M",     # encoder-decoder + copy
}

DEFAULTS = dict(
    phase1="A2", method="full",           # method: full | lora | lorma | houlsby
    rank=8, alpha=16, adapter_dim=64,
    doc_tokens=None,                      # decoder-only: 900 (900 + 1 + 620 <= 1536); D: 1024 per chunk
    n_chunks=1,                           # D only: number of 1024-token encoder chunks (fusion-in-decoder)
    keep_dec_layers=None,                 # D only: smaller student, e.g. [0, 2, 4] of the 5 decoder layers
    kd_teacher=None, kd_alpha=0.5,        # kd_teacher = config dict of the teacher experiment
    epochs=3, patience=1, per_step=16, token_budget=3072,
    lr=1e-4, weight_decay=0.01, warmup_frac=0.05, grad_clip=1.0, seed=42, ckpt_every=250,
    max_words=500, min_frac=0.5, no_repeat=8,     # no repeated 8-token sequence inside a summary (refs repeat legal phrases)
    train_limit=None, val_limit=None, eval_limit=None, test_limit=None,     # small numbers = quick tests
    name_suffix=None,
)
PEFT_METHODS = ("lora", "lorma", "houlsby")
PEFT_DEFAULTS = dict(lr=5e-4, epochs=4)

_DATA = {}   # loaded once per notebook, shared by all experiments


def run_name(c):
    m = c["method"]
    if m in ("lora", "lorma"):
        m += f"r{c['rank']}"
    if m == "houlsby":
        m += str(c["adapter_dim"])
    name = c["phase1"]
    if c["keep_dec_layers"]:
        name += "small_dec" + "".join(str(i) for i in c["keep_dec_layers"])
    name += f"_{m}_"
    name += f"enc{c['n_chunks']}x{c['doc_tokens']}" if c["phase1"] == "D" else f"doc{c['doc_tokens']}"
    if c["kd_teacher"]:
        name += f"_kd{c['kd_alpha']:g}"
    if c["train_limit"]:
        name += f"_n{c['train_limit']}"
    if c.get("name_suffix"):
        name += "_" + c["name_suffix"]
    return name


def make_steps(lengths, per_step, seed):
    """Groups of `per_step` examples of similar length, in random group order (same seed -> same groups)."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(lengths)), np.asarray(lengths)))
    groups = [order[i:i + per_step] for i in range(0, len(order), per_step)]
    return [groups[i] for i in rng.permutation(len(groups))]


def micro_batches(idx, lengths, token_budget):
    """Split a group into micro-batches with (examples) x (longest example) <= token_budget tokens."""
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


class Experiment:
    def __init__(self, root, overrides=None, quiet=False):
        self.root = Path(root)
        overrides = dict(overrides or {})
        base = {**DEFAULTS, **(PEFT_DEFAULTS if overrides.get("method") in PEFT_METHODS else {})}
        c = {**base, **overrides}
        if c["doc_tokens"] is None:
            c["doc_tokens"] = 1024 if c["phase1"] == "D" else 900
        c["phase1_run"] = PHASE1_RUNS[c["phase1"]]
        self.cfg = c
        self.name = run_name(c)
        self.dir = self.root / "checkpoints" / "phase6" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        json.dump(c, open(self.dir / "config.json", "w"), indent=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self._load_data()
        self.model = None
        if not quiet:
            print(f"experiment: {self.name} | device {self.device} | precision {self.amp}")

    # ---------------- data ----------------
    def _load_data(self):
        key = str(self.root)
        if key not in _DATA:
            data, info = su.load_summ(self.root)
            split = json.load(open(self.root / "data" / "processed" / "phase6" / "split_v1.json"))
            rule = json.load(open(self.root / "experiments" / "phase6" / "p6_01_data" / "data_summary.json"))["length_rule"]
            tok = Tokenizer.from_file(str(self.root / "data" / "tokenizer" / "bpe32k.json"))
            by_id = {d["id"]: d for d in data["train"]}
            used = [by_id[i] for i in split["train"] + split["val"]]           # documents WITH a summary
            enc = su.encode_inputs(self.root, "trainval", used, tok)
            enc_test = su.encode_inputs(self.root, "test", data["test"], tok)
            _DATA[key] = dict(by_id=by_id, test=data["test"], split=split, rule=rule, tok=tok, enc=enc, enc_test=enc_test)
            n_cut = sum(s[-1] != su.EOS_ID for _, s in enc.values())
            print(f"data: train {len(split['train'])}, val {len(split['val'])}, local_eval {len(split['local_eval'])}, "
                  f"test {len(data['test'])} | targets cut at {su.SUM_MAX} tokens: {100 * n_cut / len(enc):.0f}%")
        D = _DATA[key]
        self.by_id, self.test_docs, self.split, self.rule, self.tok = D["by_id"], D["test"], D["split"], D["rule"], D["tok"]
        self.enc, self.enc_test = D["enc"], D["enc_test"]
        c = self.cfg
        self.train_ids = self.split["train"][:c["train_limit"]]
        self.val_ids = self.split["val"][:c["val_limit"]]
        self.local_ids = self.split["local_eval"][:c["eval_limit"]]

    def is_decoder(self):
        return self.cfg["phase1"] != "D"

    def src_tokens(self, doc_ids):
        c = self.cfg
        n = c["doc_tokens"] * (c["n_chunks"] if not self.is_decoder() else 1)
        return [int(t) for t in doc_ids[:n]]

    def example(self, i):
        """(x-part, target) for one training document in this experiment's layout."""
        doc, summ = self.enc[i]
        return self.src_tokens(doc), [int(t) for t in summ]

    def lengths(self, ids):
        out = []
        for i in ids:
            d, s = self.example(i)
            out.append(len(d) + 1 + len(s) if self.is_decoder() else self.cfg["n_chunks"] * self.cfg["doc_tokens"] + len(s))
        return np.array(out)

    def collate(self, ids):
        """Training / validation batch on the device.
        decoder-only: seq = doc + [EOS] + summary; x = seq[:-1], y = seq[1:], y = IGNORE on the document part;
                      prefix_len = len(doc) + 1 (document + EOS = the bidirectional part for B).
        D: src (B, C, 1024) chunks + mask; dec_in = [EOS] + summary[:-1]; y = summary."""
        exs = [self.example(i) for i in ids]
        dev = self.device
        if self.is_decoder():
            seqs = [d + [sm.EOS_ID] + s for d, s in exs]
            T = max(len(q) for q in seqs) - 1
            x = np.full((len(exs), T), sm.PAD_ID, dtype=np.int64)
            y = np.full((len(exs), T), sm.IGNORE, dtype=np.int64)
            for r, ((d, s), q) in enumerate(zip(exs, seqs)):
                x[r, :len(q) - 1] = q[:-1]
                y[r, :len(q) - 1] = q[1:]
                y[r, :len(d)] = sm.IGNORE
            pl = torch.tensor([len(d) + 1 for d, _ in exs], device=dev)
            return {"x": torch.from_numpy(x).to(dev), "y": torch.from_numpy(y).to(dev), "prefix_len": pl}
        C, S = self.cfg["n_chunks"], self.cfg["doc_tokens"]
        src = np.full((len(exs), C, S), sm.PAD_ID, dtype=np.int64)
        msk = np.zeros((len(exs), C, S), dtype=bool)
        T = max(len(s) for _, s in exs)
        dec_in = np.full((len(exs), T), sm.PAD_ID, dtype=np.int64)
        y = np.full((len(exs), T), sm.IGNORE, dtype=np.int64)
        for r, (d, s) in enumerate(exs):
            for c in range(C):
                part = d[c * S:(c + 1) * S]
                src[r, c, :len(part)], msk[r, c, :len(part)] = part, True
            if not msk[r].any():
                msk[r, 0, 0] = True
            dec_in[r, :len(s)] = [sm.EOS_ID] + s[:-1]
            y[r, :len(s)] = s
        t = lambda a: torch.from_numpy(a).to(dev)
        return {"src": t(src), "src_mask": t(msk), "dec_in": t(dec_in), "y": t(y)}

    # ---------------- model ----------------
    def build_model(self):
        c = self.cfg
        model, mcfg = sm.load_phase1_lm(self.root / "checkpoints" / "phase1" / c["phase1_run"], c["keep_dec_layers"])
        if c["method"] in ("lora", "lorma"):
            sm.add_low_rank(model, c["method"], c["rank"], c["alpha"])
        elif c["method"] == "houlsby":
            sm.add_houlsby(model, c["adapter_dim"])
        elif c["method"] != "full":
            raise ValueError(c["method"])
        return model.to(self.device)

    def memory_check(self):
        path = self.dir / "memory_check.json"
        if path.exists():
            res = json.load(open(path))
            print("[cache] memory check:", res)
            return res
        model = self.build_model()
        total, trainable = sm.count_params(model)
        lengths = self.lengths(self.train_ids)
        longest = list(np.argsort(lengths)[-max(1, self.cfg["token_budget"] // int(lengths.max())):])
        batch = self.collate([self.train_ids[i] for i in longest])
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train()
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp != torch.float32):
            out = model.summ_forward(batch, full=bool(self.cfg["kd_teacher"]))
        out["nll"].mean().backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        res = {"params_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 3),
               "trainable_pct": round(100 * trainable / total, 2), "micro_batch": f"{len(longest)} x {int(lengths.max())}",
               "sec": round(time.time() - t0, 2),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None}
        print("memory check:", res)
        json.dump(res, open(path, "w"), indent=1)
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return res

    @torch.no_grad()
    def val_loss(self, model):
        """Mean NLL per summary token on the validation documents (cheap; used to pick the best epoch)."""
        model.eval()
        lengths = self.lengths(self.val_ids)
        tot, n, cw = 0.0, 0, []
        for part in tqdm(micro_batches(range(len(self.val_ids)), lengths, self.cfg["token_budget"] * 2),
                         desc="validation", leave=False):
            batch = self.collate([self.val_ids[i] for i in part])
            with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp != torch.float32):
                out = model.summ_forward(batch)
            tot += float(out["nll"].sum())
            n += out["nll"].numel()
            if out["copy"] is not None:
                cw.append(float(out["copy"].mean()))
        model.train()
        return {"val_loss": tot / max(n, 1), "val_copy_weight": float(np.mean(cw)) if cw else None}

    def _teacher(self):
        t = Experiment(self.root, self.cfg["kd_teacher"], quiet=True)
        hist = t.dir / "history_train.json"
        assert hist.exists() and json.load(open(hist)).get("done"), f"teacher {t.name} is not trained yet"
        model = t.build_model()
        model.load_state_dict(torch.load(t.dir / "best_train.pt", map_location=self.device))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"knowledge distillation: teacher {t.name} (alpha {self.cfg['kd_alpha']})")
        return t, model

    def train(self):
        c = self.cfg
        hist_path, best_path, ckpt_path = self.dir / "history_train.json", self.dir / "best_train.pt", self.dir / "ckpt_train.pt"
        model = self.build_model()
        self.model = model
        if hist_path.exists():
            state = json.load(open(hist_path))
            if state.get("done") and best_path.exists():
                model.load_state_dict(torch.load(best_path, map_location=self.device))
                print(f"[cache] training already done -> loaded best epoch {state['best_epoch']}")
                return state
        total, trainable = sm.count_params(model)
        print(f"parameters: {total / 1e6:.2f}M, trainable {trainable / 1e6:.3f}M ({100 * trainable / total:.2f}%)")
        teacher = self._teacher() if c["kd_teacher"] else None

        params = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": [p for p in params if p.ndim >= 2], "weight_decay": c["weight_decay"]},
                  {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
        opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=c["lr"], betas=(0.9, 0.999))
        ids = self.train_ids
        lengths = self.lengths(ids)
        n_target = np.array([len(self.example(i)[1]) for i in ids])
        steps_per_epoch = math.ceil(len(ids) / c["per_step"])
        total_steps = steps_per_epoch * c["epochs"]
        warm = max(1, int(c["warmup_frac"] * total_steps))
        lr_at = lambda s: c["lr"] * ((s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))

        state = {"epoch": 0, "step_in_epoch": 0, "step": 0, "history": [], "best": float("inf"), "best_epoch": 0,
                 "bad": 0, "done": False, "steps_per_epoch": steps_per_epoch, "losses": []}
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optim"])
            state = ck["state"]
            print(f"resuming: epoch {state['epoch'] + 1}, step {state['step_in_epoch']}/{steps_per_epoch}")
        save_ckpt = lambda: torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "state": state}, ckpt_path)
        if state["epoch"] == 0 and state["step_in_epoch"] == 0 and not state["history"]:
            v0 = self.val_loss(model)
            state["val_before_training"] = v0
            print(f"before fine-tuning: val loss {v0['val_loss']:.4f}")

        ac = dict(device_type=self.device.type, dtype=self.amp, enabled=self.amp != torch.float32)
        for epoch in range(state["epoch"], c["epochs"]):
            model.train()
            t0 = time.time()
            steps = make_steps(lengths, c["per_step"], c["seed"] + epoch)
            bar = tqdm(range(state["step_in_epoch"], steps_per_epoch), desc=f"epoch {epoch + 1}/{c['epochs']}",
                       initial=state["step_in_epoch"], total=steps_per_epoch)
            for s in bar:
                group = steps[s]
                for g in opt.param_groups:
                    g["lr"] = lr_at(state["step"])
                n_step = int(n_target[group].sum())
                step_loss, step_kd, cw = 0.0, 0.0, []
                for part in micro_batches(group, lengths, c["token_budget"]):
                    batch = self.collate([ids[i] for i in part])
                    with torch.autocast(**ac):
                        out = model.summ_forward(batch, full=teacher is not None)
                    loss = out["nll"].mean()
                    if teacher is not None:
                        t_exp, t_model = teacher
                        with torch.no_grad(), torch.autocast(**ac):
                            t_out = t_model.summ_forward(t_exp.collate([ids[i] for i in part]), full=True)
                        assert torch.equal(t_out["ys"], out["ys"]), "teacher and student targets differ"
                        kd = sm.kd_loss(out["logp"], t_out["logp"])
                        loss = (1 - c["kd_alpha"]) * loss + c["kd_alpha"] * kd
                        step_kd += kd.item() * out["nll"].numel() / n_step
                    (loss * out["nll"].numel() / n_step).backward()
                    step_loss += out["nll"].detach().mean().item() * out["nll"].numel() / n_step
                    if out["copy"] is not None:
                        cw.append(out["copy"].detach().mean().item())
                torch.nn.utils.clip_grad_norm_(params, c["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                state["step"] += 1
                state["step_in_epoch"] = s + 1
                state["losses"].append(round(step_loss, 4))
                post = {"nll": f"{np.mean(state['losses'][-50:]):.3f}"}
                if teacher is not None:
                    post["kd"] = f"{step_kd:.3f}"
                if cw:
                    post["copy"] = f"{np.mean(cw):.2f}"
                bar.set_postfix(post)
                if c["ckpt_every"] and state["step_in_epoch"] % c["ckpt_every"] == 0 and s + 1 < steps_per_epoch:
                    save_ckpt()
            bar.close()
            v = self.val_loss(model)
            row = {"epoch": epoch + 1, "train_nll": float(np.mean(state["losses"][-steps_per_epoch:])),
                   "train_min": round((time.time() - t0) / 60, 1), **v}
            improved = v["val_loss"] < state["best"]
            if improved:
                state.update(best=v["val_loss"], best_epoch=epoch + 1, bad=0)
                torch.save(model.state_dict(), best_path)
            else:
                state["bad"] += 1
            state["history"].append(row)
            state["epoch"], state["step_in_epoch"] = epoch + 1, 0
            print(f"epoch {epoch + 1} | train NLL {row['train_nll']:.4f} | val loss {v['val_loss']:.4f} | "
                  f"{row['train_min']:.0f} min" + ("  <- best" if improved else ""))
            stop = state["bad"] >= c["patience"]
            state["done"] = stop or state["epoch"] == c["epochs"]
            save_ckpt()
            json.dump(state, open(hist_path, "w"), indent=1)
            if stop:
                print(f"early stop: no improvement for {c['patience']} epoch(s)")
                break
        state["done"] = True
        json.dump(state, open(hist_path, "w"), indent=1)
        if ckpt_path.exists():
            ckpt_path.unlink()                    # resume file (with optimiser state) no longer needed
        model.load_state_dict(torch.load(best_path, map_location=self.device))
        print(f"training finished -> best epoch {state['best_epoch']} (val loss {state['best']:.4f})")
        return state

    def _model(self):
        if self.model is None or not (self.dir / "history_train.json").exists():
            self.train()
        return self.model

    # ---------------- generation ----------------
    def decode_cfg(self, decode=None):
        """Decoding settings: the experiment's defaults, overridden by `decode` (dict).
        max_words / min_frac: length control; no_repeat: no repeated n-token sequence inside the summary;
        rep_penalty: penalty on tokens already written in the summary; cut_words: after generation, cut the summary
        to this many words (ending on a full sentence if one is close)."""
        c = self.cfg
        d = {"max_words": c["max_words"], "min_frac": c["min_frac"], "no_repeat": c["no_repeat"], "rep_penalty": 1.0,
             "cut_words": None}
        d.update(decode or {})
        return d

    @staticmethod
    def cut(text, n):
        ws = text.split()
        if not n or len(ws) <= n:
            return text
        t = " ".join(ws[:n])
        last = t.rfind(". ")
        return t[:last + 1] if last > 0.8 * len(t) else t

    def _generate(self, items, out_path, desc, decode=None):
        """items: list of (id, document text, document token ids). Cached in out_path; saved every 10 documents."""
        preds = json.load(open(out_path)) if out_path.exists() else {}
        todo = [it for it in items if it[0] not in preds]
        if not todo:
            return preds
        model = self._model()
        c, d = self.cfg, self.decode_cfg(decode)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for k, (i, text, doc_ids) in enumerate(tqdm(todo, desc=desc)):
            target = su.length_rule(len(text.split()), self.rule, hi=d["max_words"])
            t0 = time.time()
            summ, info = sm.generate(model, self.src_tokens(doc_ids), self.tok, self.device, self.amp,
                                     max_words=target, min_words=int(d["min_frac"] * target), no_repeat=d["no_repeat"],
                                     chunk_len=c["doc_tokens"], n_chunks=c["n_chunks"], rep_penalty=d["rep_penalty"])
            if not summ.strip():                                    # never submit an empty summary
                summ = " ".join(su.clean_summ(text).split()[:target])
                info["stop"] = "empty->lead"
            preds[i] = {"summary": self.cut(summ, d["cut_words"]), "target_words": target,
                        "sec": round(time.time() - t0, 2), **info}
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                json.dump(preds, open(out_path, "w"), indent=1)
        if self.device.type == "cuda":
            preds["_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            json.dump(preds, open(out_path, "w"), indent=1)
        return preds

    def evaluate_local(self, lengths_analysis=False, decode=None, tag=None, limit=None):
        """Summaries for the local-eval documents + BERTScore / ROUGE and quality statistics (cached).
        tag=None: the experiment's default decoding (files local_preds.json / local_scores.json);
        tag='x': decoding `decode`, files local_preds_x.json / local_scores_x.json, first `limit` documents."""
        sfx = f"_{tag}" if tag else ""
        path = self.dir / f"local_scores{sfx}.json"
        ids = self.local_ids[:limit]
        if path.exists():
            res = json.load(open(path))
            print(f"[cache] local eval{sfx}:", {k: res[k] for k in ("bertscore_f1", "rouge1_f1", "rougeL_f1", "pred_words")})
            if not lengths_analysis or "by_length" in res:
                return res
        items = [(i, self.by_id[i]["document"], self.enc[i][0]) for i in ids]
        preds = self._generate(items, self.dir / f"local_preds{sfx}.json", f"local-eval summaries{sfx}", decode)
        refs = [su.clean_summ(self.by_id[i]["summary"]) for i in ids]
        hyp = [preds[i]["summary"] for i in ids]
        if not path.exists():
            res = su.score_summaries(hyp, refs)
            docs = [su.norm_words(su.clean_summ(self.by_id[i]["document"])) for i in ids]
            cws = [preds[i]["copy_weight"] for i in ids if preds[i]["copy_weight"] is not None]
            res.update(
                decode=self.decode_cfg(decode),
                repeated_4gram_pct=100 * float(np.mean([su.repetition_rate(h, 4) for h in hyp])),
                ref_repeated_4gram_pct=100 * float(np.mean([su.repetition_rate(r, 4) for r in refs])),
                novel_2gram_pct=100 * float(np.nanmean([su.novel_ngram_share(su.norm_words(h), d, 2) for h, d in zip(hyp, docs)])),
                ref_novel_2gram_pct=100 * float(np.nanmean([su.novel_ngram_share(su.norm_words(r), d, 2) for r, d in zip(refs, docs)])),
                copy_weight=float(np.mean(cws)) if cws else None,
                stop_reasons={str(k): int(v) for k, v in zip(*np.unique([preds[i]["stop"] for i in ids], return_counts=True))},
                sec_per_doc=float(np.mean([preds[i]["sec"] for i in ids])),
                gen_tokens_per_s=float(sum(preds[i]["tokens"] for i in ids) / max(sum(preds[i]["sec"] for i in ids), 1e-9)),
                gen_peak_gb=preds.get("_peak_gb"))
        if lengths_analysis:
            res["by_length"] = {}
            for w in (128, 256, 426):
                res["by_length"][str(w)] = su.score_summaries([" ".join(h.split()[:w]) for h in hyp], refs)
            res["by_length"]["full"] = {k: res[k] for k in ("bertscore_f1", "rouge1_f1", "rouge2_f1", "rougeL_f1", "pred_words")}
        json.dump(res, open(path, "w"), indent=1, default=float)
        print(f"local eval{sfx}:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                                    if k not in ("by_length", "decode")})
        return res

    def predict_test(self, decode=None, tag=None):
        """Test summaries -> submission.zip (tag=None: run folder; tag='x': subfolder test_x/)."""
        out_dir = self.dir / f"test_{tag}" if tag else self.dir
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / "submission.zip"
        test_ids = [d["id"] for d in self.test_docs][:self.cfg["test_limit"]]
        if not zip_path.exists():
            items = [(d["id"], d["document"], self.enc_test[d["id"]][0]) for d in self.test_docs[:self.cfg["test_limit"]]]
            preds = self._generate(items, out_dir / "test_preds.json", f"test summaries{'_' + tag if tag else ''}", decode)
            su.write_submission(out_dir, test_ids, [preds[i]["summary"] for i in test_ids])
        else:
            print("[cache] submission exists")
        chk = su.check_submission(zip_path, test_ids)
        print("submission check:", chk, "->", zip_path)
        return zip_path

    def log(self):
        log_path = self.root / "experiments" / "phase6" / "experiments.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["run"] for l in open(log_path, encoding="utf-8") if l.strip()} if log_path.exists() else set()
        if self.name in done:
            print("[cache] already logged")
            return
        hist = json.load(open(self.dir / "history_train.json"))
        loc = json.load(open(self.dir / "local_scores.json"))
        mem = json.load(open(self.dir / "memory_check.json")) if (self.dir / "memory_check.json").exists() else {}
        rec = {"run": self.name, "phase1": self.cfg["phase1"], "method": self.cfg["method"],
               "kd": bool(self.cfg["kd_teacher"]), "params_M": mem.get("params_M"), "trainable_M": mem.get("trainable_M"),
               "trainable_pct": mem.get("trainable_pct"), "train_peak_gb": mem.get("peak_gb"),
               "best_epoch": hist["best_epoch"], "epochs_run": len(hist["history"]),
               "min_per_epoch": hist["history"][0]["train_min"],
               "val_loss_before": hist.get("val_before_training", {}).get("val_loss"), "val_loss": hist["best"],
               **{k: loc.get(k) for k in ("bertscore_f1", "bertscore_p", "bertscore_r", "rouge1_f1", "rouge2_f1", "rougeL_f1",
                                          "pred_words", "repeated_4gram_pct", "novel_2gram_pct", "copy_weight",
                                          "sec_per_doc", "gen_tokens_per_s", "gen_peak_gb")},
               "date": time.strftime("%Y-%m-%d %H:%M")}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("logged:", rec)


def compare(root):
    import pandas as pd
    path = Path(root) / "experiments" / "phase6" / "experiments.jsonl"
    if not path.exists():
        print("no experiments logged yet")
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    cols = ["phase1", "method", "kd", "params_M", "trainable_pct", "val_loss_before", "val_loss", "bertscore_f1",
            "rouge1_f1", "rougeL_f1", "pred_words", "repeated_4gram_pct", "novel_2gram_pct", "copy_weight",
            "min_per_epoch", "sec_per_doc", "train_peak_gb"]
    return pd.DataFrame(rows).set_index("run")[[c for c in cols if c in rows[0]]]
