"""
ner_experiment.py  -  one Phase-2 experiment = one config dict -> Stage A, report, Stage B, log

Every Phase-2 notebook (A2 full fine-tuning, LoRA, LoRMA, distillation, ablations) uses this file,
so all experiments are prepared, trained, scored and submitted in exactly the same way.

    exp = Experiment(PROJECT_ROOT, {"phase1_run": "...", "causal": True, "head": "bilstm"})
    exp.stage_a()     # train fold 1, validate on fold 2 every epoch, keep the best epoch   (cached)
    exp.report()      # fold-2 scores per label, decoding comparison, error breakdown        (cached)
    exp.stage_b()     # retrain on fold 1 + fold 2, predict test, write submission.zip       (cached)
    exp.log()         # one line in experiments/phase2/experiments.jsonl                     (once)

Run folder: checkpoints/phase2/<run name>; the name is built from the config, so every setting
gets its own folder and nothing is overwritten.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import ner_utils as nu
import ner_models as nm

DEFAULTS = dict(
    phase1_run="A2_copy1_mtp0_d640_L10_sc70tr30_200M",
    method="full",                 # full fine-tuning (later: lora, lorma, kd)
    causal=True,                   # Phase-1 decoders read left-to-right (Step-3 ablation: causal 0.497 > bidirectional 0.419)
    head="linear", lookahead=0, lstm_hidden=256,
    normalize=True, header=0, augment=False,
    max_len=1536, stride=1024,
    epochs=20, patience=4, micro_batch=2, grad_accum=2,
    lr=1e-4, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.1, grad_clip=1.0,
    dropout=0.1, class_weights=None, seed=42,
    rank=8, alpha=16, adapter_targets=("qkv", "proj", "w1", "w2", "w3"),   # LoRA / LoRMA settings
    kd_teacher=None, kd_alpha=0.5, kd_T=2.0,                                # knowledge distillation settings
)
PEFT_DEFAULTS = dict(lr=5e-4, epochs=30, patience=5)   # adapters start from 0 -> higher lr, more epochs

_DATA = {}   # tokenised data shared by all experiments in one notebook (tokenise once)


def short_arch(phase1_run):
    for prefix, short in (("A2_", "A2"), ("A_decoder", "A"), ("B_prefix", "B"), ("D_encdec", "D")):
        if phase1_run.startswith(prefix):
            return short
    return phase1_run.split("_")[0]


def run_name(cfg):
    """Readable, unique folder name. Kept compatible with the first runs (e.g. A2_full_causal_norm_hm0)."""
    method = cfg["method"] + (f"r{cfg['rank']}" if cfg["method"] in ("lora", "lorma") else "")
    name = f"{short_arch(cfg['phase1_run'])}_{method}_{'causal' if cfg['causal'] else 'bidir'}"
    if cfg["head"] == "lookahead":
        name += f"_look{cfg['lookahead']}"
    elif cfg["head"] != "linear":
        name += f"_{cfg['head']}"
    name += f"_{'norm' if cfg['normalize'] else 'raw'}_hm{cfg['header']}"
    if cfg["max_len"] != 1536:
        name += f"_w{cfg['max_len']}"
    if cfg["augment"]:
        name += "_aug"
    if cfg.get("class_weights"):
        name += "_cw"
    if cfg.get("name_suffix"):
        name += "_" + cfg["name_suffix"]
    return name


class Experiment:
    def __init__(self, root, overrides=None):
        self.root = Path(root)
        overrides = overrides or {}
        base = {**DEFAULTS, **(PEFT_DEFAULTS if overrides.get("method") in ("lora", "lorma") else {})}
        self.cfg = {**base, **overrides}
        self.name = run_name(self.cfg)
        self.dir = self.root / "checkpoints" / "phase2" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        json.dump(self.cfg, open(self.dir / "config.json", "w"), indent=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self._load_data()
        self.model = None
        self.state_a = None
        print(f"experiment: {self.name} | device {self.device} | precision {self.amp}")

    # ---------------- data ----------------
    def _load_data(self):
        key = (str(self.root), self.cfg["normalize"])
        if key not in _DATA:
            data = nu.load_ner_data(self.root)
            tok = Tokenizer.from_file(str(self.root / "data" / "tokenizer" / "bpe32k.json"))
            tags_list, tag2id = nu.make_tags(data["labels"])
            docs = {f: nu.encode_docs(data["texts"][f], tok, self.cfg["normalize"], desc=f) for f in ("fold1", "fold2")}
            docs["test"] = nu.encode_docs(data["test"], tok, self.cfg["normalize"], desc="test")
            tags = {f: [nu.spans_to_tags(d, data["golds"][f][d.doc_id], tag2id) for d in docs[f]] for f in ("fold1", "fold2")}
            _DATA[key] = (data, tags_list, docs, tags)
        self.data, self.TAGS, self.docs, self.tags = _DATA[key]
        self.LABELS = self.data["labels"]

    def windows(self, which):
        """Training windows for 'fold1' or 'all' (fold1+fold2). With augment=True: a function epoch -> (X, Y)."""
        c = self.cfg
        docs = self.docs["fold1"] if which == "fold1" else self.docs["fold1"] + self.docs["fold2"]
        tags = self.tags["fold1"] if which == "fold1" else self.tags["fold1"] + self.tags["fold2"]
        if c["augment"]:
            return lambda epoch: nu.make_training_windows(docs, tags, c["max_len"], c["stride"], c["header"],
                                                          rng=np.random.default_rng(c["seed"] * 1000 + epoch))
        X, Y = nu.make_training_windows(docs, tags, c["max_len"], c["stride"], c["header"])
        if c["method"] == "kd":
            T = self.teacher_outputs(which, X)
            return lambda epoch: (X, Y, T)   # + teacher log-probs for the distillation loss
        return lambda epoch: (X, Y)          # same fixed windows every epoch

    def teacher_outputs(self, which, X):
        """Teacher log-probabilities on the student's training windows (computed once, cached as float16).
        Stage A ('fold1') uses the teacher trained on fold 1 only, Stage B ('all') the teacher trained on fold 1+2,
        so the teacher has never seen the fold-2 validation labels when we validate the student."""
        tag = "fold1" if which == "fold1" else "full"
        path = self.dir / f"teacher_{tag}.npy"
        if path.exists():
            print(f"[cache] teacher outputs {path.name}")
            return list(np.load(path))
        teacher, _ = nm.load_trained(self.root, self.cfg["kd_teacher"], tag, self.device)
        out = nm.make_scorer(teacher, self.device, self.amp, desc=f"teacher on {which} windows")(X)
        arr = np.stack(out).astype(np.float16)
        np.save(path, arr)
        del teacher
        torch.cuda.empty_cache()
        return list(arr)

    # ---------------- model ----------------
    def build_model(self):
        c = self.cfg
        backbone, _, info = nm.load_phase1_backbone(self.root / "checkpoints" / "phase1" / c["phase1_run"])
        model = nm.NERTagger(backbone, len(self.TAGS), c["dropout"], c["causal"], c["head"], c["lookahead"], c["lstm_hidden"])
        if c["method"] in ("lora", "lorma"):
            info["adapter_params"] = nm.add_adapters(model.backbone, c["method"], c["rank"], c["alpha"], c["adapter_targets"])
        elif c["method"] not in ("full", "kd"):        # kd = full fine-tuning of the student + distillation loss
            raise ValueError(f"unknown method {c['method']}")
        self.load_info = info
        return model.to(self.device)

    def scorer(self, model):
        return nm.make_scorer(model, self.device, self.amp)

    def predict(self, model, docs):
        c = self.cfg
        return nu.predict_spans(docs, self.scorer(model), self.TAGS, c["max_len"], c["stride"], c["header"])

    def evaluate(self, model):
        pred = self.predict(model, self.docs["fold2"])
        model.train()
        return nu.strict_scores(self.data["golds"]["fold2"], pred, self.LABELS)

    # ---------------- Stage A ----------------
    def stage_a(self):
        """Train on fold 1, validate on fold 2 after every epoch; keep the best epoch (cached)."""
        model = self.build_model()
        total, trainable = nm.count_params(model)
        print(f"parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable | loaded {self.load_info}")
        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.state_a = nm.train_tagger(model, self.windows("fold1"), None, self.dir, self.cfg, self.device,
                                       tag="fold1", eval_fn=lambda: self.evaluate(model), amp_dtype=self.amp)
        self.model = model
        self.peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None
        print(f"Stage A: best epoch {self.state_a['best_epoch']} | val macro-F1 {self.state_a['best']:.4f} | {time.time() - t0:.0f}s")
        return self.state_a

    def val_outputs(self):
        """Fold-2 log-probabilities and predictions of the best fold-1 model (cached on disk)."""
        lp_path, pred_path = self.dir / "val_logprobs.npz", self.dir / "val_pred.json"
        if lp_path.exists() and pred_path.exists():
            lp = dict(np.load(lp_path))
            pred = {k: [tuple(x) for x in v] for k, v in json.load(open(pred_path)).items()}
            return {d.doc_id: lp.get(d.doc_id) for d in self.docs["fold2"]}, pred
        if self.model is None:
            self.stage_a()
        c = self.cfg
        lp = nu.doc_log_probs(self.docs["fold2"], self.scorer(self.model), c["max_len"], c["stride"], c["header"])
        pred = nu.decode_docs(self.docs["fold2"], lp, self.TAGS)
        np.savez_compressed(lp_path, **{k: v for k, v in lp.items() if v is not None})
        json.dump(pred, open(pred_path, "w"))
        return lp, pred

    # ---------------- report ----------------
    def report(self, n_examples=2):
        if self.state_a is None:
            self.stage_a()
        lp, pred = self.val_outputs()
        gold = self.data["golds"]["fold2"]
        sc = nu.strict_scores(gold, pred, self.LABELS)
        nu.print_scores(sc, f"{self.name} - fold 2 (validation):")
        decoding = {}
        for dec in ("argmax", "viterbi"):
            for trim in (False, True):
                decoding[f"{dec}_trim{int(trim)}"] = nu.strict_scores(
                    gold, nu.decode_docs(self.docs["fold2"], lp, self.TAGS, dec, trim), self.LABELS)["macro_f1"]
        print("decoding on the same outputs:", {k: round(v, 4) for k, v in decoding.items()})
        cats, confusion, examples = nu.error_breakdown(gold, pred, self.data["texts"]["fold2"], n_examples)
        print("errors:", dict(cats.most_common()))
        print("most confused (gold -> predicted):", confusion.most_common(6))
        for cat, ex in examples.items():
            for _, s, e, lab, ctx in ex:
                print(f"  [{cat}] {lab}: ...{ctx}...")
        self.val_scores = sc
        json.dump({"val": {k: v for k, v in sc.items() if k != "per_label"},
                   "per_label": {l: round(v["f1"], 4) for l, v in sc["per_label"].items()},
                   "decoding": decoding, "errors": dict(cats), "best_epoch": self.state_a["best_epoch"],
                   "history": self.state_a["history"]}, open(self.dir / "val_scores.json", "w"), indent=1)
        return sc

    # ---------------- Stage B ----------------
    def stage_b(self):
        """Fresh Phase-1 weights, train on fold 1 + fold 2 for the best Stage-A epoch count, predict test, zip."""
        sub = self.dir / "submission"
        if (sub / "submission.zip").exists():
            nu.check_submission(sub / "submission.zip", self.data["test"], self.LABELS)
            print("[cache] submission already written")
            return sub / "submission.zip"
        if self.state_a is None:
            self.stage_a()
        model = self.build_model()
        cfg_b = {**self.cfg, "epochs": self.state_a["best_epoch"]}
        nm.train_tagger(model, self.windows("all"), None, self.dir, cfg_b, self.device, tag="full",
                        eval_fn=None, amp_dtype=self.amp)
        pred = self.predict(model, self.docs["test"])
        zp = nu.write_submission(pred, self.data["test"], self.LABELS, sub)
        counts = {}
        for spans in pred.values():
            for s, e, l in spans:
                counts[l] = counts.get(l, 0) + 1
        print("test entities per label:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
        return zp

    # ---------------- efficiency ----------------
    def speed(self, which="best_fold1"):
        """Inference cost on the fold-2 windows: tokens per second, peak GPU memory, parameters (cached in speed.json)."""
        path = self.dir / "speed.json"
        if path.exists():
            sp = json.load(open(path)); print("[cache] speed:", sp); return sp
        if self.model is None:
            self.stage_a()
        c = self.cfg
        wins = [nu.window_input(d.ids, w) for d in self.docs["fold2"]
                for w in nu.plan_windows(len(d.ids), c["max_len"], c["stride"], c["header"])]
        scorer = self.scorer(self.model)
        scorer(wins[:8])                                     # warm-up
        if self.device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        scorer(wins)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        total, trainable = nm.count_params(self.model)
        sp = {"tokens_per_sec": round(sum(len(w) for w in wins) / sec), "infer_sec_fold2": round(sec, 2),
              "peak_gb_infer": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None,
              "params_total": total, "params_trainable": trainable}
        json.dump(sp, open(path, "w"), indent=1)
        print("speed:", sp)
        return sp

    # ---------------- log ----------------
    def log(self, submitted=False):
        if self.name in {r.get("run") for r in nu.read_experiments(self.root)}:
            print("[cache] already logged")
            return
        sc = json.load(open(self.dir / "val_scores.json"))
        model = self.model or self.build_model()
        total, trainable = nm.count_params(model)
        rec = {"run": self.name, **{k: self.cfg[k] for k in ("phase1_run", "method", "causal", "head", "lookahead",
                                                             "normalize", "header", "augment", "max_len", "kd_teacher")},
               "params_total": total, "params_trainable": trainable,
               "rank": self.cfg["rank"] if self.cfg["method"] in ("lora", "lorma") else None,
               "peak_gpu_gb_last_session": getattr(self, "peak_gb", None), "best_epoch": sc["best_epoch"],
               "val_macro_f1": round(sc["val"]["macro_f1"], 4), "val_micro_f1": round(sc["val"]["micro_f1"], 4),
               "val_per_label": sc["per_label"],
               "train_sec_fold1": round(sum(h["train_sec"] for h in sc["history"]), 1),
               "submission": (self.dir / "submission" / "submission.zip").exists(),
               "time": time.strftime("%Y-%m-%d %H:%M")}
        nu.log_experiment(self.root, rec)
        print("logged:", rec["run"], "| val macro-F1", rec["val_macro_f1"])


def compare(root, names=None):
    """Table of fold-2 results of finished runs (from their val_scores.json)."""
    root = Path(root)
    rows = []
    for d in sorted((root / "checkpoints" / "phase2").glob("*")):
        f = d / "val_scores.json"
        if f.exists() and (names is None or d.name in names):
            s = json.load(open(f))
            rows.append((d.name, s["val"]["macro_f1"], s["val"]["micro_f1"], s["best_epoch"], s["per_label"]))
    rows.sort(key=lambda r: -r[1])
    labels = list(rows[0][4]) if rows else []
    print(f"{'run':42s} {'macro':>6s} {'micro':>6s} {'ep':>3s}  " + " ".join(f"{l[:6]:>6s}" for l in labels))
    for name, ma, mi, ep, per in rows:
        print(f"{name:42s} {ma:6.4f} {mi:6.4f} {ep:3d}  " + " ".join(f"{per[l]:6.3f}" for l in labels))
    return rows
