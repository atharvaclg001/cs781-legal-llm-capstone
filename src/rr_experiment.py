"""
rr_experiment.py  -  one Phase-3 experiment = one config dict -> Stage A, report, Stage B, log

    exp = Experiment(ROOT, {"phase1": "A2", "method": "full"})
    exp.stage_a()   # train on CL+IT train, check dev after every epoch, keep the best epoch   (cached)
    exp.report()    # dev scores per domain / label, confusions, comparison with the baselines (cached)
    exp.stage_b()   # fresh Phase-1 weights, train on train+dev for the best epoch count,
                    # predict the test documents, write submission.zip                     (cached)
    exp.speed()     # inference speed, GPU memory, parameters                              (cached)
    exp.log()       # one line in experiments/phase3/experiments.jsonl                     (once)

Run folder: checkpoints/phase3/<run name>; the name is built from the config, so every setting gets its
own folder and nothing is overwritten. Every step first checks whether its result already exists.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import rr_utils as ru
import rr_models as rm

PHASE1_RUNS = {
    "A2": "A2_copy1_mtp0_d640_L10_sc70tr30_200M",     # decoder + copy (best Phase-1 model)
    "A": "A_decoder_d640_L10_sc70tr30_200M",          # plain decoder
    "B": "B_prefixlm_d640_L10_sc70tr30_200M",         # prefix-LM
    "D": "D_encdec_copy1_e4d5_640_sc70tr30_200M",     # encoder-decoder (we use its 4-layer encoder)
}

DEFAULTS = dict(
    phase1="A2", method="full",
    causal=None,                  # None = natural reading direction: causal for decoders (A, A2, B), bidirectional for D's encoder
    context="doc",                # "doc": windows + BiLSTM over the document; "sentence": every sentence alone, MLP head, no position
    max_len=1536, stride=1024, max_sent_tokens=256,
    head="bilstm", hidden=256, dropout=0.2, use_pos=True, use_domain=True,
    class_weight_power=0.5,       # 0.5 = square-root class weights; 0 = no weighting
    epochs=15, patience=4, docs_per_step=2, win_batch=2,
    lr=1e-4, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.1, grad_clip=1.0, seed=42,
    rank=8, alpha=16, adapter_targets=("qkv", "proj", "w1", "w2", "w3"),
    kd_teacher=None, kd_alpha=0.5, kd_T=2.0,
)
PEFT_DEFAULTS = dict(lr=5e-4, epochs=25, patience=5)   # adapters start at 0 -> higher lr, more epochs (as in Phase 2)
PEFT_METHODS = ("lora", "lorma", "rslora", "ia3")
PEFT_LR = {"ia3": 3e-3}   # IA^3 vectors start at 1 and are tiny: the IA^3 paper uses a 10x higher learning rate

_DATA = {}   # tokenised data shared by all experiments in one notebook


def run_name(c):
    method = c["method"] + (f"r{c['rank']}" if c["method"] in ("lora", "lorma", "rslora") else "")
    name = f"{c['phase1']}_{method}_{'causal' if c['causal'] else 'bidir'}_{c['context']}"
    if c["context"] == "doc" and c["max_len"] != 1536:
        name += f"_w{c['max_len']}"
    if c["context"] == "doc" and not c["use_pos"]:
        name += "_nopos"
    if c["context"] == "doc" and not c["use_domain"]:
        name += "_nodom"
    if c["class_weight_power"] != 0.5:
        name += f"_cw{c['class_weight_power']}"
    if c.get("name_suffix"):
        name += "_" + c["name_suffix"]
    return name


class Experiment:
    def __init__(self, root, overrides=None):
        self.root = Path(root)
        overrides = dict(overrides or {})
        base = {**DEFAULTS, **(PEFT_DEFAULTS if overrides.get("method") in PEFT_METHODS else {})}
        if overrides.get("method") in PEFT_LR:
            base["lr"] = PEFT_LR[overrides["method"]]
        c = {**base, **overrides}
        if c["causal"] is None:
            c["causal"] = c["phase1"] != "D"
        if c["context"] == "sentence":                  # no document context at all
            c.update(head="mlp", use_pos=False)
            c["win_batch"] = max(c["win_batch"], 32)    # windows are single sentences (~40 tokens): batch many
        c["phase1_run"] = PHASE1_RUNS[c["phase1"]]
        self.cfg = c
        self.name = run_name(c)
        self.dir = self.root / "checkpoints" / "phase3" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        json.dump(c, open(self.dir / "config.json", "w"), indent=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self._load_data()
        self.model, self.state_a = None, None
        print(f"experiment: {self.name} | device {self.device} | precision {self.amp}")

    # ---------------- data ----------------
    def _load_data(self):
        c = self.cfg
        key = (str(self.root), c["max_sent_tokens"])
        if key not in _DATA:
            labels, data = ru.load_rr_data(self.root)
            tok = Tokenizer.from_file(str(self.root / "data" / "tokenizer" / "bpe32k.json"))
            enc = {s: ru.encode_rr_docs(data[s], tok, labels, c["max_sent_tokens"], desc=f"tokenising {s}") for s in data}
            _DATA[key] = (labels, data, enc)
        self.LABELS, self.data, self.enc = _DATA[key]
        mode = "sentence" if c["context"] == "sentence" else "context"
        self.plans = {}
        for s in self.enc:
            for d in self.enc[s]:
                wins = ru.plan_sentence_windows(d.bounds, c["max_len"], c["stride"], mode)
                self.plans[d.id] = (wins, ru.window_owners(wins, d.n_sent))

    def class_weights(self, docs):
        if self.cfg["class_weight_power"] == 0:
            return None
        w, _ = ru.class_weights(docs, len(self.LABELS), power=self.cfg["class_weight_power"])
        return w.tolist()

    # ---------------- model ----------------
    def build_model(self):
        c = self.cfg
        backbone, _, info = rm.load_phase1_backbone(self.root / "checkpoints" / "phase1" / c["phase1_run"])
        model = rm.RRModel(backbone, len(self.LABELS), c["causal"], c["head"], c["hidden"], c["dropout"],
                           c["use_pos"], c["use_domain"])
        if c["method"] in PEFT_METHODS:
            info["adapter_params"] = rm.add_peft(model.backbone, c["method"], c["rank"], c["alpha"], c["adapter_targets"])
        elif c["method"] not in ("full", "kd"):        # kd = full fine-tuning of the student + distillation loss
            raise ValueError(f"unknown method {c['method']}")
        self.load_info = info
        return model.to(self.device)

    def predict(self, model, docs, desc="predicting"):
        return rm.predict_logprobs(model, docs, self.plans, self.device, self.amp, self.cfg["win_batch"], desc)

    def evaluate(self, model):
        lp = self.predict(model, self.enc["dev"], "dev")
        model.train()
        sc = ru.score_by_domain(self.enc["dev"], rm.to_labels(lp, self.LABELS), self.LABELS)
        return {"macro_f1": sc["all"]["macro_f1"], "accuracy": sc["all"]["accuracy"],
                "CL": sc["CL"]["macro_f1"], "IT": sc["IT"]["macro_f1"]}

    def teacher_outputs(self, split_tag):
        """Teacher log-probs on the training documents (computed once, cached). Stage A uses the teacher trained
        on train only (it has never seen the dev labels); Stage B the teacher trained on train+dev."""
        path = self.dir / f"teacher_{split_tag}.npz"
        if path.exists():
            print(f"[cache] teacher outputs {path.name}")
            return dict(np.load(path))
        t_exp = Experiment(self.root, self.cfg["kd_teacher"])
        teacher = t_exp.build_model()
        best = t_exp.dir / f"best_{split_tag}.pt"
        assert best.exists(), f"train the teacher first: {best} is missing"
        teacher.load_state_dict(torch.load(best, map_location=self.device))
        docs = self.enc["train"] + (self.enc["dev"] if split_tag == "full" else [])
        lp = rm.predict_logprobs(teacher, docs, t_exp.plans, self.device, self.amp, t_exp.cfg["win_batch"], "teacher")
        np.savez_compressed(path, **lp)
        del teacher
        torch.cuda.empty_cache()
        return lp

    # ---------------- Stage A ----------------
    def stage_a(self):
        """Train on CL+IT train, validate on CL+IT dev after every epoch; keep the best epoch (cached)."""
        model = self.build_model()
        total, trainable = rm.count_params(model)
        print(f"parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.3f}M trainable | loaded {self.load_info}")
        cfg = {**self.cfg, "class_weights": self.class_weights(self.enc["train"])}
        teacher = self.teacher_outputs("train") if self.cfg["method"] == "kd" else None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        self.state_a = rm.train_rr(model, self.enc["train"], self.plans, self.dir, cfg, self.device, "train",
                                   eval_fn=lambda: self.evaluate(model), amp_dtype=self.amp, teacher=teacher)
        self.model = model
        self.peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None
        print(f"Stage A: best epoch {self.state_a['best_epoch']} | dev macro-F1 {self.state_a['best']:.4f} | "
              f"{time.time() - t0:.0f}s | peak GPU {self.peak_gb} GB")
        return self.state_a

    def dev_outputs(self):
        """Dev log-probabilities of the best Stage-A model (cached on disk)."""
        path = self.dir / "dev_logprobs.npz"
        if path.exists():
            return dict(np.load(path))
        if self.model is None:
            self.stage_a()
        lp = self.predict(self.model, self.enc["dev"], "dev")
        np.savez_compressed(path, **lp)
        return lp

    # ---------------- report ----------------
    def report(self, show_per_label=True):
        if self.state_a is None:
            self.stage_a()
        pred = rm.to_labels(self.dev_outputs(), self.LABELS)
        by_dom = ru.score_by_domain(self.enc["dev"], pred, self.LABELS)
        ru.print_rr_scores(by_dom, f"{self.name} - dev:", per_label=show_per_label)
        base_path = self.root / "experiments" / "phase3" / "p3_02_baselines.json"
        if base_path.exists():
            base = json.load(open(base_path))
            print("baselines (dev macro-F1): " + ", ".join(f"{k} {v['all']['macro_f1']:.4f}" for k, v in base.items()))
        conf = {}
        for dom in ("all",) + ru.DOMAINS:
            docs = [d for d in self.enc["dev"] if dom == "all" or d.domain == dom]
            conf[dom] = ru.top_confusions(ru.confusion_matrix(docs, pred, self.LABELS), self.LABELS, 8)
        print("most confused (gold -> predicted):", conf["all"])
        self.dev_scores = by_dom
        json.dump({"dev": {dom: {k: v for k, v in sc.items() if k != "per_label"} for dom, sc in by_dom.items()},
                   "per_label": {dom: {l: round(v["f1"], 4) for l, v in sc["per_label"].items()} for dom, sc in by_dom.items()},
                   "confusions": conf, "best_epoch": self.state_a["best_epoch"], "history": self.state_a["history"]},
                  open(self.dir / "val_scores.json", "w"), indent=1)
        return by_dom

    # ---------------- Stage B ----------------
    def stage_b(self):
        """Fresh Phase-1 weights, train on train+dev for the best Stage-A epoch count, predict test, zip."""
        sub = self.dir / "submission"
        if (sub / "submission.zip").exists():
            ru.check_submission(sub / "submission.zip", self.data["test"], self.LABELS)
            print("[cache] submission already written")
            return sub / "submission.zip"
        if self.state_a is None:
            self.stage_a()
        model = self.build_model()
        docs = self.enc["train"] + self.enc["dev"]
        cfg = {**self.cfg, "epochs": self.state_a["best_epoch"], "class_weights": self.class_weights(docs)}
        teacher = self.teacher_outputs("full") if self.cfg["method"] == "kd" else None
        rm.train_rr(model, docs, self.plans, self.dir, cfg, self.device, "full", eval_fn=None,
                    amp_dtype=self.amp, teacher=teacher)
        lp = self.predict(model, self.enc["test"], "test")
        np.savez_compressed(self.dir / "test_logprobs.npz", **lp)
        zp = ru.write_submission(rm.to_labels(lp, self.LABELS), self.data["test"], self.LABELS, sub)
        return zp

    # ---------------- efficiency ----------------
    def speed(self):
        """Inference cost on the dev documents: tokens per second, peak GPU memory, parameters (cached)."""
        path = self.dir / "speed.json"
        if path.exists():
            sp = json.load(open(path))
            print("[cache] speed:", sp)
            return sp
        if self.model is None:
            self.stage_a()
        docs = self.enc["dev"]
        self.predict(self.model, docs[:1], "warm-up")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        self.predict(self.model, docs, "timing")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        total, trainable = rm.count_params(self.model)
        n_tok = sum(len(ru.window_input(d, w)[0]) for d in docs for w in self.plans[d.id][0])
        sp = {"tokens_per_sec": round(n_tok / sec), "sentences_per_sec": round(sum(d.n_sent for d in docs) / sec),
              "infer_sec_dev": round(sec, 2),
              "peak_gb_infer": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None,
              "params_total": total, "params_trainable": trainable}
        json.dump(sp, open(path, "w"), indent=1)
        print("speed:", sp)
        return sp

    # ---------------- log ----------------
    def log(self):
        if self.name in {r.get("run") for r in ru.read_experiments(self.root)}:
            print("[cache] already logged")
            return
        sc = json.load(open(self.dir / "val_scores.json"))
        model = self.model or self.build_model()
        total, trainable = rm.count_params(model)
        keys = ("phase1", "phase1_run", "method", "causal", "context", "max_len", "head", "use_pos", "use_domain",
                "class_weight_power", "kd_teacher")
        rec = {"run": self.name, **{k: self.cfg[k] for k in keys},
               "rank": self.cfg["rank"] if self.cfg["method"] in ("lora", "lorma", "rslora") else None,
               "params_total": total, "params_trainable": trainable,
               "peak_gpu_gb_train": getattr(self, "peak_gb", None), "best_epoch": sc["best_epoch"],
               "dev_macro_f1": round(sc["dev"]["all"]["macro_f1"], 4),
               "dev_macro_f1_CL": round(sc["dev"]["CL"]["macro_f1"], 4), "dev_macro_f1_IT": round(sc["dev"]["IT"]["macro_f1"], 4),
               "dev_accuracy": round(sc["dev"]["all"]["accuracy"], 4), "dev_per_label": sc["per_label"]["all"],
               "train_sec_stage_a": round(sum(h["train_sec"] for h in sc["history"]), 1),
               "submission": (self.dir / "submission" / "submission.zip").exists(),
               "time": time.strftime("%Y-%m-%d %H:%M")}
        ru.log_experiment(self.root, rec)
        print("logged:", rec["run"], "| dev macro-F1", rec["dev_macro_f1"])


def compare(root, names=None):
    """Table of dev results of finished Phase-3 runs (from their val_scores.json)."""
    rows = []
    for d in sorted((Path(root) / "checkpoints" / "phase3").glob("*")):
        f = d / "val_scores.json"
        if f.exists() and (names is None or d.name in names):
            s = json.load(open(f))
            rows.append((d.name, s["dev"]["all"]["macro_f1"], s["dev"]["CL"]["macro_f1"], s["dev"]["IT"]["macro_f1"],
                         s["dev"]["all"]["accuracy"], s["best_epoch"], s["per_label"]["all"]))
    rows.sort(key=lambda r: -r[1])
    labels = list(rows[0][6]) if rows else []
    print(f"{'run':38s} {'macro':>6s} {'CL':>6s} {'IT':>6s} {'acc':>6s} {'ep':>3s}  " + " ".join(f"{l[:7]:>7s}" for l in labels))
    for name, ma, cl, it, acc, ep, per in rows:
        print(f"{name:38s} {ma:6.4f} {cl:6.4f} {it:6.4f} {acc:6.4f} {ep:3d}  " + " ".join(f"{per[l]:7.3f}" for l in labels))
    return rows
