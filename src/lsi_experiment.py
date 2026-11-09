"""
lsi_experiment.py  -  one Phase-5 experiment = one config dict -> train, dev report, test submission, log
(same structure as Phase 4's cjpe_experiment.py)

    exp = Experiment(ROOT, {"phase1": "A2", "method": "full"})
    exp.memory_check()      # one forward + backward on a full micro-batch: does it fit in 6 GB? (cached)
    exp.train()             # train on train, check dev after every epoch, keep the best epoch (cached)
    exp.report()            # dev: thresholds (global vs per statute), rare statutes, confusions (cached)
    exp.predict_test()      # test statute sets with the chosen thresholds -> submission.zip (cached)
    exp.teacher_outputs()   # logits on the TRAIN cases, for knowledge distillation (cached)
    exp.speed()             # inference speed, GPU memory, parameters (cached)
    exp.log()               # one line in experiments/phase5/experiments.jsonl (once)

No "Stage B" (retraining on train + dev): dev (10,181 cases) is needed to tune 100 thresholds, and a second
training run would cost another hour per experiment.

Run folder: checkpoints/phase5/<run name>; the name is built from the config, so every setting has its own folder.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import lsi_utils as lu
import lsi_models as lm

PHASE1_RUNS = {
    "A2": "A2_copy1_mtp0_d640_L10_sc70tr30_200M",     # decoder + copy (best Phase-1 model)
    "A": "A_decoder_d640_L10_sc70tr30_200M",          # plain decoder
    "B": "B_prefixlm_d640_L10_sc70tr30_200M",         # prefix-LM
    "D": "D_encdec_copy1_e4d5_640_sc70tr30_200M",     # encoder-decoder (we use its 4-layer encoder)
}

DEFAULTS = dict(
    phase1="A2", method="full",
    causal=None,                  # None = natural reading direction: causal for A/A2/B, bidirectional for D's encoder
    input="first_last", max_len=None,   # first half + last half of the budget; max_len None -> 1536 (D: 1024)
    pooling="attn", dropout=0.1,
    pos_weight_power=0.5,         # w_l = (neg_l / pos_l) ** power; 0 = plain BCE
    epochs=3, patience=2, per_step=16, token_budget=3072,   # 16 cases per optimiser step; <= 3072 tokens per micro-batch
    lr=3e-5, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.05, grad_clip=1.0, seed=42, ckpt_every=250,
    rank=8, alpha=16, adapter_targets=("qkv", "proj", "w1", "w2", "w3"),
    adapter_dim=64,               # bottleneck size of Houlsby adapters (method 'adapter')
    top_blocks=None,              # method 'partial': number of top blocks that are trained (default 4)
    kd_teacher=None, kd_alpha=0.5, kd_T=2.0,
    train_limit=None,             # e.g. 200 -> only the first 200 training cases (quick tests)
)
PEFT_METHODS = ("lora", "lorma", "rslora", "ia3", "adapter", "bitfit")
PEFT_DEFAULTS = dict(lr=5e-4, epochs=4)
PEFT_LR = {"ia3": 3e-3, "bitfit": 1e-3}

_DATA = {}   # tokenised data shared by all experiments in one notebook


def run_name(c):
    method = c["method"] + (f"r{c['rank']}" if c["method"] in ("lora", "lorma", "rslora") else "")
    if c["method"] == "adapter":
        method = f"adapter{c['adapter_dim']}"
    if c["method"] == "partial":
        method = f"partial{c['top_blocks']}"
    name = f"{c['phase1']}_{method}_{'causal' if c['causal'] else 'bidir'}_{c['input']}{c['max_len']}_{c['pooling']}"
    if c["pos_weight_power"] != 0.5:
        name += f"_pw{c['pos_weight_power']:g}"
    if c["kd_teacher"]:
        name += f"_kd{c['kd_T']:g}"
    if c["train_limit"]:
        name += f"_n{c['train_limit']}"
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
        if c["method"] == "partial" and not c["top_blocks"]:
            c["top_blocks"] = 4
        if c["max_len"] is None:
            c["max_len"] = 1024 if c["phase1"] == "D" else 1536   # D's encoder only ever read <= 1024 tokens
        c["phase1_run"] = PHASE1_RUNS[c["phase1"]]
        self.cfg = c
        self.name = run_name(c)
        self.dir = self.root / "checkpoints" / "phase5" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        json.dump(c, open(self.dir / "config.json", "w"), indent=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self._load_data()
        self.model = None
        print(f"experiment: {self.name} | device {self.device} | precision {self.amp}")

    # ---------------- data ----------------
    def _load_data(self):
        key = str(self.root)
        if key not in _DATA:
            data, info = lu.load_lsi(self.root)
            tok = Tokenizer.from_file(str(self.root / "data" / "tokenizer" / "bpe32k.json"))
            ids = {s: lu.encode_split(self.root, s, data[s], tok) for s in data}
            _DATA[key] = (data, info, ids)
            for s in data:
                n = np.array([len(x) for x in ids[s]])
                print(f"{s:5s}: {info[s]} | tokens median {int(np.median(n)):,}, max {n.max():,}")
        self.data, self.info, self.ids = _DATA[key]
        c = self.cfg
        lim = c["train_limit"]
        self.X = {s: [lu.select_tokens(x, c["input"], c["max_len"]) for x in self.ids[s][:lim if s == "train" else None]]
                  for s in self.ids}
        self.Y = {s: lu.label_matrix(self.data[s][:lim if s == "train" else None]) for s in ("train", "dev")}
        self.train_counts = self.Y["train"].sum(0)

    # ---------------- model ----------------
    def build_model(self):
        c = self.cfg
        backbone, _, info = lm.load_phase1_backbone(self.root / "checkpoints" / "phase1" / c["phase1_run"])
        self.load_info = info
        if c["method"] == "adapter":
            lm.add_bottleneck_adapters(backbone, c["adapter_dim"])
        elif c["method"] == "bitfit":
            lm.add_bitfit(backbone)
        elif c["method"] in PEFT_METHODS:
            lm.add_peft(backbone, c["method"], c["rank"], c["alpha"], c["adapter_targets"])
        elif c["method"] not in ("full", "partial"):      # partial = top blocks unfrozen inside train_ml
            raise ValueError(f"method {c['method']} not implemented")
        prior = self.train_counts / len(self.Y["train"])
        return lm.build_lsi_model(backbone, lu.N_LABELS, prior, c["causal"], c["pooling"], c["dropout"]).to(self.device)

    def memory_check(self):
        path = self.dir / "memory_check.json"
        if path.exists():
            res = json.load(open(path))
            print("[cache] memory check:", res)
            return res
        model = self.build_model()
        if self.cfg["method"] == "partial":
            lm.set_unfrozen(model, self.cfg["top_blocks"])
        total, trainable = lm.count_params(model)
        model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        L = self.cfg["max_len"]
        mb = max(1, self.cfg["token_budget"] // L)            # worst case: a micro-batch of full-length cases
        x = torch.randint(2, 32000, (mb, L), device=self.device)
        m = torch.ones(mb, L, dtype=torch.bool, device=self.device)
        t0 = time.time()
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp != torch.float32):
            logits = model(x, m).float()
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        res = {"params_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 2),
               "batch": f"{mb} x {L} tokens", "sec": round(time.time() - t0, 2),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None,
               "phase1_load": self.load_info}
        print("memory check:", res)
        json.dump(res, open(path, "w"), indent=1)
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return res

    # ---------------- training ----------------
    def _dev_eval(self, model):
        def fn():
            prob = lm.sigmoid(lm.predict_logits(model, self.X["dev"], self.device, self.amp, desc="dev"))
            sc = lu.ml_scores(self.Y["dev"], lu.decide(prob, 0.5))
            thr, f1 = lu.tune_global(self.Y["dev"], prob)
            tuned = lu.ml_scores(self.Y["dev"], lu.decide(prob, thr))
            return {"macro_f1": sc["macro_f1"], "tuned_f1": f1, "threshold": thr, "micro_f1": tuned["micro_f1"],
                    "labels_per_case": tuned["labels_per_case"]}
        return fn

    def train(self):
        c = self.cfg
        model = self.build_model()
        total, trainable = lm.count_params(model)
        print(f"parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable before freezing rules "
              f"({100 * trainable / total:.1f}%)")
        teacher = None
        if c["kd_teacher"]:
            tdir = self.root / "checkpoints" / "phase5" / c["kd_teacher"]
            teacher = np.load(tdir / "train_logits.npy")
            t_ids = json.load(open(tdir / "train_ids.json"))
            assert t_ids[:len(self.X["train"])] == [d["id"] for d in self.data["train"]][:len(self.X["train"])], \
                "teacher logits belong to different training cases"
            teacher = teacher[:len(self.X["train"])]
            print(f"knowledge distillation from {c['kd_teacher']} (alpha {c['kd_alpha']}, T {c['kd_T']})")
        state = lm.train_ml(model, self.X["train"], self.Y["train"], self.dir, c, self.device, "train",
                            self._dev_eval(model), self.amp, teacher)
        self.model = model
        return state

    def _model(self):
        if self.model is None:
            self.train()        # loads best_train.pt if training is already done
        return self.model

    def _logits(self, split):
        path = self.dir / f"{split}_logits.npy"
        if path.exists():
            return np.load(path)
        lg = lm.predict_logits(self._model(), self.X[split], self.device, self.amp, desc=f"predicting {split}")
        np.save(path, lg.astype(np.float16) if split == "train" else lg)
        if split == "train":
            json.dump([d["id"] for d in self.data["train"]][:len(lg)], open(self.dir / "train_ids.json", "w"))
        return lg

    # ---------------- analysis ----------------
    def report(self):
        """Dev analysis:
        * global threshold vs one threshold per statute, scored honestly by 2-fold cross-fitting on dev
          (tune on one half, score on the other); the better strategy is used for test
        * final thresholds tuned on the whole dev set
        * F1 by statute frequency (rare statutes), worst statutes, most common wrong additions"""
        path = self.dir / "dev_report.json"
        if path.exists():
            rep = json.load(open(path))
            print("[cache] dev report loaded")
        else:
            Y, prob = self.Y["dev"], lm.sigmoid(self._logits("dev").astype(np.float32))
            at05 = lu.ml_scores(Y, lu.decide(prob, 0.5))
            t_g, _ = lu.tune_global(Y, prob)
            t_l = lu.tune_per_label(Y, prob, min_pos=5, fallback=t_g)
            cf = lu.cross_fitted_scores(Y, prob)
            strategy = "per_label" if cf["per_label"]["macro_f1"] > cf["global"]["macro_f1"] else "global"
            P = lu.decide(prob, t_l if strategy == "per_label" else t_g)
            f1 = lu.per_label_f1(Y, P)
            worst = np.argsort(f1)[:10]
            rep = {"at_0.5": at05,
                   "global_threshold": t_g, "global_tuned_on_dev": lu.ml_scores(Y, lu.decide(prob, t_g)),
                   "per_label_tuned_on_dev": lu.ml_scores(Y, lu.decide(prob, t_l)),
                   "cross_fitted": cf, "strategy": strategy, "per_label_thresholds": [float(t) for t in t_l],
                   "chosen": lu.ml_scores(Y, P),
                   "f1_by_frequency": lu.f1_by_frequency(Y, P, self.train_counts),
                   "per_label_f1": [round(float(x), 4) for x in f1],
                   "worst_statutes": [{"id": int(l), "f1": round(float(f1[l]), 4), "train_cases": int(self.train_counts[l]),
                                       "dev_cases": int(Y[:, l].sum())} for l in worst],
                   "top_confusions": lu.top_confusions(Y, P)}
            json.dump(rep, open(path, "w"), indent=1)
        print(f"dev @0.5                     : {rep['at_0.5']}")
        print(f"dev global threshold {rep['global_threshold']:.2f}   : {rep['global_tuned_on_dev']}")
        print(f"dev per-statute thresholds   : {rep['per_label_tuned_on_dev']}  (tuned and scored on the same cases)")
        print(f"cross-fitted (honest) macro-F1: global {rep['cross_fitted']['global']['macro_f1']} | "
              f"per statute {rep['cross_fitted']['per_label']['macro_f1']} -> using {rep['strategy']}")
        print("F1 by statute frequency:", rep["f1_by_frequency"])
        print("worst statutes:", [(w["id"], w["f1"]) for w in rep["worst_statutes"]])
        print("most common wrong additions (gold A -> wrongly added B):",
              [(c["gold"], c["wrongly_added"], c["cases"]) for c in rep["top_confusions"][:5]])
        return rep

    def predict_test(self):
        zip_path = self.dir / "submission.zip"
        test_ids = [d["id"] for d in self.data["test"]]
        if zip_path.exists():
            print("[cache] submission exists")
        else:
            rep = self.report()
            thr = rep["per_label_thresholds"] if rep["strategy"] == "per_label" else rep["global_threshold"]
            prob = lm.sigmoid(self._logits("test").astype(np.float32))
            zip_path = lu.write_submission(test_ids, lu.decide(prob, np.asarray(thr)), self.dir)
            print(f"test predictions with {rep['strategy']} thresholds tuned on dev")
        lu.check_submission(zip_path, test_ids)
        return zip_path

    def teacher_outputs(self):
        """Logits of this model on every TRAINING case (train_logits.npy, float16) -> soft labels for KD."""
        lg = self._logits("train").astype(np.float32)
        print(f"train logits: {lg.shape}, mean predicted statutes/case at 0.5: {(lg > 0).sum(1).mean():.2f}")
        return lg

    def speed(self):
        path = self.dir / "speed.json"
        if path.exists():
            res = json.load(open(path))
            print("[cache] speed:", res)
            return res
        model = self._model()
        X = self.X["dev"][:500]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        lm.predict_logits(model, X, self.device, self.amp, desc="speed")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        total, trainable = lm.count_params(model)
        res = {"cases_per_s": round(len(X) / sec, 1), "tokens_per_s": round(sum(len(x) for x in X) / sec),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None,
               "params_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 2)}
        print("speed:", res)
        json.dump(res, open(path, "w"), indent=1)
        return res

    def log(self):
        if any(r.get("run") == self.name for r in lu.read_experiments(self.root)):
            print("[cache] already logged")
            return
        hist = json.load(open(self.dir / "history_train.json"))
        rep = json.load(open(self.dir / "dev_report.json"))
        mem = json.load(open(self.dir / "memory_check.json")) if (self.dir / "memory_check.json").exists() else {}
        rec = {"run": self.name, "phase1": self.cfg["phase1"], "method": self.cfg["method"],
               "input": f"{self.cfg['input']}{self.cfg['max_len']}", "best_epoch": hist["best_epoch"],
               "epochs_run": len(hist["history"]), "train_min_per_epoch": hist["history"][0]["train_min"],
               "dev_macro_f1_at_0.5": rep["at_0.5"]["macro_f1"],
               "dev_macro_f1_global": rep["cross_fitted"]["global"]["macro_f1"],
               "dev_macro_f1_per_label": rep["cross_fitted"]["per_label"]["macro_f1"],
               "strategy": rep["strategy"], "dev_micro_f1": rep["chosen"]["micro_f1"],
               "dev_subset_acc": rep["chosen"]["subset_acc"], "trainable_M_train": mem.get("trainable_M"),
               **{k: v for k, v in self.speed().items()}, "date": time.strftime("%Y-%m-%d %H:%M")}
        lu.log_experiment(self.root, rec)
        print("logged:", rec)


def compare(root):
    import pandas as pd
    rows = lu.read_experiments(root)
    if not rows:
        print("no experiments logged yet")
        return None
    df = pd.DataFrame(rows).drop_duplicates("run", keep="last").set_index("run")
    cols = [c for c in ["phase1", "method", "input", "trainable_M_train", "best_epoch", "train_min_per_epoch",
                        "dev_macro_f1_at_0.5", "dev_macro_f1_global", "dev_macro_f1_per_label", "strategy",
                        "dev_micro_f1", "dev_subset_acc", "cases_per_s", "peak_gb"] if c in df.columns]
    return df[cols].sort_values("dev_macro_f1_per_label", ascending=False)
