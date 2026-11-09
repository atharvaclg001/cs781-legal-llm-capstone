"""
cjpe_experiment.py  -  one Phase-4 experiment = one config dict -> train, dev report, test submission, log

    exp = Experiment(ROOT, {"phase1": "A2", "method": "full"})
    exp.memory_check()      # one forward + backward on a full batch: does it fit in 6 GB? (cached)
    exp.train()             # train on train (31,652 cases), check dev after every epoch, keep the best (cached)
    exp.report()            # dev scores (threshold 0.5 and tuned), confusion, errors by length (cached)
    exp.predict_test()      # test predictions with the dev-tuned threshold -> submission.zip (cached)
    exp.teacher_outputs()   # logits on the TRAIN cases, for knowledge distillation later (cached)
    exp.speed()             # inference speed, GPU memory, parameters (cached)
    exp.log()               # one line in experiments/phase4/experiments.jsonl (once)

Why no "Stage B" (retraining on train + dev) as in Phases 2-3?  Here train has 31,652 cases and dev only 994:
adding dev would change the training data by 3% but cost a whole second training run (hours). So the model
trained on train, selected on dev, directly predicts test.

Run folder: checkpoints/phase4/<run name>; the name is built from the config, so every setting has its own
folder and nothing is overwritten.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import cjpe_utils as cu
import cjpe_models as cm

PHASE1_RUNS = {
    "A2": "A2_copy1_mtp0_d640_L10_sc70tr30_200M",     # decoder + copy (best Phase-1 model)
    "A": "A_decoder_d640_L10_sc70tr30_200M",          # plain decoder
    "B": "B_prefixlm_d640_L10_sc70tr30_200M",         # prefix-LM
    "D": "D_encdec_copy1_e4d5_640_sc70tr30_200M",     # encoder-decoder (we use its 4-layer encoder)
}

DEFAULTS = dict(
    phase1="A2", method="full",
    causal=None,                 # None = natural reading direction: causal for A/A2/B, bidirectional for D's encoder
    input="last", max_len=None,  # which part of the case: 'last' / 'first' / 'first_last'; max_len None -> 1536 (D: 1024)
    pooling="attn", dropout=0.1,
    class_weights="balanced",    # 'balanced': weight_c = N / (2 N_c) (train is 41% label 1); None = no weighting
    epochs=3, patience=2, micro_batch=2, grad_accum=8,      # 16 cases per optimiser step
    lr=3e-5, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.05, grad_clip=1.0, seed=42, ckpt_every=250,
    rank=8, alpha=16, adapter_targets=("qkv", "proj", "w1", "w2", "w3"),
    adapter_dim=64,              # bottleneck size of Houlsby adapters (method 'adapter')
    unfreeze_schedule=None,      # method 'gradual': trainable top blocks per epoch, default [3, 6, 'all']
    kd_teacher=None, kd_alpha=0.5, kd_T=2.0,
    train_limit=None,            # e.g. 200 -> use only the first 200 training cases (quick tests)
)
PEFT_METHODS = ("lora", "lorma", "rslora", "ia3", "adapter", "bitfit")
PEFT_DEFAULTS = dict(lr=5e-4)
PEFT_LR = {"ia3": 3e-3, "bitfit": 1e-3}   # IA3 / BitFit train only vectors -> higher lr (BitFit paper: 1e-4 .. 1e-3)

_DATA = {}   # tokenised data shared by all experiments in one notebook


def run_name(c):
    method = c["method"] + (f"r{c['rank']}" if c["method"] in ("lora", "lorma", "rslora") else "")
    if c["method"] == "adapter":
        method = f"adapter{c['adapter_dim']}"
    name = f"{c['phase1']}_{method}_{'causal' if c['causal'] else 'bidir'}_{c['input']}{c['max_len']}_{c['pooling']}"
    if c["class_weights"] != "balanced":
        name += "_nocw"
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
        if c["method"] == "gradual" and not c["unfreeze_schedule"]:
            c["unfreeze_schedule"] = [3, 6, "all"]
        if c["max_len"] is None:
            c["max_len"] = 1024 if c["phase1"] == "D" else 1536   # D's encoder only ever read <= 1024 tokens
        c["phase1_run"] = PHASE1_RUNS[c["phase1"]]
        self.cfg = c
        self.name = run_name(c)
        self.dir = self.root / "checkpoints" / "phase4" / self.name
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
            data, info = cu.load_cjpe(self.root)
            tok = Tokenizer.from_file(str(self.root / "data" / "tokenizer" / "bpe32k.json"))
            ids = {s: cu.encode_split(self.root, s, data[s], tok) for s in data}
            _DATA[key] = (data, info, ids)
            for s in data:
                n = np.array([len(x) for x in ids[s]])
                print(f"{s:5s}: {info[s]['records']:,} records -> {info[s]['kept']:,} unique cases | "
                      f"label-1 rate {info[s]['label_1_rate']} | tokens median {int(np.median(n)):,}, max {n.max():,}")
        self.data, self.info, self.ids = _DATA[key]
        c = self.cfg
        lim = c["train_limit"]
        self.X = {s: [cu.select_tokens(x, c["input"], c["max_len"]) for x in self.ids[s][:lim if s == "train" else None]]
                  for s in self.ids}
        self.y = {s: np.array([d["label"] for d in self.data[s][:lim if s == "train" else None]]) for s in ("train", "dev")}

    def class_weights(self):
        if self.cfg["class_weights"] != "balanced":
            return None
        y = self.y["train"]
        n1 = y.sum()
        return [round(float(len(y) / (2 * (len(y) - n1))), 4), round(float(len(y) / (2 * n1)), 4)]

    # ---------------- model ----------------
    def build_model(self):
        c = self.cfg
        backbone, _, info = cm.load_phase1_backbone(self.root / "checkpoints" / "phase1" / c["phase1_run"])
        self.load_info = info
        if c["method"] == "adapter":
            cm.add_bottleneck_adapters(backbone, c["adapter_dim"])
        elif c["method"] == "bitfit":
            cm.add_bitfit(backbone)
        elif c["method"] in PEFT_METHODS:
            cm.add_peft(backbone, c["method"], c["rank"], c["alpha"], c["adapter_targets"])
        elif c["method"] not in ("full", "gradual"):    # gradual = full model, unfrozen step by step in train_cls
            raise ValueError(f"method {c['method']} not implemented yet")
        return cm.CJPEModel(backbone, c["causal"], c["pooling"], c["dropout"]).to(self.device)

    def memory_check(self):
        path = self.dir / "memory_check.json"
        if path.exists():
            res = json.load(open(path))
            print("[cache] memory check:", res)
            return res
        model = self.build_model()
        total, trainable = cm.count_params(model)
        model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mb, L = self.cfg["micro_batch"], self.cfg["max_len"]
        x = torch.randint(2, 32000, (mb, L), device=self.device)
        m = torch.ones(mb, L, dtype=torch.bool, device=self.device)
        t0 = time.time()
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp != torch.float32):
            logits = model(x, m).float()
        torch.nn.functional.cross_entropy(logits, torch.zeros(mb, dtype=torch.long, device=self.device)).backward()
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
            p = cm.probs_1(cm.predict_logits(model, self.X["dev"], self.device, self.amp, desc="dev"))
            sc = cu.f1_scores(self.y["dev"], (p >= 0.5).astype(int))
            sc["tuned_f1"] = cu.best_threshold(self.y["dev"], p)[1]
            return sc
        return fn

    def train(self):
        c = self.cfg
        model = self.build_model()
        total, trainable = cm.count_params(model)
        print(f"parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable ({100 * trainable / total:.1f}%)")
        cfg = {**c, "class_weights": self.class_weights()}
        print("class weights:", cfg["class_weights"])
        teacher = None
        if c["kd_teacher"]:
            tpath = self.root / "checkpoints" / "phase4" / c["kd_teacher"] / "train_logits.npy"
            teacher = np.load(tpath)[:len(self.X["train"])]
            print(f"knowledge distillation from {c['kd_teacher']} (alpha {c['kd_alpha']}, T {c['kd_T']})")
        state = cm.train_cls(model, self.X["train"], self.y["train"], self.dir, cfg, self.device, "train",
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
        lg = cm.predict_logits(self._model(), self.X[split], self.device, self.amp, desc=f"predicting {split}")
        np.save(path, lg)
        return lg

    # ---------------- analysis ----------------
    def report(self, n_examples=5):
        path = self.dir / "dev_report.json"
        if path.exists():
            rep = json.load(open(path))
            print("[cache] dev report loaded")
        else:
            y, p = self.y["dev"], cm.probs_1(self._logits("dev"))
            thr, _ = cu.best_threshold(y, p)
            pred = (p >= thr).astype(int)
            n_tok = np.array([len(x) for x in self.ids["dev"]])
            bins = np.quantile(n_tok, [0, 0.25, 0.5, 0.75, 1.0])
            by_len = []
            for a, b in zip(bins[:-1], bins[1:]):
                sel = (n_tok >= a) & (n_tok <= b)
                by_len.append({"tokens": f"{int(a)}-{int(b)}", "cases": int(sel.sum()),
                               **cu.f1_scores(y[sel], pred[sel])})
            ids = [d["id"] for d in self.data["dev"]]
            conf_wrong = lambda mask: [ids[i] for i in np.argsort(-np.abs(p - 0.5)) if mask[i]][:n_examples]
            rep = {"at_0.5": cu.f1_scores(y, (p >= 0.5).astype(int)), "threshold": thr, "tuned": cu.f1_scores(y, pred),
                   "confusion_tuned": cu.confusion(y, pred), "by_length_tuned": by_len,
                   "mean_p1_gold0": round(float(p[y == 0].mean()), 3), "mean_p1_gold1": round(float(p[y == 1].mean()), 3),
                   "most_confident_FP": conf_wrong((pred == 1) & (y == 0)),
                   "most_confident_FN": conf_wrong((pred == 0) & (y == 1))}
            json.dump(rep, open(path, "w"), indent=1)
        print(f"dev @0.5      : {rep['at_0.5']}")
        print(f"dev @{rep['threshold']:.2f} (tuned): {rep['tuned']}")
        print(f"confusion (tuned): {rep['confusion_tuned']} | mean P(1): gold 0 -> {rep['mean_p1_gold0']}, gold 1 -> {rep['mean_p1_gold1']}")
        for r in rep["by_length_tuned"]:
            print(f"   length {r['tokens']:>13s} tokens: {r['cases']:4d} cases, macro-F1 {r['macro_f1']:.4f}, acc {r['acc']:.4f}")
        print("most confident false positives:", rep["most_confident_FP"])
        print("most confident false negatives:", rep["most_confident_FN"])
        return rep

    def predict_test(self):
        zip_path = self.dir / "submission.zip"
        test_ids = [d["id"] for d in self.data["test"]]
        if zip_path.exists():
            print("[cache] submission exists")
        else:
            thr = self.report()["threshold"]
            p = cm.probs_1(self._logits("test"))
            zip_path = cu.write_submission(test_ids, (p >= thr).astype(int), self.dir)
            print(f"test predictions with dev-tuned threshold {thr}")
        cu.check_submission(zip_path, test_ids)
        return zip_path

    def teacher_outputs(self):
        """Logits of this model on every TRAINING case (saved as train_logits.npy) -> soft labels for KD."""
        lg = self._logits("train")
        print(f"train logits: {lg.shape}, mean P(1) {cm.probs_1(lg).mean():.3f}")
        return lg

    def speed(self):
        path = self.dir / "speed.json"
        if path.exists():
            res = json.load(open(path))
            print("[cache] speed:", res)
            return res
        model = self._model()
        X = self.X["dev"][:200]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        cm.predict_logits(model, X, self.device, self.amp, desc="speed")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        total, trainable = cm.count_params(model)
        res = {"cases_per_s": round(len(X) / sec, 1), "tokens_per_s": round(sum(len(x) for x in X) / sec),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None,
               "params_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 2)}
        print("speed:", res)
        json.dump(res, open(path, "w"), indent=1)
        return res

    def log(self):
        if any(r.get("run") == self.name for r in cu.read_experiments(self.root)):
            print("[cache] already logged")
            return
        hist = json.load(open(self.dir / "history_train.json"))
        rep = json.load(open(self.dir / "dev_report.json"))
        rec = {"run": self.name, "phase1": self.cfg["phase1"], "method": self.cfg["method"],
               "input": f"{self.cfg['input']}{self.cfg['max_len']}", "pooling": self.cfg["pooling"],
               "best_epoch": hist["best_epoch"], "epochs_run": len(hist["history"]),
               "dev_f1_at_0.5": rep["at_0.5"]["macro_f1"], "dev_f1_tuned": rep["tuned"]["macro_f1"],
               "threshold": rep["threshold"], **{k: v for k, v in self.speed().items()},
               "date": time.strftime("%Y-%m-%d %H:%M")}
        cu.log_experiment(self.root, rec)
        print("logged:", rec)


def compare(root):
    import pandas as pd
    rows = cu.read_experiments(root)
    if not rows:
        print("no experiments logged yet")
        return None
    df = pd.DataFrame(rows).set_index("run")
    cols = [c for c in ["phase1", "method", "input", "pooling", "trainable_M", "best_epoch", "dev_f1_at_0.5",
                        "dev_f1_tuned", "threshold", "tokens_per_s", "peak_gb"] if c in df.columns]
    return df[cols].sort_values("dev_f1_tuned", ascending=False)
