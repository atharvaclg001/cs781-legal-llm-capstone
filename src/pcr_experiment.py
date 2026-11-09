"""
pcr_experiment.py  -  one Phase-7 experiment = one config dict -> train, dev report, test submission, log.

    prepare(root)          once for all experiments (cached in data/processed/phase7/):
                           BM25 first stage (best P7_01 setting) -> top 200 per query for train/dev/test,
                           token cache (candidate start, query citation contexts)
    Experiment(root, cfg)  .run() = memory_check -> train (checkpoints + resume) -> dev scores -> report
                           (fusion weight with BM25 tuned on dev) -> test scores -> submission.zip (checked) -> log
    ensemble(root, names)  average of several runs' scores + BM25 (dev-tuned weight) -> submission
    compare(root)          table of all logged runs

Every stage first looks for its result file and loads it instead of recomputing.

Final score of a candidate (per query, over its top-K candidates):
    s = w * z(neural score) + (1 - w) * z(BM25 score),   z = (x - mean) / std over the query's K candidates
    then candidates clearly newer than the query (P7_01 year filter) go to the bottom; top 7 are submitted.
    w = 1: neural model only; w = 0: BM25 only (= the P7_01 reference, dev 0.4247). w is tuned on dev.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm.auto import tqdm

import pcr_utils as pu
import pcr_models as pm

PHASE1_RUNS = {
    "A2": "A2_copy1_mtp0_d640_L10_sc70tr30_200M",     # decoder + copy (best Phase-1 model)
    "A": "A_decoder_d640_L10_sc70tr30_200M",          # plain decoder
    "B": "B_prefixlm_d640_L10_sc70tr30_200M",         # prefix-LM
    "D": "D_encdec_copy1_e4d5_640_sc70tr30_200M",     # encoder-decoder (we use its 4-layer encoder)
}
BM25_DEFAULT = {"c_words": 2048, "k1": 8.0, "b": 1.0, "mode": "cit", "year_margin": 0}   # P7_01 best (dev 0.4247)
PREP_VERSION = "v1"
STORE_DEPTH = 200          # BM25 candidates kept per query (train hard negatives come from the top 100)
C_STORE, Q_STORE = 1500, 512   # tokens kept per candidate start / query snippet in the cache
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)

DEFAULTS = dict(
    phase1="A2", method="full", causal=None,      # None: causal for A / A2 / B, bidirectional for D's encoder
    max_len=1024, q_max=320,                      # pair = candidate (max_len - query - 2 tokens) + query (<= q_max)
    depth=100,                                    # re-rank BM25's top 100
    epochs=2, patience=1, groups_per_step=4,      # 4 groups (queries' positives) per optimiser step
    max_pos=8, n_hard=6, n_rand=1,                # group = 1 positive + 6 BM25 hard negatives + 1 random negative
    lr=3e-5, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.05, grad_clip=1.0, seed=42, ckpt_every=250,
    rank=8, alpha=16, top_blocks=None, dropout=0.1,
    kd_teacher=None, kd_alpha=0.5, kd_T=2.0,
    train_limit=None, infer_rows=16, name_suffix=None,
)
METHOD_LR = {"lorma": 5e-4, "lora": 5e-4, "ia3": 3e-3, "partial": 5e-5}

_CACHE = {}


def find_root():
    return pu.find_project_root()


# ----------------------------------------------------------------------------------------
# 1. Preparation (shared by all experiments)
# ----------------------------------------------------------------------------------------
def _bm25_settings(root):
    p = Path(root) / "experiments" / "phase7" / "p7_01_data" / "bm25_dev.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))["_best"]["settings"]
    return dict(BM25_DEFAULT)


def _bm25_topk(data, split, st, depth):
    """BM25 (setting st) for every query of a split -> {qid: {"ids", "bm25" (raw score), "newer" (0/1)}}.
    Ranking = not-newer candidates first, then by BM25 score; the query itself is never included."""
    Q, C = data[split]["queries"], data[split]["pool"]
    ids = [d["id"] for d in C]
    bm = pu.BM25((pu.bm25_tokens(pu.first_words(d, st.get("c_words"))) for d in C), k1=st["k1"], b=st["b"],
                 desc=f"BM25 index {split}")
    cy = np.array([pu.doc_year(pu.doc_text(d)) or 0 for d in C])
    out = {}
    for q in tqdm(Q, desc=f"BM25 {split}", leave=False):
        qt = pu.cit_text(q) if st.get("mode") == "cit" else pu.first_words(q, st.get("q_words"))
        s = bm.score(pu.bm25_tokens(qt), query_tf=st.get("qtf", False))
        qy = pu.doc_year(pu.doc_text(q))
        newer = (cy > qy + st["year_margin"]) if (qy and st.get("year_margin") is not None) else np.zeros(len(ids), bool)
        order = np.lexsort((-s, newer))                   # primary key: newer (False first), then score
        top = [j for j in order[:depth + 1] if ids[j] != q["id"]][:depth]
        out[q["id"]] = {"ids": [ids[j] for j in top], "bm25": [float(s[j]) for j in top],
                        "newer": [int(newer[j]) for j in top]}
    return out


def prepare(root):
    """Returns P = dict(data, gold, top, cand, query, settings). Cached files:
       data/processed/phase7/bm25_top_v1.json, tokens_v1.npz (read ONCE, then kept in memory)."""
    root = Path(root)
    key = str(root)
    if key in _CACHE:
        return _CACHE[key]
    out = root / "data" / "processed" / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    data, _ = pu.load_pcr(root)
    gold = {s: {q["id"]: list(dict.fromkeys(q["rel"])) for q in data[s]["queries"]} for s in ("train", "dev")}

    top_path = out / f"bm25_top_{PREP_VERSION}.json"
    if top_path.exists():
        top = json.load(open(top_path, encoding="utf-8"))
        print(f"[cache] {top_path.name}")
    else:
        st = _bm25_settings(root)
        print("BM25 first stage, setting:", st)
        top = {"settings": st, "pool": {s: [d["id"] for d in data[s]["pool"]] for s in data}}
        for s in data:
            top[s] = _bm25_topk(data, s, st, STORE_DEPTH)
        json.dump(top, open(top_path, "w", encoding="utf-8"))
    # sanity: BM25 top 7 of the cached ranking = P7_01's best dev score
    pred = {q: v["ids"][:7] for q, v in top["dev"].items()}
    m = pu.micro_prf_at_k(pred, gold["dev"])
    print(f"BM25 first stage on dev: Micro-F1@7 {m['micro_f1']:.4f}  (P7_01 best: 0.4247)")
    rec = {}
    for s in ("train", "dev"):
        hit = [c in set(top[s][q]["ids"][:100]) for q, g in gold[s].items() for c in g]
        rec[s] = round(float(np.mean(hit)), 4)
    print("share of gold pairs inside the BM25 top 100 (= best possible after re-ranking):", rec)

    tok_path = out / f"tokens_{PREP_VERSION}.npz"
    if not tok_path.exists():
        tok = Tokenizer.from_file(str(root / "data" / "tokenizer" / "bpe32k.json"))
        cands = {}
        for s in data:
            for d in data[s]["pool"]:
                cands.setdefault(d["id"], d)
        c_ids = sorted(cands)
        q_items = [(q["id"], q) for s in data for q in data[s]["queries"]]
        q_ids = [i for i, _ in q_items]
        assert len(set(q_ids)) == len(q_ids), "a query id appears in two splits"

        def encode(texts, limit, desc):
            res = []
            for s in tqdm(range(0, len(texts), 256), desc=desc):
                res += [np.asarray(e.ids[:limit], dtype=np.int32) for e in tok.encode_batch(texts[s:s + 256])]
            return res
        c_tok = encode([pu.clean_pcr(pu.first_words(cands[i], 1600)) for i in c_ids], C_STORE, "tokenise candidates")
        q_tok = encode([pu.clean_pcr(pu.query_snippet(q)) for _, q in q_items], Q_STORE, "tokenise query snippets")
        flat = lambda arrs: (np.concatenate(arrs), np.cumsum([0] + [len(a) for a in arrs]))
        cf, co = flat(c_tok)
        qf, qo = flat(q_tok)
        np.savez(tok_path, c_ids=np.array(c_ids), c_flat=cf, c_off=co, q_ids=np.array(q_ids), q_flat=qf, q_off=qo)
    z = np.load(tok_path)
    arrs = {k: z[k] for k in z.files}                  # read every array ONCE (z[k] re-reads the file each time)
    z.close()
    cand = {i: arrs["c_flat"][arrs["c_off"][j]:arrs["c_off"][j + 1]] for j, i in enumerate(arrs["c_ids"].tolist())}
    query = {i: arrs["q_flat"][arrs["q_off"][j]:arrs["q_off"][j + 1]] for j, i in enumerate(arrs["q_ids"].tolist())}
    ql = np.array([len(v) for v in query.values()])
    cl = np.array([len(v) for v in cand.values()])
    print(f"tokens: {len(cand):,} candidates (start, median {int(np.median(cl))}), "
          f"{len(query):,} query snippets (median {int(np.median(ql))}, p90 {int(np.percentile(ql, 90))})")
    P = {"data": data, "gold": gold, "top": top, "cand": cand, "query": query, "settings": top["settings"],
         "bm25_dev_f1": m["micro_f1"], "gold_in_top100": rec}
    _CACHE[key] = P
    return P


# ----------------------------------------------------------------------------------------
# 2. Fusion with BM25 + evaluation
# ----------------------------------------------------------------------------------------
def zrows(a):
    """Per-row z-score over the finite entries."""
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a, axis=1, keepdims=True)
    sd = np.nanstd(a, axis=1, keepdims=True)
    return (a - mu) / np.where(sd > 0, sd, 1.0)


def topk_arrays(P, split, qids, depth):
    """BM25 ids / scores / newer flags of the first `depth` candidates, as (Q, depth) arrays."""
    T = P["top"][split]
    ids = [T[q]["ids"][:depth] for q in qids]
    bm = np.full((len(qids), depth), np.nan)
    nw = np.zeros((len(qids), depth), bool)
    for i, q in enumerate(qids):
        n = len(ids[i])
        bm[i, :n] = T[q]["bm25"][:depth]
        nw[i, :n] = np.asarray(T[q]["newer"][:depth], bool)
    return ids, bm, nw


def fused_preds(neural, bm, newer, ids, qids, w, k=pu.TOP_K):
    s = w * np.nan_to_num(zrows(neural), nan=-1e9) + (1 - w) * np.nan_to_num(zrows(bm), nan=-1e9)
    s = s - 1e6 * newer
    preds = {}
    for i, q in enumerate(qids):
        order = np.argsort(-s[i, :len(ids[i])], kind="stable")
        preds[q] = [ids[i][j] for j in order[:k]]
    return preds


def tune_fusion(P, neural, qids, depth):
    """Dev: Micro-F1@7 for every fusion weight w -> (best w, table)."""
    ids, bm, nw = topk_arrays(P, "dev", qids, depth)
    gold = {q: P["gold"]["dev"][q] for q in qids}
    table = {}
    for w in W_GRID:
        m = pu.micro_prf_at_k(fused_preds(neural, bm, nw, ids, qids, float(w)), gold)
        table[float(w)] = {k: round(v, 4) for k, v in m.items() if k.startswith("micro")}
    best = max(table, key=lambda w: (table[w]["micro_f1"], w))
    return best, table


# ----------------------------------------------------------------------------------------
# 3. One experiment
# ----------------------------------------------------------------------------------------
def run_name(c):
    m = c["method"] + (f"r{c['rank']}" if c["method"] in ("lora", "lorma", "rslora") else "")
    if c["method"] == "partial":
        m = f"partial{c['top_blocks']}"
    name = f"{c['phase1']}_{m}_{'causal' if c['causal'] else 'bidir'}_L{c['max_len']}"
    if c["kd_teacher"]:
        name += f"_kd{c['kd_T']:g}"
    if c["train_limit"]:
        name += f"_n{c['train_limit']}"
    if c.get("name_suffix"):
        name += "_" + c["name_suffix"]
    return name


def technique(c):
    t = {"full": "full fine-tuning", "partial": f"partial (layer-wise) FT, top {c['top_blocks']} blocks",
         "lorma": f"LoRMA r{c['rank']}", "lora": f"LoRA r{c['rank']}", "ia3": "IA3"}[c["method"]]
    return t + (f" + KD from {c['kd_teacher']}" if c["kd_teacher"] else "")


class Experiment:
    def __init__(self, root, overrides=None):
        self.root = Path(root)
        ov = dict(overrides or {})
        base = dict(DEFAULTS)
        if ov.get("method") in METHOD_LR:
            base["lr"] = METHOD_LR[ov["method"]]
        c = {**base, **ov}
        if c["causal"] is None:
            c["causal"] = c["phase1"] != "D"
        if c["method"] == "partial" and not c["top_blocks"]:
            c["top_blocks"] = 4
        c["phase1_run"] = PHASE1_RUNS[c["phase1"]]
        self.cfg = c
        self.name = run_name(c)
        self.dir = self.root / "checkpoints" / "phase7" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        json.dump(c, open(self.dir / "config.json", "w"), indent=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self.P = prepare(self.root)
        self.model = None
        print(f"experiment: {self.name} | {technique(c)} | device {self.device} | precision {self.amp}")

    # ---------------- data ----------------
    def get_pair(self, qid, cid):
        return pm.make_pair(self.P["cand"][cid], self.P["query"][qid], self.cfg["max_len"], self.cfg["q_max"])

    def train_queries(self):
        T, pool = self.P["top"]["train"], self.P["top"]["pool"]["train"]
        qs = [{"id": q, "gold": g, "top": T[q]["ids"], "pool": pool} for q, g in self.P["gold"]["train"].items() if g]
        return qs[:self.cfg["train_limit"]] if self.cfg["train_limit"] else qs

    # ---------------- model ----------------
    def build_model(self):
        c = self.cfg
        model, info = pm.build_scorer(self.root / "checkpoints" / "phase1" / c["phase1_run"], c["method"], c["causal"],
                                      c["rank"], c["alpha"], c["top_blocks"], c["dropout"])
        self.load_info = info
        return model.to(self.device)

    def load_teacher(self):
        tdir = self.root / "checkpoints" / "phase7" / self.cfg["kd_teacher"]
        tc = json.load(open(tdir / "config.json"))
        t, _ = pm.build_scorer(self.root / "checkpoints" / "phase1" / tc["phase1_run"], tc["method"], tc["causal"],
                               tc["rank"], tc["alpha"], tc["top_blocks"], tc["dropout"])
        t.load_state_dict(torch.load(tdir / "best_train.pt", map_location="cpu"))
        for p in t.parameters():
            p.requires_grad_(False)
        print(f"teacher {self.cfg['kd_teacher']} loaded (dev F1 fused {json.load(open(tdir / 'report.json'))['dev_fused_f1']})")
        return t.to(self.device).eval()

    def memory_check(self):
        path = self.dir / "memory_check.json"
        if path.exists():
            res = json.load(open(path))
            print("[cache] memory check:", res)
            return res
        model = self.build_model()
        total, trainable = pm.count_params(model)
        teacher = self.load_teacher() if self.cfg["kd_teacher"] else None
        model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        qs = self.train_queries()
        q = max(qs, key=lambda r: len(self.P["query"][r["id"]]))           # longest query snippet
        G = 1 + self.cfg["n_hard"] + self.cfg["n_rand"]
        cands = sorted(self.P["cand"], key=lambda c: -len(self.P["cand"][c]))[:G]
        pairs = [self.get_pair(q["id"], c) for c in cands]
        t0 = time.time()
        x, mk, pmask = (t.to(self.device) for t in pm.collate_pairs(pairs, model.causal))
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp != torch.float32):
            sc = model(x, mk, pmask).float()
            if teacher is not None:
                with torch.no_grad():
                    teacher(x, mk, pmask)
        torch.nn.functional.cross_entropy(sc.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=self.device)).backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        n_groups = len(pm.epoch_groups(qs, self.cfg["seed"], self.cfg["max_pos"], self.cfg["n_hard"], self.cfg["n_rand"]))
        res = {"params_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 3),
               "trainable_%": round(100 * trainable / total, 2), "group": f"{G} pairs x {x.shape[1]} tokens",
               "sec_per_group (first, incl. warm-up)": round(sec, 2), "groups_per_epoch": n_groups,
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None,
               "phase1_load": self.load_info}
        print("memory check:", res)
        json.dump(res, open(path, "w"), indent=1)
        del model, teacher
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return res

    # ---------------- scores ----------------
    def _score(self, model, split, desc):
        qids = list(self.P["top"][split])
        queries = [(q, self.P["top"][split][q]["ids"]) for q in qids]
        t0 = time.time()
        S = pm.score_topk(model, queries, self.get_pair, self.cfg["depth"], self.device, self.amp,
                          self.cfg["infer_rows"], desc=desc)
        sec = time.time() - t0
        n_tok = sum(len(self.get_pair(q, c)[0]) for q, cs in queries for c in cs[:self.cfg["depth"]])
        return qids, S, {"pairs": int(np.isfinite(S).sum()), "sec": round(sec, 1), "tokens_per_s": round(n_tok / max(sec, 1e-9))}

    def dev_eval(self, model):
        """Called after every epoch: dev re-ranking -> fused F1 (picks the best epoch). The scores of the best epoch
        so far are saved as scores_dev.npz (same rule as the trainer: strictly better), so they are not recomputed."""
        qids, S, sp = self._score(model, "dev", "dev re-ranking")
        w, table = tune_fusion(self.P, S, qids, self.cfg["depth"])
        f1 = table[w]["micro_f1"]
        hist = self.dir / "history_train.json"
        best = json.load(open(hist))["best"] if hist.exists() else -1.0
        if f1 > max(best, getattr(self, "_dev_best", -1.0)):
            self._dev_best = f1
            np.savez(self.dir / "scores_dev.npz", qids=np.array(qids), S=S)
            json.dump(sp, open(self.dir / "speed_dev.json", "w"), indent=1)
        return {"f1": f1, "f1_neural_only": table[1.0]["micro_f1"], "w": w}

    def scores(self, split):
        """(qids, (Q, depth) neural scores) for a split, cached in scores_{split}.npz."""
        path = self.dir / f"scores_{split}.npz"
        if path.exists():
            z = np.load(path)
            out = (z["qids"].tolist(), z["S"].copy())
            z.close()
            return out
        if self.model is None:
            self.train()
        qids, S, sp = self._score(self.model, split, f"{split} re-ranking")
        np.savez(path, qids=np.array(qids), S=S)
        json.dump(sp, open(self.dir / f"speed_{split}.json", "w"), indent=1)
        print(f"{split}: {sp['pairs']:,} pairs scored in {sp['sec']:.0f} s ({sp['tokens_per_s']:,} tokens/s)")
        return qids, S

    # ---------------- stages ----------------
    def train(self):
        model = self.build_model()
        teacher = self.load_teacher() if self.cfg["kd_teacher"] else None
        state = pm.train_ranker(model, self.train_queries(), self.get_pair, self.dir, self.cfg, self.device, self.amp,
                                eval_fn=self.dev_eval, teacher=teacher)
        del teacher
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model, self.state = model, state
        return state

    def report(self):
        path = self.dir / "report.json"
        if path.exists():
            r = json.load(open(path))
            print(f"[cache] report: dev fused {r['dev_fused_f1']} (w {r['w']}), neural only {r['dev_neural_only_f1']}")
            return r
        qids, S = self.scores("dev")
        w, table = tune_fusion(self.P, S, qids, self.cfg["depth"])
        assert abs(table[0.0]["micro_f1"] - round(self.P["bm25_dev_f1"], 4)) < 1e-3, "w=0 must equal BM25"
        by_depth = {}
        for d in (20, 50, 100):
            if d <= self.cfg["depth"]:
                wd, td = tune_fusion(self.P, S[:, :d], qids, d)
                by_depth[d] = {"w": wd, **td[wd]}
        # per-query analysis at the chosen w: by number of gold, hits
        ids, bm, nw = topk_arrays(self.P, "dev", qids, self.cfg["depth"])
        pred = fused_preds(S, bm, nw, ids, qids, w)
        bm_pred = fused_preds(S, bm, nw, ids, qids, 0.0)
        g = self.P["gold"]["dev"]
        buckets = {"1": (1, 1), "2-3": (2, 3), "4-7": (4, 7), "8+": (8, 10 ** 6)}
        by_gold = {}
        for b, (lo, hi) in buckets.items():
            qs = [q for q in qids if lo <= len(g[q]) <= hi]
            if qs:
                by_gold[b] = {"queries": len(qs),
                              "model": round(pu.micro_prf_at_k({q: pred[q] for q in qs}, {q: g[q] for q in qs})["micro_f1"], 4),
                              "bm25": round(pu.micro_prf_at_k({q: bm_pred[q] for q in qs}, {q: g[q] for q in qs})["micro_f1"], 4)}
        better = sum(len(set(pred[q]) & set(g[q])) > len(set(bm_pred[q]) & set(g[q])) for q in qids)
        worse = sum(len(set(pred[q]) & set(g[q])) < len(set(bm_pred[q]) & set(g[q])) for q in qids)
        hist = json.load(open(self.dir / "history_train.json"))
        mem = json.load(open(self.dir / "memory_check.json")) if (self.dir / "memory_check.json").exists() else {}
        sp = json.load(open(self.dir / "speed_dev.json")) if (self.dir / "speed_dev.json").exists() else {}
        r = {"name": self.name, "technique": technique(self.cfg), "phase1": self.cfg["phase1"],
             "trainable_M": mem.get("trainable_M"), "trainable_%": mem.get("trainable_%"), "peak_gb_train": mem.get("peak_gb"),
             "best_epoch": hist["best_epoch"], "history": hist["history"],
             "dev_bm25_f1": table[0.0]["micro_f1"], "dev_neural_only_f1": table[1.0]["micro_f1"],
             "dev_neural_only": table[1.0], "w": w, "dev_fused_f1": table[w]["micro_f1"], "dev_fused": table[w],
             "fusion_table": table, "by_depth": by_depth, "by_n_gold": by_gold,
             "queries_better_than_bm25": int(better), "queries_worse_than_bm25": int(worse),
             "infer_tokens_per_s": sp.get("tokens_per_s"), "min_per_epoch": [h["train_min"] for h in hist["history"]]}
        json.dump(r, open(path, "w"), indent=1)
        print(f"dev Micro-F1@7: BM25 {r['dev_bm25_f1']} | neural only {r['dev_neural_only_f1']} | "
              f"fused {r['dev_fused_f1']} (w {w}) | queries better / worse than BM25: {better} / {worse}")
        return r

    def predict_test(self):
        zpath = self.dir / "submission.zip"
        r = self.report()
        if not zpath.exists():
            qids, S = self.scores("test")
            ids, bm, nw = topk_arrays(self.P, "test", qids, self.cfg["depth"])
            pu.write_submission(self.dir, fused_preds(S, bm, nw, ids, qids, r["w"]))
        info = pu.check_submission(zpath, [q["id"] for q in self.P["data"]["test"]["queries"]],
                                   self.P["top"]["pool"]["test"])
        print(f"submission checked: {info} -> {zpath}")
        return zpath

    def log(self):
        r = json.load(open(self.dir / "report.json"))
        row = {k: r[k] for k in ("name", "technique", "phase1", "trainable_M", "trainable_%", "best_epoch",
                                 "dev_bm25_f1", "dev_neural_only_f1", "w", "dev_fused_f1", "infer_tokens_per_s",
                                 "min_per_epoch", "peak_gb_train")}
        row["dev_fused_P"], row["dev_fused_R"] = r["dev_fused"]["micro_p"], r["dev_fused"]["micro_r"]
        row["submission"] = str((self.dir / "submission.zip").relative_to(self.root))
        _log(self.root, row)

    def run(self):
        """The whole pipeline; every stage loads its result if it already exists."""
        if (self.dir / "report.json").exists() and (self.dir / "submission.zip").exists():
            print(f"[cache] {self.name} already finished")
            r = self.report()
            self.predict_test()
            self.log()
            return r
        self.memory_check()
        self.train()
        r = self.report()
        self.predict_test()
        self.log()
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return r


# ----------------------------------------------------------------------------------------
# 4. Ensemble and comparison
# ----------------------------------------------------------------------------------------
def _log(root, row):
    p = Path(root) / "experiments" / "phase7" / "log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(p, encoding="utf-8")] if p.exists() else []
    rows = [x for x in rows if x["name"] != row["name"]] + [row]          # one row per run (latest)
    with open(p, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")


def ensemble(root, names, out_name="ensemble", depth=100):
    """Score fusion of several runs: z-scores averaged over the models, then fused with BM25 (w tuned on dev)."""
    root = Path(root)
    d = root / "checkpoints" / "phase7" / out_name
    d.mkdir(parents=True, exist_ok=True)
    P = prepare(root)

    def load(split):
        mats, q0 = [], None
        for n in names:
            z = np.load(root / "checkpoints" / "phase7" / n / f"scores_{split}.npz")
            q, S = z["qids"].tolist(), z["S"].copy()
            z.close()
            assert q0 is None or q == q0
            q0 = q
            mats.append(zrows(S[:, :depth]))
        return q0, np.mean(mats, axis=0)

    path = d / "report.json"
    if path.exists():
        r = json.load(open(path))
        print(f"[cache] ensemble report: dev fused {r['dev_fused_f1']} (w {r['w']})")
    else:
        qids, S = load("dev")
        w, table = tune_fusion(P, S, qids, depth)
        singles = {n: json.load(open(root / "checkpoints" / "phase7" / n / "report.json"))["dev_fused_f1"] for n in names}
        r = {"name": out_name, "technique": "score ensemble of " + ", ".join(names), "phase1": "+".join(sorted({json.load(open(root / "checkpoints" / "phase7" / n / "config.json"))["phase1"] for n in names})),
             "members": names, "members_dev_fused_f1": singles, "dev_bm25_f1": table[0.0]["micro_f1"],
             "dev_neural_only_f1": table[1.0]["micro_f1"], "w": w, "dev_fused_f1": table[w]["micro_f1"],
             "dev_fused": table[w], "fusion_table": table}
        json.dump(r, open(path, "w"), indent=1)
        print(f"ensemble dev Micro-F1@7: neural only {r['dev_neural_only_f1']} | fused {r['dev_fused_f1']} (w {w}) | members {singles}")
    zpath = d / "submission.zip"
    if not zpath.exists():
        qids, S = load("test")
        ids, bm, nw = topk_arrays(P, "test", qids, depth)
        pu.write_submission(d, fused_preds(S, bm, nw, ids, qids, r["w"]))
    print("submission checked:", pu.check_submission(zpath, [q["id"] for q in P["data"]["test"]["queries"]],
                                                     P["top"]["pool"]["test"]), "->", zpath)
    _log(root, {"name": out_name, "technique": r["technique"], "phase1": r["phase1"], "trainable_M": None,
                "trainable_%": None, "best_epoch": None, "dev_bm25_f1": r["dev_bm25_f1"],
                "dev_neural_only_f1": r["dev_neural_only_f1"], "w": r["w"], "dev_fused_f1": r["dev_fused_f1"],
                "dev_fused_P": r["dev_fused"]["micro_p"], "dev_fused_R": r["dev_fused"]["micro_r"],
                "infer_tokens_per_s": None, "min_per_epoch": None, "peak_gb_train": None,
                "submission": str(zpath.relative_to(root))})
    return r


def compare(root):
    import pandas as pd
    p = Path(root) / "experiments" / "phase7" / "log.jsonl"
    rows = [json.loads(line) for line in open(p, encoding="utf-8")] if p.exists() else []
    cols = ["name", "technique", "trainable_M", "trainable_%", "best_epoch", "dev_neural_only_f1", "w", "dev_fused_f1",
            "dev_fused_P", "dev_fused_R", "min_per_epoch", "infer_tokens_per_s", "peak_gb_train"]
    return pd.DataFrame(rows, columns=cols).set_index("name") if rows else pd.DataFrame(columns=cols)
