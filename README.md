# CS-781 Capstone: In-House Legal Intelligence System

**Atharva Chouhan · 261110017 · IIT Kanpur**

A legal language model trained **from scratch** (no pretrained weights, about 70 M parameters) on one laptop GPU (RTX 4050, 6 GB), then adapted to six Indian legal NLP tasks. Every phase after Phase 1 starts from our own Phase-1 models.

| Phase | Task | Metric | Best test score | Best model |
|---|---|---|---|---|
| 1 | Legal LM pre-training (continue a 512-word judgment with 128 words) | BERTScore F1 | 0.8283 | A2 = decoder + copy mechanism |
| 2 | Legal NER (12 entity types, character spans) | strict macro-F1 | 0.5498 | A2 full fine-tuning + BiLSTM head + Viterbi |
| 3 | Rhetorical role per sentence (13 roles) | macro-F1 | 0.5735 | A + LoRA + document-level BiLSTM |
| 4 | Court judgment prediction (0/1) | macro-F1 | 0.8063 | frozen A2 reading the whole case in chunks |
| 5 | Statute identification (multi-label, 100 statutes) | macro-F1 | see report | A2 full fine-tuning / whole-case model |
| 6 | Abstractive summarisation | BERTScore F1 | 0.8360 | D (encoder–decoder) + LoRA + fusion-in-decoder |
| 7 | Prior-case retrieval (top 7) | Micro-F1@7 | see report | BM25 + Phase-1 cross-encoder re-ranker |

The full method and analysis are in the report (`report/CS781-CP-Atharva-Chouhan-261110017-report.pdf`).

---

## 1. Repository layout

```
code/
├── README.md                this file
├── requirements.txt
├── .gitignore               also marks the project ROOT (see below)
├── data/tokenizer/bpe32k.json   our 32k byte-level BPE tokenizer (made in phase1/02_tokenizer.ipynb)
├── src/                     shared Python modules used by Phases 2-7
│   ├── ner_*.py             Phase 2 (+ the Phase-1 backbone loader and LoRA/LoRMA used by every phase)
│   ├── rr_*.py              Phase 3 (+ IA3 / rsLoRA / Houlsby adapters)
│   ├── cjpe_*.py            Phase 4 (+ attention pooling, partial fine-tuning, hierarchical head)
│   ├── lsi_*.py             Phase 5
│   ├── summ_*.py            Phase 6 (generation, fusion-in-decoder, decoding)
│   └── pcr_*.py             Phase 7 (BM25, cross-encoder, fusion)
├── scripts/                 small helper scripts (Phase-1 local evaluation, Phase-4 overnight runner)
└── phase1/ ... phase7/      the notebooks of every phase, in the order they are run
```

**How paths work.** Every notebook walks up from its own folder until it finds `.gitignore`. That folder (`code/`) is the project ROOT, and everything lives under it:

- `ROOT/src`: the modules
- `ROOT/data/raw/...`: the downloaded data
- `ROOT/data/processed/...`: caches
- `ROOT/checkpoints/phaseN/...`: models
- `ROOT/experiments/phaseN/...`: results and submissions

These folders are created automatically. Data and checkpoints are **not** in the repository because they are too large.

## 2. Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu132   # we used torch 2.14.0+cu132; pick yours
pip install -r requirements.txt
hf auth login                     # the dataset Exploration-Lab/CS-781-Capstone is gated
```

The first notebook of each phase downloads that phase's data from `Exploration-Lab/CS-781-Capstone` into `data/raw/` and skips the download if the files already exist.

## 3. How to run

Open the notebooks of a phase **in numerical order** and use **Run All**. The design rules are the same in every phase:

- **Everything is cached.** A finished model, score file or submission is loaded instead of recomputed, so you can safely re-run a notebook.
- **Training is checkpointed** every 250 optimiser steps and resumes in the middle of an epoch after a crash.
- **Every long step has a progress bar.**
- **Each `submission.zip` is checked** against the official format before it is written.

Phases 2–7 need the Phase-1 checkpoints in `checkpoints/phase1/<run name>/model_final.pt` + `config.json`, so run Phase 1 first.

---

## 4. Phase by phase

### Phase 1: pre-training (`phase1/`)
| Notebook | What it does |
|---|---|
| `01_data_exploration.ipynb` | statistics of the four sources (Supreme Court, Tribunals, High Courts, Acts) and of the test prompts |
| `02_tokenizer.ipynb` | trains our 32k byte-level BPE tokenizer and compares it with GPT-2 |
| `03_prepare_data.ipynb` | cleaning (ASCII, no signature stamps, whitespace), MD5 deduplication, 1% validation split, token files |
| `A_decoder.ipynb` | **A**: LLaMA-style decoder (RoPE, RMSNorm, SwiGLU), 10 layers, width 640, 70.06 M params |
| `B_prefix_lm.ipynb` | **B**: prefix-LM (prompt read in both directions, continuation left to right) |
| `A2_decoder_copy_mtp.ipynb` | **A2 (best)**: decoder A plus a **copy head** that can copy tokens from the prompt; also the copy/MTP ablation |
| `D_encdec_copy.ipynb` | **D**: encoder–decoder (4 + 5 layers) with a copy head |
| `Starter_GPT2_Style.ipynb` | the course starter (GPT-2 style) used as a baseline |

All four models see the same 200 M tokens (70% Supreme Court, 30% Tribunals) in the same order. The copy head idea: 68% of next tokens already appear earlier in a legal document (names, dates, section numbers). The model therefore mixes "generate from the vocabulary" and "copy from the input" with a learned gate:
`p(word) = g * p_vocab(word) + (1 - g) * p_copy(word)`.
Decoding uses a KV cache, repetition penalty 1.1, no repeated 6-grams, and an exact cut to 128 words. `scripts/evaluate_phase1.py` is the local Phase-1 evaluator from the course kit (BERTScore, ROUGE, BLEU); we used it on 200 held-out prompts.

### Phase 2: Legal NER (`phase2/`, `src/ner_*.py`)
- **Notebooks:** `P2_01` data → `P2_02` pipeline tests → `P2_03` A2 full fine-tuning → `P2_04` improvements (causal + BiLSTM = **best**) → `P2_05` PEFT (LoRA, LoRMA) → `P2_06` knowledge distillation into the 4-layer encoder.
- **Method:**
  1. *Offset-preserving cleaning:* the text is cleaned character by character with a map back to the original positions.
  2. Windows of 1,536 tokens (stride 1,024) are tagged with 25 BIO tags.
  3. A BiLSTM head adds right-hand context.
  4. *Constrained Viterbi* keeps the tags valid (+0.13 macro-F1).
  5. The final model is retrained on both folds.

### Phase 3: Rhetorical roles (`phase3/`, `src/rr_*.py`)
- **Notebooks:** `P3_01` data → `P3_02` pipeline → `P3_03` A2 full fine-tuning → `P3_04` PEFT (LoRA = **best**, LoRMA, IA3) → `P3_05` encoder + rsLoRA + KD → `P3_06` ensembles and ablations.
- **Method:**
  1. Each sentence becomes one vector, the mean of its tokens.
  2. A BiLSTM reads **all** sentence vectors of the document, so every sentence sees its whole context (+0.19 macro-F1).
  3. Corporate Law and IT documents are trained jointly with a domain embedding.

### Phase 4: Judgment prediction (`phase4/`, `src/cjpe_*.py`)
- **Notebooks:** `P4_01_v2` data → `P4_02` A2 full fine-tuning → `P4_03` B + LoRMA → `P4_04` D-encoder + KD → `P4_05` A + adapters → `P4_06` gradual unfreezing → `P4_07` **whole-case hierarchical model (best)** → `P4_08` BitFit.
- **Runners:** `scripts/p4_run.py` and `scripts/p4_overnight.py` run P4_02–P4_06 one after another.
- **Method:**
  1. Text is lower-cased to match the dev/test style; duplicates are removed.
  2. The flat models read the last 1,536 tokens (the outcome is at the end).
  3. The best model cuts the whole case into chunks, encodes each chunk with the fine-tuned A2, and combines them with a small attention head (0.28 M params, 5-fold cross-validation on dev).
  4. The decision threshold is tuned on dev.

### Phase 5: Statute identification (`phase5/`, `src/lsi_*.py`)
- **Notebooks:** `P5_01` data → `P5_02` **all experiments in one notebook**: A2 full fine-tuning, D-encoder + KD (and its control), B + LoRMA, A + adapters, A2 partial fine-tuning, whole-case hierarchical head.
- **Method:**
  1. 100 sigmoid outputs trained with BCE.
  2. `pos_weight = sqrt(neg/pos)` gives more weight to rare positives.
  3. Each model reads the first 768 + last 768 tokens.
  4. Each statute gets its own threshold, chosen by cross-fitting on dev.

### Phase 6: Summarisation (`phase6/`, `src/summ_*.py`)
- **Notebooks:** `P6_01_v2` data and own validation split → `P6_02` **all models** (A2 full, B + LoRMA, A + Houlsby adapters, D + LoRA, D + LoRA + FiD = **best**, small D + KD and its control) → `P6_03` decoding study and final submissions.
- **Method:**
  1. D's encoder reads 3 chunks of 1,024 tokens (fusion-in-decoder), and the copy layers stay trainable.
  2. *Anti-repeat decoding:* no repeated 4-grams, a repetition penalty of 1.2 on words already in the summary, and a cut at about 320 words.

### Phase 7: Prior-case retrieval (`phase7/`, `src/pcr_*.py`)
- **Notebooks:** `P7_01_pcr_data_v2` data exploration and the tuned BM25 baseline → `P7_02_pcr_all` **all experiments in one notebook**.
- **Method:**
  1. **Stage 1, BM25** (own numpy implementation): candidates are cut to their first 2,048 words, the query is the sentences around each `<CITATION>`, k1 = 8 and b = 1, plus a year filter (a prior case cannot be newer than the query). Dev Micro-F1@7 is 0.4247.
  2. **Stage 2, re-ranker:** a Phase-1 model reads `[candidate start] EOS [query citation contexts] EOS` (≤ 1,024 tokens) for BM25's top 100. It is trained with a listwise softmax loss (1 relevant case + 6 BM25 hard negatives + 1 random negative per group).
  3. The final score is `w * z(model) + (1 - w) * z(BM25)`, with `w` tuned on dev.
- **Experiments:** A2 + LoRMA, A2 full fine-tuning (teacher), D-encoder + KD (best on dev, 0.438) and its control, A + IA3, A2 partial fine-tuning, and a score ensemble.

---

## 5. Fine-tuning techniques implemented

All techniques are written by hand in `src/`; no PEFT library is used.

| Technique | Where | Idea |
|---|---|---|
| Full fine-tuning | all | train every weight |
| Partial / layer-wise, gradual unfreezing | `cjpe_models.set_unfrozen` | train only the top blocks / unlock blocks step by step |
| LoRA | `ner_models` | `h = W0 x + (alpha/r) B A x` |
| **LoRMA** (compulsory) | `ner_models` | `h = (I + (alpha/r) B A) W0 x` (multiplicative) |
| rsLoRA | `rr_models` | LoRA with scale `alpha / sqrt(r)` |
| IA3 | `rr_models` | learned scaling of keys, values and FFN activations |
| Bottleneck / Houlsby adapters | `cjpe_models`, `summ_models` | small bottleneck layers inside each block |
| Knowledge distillation | every phase | `(1 - a) CE + a T^2 KL(teacher || student)`; student = D's 4-layer encoder; every KD run has a no-teacher control |

Every PEFT method starts as the identity: at step 0 it outputs exactly the Phase-1 model's values (checked in CPU tests).

## 6. Hardware and run time
- One NVIDIA RTX 4050 laptop GPU (6 GB) with bf16.
- Each Phase-1 model took about 2–4 h.
- Each later phase ran all its experiments overnight in one notebook (4–8 h).
- Before every GPU run, all code was tested on CPU with tiny fake models that have the real weight names.

## 7. Data and rules
- Only the official dataset `Exploration-Lab/CS-781-Capstone` was used.
- No pretrained weights or external legal corpora were used.
- The 32k tokenizer was trained on the official corpus only.
