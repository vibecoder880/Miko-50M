---
title: "Miko-50M: M3 groundwork — Adafactor 1GB fit, fixed eval suite, real Stage-1 pretraining"
date: 2026-11-16
summary: "Added optimizer factory (Adafactor cuts 50M RSS 1375->937MB, 38M variant 798MB), fixed eval suite (36 items + ppl), fetched real Wikipedia mini corpus (3.84M tokens), Stage-1 training shows Gate C pass (val loss 12.53->7.77 monotonic)"
---

# Miko-50M: M3 groundwork — Adafactor 1GB fit, fixed eval suite, real Stage-1 pretraining

## What happened

Continued the Miko 50M plan (ak-cook --auto) into milestone M3 on top of
commit 292fa09:

1. **Memory-saving optimizer (plan §17/§46)** — `miko/optim.py` factory:
   `adamw | adafactor`. torch 2.15's Adafactor has a simplified signature
   (no clip_threshold/decay_rate/beta1 kwargs — relative step
   `min(lr, 1/sqrt(t))` is built in; pass explicit lr to keep the cosine
   schedule driving it). Measured RSS (seq 128, micro-batch 1):
   - 50M AdamW: 1375 MB (state 405 MB) @ 46.8 tok/s
   - 50M Adafactor: **937 MB** (state 0.7 MB) @ 46.7 tok/s
   - 38M (16K vocab, 12L) Adafactor: **798 MB** @ 58.0 tok/s → the 1GB-box
     fit variant `configs/miko_38m_1gb.yaml`.

2. **Fixed eval suite (plan §24-25/§49)** — `miko/evaluation.py` +
   `tools/evaluate.py` + `evals/{vi_basic:16,code_basic:12,chat_basic:8}.jsonl`
   (greedy completion, accepted-answer containment, per-category scores,
   val loss/perplexity, §49-style summary JSON). Tiny-checkpoint smoke:
   vi 0.13 / code 0.17 / chat 0.13, val ppl 15.

3. **Real corpus (dataset-v0.1-mini)** — `scripts/fetch_corpus.py` streams
   `wikimedia/wikipedia` with §12 cleaning + provenance manifests (CC BY-SA):
   vi-wiki 500 docs (10.1 MB) + en-wiki 150 (4.7 MB) → 3.84M packed tokens
   (3.65M train / 183K val). Real tokenizer vocab 4096 achieved.

4. **Stage-1 pretraining (Gate C)** — Miko Tiny ~1M, 1000 steps / 512K
   tokens in 158 s: train loss 20.3 → 1.93, val loss monotonic
   12.53 → 7.77 → **Gate C PASS**. Eval baseline on real corpus:
   vi 0.00 / code 0.08 / chat 0.00, val ppl 2368 — expected floor for a 1M
   model on 512K tokens; generation shows Vietnamese function words
   emerging. Full suite: **55/55 tests pass**.

## Decision

- Benchmark numbers vary run-to-run on this box (tiny measured 220 → 524
  tok/s across runs; training CSV averaged 3241). Treat benchmark JSONs as
  order-of-magnitude; RM-fit numbers (RSS) are the reliable ones.
- `scripts/fetch_corpus.py` ends with `os._exit(0)`: pyarrow/datasets
  streaming crashes interpreter teardown (PyGILState fatal) after a fully
  successful fetch — output is flushed before the hard exit.
- 38M/Adafactor is the recommended 1GB-box config; 50M+Adafactor needs
  ~1.3GB total. Both documented in README + MODEL_CARD.

## Next steps

1. Scale corpus: full vi-wiki slice + code sources (plan §11-12), manifest
   per source; retrain tokenizer at 16K vocab.
2. Longer pretraining of the 38M/50M model on dataset-v0.2 (Stage 2-3,
   plan §21) — budget ~2.5-10 days CPU at measured tok/s.
3. Instruction tuning phase (M4) once base is stable; then tool calling
   (M5) using the reserved special tokens.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
