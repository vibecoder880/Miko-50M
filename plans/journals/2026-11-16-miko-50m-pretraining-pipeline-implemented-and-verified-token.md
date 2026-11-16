---
title: "Miko-50M: pretraining pipeline implemented and verified (tokenizer -> shards -> train -> checkpoint -> resume -> generate)"
date: 2026-11-16
summary: Implemented Miko v0.1 pretraining pipeline per plan §9-23; 42/42 tests pass incl. Gate A tiny-overfit; smoke train+resume verified; benchmark shows 50M needs ~1.4GB RSS (exceeds 1GB box)
---

# Miko-50M: pretraining pipeline implemented and verified (tokenizer -> shards -> train -> checkpoint -> resume -> generate)

## What happened

Cloned https://github.com/vibecoder880/Miko-50M, merged remote branch
`agent/model-core` (transformer core + branch tests) into main, then
implemented the full pretraining pipeline per the Miko 50M training plan:

- `miko/tokenizer.py` — byte-level BPE, special tokens reserved ids 0-9,
  lossless Vietnamese/code roundtrip; `tools/train_tokenizer.py` CLI.
- `miko/dataset.py` — `<bos> doc <eos>` packing into uint16 shards, mmap
  windows, per-epoch seeded shuffle; `tools/pack_dataset.py` CLI.
- `miko/checkpoint.py` — safetensors weights + optimizer + trainer_state
  (step/metrics/torch/numpy/python RNG); `miko/generation.py` — greedy/
  temperature/top-k/top-p/repetition-penalty.
- `miko/trainer.py` + `tools/train.py` — AdamW(0.9/0.95), warmup+cosine,
  grad accumulation, clip, NaN guard with emergency checkpoint, CSV log
  (plan §21 fields), milestone checkpoints + pruning, `--resume auto`.
- `configs/` — miko_50m.yaml = 50,636,928 params (verified via
  `tools/count_params.py`), miko_tiny.yaml ~1M, two train configs.

Verification evidence:
- 42/42 tests pass (221s) incl. **Gate A** tiny-overfit (~230k-param model
  memorises corpus, loss < 0.5 from ln(vocab)≈6.1 baseline).
- Smoke: tiny train 160 steps loss 16.5 -> 0.9, peak RSS 356MB;
  `--resume auto` continued 150 -> 200 (val 2.80) — **Gate B** pass;
  `tools/generate.py` produces text (repetition expected on 33K-token smoke
  corpus, plan §61).
- Benchmark (`scripts/benchmark_cpu.py`, 3 cores): tiny 221 tok/s @ 344MB;
  **50M floor 46.8 tok/s @ 1375MB RSS** (seq 128, grad_accum 1).

## Bugs found and fixed along the way

- Branch tests used hidden_size=64 with default num_heads=6 (indivisible)
  after the config-defaults fix commit → added num_heads=8.
- `BpeTrainer(limit_alphabet=0)` literally capped the alphabet at 0 chars
  → tokenizer trained 10 specials only; removed the param.
- Shard writer rolled a trailing empty shard; fixed roll-on-next-write.
- numpy/python RNG restore in checkpoints stored partial state →
  set_state/ValueError; now stores full MT19937 + version-3 tuples.
- Off-by-one in the overfit *test* (same window as input+target lets causal
  attention copy inputs → fake zero loss); dataset path was already correct.
- pytest.approx(0-d tensor) iterates → wrap float() in tests.

## Decision

- Committed everything on `main` as `292fa09` (matches repo's existing
  direct-to-main convention); training artifacts gitignored
  (checkpoints_*/, data/shards/, data/raw/seed/).
- 50M training verified to RUN on this 6GB box (~47 tok/s) but does NOT fit
  the plan's 1GB target box (1375MB RSS) — deferred per plan §17:
  memory-saving optimizer (Adafactor-style/offloaded AdamW) is the next
  prerequisite before any 1GB-only run.

## Next steps

1. Memory-saving optimizer for 1GB fit (plan §17 / §46 Risk 2).
2. Real corpus build: Vietnamese Wikipedia slice + open code, manifests,
   dataset-v0.1 (plan §10-12, §22 Stage 1-2).
3. Longer tiny/small pretraining runs + `evals/` fixed suite (§24-25).

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
