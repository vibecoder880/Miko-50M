---
title: "Miko-50M: SFT seed doubled to 533 conversations; first SFT run fully staged"
date: 2026-11-19
summary: "Expanded seed banks (identity/vi-en/daily/concepts), wrote the first real SFT config + guarded launcher that seeds from the base checkpoint; v3 pretraining continues slowly under external load"
---

# Miko-50M: SFT seed doubled to 533 conversations; first SFT run fully staged

## What happened

Eighth session (commit dc23fd6, pushed). Low-CPU session by design —
the box is running a heavy external job, v3 pretraining crawls at
~85-115 tok/s (load 5/3 cores), so no test suites or evals were run
against it.

1. **SFT seed doubled** (533 conversations / 508 train + 25 val masked
   windows): IDENTITY 20, VI_EN 28 two-way pairs, new DAILY bank (20),
   new CONCEPTS bank (30 ML/programming explanations), arithmetic
   density 0.22 -> 0.45. Verified-answer property kept (computed +
   asserted).

2. **First real SFT run staged**:
   - configs/train_sft_38m.yaml — LR 1e-4 (3x below pretrain peak),
     400 steps x grad_accum 8 (~6 epochs over seed), AdamW, separate
     checkpoints_sft_38m/ lineage.
   - scripts/train_sft_38m.sh — guarded (refuses while pretraining
     runs), seeds SFT lineage from checkpoints_38m/latest (step reset
     0, weights kept) on first launch, auto-restart.
   - Guard verified live: refused correctly while v3 runs.

3. v3 progress: step 40250/60000 at ~85 tok/s — ETA stretches to days
   under the external load; will speed up when the box frees.

## Decision

- SFT waits for v3 to finish (guarded launcher, one command after).
- Seed scale (533 convs / ~22K tokens) is a first-signal size; the
  next seed expansion targets ~2-5K conversations once the first SFT
  curve is seen.

## Next steps

1. When v3 hits 60000 (or is stopped early): ./scripts/train_sft_38m.sh
   then evaluate with evals/chat_sft.jsonl (baseline 0/14 pre-SFT).
2. If SFT val rises early, cut steps; if healthy, expand banks and
   rerun longer.
3. Keep v3 milestones: eval at 50000 and 60000 for the vi-heavy corpus
   effect on vi_basic.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
