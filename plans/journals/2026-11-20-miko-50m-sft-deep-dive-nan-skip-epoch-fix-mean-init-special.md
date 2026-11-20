---
title: "Miko-50M: SFT deep-dive — nan-skip, epoch fix, mean-init special embeddings"
date: 2026-11-20
summary: "Found why SFT stalled: never-trained special-token embeddings are random noise; added mean-init seeding, nan-skip batching, epoch property fix, extended to 2000 steps; first 400-step cycle taught format only partially (chat_sft 0/14)"
---

# Miko-50M: SFT deep-dive — nan-skip, epoch fix, mean-init special embeddings

## What happened

ak-plan session (plans/20261120-sft-completion-and-eval/, 5 commits):

1. **Phase 1 hardening executed**: lr 3e-5, warmup 25%, checkpoint
   every 25 steps, nan-skip (1 bad batch skipped, 3 consecutive ->
   emergency), attempts 30. This unblocked the step-37 deterministic
   NaN death that killed all 10 original attempts.

2. **First 400-step cycle completed** (train 315 -> 19.8, val 195 ->
   37.5 monotonic) but post-SFT eval was flat: chat_sft 0/14. Manual
   generation debug showed garbage after '<assistant>' — the special
   tokens (ids 4-9) NEVER appeared in pretraining, so their embeddings
   are random noise; 400 steps merely began learning the format.

3. **Extended to 2000 steps**; resume then crashed on
   SFTShardedDataset missing the epoch property (read by resume
   alignment) — fixed; the restart loop had been burning ~1 attempt
   per minute on that AttributeError.

4. **Structural fix — mean-init seeding**: ids 0,1,4-9 initialized to
   the mean of trained vocab embeddings in the SFT copy (bos/eos kept,
   they were trained by document packing; tied lm_head clone patched
   identically). Relaunched from step 0.

5. Current run: steps 0-140 done (loss 320 -> 141, LR in 500-step
   warmup), box intermittently throttled to ~70 tok/s by an external
   job. ETA ~3-6h; auto-restart armed (30 attempts, 25-step
   checkpoints preserve progress).

## Decision

- Restart from step 0 with mean-init rather than continuing the
  step-400 checkpoint: a neutral in-distribution start beats
  partially-dragged noise rows, and deterministic seeding makes the
  comparison clean.
- v3-to-60000 resume stays deferred (non-goal of this plan).

## Next steps

1. When SFT completes 2000 steps: tools/evaluate.py on
   checkpoints_sft_38m/latest -> chat_sft (baseline 0/14) +
   tool_basic (0.25 baseline = negatives only).
2. tools/chat.py --tools smoke: '123*456?' should trace
   calculator -> 56088.
3. If chat_sft still ~0 with loss < 5: the format is learned but
   content is weak -> expand seed data (not more training).
4. MODEL_CARD v0.4 row + archive the plan.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
