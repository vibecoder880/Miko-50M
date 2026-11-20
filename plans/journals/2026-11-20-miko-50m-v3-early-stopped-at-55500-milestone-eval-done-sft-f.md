---
title: "Miko-50M: v3 early-stopped at 55500, milestone eval done, SFT fighting external load"
date: 2026-11-20
summary: "v3 stopped at plateau (val 6.67-6.68, ppl 917 on v0.3 val); tool_basic 0.25 pre-SFT (negatives pass); SFT optimizer-seed bug fixed (Adafactor state into AdamW); SFT auto-restarting under heavy external load"
---

# Miko-50M: v3 early-stopped at 55500, milestone eval done, SFT fighting external load

## What happened

Twelfth session (commits 61728d2, a3d979c, d0ced16, pushed):

1. **Disk-full crash diagnosed** — v3 died at step 56000: checkpoint
   write hit 'No space left on device' while an external job filled
   the disk; restart loop then couldn't even import torch. Added a
   disk guard to the launcher (wait until >=700MB free, 5-min backoff,
   not counted against restart attempts). Resumed from step-00055500.

2. **v3 early stop** — box re-throttled to ~220-290 tok/s; remaining
   4150 steps (cosine LR tail 3.3e-05 -> 3e-05) at ~25h ETA for ~zero
   expected gain with val flat 6.67-6.68 since step 55000. Stopped;
   resumable anytime via ./scripts/train_38m_v3.sh.

3. **Milestone eval (actual step 55500, v0.3 val split)**:
   val loss 6.82 / ppl 917 (better than v2's 6.93/1023 despite earlier
   stop — the vi-heavy mixture helped). Suite: vi 0.0625, code 0.167,
   chat 0, tool_basic 0.25 — the tool score is the 4 negative items
   passing (base model correctly doesn't call tools it has never
   seen); chat_sft still 0/14 pre-SFT as expected.

4. **SFT seed bug caught** — seeding copied the base checkpoint
   wholesale, importing its Adafactor optimizer.pt into the AdamW SFT
   run (polluted param groups beta2_decay/d/eps; unknown state keys
   col_var/row_var). AdamW silently re-initialized moments — the loss
   curve was identical before/after the fix — but the seed now deletes
   optimizer.pt for a genuinely fresh optimizer. Correct hygiene.

5. **SFT vs the box** — under external load ~10, python processes die
   every ~17-20 min (before reaching the step-100 checkpoint). The
   auto-restart loop is on attempt 3+ of 10, re-running the
   deterministic step-0 seed each time: per-attempt loss 309 -> 250 ->
   129 by step 30 (healthy halving). The run completes when the box
   frees; stage-3 eval is one command afterwards.

## Decision

- Early-stopping a plateaued run beats waiting 25h for a cosine tail.
- The SFT restart loop is left armed; no code changes to the running
  bash script (editing a running bash script risks misreading).

## Next steps

1. When the box frees: SFT attempt survives past step 100, completes
   400 steps; then tools/evaluate.py --checkpoint
   checkpoints_sft_38m/latest (chat_sft baseline 0/14 to beat).
2. If attempts exhaust (10): just re-run ./scripts/train_sft_38m.sh.
3. Optional: resume v3 to 60000 later for the last ~0.1 val loss.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
