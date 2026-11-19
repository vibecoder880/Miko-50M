---
title: "Miko-50M: base run finished at step 40000, v0.3 continuation live"
date: 2026-11-19
summary: "40000 steps / 81.92M tokens done: val 6.93 ppl 1023 (-38% vs step 20000); benchmark flat as expected pre-SFT; v0.3 continuation resumed on 67.7M-token corpus"
---

# Miko-50M: base run finished at step 40000, v0.3 continuation live

## What happened

Seventh session (commit 585c1e7, pushed):

1. **Base v2 run finished** — 40000 steps / 81.92M tokens (~1.7 epochs
   of dataset-v0.2), clean exit, final val 6.88 (in-training).
2. **Milestone eval** — spread-sampled val 6.93 / ppl 1023 (vs 7.41 /
   1646 at 20000). Eval-suite scores flat (vi 0.06, code 0.08, chat 0):
   a base LM without instruction tuning continues text rather than
   answering QA — consistent with plan §56 positioning; SFT is the
   lever, and the chat_sft items (0/14) quantify the pre-SFT floor.
3. **v0.3 continuation launched** — ./scripts/train_38m_v3.sh resumed
   from step-00040000 onto dataset-v0.3 (67.7M tokens, vi 56.6%) with
   the auto-restart wrapper; +20000 steps to 60000 (~14-30h on the
   shared box).

## Decision

- Treat benchmark 'contains' scores as SFT-sensitive metrics; the base
  model's health is judged by val loss (healthy: 67 -> 8.23 -> 6.88).
- v3 exists to deepen Vietnamese before SFT (mixture vi 56.6% vs 38.9%).

## Next steps

1. Monitor v3 (logs/train_38m_v3.csv); milestone eval at 50000/60000.
2. After v3: first real masked SFT run from the base checkpoint
   (low LR ~1e-4, 2-4 epochs over the seed, mix pretraining slice).
3. Expand SFT banks toward ~5K conversations to make SFT meaningful.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
