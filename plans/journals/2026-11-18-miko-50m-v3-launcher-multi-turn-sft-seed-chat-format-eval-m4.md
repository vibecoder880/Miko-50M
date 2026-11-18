---
title: "Miko-50M: v3 launcher, multi-turn SFT seed, chat-format eval — M4 fully staged"
date: 2026-11-18
summary: "Added guarded v3 continuation launcher, multi-turn conversations in the verified seed (285 convs), chat-format eval items for the SFT tier; 73/73 tests; base run at step ~36900/40000 under heavy external load"
---

# Miko-50M: v3 launcher, multi-turn SFT seed, chat-format eval — M4 fully staged

## What happened

Sixth continuation session (commit 0aff543 + journal, pushed):

1. **scripts/train_38m_v3.sh** — guarded launcher for the v0.3
   continuation: refuses to start while the v2 run is still alive
   (both write checkpoints_38m/), auto-restart loop, one command to
   switch when the base hits 40000.

2. **Multi-turn SFT data** — make_sft_seed.py gained add_multi():
   math follow-up chains (2+3=5, double it, thanks), code sessions
   where the example output is COMPUTED via the verified-exec helper,
   fact-clarification chains. Seed: 285 conversations -> 273 train +
   12 val masked windows. Real multi-turn coverage was the missing
   piece for SFT realism.

3. **Chat-format evaluation** — EvalItem.format=raw|chat: chat items
   render system(+optional)+user through the chat template ending in
   the assistant header, so the SFT model is scored AS AN ASSISTANT.
   evals/chat_sft.jsonl ships 14 items (identity w/ system, facts,
   math, code, politeness, closings). Unknown formats rejected at load.

4. **Tests**: 73/73 pass (33.7 min wall under load ~11 — the box is
   running a heavy external job alongside training).

5. **Base run**: step 36900/40000 at ~283 tok/s under that same load;
   ETA ~6h. val 6.97 @ 35000, trajectory healthy. Auto-restart armed.

## Decision

- Keep raw-format items for base-model evals; chat items only score the
  SFT tier — categories stay separate in the results JSON.
- The v3 switch is deliberately a manual command (guarded), not an
  automatic chain, so the operator sees the step-40000 eval first.

## Next steps

1. Base hits 40000 -> tools/evaluate.py milestone eval -> ./scripts/
   train_38m_v3.sh.
2. Expand SFT banks toward ~5K conversations (multi-turn generator is
   the template to scale).
3. First masked SFT run from the 40000-step base (low LR, few epochs,
   mixed with pretraining slice).

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
