---
title: "Miko-50M: tool_call eval metric + 16-item tool benchmark"
date: 2026-11-19
summary: "Added the static tool-call eval tier (name+args scoring, negatives, whitespace-insensitive expressions); v3 at step 56000/60000 val 6.68"
---

# Miko-50M: tool_call eval metric + 16-item tool benchmark

## What happened

Tenth session (commit pushed). v3 near completion: step 56000/60000,
val falling steadily on the vi-heavy corpus (6.80 -> 6.68).

1. **metric=tool_call** in miko/evaluation.py — scores the generated
   text as the expected tool call: parse_tool_call + exact tool name +
   argument match (whitespace-insensitive for string args so
   '123 * 456' == '123*456'). expected_tool=__none__ negative items
   check the model does NOT call for trivial/identity questions.
2. **evals/tool_basic.jsonl** — 16 items: 10 calculator expressions
   (multi-digit ops, parens, powers), 2 time calls, 4 negatives.
3. Loader bug caught: 'needs prompt and expected' fired for tool_call
   rows before the metric branch — check order fixed.

## Next steps

1. v3 completes (~4h) -> milestone eval -> ./scripts/train_sft_38m.sh
   (trains on the 968-conv chat+tool set) -> score chat_sft (base
   0/14) and tool_basic (base 0/16 expected).
2. After first SFT numbers, tune seed mixture / LR / steps.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
