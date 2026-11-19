---
title: "Miko-50M: bounded agent loop (M8 foundation) with repair path"
date: 2026-11-19
summary: run_agent with max_steps/max_tool_calls bounds and the plan §36 repair path; the malformed-JSON test caught the loop accepting broken calls as final answers — fixed by feeding errors back
---

# Miko-50M: bounded agent loop (M8 foundation) with repair path

## What happened

Eleventh session (commit pushed). v3 at step 56000/60000, val plateau
~6.68 (falling from 6.80) — ~4h left.

Built miko/agent.py — the M8 agent-loop foundation:

- run_agent(generate_fn, registry, user_message, config): bounded loop,
  injected generator (testable without a live model), tool calls via
  the fail-safe executor, context trimming, forced plain answer when
  max_steps trips.
- **Design bug caught by test**: the first implementation treated a
  malformed <tool_call> JSON block as a *final answer* (parse returns
  an error dict; the loop's 'no call => final' branch swallowed it).
  Exactly the failure mode plan §36 warns about. Fixed: parse errors
  are fed back as tool turns (repair), bounded at 2 consecutive errors.
- encode_for_model: token-budgeted encoding preserving the system turn.
- 7 scripted-generator tests, all pass.

The stack now: tooling (registry/parser/safe_calc) + chat template
(system/user/assistant/tool) + agent loop + eval tiers (chat_sft,
tool_basic with negatives). Only the model behind generate_fn is
missing — SFT runs next once v3 completes.

## Next steps

1. v3 completes -> milestone eval -> ./scripts/train_sft_38m.sh.
2. Wire the real model into run_agent via a generate_fn built from
   generate() + encode_for_model; smoke the calculator flow end-to-end.
3. Score chat_sft + tool_basic; iterate seed mixture.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
