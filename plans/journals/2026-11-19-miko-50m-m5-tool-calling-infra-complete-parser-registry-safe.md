---
title: "Miko-50M: M5 tool-calling infra complete — parser, registry, safe_calc, verified tool seed"
date: 2026-11-19
summary: "Built the full tool tier (plan §28-29, §36): <tool_call> parser failing safe, schema-validating registry, ast-based safe_calc (rejects code injection), tool role in chat template with <tool_result>, 289 verified tool conversations; combined SFT set now 968 convs"
---

# Miko-50M: M5 tool-calling infra complete — parser, registry, safe_calc, verified tool seed

## What happened

Ninth session (commit 3f20bfc, pushed). v3 pretraining meanwhile
improved to ~415 tok/s (external job ended) and its val is falling on
the vi-heavy corpus: 6.91 -> 6.85 -> 6.81 -> 6.80 by step 45500 —
evidence the v0.3 mixture helps Vietnamese. ~14K steps left.

1. **miko/tooling.py** — plan §28-29 + §36:
   - ToolRegistry: schema_prompt() renders tool definitions for the
     model context; validate_call implements plan §36 checks 1-4
     (existence, shape, schema types, required args), with int->float
     leniency for numeric params.
   - parse_tool_call: extracts the first <tool_call>{json}</tool_call>
     from assistant text; any malformation (bad JSON, non-object,
     missing name) returns an error dict — the runtime never crashes.
   - safe_calc: ast-walk evaluator allowing only numeric constants and
     + - * / // % ** with parens; names, calls, attributes, conditionals
     are structurally impossible (test-fuzzed with injection strings).
   - run_tool: validate -> execute -> return error strings instead of
     raising into the agent loop.

2. **Chat template M5 extension** — role 'tool' valid, rendered via the
   day-one <tool_result> token (plan §33); masked in SFT labels (tool
   results are runtime evidence, not model output). The plan §33 full
   shape (user -> assistant+tool_call -> tool -> assistant) round-trips
   through encode/decode with correct masking.

3. **Verified tool seed** (scripts/make_tool_seed.py): 289
   conversations. Large-operand arithmetic (3-4 digit multiplication,
   6-digit sums, parens, powers) forces the model to delegate; two-call
   chains; error-recovery examples (truncated JSON, wrong arg name)
   teaching the repair path of plan §36; time examples; no-tool
   negatives (model must not call when trivially answerable).
   Verification pass re-executes every parseable call in the data and
   asserts equality with the stored tool result.

4. **Combined SFT set**: 968 conversations / 66K tokens -> 1007 train +
   39 val masked windows.

## Decision

- Tool results as role 'tool' with <tool_result> header (not inline in
  assistant content): keeps assistant spans (what the model must learn
  to emit) cleanly separated from runtime evidence (masked).
- calculator stays the only real dependency of the seed; time added as
  the second no-side-effect tool. search/fetch (M6) come later with
  their own adapters and much sparser training data.

## Next steps

1. v3 completes (~7h at current rate) -> ./scripts/train_sft_38m.sh
   (now trains on chat+tool data together — M4+M5 in one SFT pass).
2. Add tool eval items (evals/tool_basic.jsonl) scoring: does the
   model emit a parseable call, correct tool, correct expression, and
   use the result in the final answer.
3. After first SFT: measure chat_sft + tool scores; iterate.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
