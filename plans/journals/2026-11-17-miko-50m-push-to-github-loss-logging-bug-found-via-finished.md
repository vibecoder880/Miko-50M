---
title: "Miko-50M: push to GitHub, loss-logging bug found via finished 38M run, training extended"
date: 2026-11-17
summary: "Pushed all 8 commits; 38M run finished 20000 steps (41M tokens, 14.2h, ~802 tok/s); caught double-scaled train-loss logging (0.44 was really 7.15) and biased first-N val sampling; fixed both, extended run to 40000 steps"
---

# Miko-50M: push to GitHub, loss-logging bug found via finished 38M run, training extended

## What happened

1. **Push** — user asked for detailed commits + push: 8 commits
   (621b66a..0b063b5) are on origin/main.

2. **38M run finished** — 20000 steps / 40.96M tokens in 51068s
   (~802 tok/s avg on the shared box). Val trajectory healthy:
   67.2 -> 8.23 monotonic-ish plateau.

3. **Measurement bug caught** — CSV train loss 0.44 was impossible for
   <1 epoch of unseen data. Re-measured the final checkpoint: train
   windows 7.27, val windows 7.35 (spread) — model did NOT memorize;
   the logged value was CE/grad_accum (division applied twice:
   loss/grad_accum accumulated, then /accum_steps). 0.44 x 16 = 7.15
   reconciles. Past logs corrected retroactively: tiny-v2 '1.69' ->
   6.76 real; stage-1 '1.93' -> 7.72 real.

4. **Eval sampling bias** — evaluate() averaged the first N val windows
   (shard-0 head = one long article): 8.23 vs 7.35 spread. Fixed with
   deterministic evenly-spaced positions (np.linspace).

5. Both fixed (commit 9db2823), 55/55 tests pass (35 min under load ~8
   from another process), full run CSV preserved as evidence
   (evals/train_38m_full.csv), MODEL_CARD v0.2 row added.

6. **Training extended** — resumed from step-00020000 with
   --max-steps 40000 (nohup nice 10). Cosine schedule re-anchors on the
   new horizon (LR rises 3e-5 -> ~1.7e-4 mid-curve, standard practice).

## Decision

- Trust RSS numbers and long-run CSV averages; short benchmark tok/s on
  this shared box varies 3-14x with load.
- Keep per-run raw CSVs in evals/ as versioned evidence.

## Next steps

1. ~14h: 40000-step run completes (~82M tokens, 1.7 epochs); then
   re-run tools/evaluate.py + eval suite on the new milestone.
2. dataset-v0.3: more vi text (target 50% mixture) + JS/Go/SQL.
3. M4 instruction tuning prep once base val < ~6.5.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
