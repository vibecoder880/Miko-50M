---
title: "Miko-50M: dataset-v0.3 + chat template + SFT packer + run-resilience fixes"
date: 2026-11-18
summary: "Built dataset-v0.3 (67.7M tokens, vi 56.6%), M4 chat template with SFT masking + packer + 61-conv seed, fixed silent-crash resume (max_steps pin), 38M run continuing to 40000"
---

# Miko-50M: dataset-v0.3 + chat template + SFT packer + run-resilience fixes

## What happened

Fourth continuation session (all pushed to origin/main):

1. **dataset-v0.3 (67.69M tokens)** — scripts/fetch_corpus.py gained
   --skip N to fetch fresh rows from an already-partially-fetched
   stream; fetched 11000 new vi-wiki docs (skip 7000, zero overlap
   with v0.2), reused v0.2 en/code files. Mixture measured: vi 56.6% /
   code 26.2% / en 17.2% (target 50/20/25/5 — vi a bit over). Same 16K
   tokenizer -> the 38M checkpoint continues directly.
   configs/train_38m_v3.yaml prepared (+20000 steps, same run lineage).

2. **M4 prep** — miko/chat_template.py: renders system/user/assistant
   with the reserved tokens (ids 4-6), per-turn <eos>,
   encode_chat_masked() produces SFT labels aligned 1:1 (assistant
   content + closing eos labeled; first implementation had a
   labels/ids misalignment — caught before commit). tools/
   pack_sft_dataset.py packs chat JSONL into shards (full-LM SFT, no
   loop changes needed). data/sft/chat_seed_v0.1.jsonl: 61
   hand-written Vietnamese conversations (~2K tokens). 9 new tests.

3. **Run resilience** — the 38M run died silently right after the
   step-35000 checkpoint during heavy box load (silent-kill, likely
   OOM; journal needs root). train_38m.sh now auto-restarts (10
   attempts, 60s backoff). Second failure mode caught immediately: the
   restart omitted --max-steps, so the run read max_steps=20000 from
   the yaml and exited at step 35001 with 'training complete' — a
   silent one-step run. Pinned --max-steps 40000 in the script.

4. Run status: resumed at 35000, continuing to 40000 (~400 tok/s under
   shared-box load; ETA 5-7h). Val at 35000: 6.97 (ppl ~1060) — down
   from 8.23 at 20000. Trajectory healthy.

## Decision

- v0.3 continuation waits until the v2 run hits 40000 (one command:
  python tools/train.py --config configs/train_38m_v3.yaml --resume auto).
- Full-LM SFT on chat streams first (reuses the whole pipeline);
  masked-label SFT deferred as an M4 refinement.

## Next steps

1. At step 40000: tools/evaluate.py + eval suite on the milestone,
   then launch the v3 continuation (above command).
2. Expand SFT seed toward plan §27 scale (10K conversations) before M4
   fine-tuning; consider mixing pretraining data into SFT to avoid
   forgetting.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
