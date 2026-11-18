---
title: "Miko-50M: masked SFT training path complete, verified seed data, suite at 70 tests"
date: 2026-11-18
summary: "Built the full masked-label SFT path (writer/dataset/--sft flag/packer), verified-answer seed generator (277 convs), fixed window-count reporting and shard-glob mixing; 70/70 tests; base run finishing step 40000"
---

# Miko-50M: masked SFT training path complete, verified seed data, suite at 70 tests

## What happened

Fifth continuation session (commits fe45496..cf05b27, pushed):

1. **Masked-label SFT end-to-end** (the tracked M4 refinement):
   - miko/dataset.py: SFTShardWriter chunks (ids, labels) conversations
     into uniform seq_len windows with SHIFTED targets (targets[i] =
     label of token i+1), uint16 ids + int16 labels parallel files,
     padding with pad_id/-100; SFTShardedDataset (mmap, seeded shuffle).
   - tools/train.py --sft swaps the dataset class; next_token_loss
     already ignores -100, so the loop needed no other change.
   - tools/pack_sft_dataset.py: masked mode default; --full-lm legacy.
   - Smoke: tiny model, 30 steps — loss 78->32 (masked CE averages only
     the few assistant tokens per window; mechanics verified).

2. **Bugs caught during SFT wiring**:
   - pack_sft_dataset reported windows as bytes/2 (128x overcount:
     '101760 windows' was really 265).
   - find_shards('*.bin') would mix legacy full-LM part-00000.bin with
     new part-00000.ids.bin in the same directory — added
     find_sft_shards('*.ids.bin') and repacked clean.

3. **Verified seed generator** (scripts/make_sft_seed.py): curated
   banks (facts/smalltalk/rewrite/titles/identity/vi-en) + arithmetic
   and code tasks whose answers are COMPUTED at generation time (exec
   in a scratch namespace with expected outputs asserted). Generated
   chat_seed_v0.2.jsonl (216 convs); combined with v0.1: 277
   conversations -> 265 train + 12 val masked windows.

4. **Tests**: 70/70 pass (26.5 min wall under load ~9 from the training
   run + external processes).

5. **Base run**: step ~36850/40000 at ~318 tok/s (shared box),
   train ~7.0, val 6.97 at 35000; ETA ~5-6h with auto-restart armed.

## Decision

- Keep full-LM packing as an option; masked SFT is the M4 default.
- Seed scale (277 convs) is a smoke scale; real SFT needs ~5-10K
  conversations — the generator banks make that a config change.

## Next steps

1. When the base hits 40000: tools/evaluate.py on the milestone, then
   launch v3 continuation (train_38m_v3.yaml --resume auto).
2. Expand seed banks toward ~5K conversations; then first masked SFT
   run from the 40000-step base (low LR, 1-3 epochs over SFT shards,
   mixed with a slice of pretraining data to limit forgetting).
3. Chat-tier eval items for the SFT model (role-format aware).

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
