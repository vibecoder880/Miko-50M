# Plan: Hoàn tất SFT + đánh giá chat/tool tier — Miko-38M

```yaml
plan: sft-completion-and-eval
created: 2026-11-20
status: ready
mode: auto (focused continuation — codebase fully known)
outcome: Miko-38M chat tier trained (SFT complete) and scored on chat_sft
         + tool_basic benchmarks; chat CLI verified with real tool calls.
constraints:
  - shared box (external jobs throttle CPU at will; load spikes to ~10)
  - SFT lineage checkpoints_sft_38m/ currently poisoned by the NaN loop
    (every attempt dies at step 37 — deterministic, seed-42 data order)
  - v3 base checkpoint (step 55500, val ppl 917) is the SFT source
non-goals:
  - no resume of v3 to 60000 (cosine tail, ~zero gain; revisit later)
  - no M6 web/search work in this plan
  - no seed expansion beyond the current 968 conversations
acceptance:
  - SFT completes 400 steps with loss < 5 (from 309 start)
  - evals/results_sft_38m.json written with chat_sft + tool_basic scores
  - chat_sft score strictly above the 0/14 pre-SFT baseline
  - tools/chat.py --tools answers a calculator question end-to-end
```

## Bối cảnh & chẩn đoán

SFT run của M4/M5 hiện **chết lặp lại ở step 37** trên mọi attempt
(`non-finite loss`, emergency checkpoint lưu đúng nguyên nhân). Ba yếu tố
gây explosion:

1. **LR 1e-4 quá cao** cho landscape chưa từng thấy chat tokens
   (initial masked CE ~309 — base model confidently wrong).
2. **Warmup 10% (40 steps)** — explosion xảy ra đúng cuối warmup (step 37).
3. **Emergency-stop dừng ngay** ở NaN đầu tiên → mất toàn bộ tiến trình
   vì `checkpoint_every=100` > lifetime (~37 steps).

Cùng lúc, box đã rảnh (load 0.83, 339 tok/s) — cửa sổ hoàn tất SFT đang mở.

## Phases

| Phase | File | Nội dung |
|---|---|---|
| 1 | [phase-01-sft-stability.md](phase-01-sft-stability.md) | Hardening: LR/warmup, nan-skip, checkpoint dày, attempts |
| 2 | [phase-02-sft-run.md](phase-02-sft-run.md) | Chạy SFT 400 steps đến hoàn tất |
| 3 | [phase-03-eval-and-cli.md](phase-03-eval-and-cli.md) | Eval chat/tool tier + CLI smoke + finalize |

## Execution order

```
Phase 1 (code + tests, ~30 min)
   ↓ wipe lineage, relaunch
Phase 2 (training ~1-2h khi box rảnh; auto-restart giữ tiến trình)
   ↓ 400 steps xong
Phase 3 (eval + smoke + commit ~30 min)
```

## Verification tổng

- `logs/train_sft_38m.csv`: 400 rows train, loss cuối < 5
- `evals/results_sft_38m.json`: chat_sft > 0 (baseline 0/14)
- `python tools/chat.py --checkpoint checkpoints_sft_38m/latest --tools
  --prompt "123*456=?"`: in ra `[tool] calculator... -> 56088`
- Full test suite vẫn pass (~98 tests)
