---
title: "SFT v2 eval gates: không đạt — copy-binding là giới hạn cấu trúc ở 38M"
date: 2026-11-21
summary: "v2 (restate + mix 70/30, fresh lineage): val 11.23 nhưng tool positives vẫn 0/12, base-val 31.96 — hai scaffold xác nhận number-copy không hình thành ở 38M/2M SFT tokens; dừng theo gate, 3 lựa chọn cho user"
---

# SFT v2 eval gates: không đạt — copy-binding là giới hạn cấu trúc ở 38M

## What happened

SFT v2 hoàn tất (2000 steps, ~12.6h trên box chia sẻ, val 11.23 vs 27.57
của v0.4 cùng mốc). Eval gates trên checkpoints_sft_v2/latest:

| Gate | Kết quả |
|---|---|
| chat_sft > 0 | 1/14 ✓ (formal) |
| tool positives > 0/12 | **0/12 ✗** |
| vi ≥ 0.03 | 0/16 ✗ (mất đúng 1 item so với base) |
| code ≥ 0.04 | 1/12 = base ✓ |
| base-val (forgetting) | **31.96** vs 6.93 base ✗ (30% mix không đủ) |
| smoke 123*456 | ✗ (bỏ call, "Kết quả là 10 server server...") |

Điểm sáng: restate-scaffold ĐÃ được emit ("Để tính (67 + 93 × 3 ...") —
surface pattern học được; chỉ có nội dung số là không bind được.

## Đọc kết quả (2 scaffolds, 2 data scales, cùng kết cục)

1. **Number-copy binding không hình thành ở 38M / ~2M SFT tokens**:
   model tạo được mọi surface pattern (preface, JSON scaffold, restate)
   nhưng không thể copy chuỗi số từ context vào slot. Đây là hạn chế
   của induction heads yếu ở quy mô này + lượng SFT tokens quá nhỏ so
   với chuẩn (thường cần 10-100M+).
2. **Forgetting vẫn còn dù mix 30% LM**: lr 3e-5 trên 2M tokens
   áp đảo 600K replay tokens. Cần 50/50+ hoặc lr thấp hơn nhiều.

## Decision (gate f — không tự mở chu kỳ mới)

Dừng huấn luyện, ghi nhận trung thực, đưa 3 lựa chọn:
(a) chấp nhận format-tier đạt được, archive M4, review M5/M8;
(b) scale data mạnh (10-20K tool convs + copy-task pretraining +
    50/50 mix, lr 1e-5) — ~1-2 ngày box chia sẻ;
(c) strengthen base trước: resume pretraining v3 → 60K steps
    (induction heads tốt hơn là root-fix cho copy) rồi SFT lại.

## Artifacts

- evals/results_sft_v2_part{1,2,3}.json, MODEL_CARD v0.5 row
- checkpoints_sft_v2/latest (step 2000, val 11.23)

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
