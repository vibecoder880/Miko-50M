---
title: "Phase-04 hoàn tất: token <tool_call> lần đầu xuất hiện — v0.6 đóng"
date: 2026-11-24
summary: "1100 steps random-direction init: 2/6 probe emit id-8 thật (lần đầu dự án); gates vẫn 0 positives (expression binding) nhưng cơ chế call đã tồn tại; e2e demo cuối từ bundle step-1100; MODEL_CARD v0.6 + full tests 117 pass"
---

# Phase-04 hoàn tất: token <tool_call> lần đầu xuất hiện — v0.6 đóng

## What happened

SFT random-init lineage hoàn tất (1100 steps, ~1h box rảnh, val 4.05).

Probe đột phá: 2/6 emit token <tool_call> (id 8) THẬT — lần đầu sau
6 chu kỳ (trước: 0 suốt). Fix per-special random-direction init mở
khóa token (trước đó specials giống hệt nhau → model phát <pad>).

Gates 56-token: tool 4/16 (positives 0/12 — expression vẫn là template
rác "(self):"), chat_sft 0/14, regression 2/9 (2 negatives vacuous).
E2E demo cuối (bundle step-1100): contract pass đầy đủ, server-mode
nội dung degenerate (sampling + seed cố định khác greedy probe — ghi
nhận trung thực). 117 tests pass.

## Phân tích còn thiếu gì để đạt tool positives

1. Expression binding: model copy CHƯA form. Data copy-gym v3 chu kỳ
   trước dạy echo/restate với lr 1e-5 chưa đủ; giờ token đúng rồi,
   cần chạy dài hơn (3000-4000 steps) trên mix nhiều gym hơn.
2. Server-mode degenerate khác greedy probe: temperature 0.7 + seed 42
   cố định trúng chuỗi xấu; nên hỗ trợ temperature=0 qua API và demo
   dùng greedy.

## Next steps (kế thừa plan)

- Chu kỳ 3000+ steps trên mix gym-nặng (giờ token reachable) HOẶC
  scale base trước (v3 → 60K).
- server: cho phép temperature=0/greedy qua request; demo greedy.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
