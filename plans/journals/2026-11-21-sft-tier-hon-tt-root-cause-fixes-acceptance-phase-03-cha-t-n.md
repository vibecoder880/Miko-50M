---
title: SFT tier hoàn tất + root-cause fixes; acceptance phase-03 chưa đạt — next cycle đã thiết kế
date: 2026-11-21
summary: "SFT 2800 steps: format tool-call học trọn vẹn; 3 bug thật sửa (scoring specials, eval system prompts, generation prompt newline); copy-binding + forgetting là 2 gap còn lại; MODEL_CARD v0.4 honest"
---

# SFT tier hoàn tất + root-cause fixes; acceptance phase-03 chưa đạt — next cycle đã thiết kế

## What happened

SFT chain hoàn tất: 2000-step masked chat SFT (mean-init special embeddings,
val 222→27.57) + 800-step tool-only continuation (seed v0.2, 1241 verified
convs, 6 operator families). Phase-3 eval với fair harness (56-token budget):

| Suite | step-2000 (cũ, prompt lệch) | step-2800 (fair) |
|---|---|---|
| chat_sft | 1/14 | 0/14 |
| chat_basic | 0/8 | 0/8 |
| vi_basic | 2/16 | 0/16 |
| code_basic | 2/12 | 1/12 |
| tool_basic | 4/16 (negatives) | 4/16 (positives 0/12) |
| base-val | 6.93 (base) | 37.77 |
| tool-val | — | 3.54 (masked CE) |

Điểm sáng: format tier học trọn vẹn — model emit đúng
`Để tính chính xác, Miko dùng máy tính.\n<tool_call>\n{"name":
"calculator", "arguments": {"expression": "..."}}\n</tool_call>`,
có item hoàn hảo tới </tool_call>.

## Root causes tìm được (3 bug thật, đã fix + push)

1. **evaluate.py score trên specials-stripped text** → parse_tool_call
   không bao giờ thấy <tool_call> → positives unwinnable (e910374).
2. **Eval items thiếu system prompt** training luôn có → OOD (e910374).
3. **Generation prompt mismatch (root cause degenerate output)**:
   encode_chat(add_generation_prompt=True) + chat.py append
   `<assistant>` + NEWLINE, nhưng encode_chat_masked (training) là
   [<assistant>, *content, <eos>] KHÔNG newline → transition chưa từng
   được train → model rơi vào `{` P=0.996. Diagnostic: P(id8)=0.69 khi
   context khớp system-prompt text; P({)=0.996 sau header+newline sai.
   Fix: header-only prompt (07d8fd9). Sau fix model emit preface + id8
   + JSON scaffold chuẩn.

## Two capability gaps còn lại (không phải bug)

1. **Copy-binding chưa hình thành**: expression slot emit digit soup,
   không copy số từ câu hỏi — dù 800 steps thêm trên 1241 convs,
   train loss 3.5 (fit được data, không generalize sang số mới).
   Chu kỳ sau: restate-then-call preface ("Để tính 123*456 chính
   xác,...") — copy ngắn into preface rồi JSON copy near-adjacent.
2. **Forgetting**: tool-only cycle đổi chat/vi/code lấy tool format
   (base-val 6.93→37.8). Checkpoint step-2000 đã bị prune (milestone
   per-process). Chu kỳ sau bắt buộc mix 70% SFT + 30% pretraining.

## Decision

- MODEL_CARD v0.4 ghi số honest + 2 gap + next-cycle design.
- Chấp nhận phase-03: chat_sft>0 ✓ (formal), tool ≥12/16 ✗, smoke ✗,
  tests ✓, push ✓ — automation dừng sau finalize; plan archive.
- SIGILL thoáng qua trên base-val với OMP=2 (OMP=1 OK) — ghi nhận,
  không chặn.

## Next steps

1. Chu kỳ SFT v2: seed v0.3 (restate-then-call) + mix 70/30, train
   từ base step-55500 (không kế thừa forgetting), 2000 steps, lr 3e-5.
2. Đánh giá lại; nếu tool positives > 0 → smoke + MODEL_CARD v0.5.
3. Sau đó: M8 agent-loop polish / resume pretraining v3 nếu muốn base
   mạnh hơn (optional deferred).

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
