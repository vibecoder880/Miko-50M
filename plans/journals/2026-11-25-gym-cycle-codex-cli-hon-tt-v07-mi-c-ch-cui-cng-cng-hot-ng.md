---
title: "Gym cycle + Codex CLI hoàn tất — v0.7, mọi cơ chế cuối cùng cũng hoạt động"
date: 2026-11-25
summary: "3000 steps val 4.44 (resume từ OOM-kill step 2170 nhờ fix latest-pointer); chat_sft 2/14 kỷ lục; Codex demo 0 lỗi; sản phẩm: server + OpenAI-compat + Responses SSE + ./miko-cli"
---

# Gym cycle + Codex CLI hoàn tất — v0.7, mọi cơ chế cuối cùng cũng hoạt động

## What happened

Gym cycle hợp lệ đầu tiên hoàn tất trên lineage đã sửa (resume + per-special
init): 3000 steps, ~3.1M tokens, val 29.9→4.44. Giữa chừng bị OOM-kill ở
step 2170 (box 3 lõi chia sẻ, swap 1.3GB) — resume-from-latest nối lại
chính xác, fix 9624a23 chứng minh giá trị thực chiến.

Kết quả gates (56-token):
- chat_sft 2/14 — KỶ LỤC dự án (v0.4: 0, v2: 1, v3-random: 0)
- regression 3/9, tool 4/16 (positives 0/12 — binding còn là frontier)
- probe: preface scaffold sạch + <eos> mọi prompt (không còn loop)

Sản phẩm CLI chốt: Codex 0.172 chạy trọn trên Miko — Responses API
SSE chain chuẩn (0 lỗi, 305 tokens), bundle hot-swap, greedy serving
mặc định. Transcript evals/codex_gym_final.txt.

## Quyết định kỹ đáng nhớ

- wire_api 'chat' đã bị xóa khỏi Codex 0.172 → phải hiện thực /v1/responses
  (SSE event chain có item_id/content_part) — hợp đồng test 7/7
- Codex gửi temperature=None → server cần --temperature force flag
- Codex hang hôm qua là box contention, không phải protocol (in-process
  test 121 events chứng minh)

## Next steps

1. Tool-call-heavy epochs (seed v0.4 5736 convs) trên lineage gym —
   binding là gap cuối; mọi cơ chế giờ hoạt động đúng.
2. (tùy chọn) resume pretraining v3 → 60K để base mạnh hơn trước SFT
   scale lớn.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
