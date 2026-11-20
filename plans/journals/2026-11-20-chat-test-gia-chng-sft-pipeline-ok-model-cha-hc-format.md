---
title: "Chat test giữa chừng SFT: pipeline OK, model chưa học format"
date: 2026-11-20
summary: "Test chat.py + agent loop với checkpoint SFT step ~160; pipeline hoạt động đúng, model output newline-only vì warmup chưa xong; ETA đà tăng do box chia sẻ với train_detector"
---

# Chat test giữa chừng SFT: pipeline OK, model chưa học format

## What happened

User yêu cầu test chat với Miko giữa chừng SFT run (mean-init relaunch,
max_steps 2000). Kết quả với `checkpoints_sft_38m/latest` (step ~160):

| Test | Output |
|---|---|
| SFT + chat template "Xin chào, bạn là ai?" | toàn newline |
| SFT + `--tools` "123*456?" | 1 step, 0 tool_call, stopped_reason=final — agent loop cơ chế OK |
| Base(step-55500) + chat template | ` # # # #` (special tokens 4-9 chưa từng có trong pretraining) |
| Base + prompt in-distribution "Hà Nội là thủ đô của" | `Các một số lượng của các nước và các nước...` — có cấu trúc tiếng Việt cục bộ, vô nghĩa (đúng val 6.82/ppl 917) |

Pipeline chat hoàn chỉnh chạy không lỗi: chat.py → encode_for_model →
generate → decode; agent loop → parse_tool_call → registry. Mystery "CLI
không output" thực ra là `Miko:` + 96 newline — tail chỉ hiện dòng trống.

## Decision

- Không dry-run lại evaluate.py trước khi train xong: pipeline đã được
  verify từ baseline run (vi 0.06 / code 0.08-0.17 / chat 0 / tool 0.25)
  — tiết kiệm CPU cho trainer.
- Không can thiệp job ngoài `src.training.train_detector` (99% CPU, project
  khác của user) — trainer Miko đã nice 10, đúng thiết kế chia sẻ box.
- Kỳ vọng format chat bám sau khi warmup 500 steps xong (hiện LR ~38% peak).

## Observations

- Loss SFT giảm đều: 320 → 141 → 124 → 99 → 91 → 71 → 74 (step 190),
  chưa có emergency checkpoint (nan-skip không kích hoạt).
- Throughput: 70 → 39 tok/s khi job ngoài tăng tải. 30 steps gần nhất
  ≈ 55s/step → 1810 steps còn lại ≈ 18-28 giờ (hơn ước tính 3-6h ban đầu).
- Disk 2.7GB free (95%): đủ cho steady-state SFT (keep_last=2 + latest +
  milestones ≈ 1.9GB); prune đã hoạt động (step-00000125 bị prune).

## Next steps

- Chờ run 2000 steps hoàn tất (auto-restart 30 attempts armed).
- Phase 3: evaluate.py full suite → so chat_sft vs 0/14, tool_basic vs 0.25.
- CLI smoke `123*456` end-to-end kỳ vọng `[tool] calculator -> 56088`.
- MODEL_CARD v0.4 + archive plan + push.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
