# Plan status — SFT completion & eval (20261120)

Cập nhật: 2026-11-22 (chu kỳ v3 đang chạy)

## Phases

| Phase | Trạng thái | Kết quả |
|---|---|---|
| 01 — SFT stability hardening | ✅ done | nan-skip, mean-init specials, lr 3e-5, ckpt-25 (commit 5da1edd/b81963f) |
| 02 — SFT run 2000 steps | ✅ done | val 222→27.57, không emergency (checkpoints_sft_38m/latest step-2000) |
| 03 — eval + CLI + finalize | ⚠️ executed | Formal acceptance một phần: chat_sft 1/14 > 0 ✓; tool ≥12/16 ✗ (positives 0/12); smoke ✗. 3 bug thật sửa + push (e910374, e59c3b6, 07d8fd9). MODEL_CARD v0.4/v0.5, journals |

## Chu kỳ gắn thêm (ngoài plan gốc, theo ủy quyền tự chủ)

- **v0.4 continuation**: +800 steps tool-only (seed v0.2, 1241 convs) → format
  tool-call học trọn; copy số + forgetting chưa giải quyết (base-val 37.8)
- **v0.5 / SFT v2**: fresh lineage + restate-then-call + mix 70/30 (2000 steps,
  val 11.23) → gates vẫn fail: tool positives 0/12, base-val 31.96
- **v0.6 / SFT v3 (ĐANG CHẠY)**: copy-gym 7999 convs + tool v0.4 5736 + mix
  50/50, lr 1e-5, 3000 steps từ base. Decision rule: val < ~15 mà copy probe
  fail + val còn giảm → kéo +1000 steps; gates đạt → v0.6 finalize; fail →
  kích contingency (c): resume pretraining v3 → 60K steps
  (`scripts/train_38m_v3_60k.sh`)

## Số liệu tổng hợp các chu kỳ

| Cycle | val | chat_sft | tool positives | base-val |
|---|---|---|---|---|
| v0.4 (step 2800) | 27.57* | 0/14 | 0/12 | 37.77 |
| v2 (step 2000) | 11.23 | 1/14 | 0/12 | 31.96 |
| v3 (step 3000) | đang chạy | — | — | — |

*v0.4 val tính trên shards_sft (chat seed); v2/v3 val trên mix shards — không
so sánh trực tiếp được, chỉ dùng nội bộ chu kỳ.

## Engineering rút ra (root-cause fixes đã push)

1. e910374 — score trên specials-visible text + eval system prompts khớp training
2. e59c3b6 — evaluate() bỏ qua window toàn -100 (val NaN)
3. 07d8fd9 — generation prompt header-only (khớp encode_chat_masked; đây là
   root cause của degenerate continuations trước đó)
4. b602ffc — regression suite (9 invariants) + gates runner một-lệnh
