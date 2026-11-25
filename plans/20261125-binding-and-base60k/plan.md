# Plan — Binding epochs (a) + Base 60K (b): đóng gap expression-binding

```yaml
plan: 20261125-binding-and-base60k
mode: two-approaches
goal: tool positives > 0/12 — gap cuối của Miko; 2 đường độc lập có thể
      nối tiếp nhau, mỗi đường có gate riêng trước khi tiêu tài nguyên đường sau
owner: single
status: pending-cook   # user yêu cầu: CHƯA cook, chỉ plan
```

## Bối cảnh (bằng chứng đo được, không phỏng đoán)

| Chu kỳ | val | tool positives | Ghi chú |
|---|---|---|---|
| v0.4 (random-init, tool-only) | — | 0/12 | vô hiệu — bug resume |
| v2 (random-init, mix) | — | 0/12 | vô hiệu — bug resume |
| v2 fix (base thật, 1100 st) | 4.05 | 0/12 | scaffold tốt, phát nhầm `<pad>` (init chung hướng) |
| v2 rand-init (base thật) | 4.05→? | 0/12 | **phát được `I然的` id-8 (2/6 probe)** — token mở khóa |
| gym v3 (base thật, 3000 st) | **4.44** | 0/12 | chat_sft 2/14 kỷ lục; scaffold+eos sạch; binding chưa form |

Ba lớp root-cause (resume, init-norm, init-hướng) đã sửa hết; format + eos
đã chắc. Còn đúng MỘT gap: **copy số từ câu hỏi vào slot expression**.

## Hai đường — trade-off trung thực (--two)

| | **(a) Binding epochs** trên lineage gym | **(b) Base v3 → 60K** rồi SFT lại |
|---|---|---|
| Cơ sở | Tool shards 1334 w + gym v4 90737 w **đã đóng gói sẵn**; mọi cơ chế đã chạy | 6.93→~6.7 kỳ vọng; induction mạnh hơn là điều kiện nền của copy |
| Chi phí | ~3-4h train (lr 2e-5, 3000 st trên tool+v4 shards) | ~10-14h pretrain (4500 st × box chia sẻ) **+ sau đó vẫn phải SFT lại** (~4h) |
| Giả định phụ thuộc nhất | copy mechanism có thể form chỉ bằng gradient thừa (đủ epochs × data đa dạng) | val 6.7 đủ tạo induction heads render copy dễ form hơn |
| Fail-first | data tool còn non (5736 convs, số 2-6 chữ số) — nếu binding cần nhiều hơn nữa thì val giảm mà positives vẫn 0 | +4.5K steps chỉ dịch val 6.93→~6.7 (đã đo xu hướng) — có thể vẫn thiếu, phí ~14h |
| Điểm bật | rẻ, nhanh có tín hiệu, re-uses mọi thứ | root-fix thực sự nếu (a) fail |

**Khuyến nghị: (a) TRƯỚC, (b) chỉ khi (a) fail với val tốt.** (a) rẻ và
chính nó là gate quyết định cho (b): nếu sau (a) mà positives vẫn 0 với
val chat thấp → bằng chứng model cần base mạnh hơn → (b) đáng giá.

## Phases (chưa hydrate — chờ cook)

| Phase | Nội dung | Gate |
|---|---|---|
| 01 — (a) binding epochs | Continuation từ `checkpoints_sft_v3/latest` (không fresh!): data = `shards_sft_tool` (1334) mix `shards_sft_v4` (90737, đã chứa gym), lr 2e-5 cosine, warmup 5%, 3000 steps, out_dir riêng `checkpoints_binding_a`; milestone 500 để checkpoint sớm | **tool positives > 0** hoặc val tool < 3.0 tại step 500/1500/3000 |
| 02 — (a) finalize | Probe + full gates + export `dist/miko-binding-a` + codex demo; nếu positives > 0 → MODEL_CARD v0.8 + đóng plan; nếu 0 → chuyển (b) | positives > 0 |
| 03 — (b) base 60K | `./scripts/train_38m_v3.sh` (launcher + guard đã sẵn) resume 55500 → 60000; theo dõi val 6.93→? | val ≤ 6.80 (không chấp nhận flat) |
| 04 — (b) SFT lại trên base mới | Fresh lineage từ base 60K (init per-special random chuẩn), data = tool + v4 + chat, lr 2e-5, 3000-4000 steps | positives > 0 + chat_sft ≥ 2/14 |

## Kịch bản thất bại

| Hiện tượng | Hành động |
|---|---|
| (a) val tool không giảm dưới 4 | data binding quá khó → scale số operands 1-3 chữ số trước, tăng dần (curriculum) |
| (a) positives 0 dù val < 3 | copy cần induction base → kích (b) với bằng chứng này |
| (b) val flat ~6.9 sau 2000 steps | dừng sớm, giữ 55500, ghi journal — 67M tokens là trần của dataset hiện có |
| OOM-kill lại giữa chừng | resume-from-latest (đã chứng minh) + giảm eval_batches |

## Không-goals

- Không đụng server/CLI (đã xong v0.7)
- Không scale dataset mới (dùng data có sẵn; curriculum số là thay đổi
  generator duy nhất nếu cần)
- Không multi-cycle tự động vượt 3000 steps (mỗi phase có gate dừng rõ)
