# Phase 03 — (b) Base v3 → 60K: pretraining continuation

```yaml
phase: 03
gated-by: "phase-02 quyết định (a) fail — KHÔNG chạy nếu (a) đạt"
files:
  - scripts/train_38m_v3.sh           # đã có, guard trainer-khác đã thêm
  - logs/train_38m_v3.csv             # artifact (append)
depends: [phase-02]
```

## Quy trình

1. Xác nhận không còn trainer nào: `pgrep -f tools/train[.]py` rỗng
   (launcher đã tự guard + từ chối).
2. `nohup ./scripts/train_38m_v3.sh` — resume 55500 → 60000 trên
   data/shards_v3 (67.7M tokens, vi 56.6%), Adafactor, disk guard 700MB.
3. Giám sát: mỗi 500 steps ghi val; **gate val ≤ 6.80 tại 60000**.
   Nếu val flat (≥ 6.88) sau 2000 steps tiếp → early-stop có chủ đích,
   giữ base 55500, journal "dataset đã tới trần".
4. Milestone 60000 tự giữ (`milestone_every` cấu hình sẵn).

## Chấp nhận

- [ ] "resumed from ... step 55500" trong log
- [ ] val tại 60000 ≤ 6.80 (hoặc early-stop có bằng chứng flat ≥ 3 mốc)
- [ ] không emergency checkpoint

## Lưu ý box

- 4500 steps × ~80-290 tok/s (chia sẻ) = 10-14h. Chạy qua đêm là hợp lý.
- Adafactor RSS ~800MB — an toàn hơn AdamW cho run dài trên box 6GB.
