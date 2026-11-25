# Phase 04 — (b) SFT lại trên base 60K: cơ hội binding cuối

```yaml
phase: 04
gated-by: "phase-03 đạt gate val ≤ 6.80"
files:
  - configs/train_sft_v5.yaml          # mới — fresh lineage từ base 60K
  - scripts/train_sft_v5.sh             # mới (seed từ checkpoints_38m/latest sau 60K)
  - evals/results_v5_*.json             # artifacts
depends: [phase-03]
```

## Quy trình

1. Seed fresh lineage: copy `checkpoints_38m/latest` (60K) →
   `checkpoints_sft_v5/latest`, drop optimizer, **per-special random
   init norm 19.43** (launcher v2 đã đúng — copy pattern đó).
2. Data: dùng lại `data/shards_sft_v4` (90737 w — gym + tool + chat +
   LM mix) — dữ liệu này chưa từng được thử trên base mạnh hơn.
3. `lr: 2.0e-5`, warmup 8%, `max_steps: 3000` (data ~1.15 epoch với
   base mới; giữ steps khiêm tốn vì mục tiêu là binding, không phải
   re-learn mọi thứ).
4. Gates giống phase 02 (probe + tool positives + chat_sft vs 2/14).

## Chấp nhận

- [ ] resume log step 0 với seeding chuẩn (norm + distinct hướng)
- [ ] tool positives > 0 → MODEL_CARD v0.8 + bundle `dist/miko-v5` + demo
- [ ] positives = 0 → kết luận kiến trúc/thang đo: 38M + 67M-token corpus
      không đủ cho copy binding dù mọi thứ khác chuẩn — MODEL_CARD ghi
      rõ, hướng duy nhất còn lại là dataset lớn hơn (ngoài scope hiện tại)

## Kết thúc chuỗi

Dù kết quả nào, phase 04 khép plan: journal tổng kết (a)+(b) với số liệu
đầy đủ, push, archive. Không tự mở phase mới.
