# Phase 02 — SFT run đến hoàn tất

```yaml
phase: 02
goal: 400-step masked SFT completes into checkpoints_sft_38m/latest
owner: single
files:
  - logs/train_sft_38m.csv          # evidence
  - checkpoints_sft_38m/latest/     # artifact
depends: phase-01
```

## Quy trình

1. Wipe lineage poisoned: `rm -rf checkpoints_sft_38m`
2. `./scripts/train_sft_38m.sh` (auto-restart + disk guard kế thừa)
3. Monitor `logs/train_sft_38m.csv` mỗi ~10 phút:
   - kỳ vọng loss: ~309 → <5 qua 400 steps (LR 3e-5 sẽ chậm hơn cũ ở
     đầu nhưng không explosion)
   - val ở step 50/100/.../400: phải giảm đơn điệu (hoặc flat)
   - nếu val **tăng sớm** (overfit 968 convs): cắt sớm tại milestone
     gần nhất, dùng checkpoint đó cho phase 3 và ghi nhận
4. Restart-safe: checkpoint 25-step nghĩa là death ≤ mất 25 steps

## Chấp nhận

- [ ] CSV có step 400 (hoặc early-stop có chủ đích ≥ 200 steps với val flat/giảm)
- [ ] `checkpoints_sft_38m/latest/trainer_state.json` step ≥ 200
- [ ] Không emergency checkpoint mới (hoặc nếu có: reason 3x-NaN là
      dấu hiệu cần hạ LR thêm — quay lại phase 1 với lr 1e-5)

## Risks

| Risk | Mitigation |
|---|---|
| Job ngoài quay lại chiếm box | auto-restart + checkpoint 25 + attempts 30 |
| NaN quay lại cả với 3e-5 | nan-skip (phase 1) + hạ xuống 1e-5 nếu 3x-NaN |
| Loss không xuống dưới 5 | vẫn chấp nhận nếu eval chất lượng (lossmasked |
| | trên 16K vocab với tokenizer mới là chỉ báo phụ) |
