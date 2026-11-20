# Phase 01 — SFT stability hardening

```yaml
phase: 01
goal: SFT training survive its own loss landscape and external restarts
owner: single (this session)
files:
  - configs/train_sft_38m.yaml      # LR, warmup, checkpoint interval
  - tools/train.py                  # nan-skip instead of instant abort
  - scripts/train_sft_38m.sh        # attempts budget
tests:
  - pytest tests/ -q                # no regression (~98 tests)
  - manual: relaunch SFT, pass step 37-50 without emergency stop
```

## Thay đổi

### 1. configs/train_sft_38m.yaml

```yaml
train:
  lr: 3.0e-5            # was 1.0e-4 — SFT adapts format, not capabilities;
                        # 3e-5 is 3x below pretrain peak... actually 10x below
                        # the exploding 1e-4; standard SFT range
  warmup_pct: 0.25      # was 0.10 — explosion hit at warmup end (step 37/40)
  checkpoint_every: 25  # was 100 — attempts died at ~37 steps; a 25-step
                        # checkpoint preserves progress across restarts
```

### 2. tools/train.py — nan-skip (bounded)

Emergency-stop trên NaN **đầu tiên** quá gay gắt với SFT (OOD loss spike
là bình thường). Đổi sang:

- NaN loss đơn lẻ → bỏ batch (`opt.zero_grad`, không backward), log warning,
  tăng counter.
- **3 lần NaN liên tiếp** → emergency-stop như cũ (mất ổn định thật).

Vị trí: trong loop `for x, y in batches`, trước `(ce / grad_accum).backward()`.
Giữ nguyên emergency checkpoint cho trường hợp 3x.

### 3. scripts/train_sft_38m.sh — attempts 10 → 30

10 attempts × ~20 phút lifetime không đủ vượt một đợt load kéo dài của
job ngoài. 30 attempts ≈ 10 giờ cửa sổ.

## Verification

- [ ] `pytest tests/ -q` pass (không regression)
- [ ] SFT chạy qua step 37-60 không emergency (loss giảm đều, < 129 mốc cũ)
- [ ] CSV có val row ở step 50 (eval_every=50)

## Rollback

Revert 3 file; lineage cũ đã poisoned nên phải wipe
`checkpoints_sft_38m/` bất kể rollback hay không.
