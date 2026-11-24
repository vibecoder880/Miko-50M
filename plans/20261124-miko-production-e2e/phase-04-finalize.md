# Phase 04 — E2E trên model cuối + finalize

```yaml
phase: 04
files:
  - evals/e2e_demo_transcript.txt   # final artifact
  - MODEL_CARD.md                   # v0.6 row
depends: [03, "SFT v2 rerun (2 root-cause fixes) hoàn tất"]
```

## Quy trình

1. Chờ SFT run xong (watcher tự đánh thức; `logs/train_sft_v2.csv` đạt
   2000 và completion count tăng).
2. Probe copy (`scripts/probe_copy.py`) + eval gates (tách lệnh ≤560s).
3. `scripts/export_bundle.py --checkpoint checkpoints_sft_v2/latest
   --out dist/miko-sft-v2` → chạy `scripts/e2e_demo.sh` → transcript.
4. MODEL_CARD v0.6: SFT số thật (sau 2 fix) + serve product row.
5. `pytest tests/ -q` full pass → commit chi tiết → push → journal.

## Chấp nhận

- [ ] transcript e2e từ model cuối tồn tại (3 response)
- [ ] MODEL_CARD v0.6 với số đo thật
- [ ] full test suite pass
- [ ] pushed + journal

## Kịch bản thất bại

| Hiện tượng | Hành động |
|---|---|
| Model vẫn không emit `<tool_call>` | e2e vẫn pass băng cơ khí; MODEL_CARD ghi hạn chế + nguyên nhân (đã biết norm-matched init là fix đúng hướng, có thể cần chu kỳ nữa) |
| Server lỗi RAM/thấp | giảm threads=1, max_new_tokens mặc định 64 |
| Box bận làm e2e chậm | giảm max_new_tokens trong demo, chạy đêm |
