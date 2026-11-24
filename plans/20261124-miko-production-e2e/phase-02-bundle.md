# Phase 02 — Model bundle versioned: `scripts/export_bundle.py`

```yaml
phase: 02
files:
  - scripts/export_bundle.py
depends: []
```

## Thiết kế

- `python scripts/export_bundle.py --checkpoint DIR --out dist/<name>`
  copy vào bundle: `model.safetensors`, `config.json`, `trainer_state.json`,
  `tokenizer.json` (từ --tokenizer, default real_v0.2), `VERSION` (1 dòng:
  name + date + step), `MODEL_CARD.md` (copy nếu có).
- Bundle là **dir phẳng** — server đọc trực tiếp, không nén (tar chỉ là
  bước tuỳ chọn ngoài plan).
- `checkpoints_sft_v2/latest` → `dist/miko-sft-v2/` (server dùng qua
  `--bundle`).

## Chấp nhận

- [ ] export xong: ls bundle đủ 5 file; VERSION chứa step đúng
- [ ] `load_model(bundle)` + tokenizer load từ bundle hoạt động
