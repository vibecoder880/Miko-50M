# Phase 03 — Eval chat/tool tier + CLI smoke + finalize

```yaml
phase: 03
goal: score the SFT model, verify the tool loop end-to-end, ship
owner: single
files:
  - evals/results_sft_38m.json      # artifact
  - MODEL_CARD.md                   # version row update
depends: phase-02
```

## Quy trình

### 1. Post-SFT eval

```bash
.venv/bin/python tools/evaluate.py \
  --checkpoint checkpoints_sft_38m/latest \
  --tokenizer tokenizer/real_v0.2/tokenizer.json \
  --data-dir data/shards_sft/val --eval-batches 16 \
  --out evals/results_sft_38m.json
```

So sánh với baseline pre-SFT (step-55500 base):
- chat_sft: 0/14 → mục tiêu ≥ 7/14 (0.5)
- tool_basic: 0.25 (chỉ negatives) → mục tiêu: 12+/16 (gọi đúng tool
  + expression khớp; negatives vẫn phải pass)
- vi/code: không được tụt thê thảm so với base (catastrophic forgetting
  check; nếu tụt mạnh → mix pretraining data vào SFT ở lần sau)

### 2. CLI smoke với tool thật

```bash
python tools/chat.py --checkpoint checkpoints_sft_38m/latest \
  --tokenizer tokenizer/real_v0.2/tokenizer.json --tools \
  --prompt "123*456 bằng bao nhiêu?"
```

Kỳ vọng: `[tool] calculator(...) -> 56088` + câu trả lời chứa 56088.
Nếu model emit call sai format: parser/repair path (agent loop) xử lý —
ghi nhận số lần repair trong smoke.

### 3. Finalize

- MODEL_CARD.md: thêm row v0.4 (SFT tier) với số eval
- Commit eval JSON + code changes (phase 1) + MODEL_CARD
- Push origin/main
- /ak-journal

## Chấp nhận (mapping plan.md)

- [ ] chat_sft > 0
- [ ] tool_basic ≥ 12/16
- [ ] CLI smoke end-to-end (tool trace + đáp án đúng)
- [ ] Full test suite pass trước commit cuối
- [ ] Đã push

## Kịch bản thất bại

| Hiện tượng | Hành động |
|---|---|
| chat_sft vẫn ~0 sau SFT | kiểm tra: model có emit format `<assistant>` đúng không |
| | qua tools/chat.py; nếu format đúng mà nội dung sai → cần |
| | nhiều data (mở rộng seed) chứ không phải train thêm |
| tool calls sai expression thường xuyên | thêm cặp Q→call trong seed với biểu thức |
| | đa dạng hơn; 1 epoch SFT thêm chỉ trên tool data |
| Catastrophic forgetting (vi/code tụt >50%) | SFT lại với mix 70% SFT + 30% |
| | pretraining shards |
