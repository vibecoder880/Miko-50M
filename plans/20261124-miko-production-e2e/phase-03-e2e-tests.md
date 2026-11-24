# Phase 03 — E2E tests + demo transcript

```yaml
phase: 03
files:
  - tests/test_e2e_server.py
  - scripts/e2e_demo.sh
  - evals/e2e_demo_transcript.txt   # artifact (git-tracked)
depends: [01, 02]
```

## Test e2e (pytest, in-process)

- Fixture: start server trong thread trên port 0 (HTTP server thật,
  model thật từ bundle mặc định `dist/miko-sft-v2` — skip nếu bundle
  chưa export khi chạy trước phase 04), dùng `urllib.request` (không
  dependency mới).
- Cases:
  1. `/health` 200 + status ok
  2. `/v1/info` 200 + có params, tools chứa "calculator"
  3. `/v1/chat` no-tools: 200, reply str không rỗng, usage > 0 tokens
  4. `/v1/chat` tools với câu toán: 200, `tool_calls` là list; nếu model
     emit call hợp lệ → có "result" từ registry (assert nhẹ để không phụ
     thuộc chất lượng model: chỉ assert shape + stopped_reason str)
  5. body rác → 400; path lạ → 404
- Runtime budget: mỗi test < 60s (max_new_tokens nhỏ 24).

## Demo script

`scripts/e2e_demo.sh`: export bundle (nếu chưa) → start server background
(port 8765) → 3 request (health, chào hỏi tiếng Việt, "123*456 bằng bao
nhiêu?" với tools) → lưu body + status vào transcript → kill server.
Transcript là evidence cho phase 04.

## Chấp nhận

- [ ] `pytest tests/test_e2e_server.py -q` pass (hoặc skip rõ lý do khi
      bundle chưa có)
- [ ] transcript chứa ≥ 3 response, có dòng tool call (hoặc note model
      chưa emit call — trung thực ghi)
