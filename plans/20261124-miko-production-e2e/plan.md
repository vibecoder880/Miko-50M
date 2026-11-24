# Plan — Miko Production E2E (serve + tool-calling product)

```yaml
plan: 20261124-miko-production-e2e
mode: fast
goal: Miko chạy như sản phẩm: HTTP API chat + tool thật, bundle model
      versioned, e2e test chứng minh full path HTTP → model → tool → answer
owner: single
status: completed
```

## Brainstorm contract

- **Outcome**: `python -m miko.server` phục vụ `/v1/chat` (chat tiếng Việt +
  tool-calling calculator/time chạy thật) trên box CPU 1GB; model đóng gói
  thành bundle versioned; e2e test + demo transcript chứng minh đường full.
- **Constraints**: CPU-only (box đang train song song), RAM serving ~1GB,
  venv hiện có (torch/tokenizers/safetensors — KHÔNG thêm dependency),
  single-model instance (serialize generate bằng lock), chất lượng model
  còn giới hạn → e2e gate theo **cấu trúc**, chất lượng tách băng eval suites.
- **Non-goals**: web UI, auth/multi-tenant, streaming SSE, Docker/K8s,
  GPU, multi-model serving.
- **Acceptance**:
  1. `GET /health` 200; `GET /v1/info` trả version + params + context
  2. `POST /v1/chat` (không tools) → reply str + usage, tiếng Việt, có eos
  3. `POST /v1/chat` (tools) → `tool_calls[]` chứa kết quả thực thi registry
     khi model emit call hợp lệ; stopped_reason có nghĩa
  4. `tests/test_e2e_server.py` pass (in-process, port 0, không cần network)
  5. `scripts/e2e_demo.sh` xuất transcript vào `evals/e2e_demo_transcript.txt`
  6. MODEL_CARD v0.6 row (SFT + serve) + journal + push
- **Trade-offs**: stdlib `http.server` vs FastAPI — chọn stdlib: 0 dependency
  mới, đủ cho 1 box, tránh network install; FastAPI chỉ thắng khi cần
  OpenAPI/validation đa client (chưa có case thứ hai). GIả định chính:
  single-box nội bộ; fail-first: khi cần expose public thì phải thêm
  reverse proxy + auth (ngoài scope).

## Phases

| Phase | Nội dung | Phụ thuộc |
|---|---|---|
| 01 | `miko/server.py` — stdlib HTTP, load model 1 lần, lock, 3 endpoints | — |
| 02 | `scripts/export_bundle.py` — bundle versioned `dist/miko-*` | — |
| 03 | E2E pytest + `scripts/e2e_demo.sh` (transcript artifact) | 01, 02 |
| 04 | Chạy e2e trên model SFT cuối (sau khi train xong) + MODEL_CARD v0.6 + journal + push | 03 + SFT run |

## Chất lượng model vs e2e

E2E gate cơ khí (contract HTTP + tool execution + shape) **phải pass**.
Băng chất lượng (copy số đúng, chat_sft score) do `scripts/eval_gates_v3.sh`
đo riêng và ghi vào MODEL_CARD — không chặn sản phẩm.
