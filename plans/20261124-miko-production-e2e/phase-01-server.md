# Phase 01 — Serving core: `miko/server.py`

```yaml
phase: 01
files:
  - miko/server.py            # stdlib HTTP server, module entry (python -m miko.server)
  - tests/test_server_unit.py # handler logic không cần HTTP thật
depends: []
```

## Thiết kế

- `ThreadingHTTPServer` + `BaseHTTPRequestHandler`; JSON in/out; UTF-8.
- Server state (load 1 lần khi start):
  - model = `load_model(bundle_or_checkpoint)`; `model.eval()`; `torch.set_num_threads(2)`
  - tokenizer = `MikoTokenizer.load`
  - `threading.Lock` bao mọi lần generate (single model, CPU box)
  - registry = `make_default_registry()`
- Endpoints:
  - `GET /health` → `{"status":"ok","step":<trainer_state.step>}`
  - `GET /v1/info` → `{"name":"miko","version":<VERSION file hoặc "dev">,"params":N,"context_length":512,"tools":["calculator","time"]}`
  - `POST /v1/chat` body:
    `{"messages":[{"role","content"}...], "tools":bool, "max_new_tokens":int(96), "temperature":0.7}`
    → `{"reply":str, "tool_calls":[{"name","arguments","result"}...], "stopped_reason":str, "usage":{"prompt_tokens","new_tokens","elapsed_s"}}`
    `messages` rỗng hoặc sai shape → 400; lỗi generate → 500 + `{"error":...}`.
- Chat path tái dùng y nguyên `tools/chat.py` logic (không copy-paste:
  tách `make_generate_fn` vào dùng chung qua import; nếu chat.py cản trở
  thì import hàm sẵn có `miko.agent.run_agent` + `encode_for_model` như chat.py đang làm).
- Generation config: temperature/top_k/top_p/repetition_penalty mặc định
  như chat.py; request có thể override temperature & max_new_tokens (clamp
  8..256).
- CLI: `python -m miko.server --bundle dist/miko-xxx | --checkpoint DIR
  --tokenizer FILE [--host 127.0.0.1] [--port 8000]`; log 1 dòng/request
  (method, path, ms, status) ra stderr.

## Chấp nhận

- [ ] `python -m miko.server --help` chạy, args đúng
- [ ] unit test: request parse (hợp lệ/400), response shape, lock bao generate
- [ ] không thêm dependency mới (import list chỉ stdlib + miko + torch)
