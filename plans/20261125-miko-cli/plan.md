# Plan — Miko CLI theo kiểu "mượn Codex CLI" (pivot theo user)

```yaml
plan: 20261125-miko-cli
mode: fast
goal: CLI agent như Codex/Claude Code chạy trên model Miko — KHÔNG viết
      lại từ đầu: mượn @openai/codex làm frontend, Miko server nói
      giao thức OpenAI-compatible làm backend; ./miko wrapper gắn 2 bên
owner: single
status: in-progress
```

## Brainstorm contract (pivot 2026-11-25)

- **Outcome**: gõ `./miko` → server Miko tự khởi động (nói OpenAI-compat,
  streaming SSE) → Codex CLI (npm) khởi động với provider trỏ localhost →
  toàn bộ TUI của Codex (chọn model, session, approval, agent) chạy trên
  model Miko. Chọn model = `GET /v1/models` list bundles, request field
  `model` lazy-load/switch.
- **Constraints**: npm/node24 sẵn (verified 0.172.0 codex); cargo backup;
  CPU box đang train v3 song song (smoke test max_new nhỏ); tools chạy
  **server-side trong registry của mình** (transparent mode) — không trả
  `tool_calls` cho client thực thi (model 38M chưa đủ tin giao agent
  chạy shell; Codex approval vẫn canh hành động của chính nó).
- **Non-goals**: viết TUI riêng (đã bỏ sau pivot), fork+sửa source Codex
  (Rust — chỉ cấu hình qua CODEX_HOME), trả tool_calls chuẩn client-side,
  PyPI packaging.
- **Acceptance**:
  1. `GET /v1/models` trả list bundle `dist/*` dạng OpenAI
  2. `POST /v1/chat/completions` non-stream: trả đúng shape OpenAI
     (id/object/created/model/choices[0].message.content + usage)
  3. `stream: true` → SSE chunks `data: {...delta...}` + `data: [DONE]`
  4. Đổi `model` trong request → server lazy-load bundle khác (lock an
     toàn), đúng model trả trong response
  5. `./miko` wrapper: start server port auto → CODEX_HOME repo-local
     `.miko-codex/` (config.toml provider miko + base_url) → npx codex
     chạy được profile miko → thoát thì kill server sạch
  6. e2e pytest: contract OpenAI (urllib + parse SSE) pass; smoke
     transcript từ Codex session
- **Trade-offs**: 
  - Codex CLI (Rust/npm) vs Gemini CLI (TS) vs Aider (Python) — chọn
    Codex: đúng cái user gọi tên, Apache-2.0, cấu hình provider sạch
    nhất qua config.toml, không cần build (npm binary). Gemini CLI là
    plan-B nếu Codex chặn custom provider. Aider là plan-C (code-edit
    agent, nặng deps).
  - Transparent server-side tools vs protocol-faithful tool_calls —
    chọn transparent: an toàn với model nhỏ, client chỉ thấy text
    (Codex vẫn chạy shell của chính nó khi user yêu cầu — có approval).

## Phases

| Phase | Nội dung | Phụ thuộc |
|---|---|---|
| 01 | OpenAI-compat layer trên server: /v1/models, /v1/chat/completions (non-stream + SSE), lazy model switch | — |
| 02 | on_token callback trong generate + SSE stream thật; test contract | 01 |
| 03 | Codex CLI tích hợp: .miko-codex/config.toml, ./miko wrapper, smoke e2e, README | 02 |
| 04 | Gated v3 gym xong: default model = model cuối, demo transcript, journal, push | 03 + v3 |

## Cảnh báo chất lượng (trung thực)

Codex là agent CLI đầy đủ — nó sẽ gửi system prompt dài + tools schema
dạng OpenAI. Model 38M xử lý được cấu trúc nhưng nội dung agent thật
(yêu cầu sửa file, chạy lệnh) ngoài năng lực hiện tại. Sản phẩm đúng
nghĩa "CLI chạy được trên model mình" — chất lượng agent đi theo
chất lượng model khi v3+ hoàn thiện.
