# Miko CLI — Codex chạy trên model Miko

`./miko-cli` mở **OpenAI Codex CLI** (mượn nguyên, không fork) chạy trên
model Miko cục bộ: server `miko/server.py` nói giao thức
OpenAI-compatible (`/v1/models`, `/v1/chat/completions`, `/v1/responses`
SSE), Codex gọi vào qua provider tự cấu hình.

## Chạy

```bash
./miko-cli                    # agent TUI (mặc định model dist/miko-sft-v2)
./miko-cli exec "2+2?"        # one-shot, không TUI
MIKO_PORT=8800 ./miko-cli     # đổi port
```

Lần đầu: tự cài `@openai/codex` vào `.miko-codex-cli/` (git-ignored),
tự export bundle nếu chưa có, tự viết `$CODEX_HOME/config.toml`
(repo-local `.miko-codex/`, cũng git-ignored). Thoát Codex → server tự tắt.

## Chọn model

Model list = các bundle trong `dist/` (mỗi bundle là output của
`scripts/export_bundle.py`). Hai cách:

```bash
# 1. Trong TUI Codex: /model -> chọn "miko-sft-v2" (hoặc bundle khác)

# 2. Request thẳng:
curl -s localhost:8877/v1/models | jq '.data[].id'
curl -s localhost:8877/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"miko-sft-v2","messages":[{"role":"user","content":"xin chào"}],"temperature":0}'
```

Server lazy-load + hot-swap model theo field `model` của từng request
(an toàn luồng: lock bao cả swap lẫn generate).

## Kiến trúc

```
Codex TUI (npm, .miko-codex-cli/)
   │  wire_api = "responses" (SSE)
   ▼
miko/server.py  ── /v1/responses, /v1/chat/completions, /v1/models
   │  lazy model registry (dist/*)
   ▼
miko/openai_compat.OpenAICompatService ── GenerationConfig(on_token=…)
   │
   ▼
MikoModel 38M + tooling registry (tools chạy server-side, transparent)
```

- **Tools** (calculator/time) chạy trong registry server — client chỉ
  thấy text cuối. Trả `tool_calls` cho client tự chạy là non-goal có
  chủ đích (model 38M chưa đủ tin điều phối shell của agent).
- **Codex vẫn chạy shell/sandbox của chính nó** với approval mặc định —
  phần agent-filesystem là của Codex, không phải model.

## Giới hạn trung thực

Codex gửi system prompt dài + tool schema OpenAI; model 38M hiểu cấu
trúc nhưng năng lực agent thật (sửa file, chạy lệnh đúng) phụ thuộc
chất lượng model — đang cải thiện qua các chu kỳ SFT (xem MODEL_CARD
v0.6 và chu kỳ gym đang chạy). CLI này là **sản phẩm dây chuyền**:
model mạnh tới đâu, agent tới đó.

## Kiểm thử

```bash
.venv/bin/python -m pytest tests/test_openai_compat.py -q   # 7 contract tests
./miko-cli exec --skip-git-repo-check -s read-only "Nói xin chào"   # e2e thật
```

Transcript e2e chuẩn: `evals/codex_cli_transcript.txt` (0 protocol
error, tokens used 305).

## Bundles hiện có

| Bundle | Nguồn | Ghi chú |
|---|---|---|
| `miko-sft-v2` | SFT 1100 steps (chat+tool mix) | lần đầu emit `<tool_call>` |
| `miko-gym-interim` | gym cycle step-600 (tạm, frozen) | chỉ để so sánh tiến trình |

Sau khi chu kỳ gym 3000 steps hoàn tất, bundle cuối sẽ thay thế làm
mặc định (`dist/miko-sft-gym`).
