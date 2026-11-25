"""OpenAI-compatible API layer for Miko (plan 20261125-miko-cli phase 01).

Lets existing CLIs (Codex, Gemini CLI, Aider, ...) drive the Miko model
by speaking the de-facto standard protocol:

    GET  /v1/models              -> {"data": [{"id": "miko-sft-v2", ...}]}
    POST /v1/chat/completions    -> OpenAI chat completion (stream optional)

Design notes:
- ``model`` field selects a serving bundle under ``bundles_root`` (default
  ``dist/``); the service lazy-loads and hot-swaps under a lock.
- Streaming uses SSE ``data: {...}`` chunks with ``delta.content`` and a
  terminating ``data: [DONE]``.
- Tools execute SERVER-SIDE through ``miko.tooling`` (transparent mode):
  the client only sees the final text. Returning protocol-faithful
  ``tool_calls`` for client-side execution is a deliberate non-goal —
  the 38M model is not trusted to drive a client agent's shell.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Optional

import torch

from .agent import AgentConfig, encode_for_model, run_agent
from .chat_template import ChatMessage
from .checkpoint import load_model
from .generation import GenerationConfig, generate
from .tokenizer import MikoTokenizer
from .tooling import make_default_registry
from .utils import get_logger, seed_everything

logger = get_logger("miko.openai")

DEFAULT_SYSTEM = (
    'Bạn là Miko, một trợ lý nhỏ nói tiếng Việt. Trả lời ngắn gọn, rõ ràng. Khi cần tính toán chính xác, dùng công cụ calculator theo định dạng: <tool_call>\n{"name": "calculator", "arguments": {"expression": "..."}}\n</tool_call>'
)


def _read_version(bundle_dir: str) -> tuple[str, int]:
    ver = "dev"
    step = 0
    path = os.path.join(bundle_dir, "VERSION")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            ver = fh.readline().strip() or "dev"
    state = os.path.join(bundle_dir, "checkpoint", "trainer_state.json")
    if os.path.exists(state):
        with open(state, encoding="utf-8") as fh:
            step = int(json.load(fh).get("step", 0))
    return ver, step


class OpenAICompatService:
    """Model registry + chat-completion engine for the OpenAI wire format."""

    def __init__(self, bundles_root: str = "dist", threads: int = 2) -> None:
        torch.set_num_threads(threads)
        seed_everything(42)
        self.bundles_root = bundles_root
        self.registry = make_default_registry()
        self._lock = __import__("threading").Lock()
        self._loaded: Optional[str] = None
        self._model = None
        self._tokenizer: Optional[MikoTokenizer] = None
        self.max_context = 448

    # -- registry -------------------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        data = []
        if os.path.isdir(self.bundles_root):
            for name in sorted(os.listdir(self.bundles_root)):
                ckpt = os.path.join(self.bundles_root, name, "checkpoint")
                if os.path.exists(os.path.join(ckpt, "model.safetensors")):
                    data.append({
                        "id": name,
                        "object": "model",
                        "owned_by": "miko",
                        "created": 0,
                    })
        return data

    def _ensure_model(self, model_id: str) -> str:
        """Load the requested bundle if needed; returns the loaded id."""
        if self._loaded == model_id and self._model is not None:
            return model_id
        bundle = os.path.join(self.bundles_root, model_id)
        ckpt = os.path.join(bundle, "checkpoint")
        if not os.path.exists(os.path.join(ckpt, "model.safetensors")):
            raise KeyError(f"unknown model: {model_id}")
        model = load_model(ckpt)
        model.eval()
        tok = MikoTokenizer.load(os.path.join(bundle, "tokenizer.json"))
        with self._lock:
            self._model, self._tokenizer = model, tok
            self._loaded = model_id
        logger.info("loaded model %s", model_id)
        return model_id

    # -- generation -----------------------------------------------------
    def _generate_fn(self, max_new_tokens: int, temperature: float,
                     on_token=None):
        model, tok = self._model, self._tokenizer
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40, top_p=0.9, repetition_penalty=1.1,
            eos_token_id=tok.eos_id,
            on_token=on_token,
        )

        def generate_fn(messages: list[ChatMessage]) -> str:
            ids = encode_for_model(tok, list(messages),
                                   max_tokens=self.max_context)
            ids = ids + [tok.token_to_id("<assistant>")]
            input_ids = torch.tensor([ids], dtype=torch.long)[
                ..., -model.config.context_length:]
            out = generate(model, input_ids, cfg=cfg)
            new_ids = out[0, input_ids.shape[1]:].tolist()
            return tok.decode(new_ids, skip_special_tokens=False)

        return generate_fn

    @staticmethod
    def _parse_messages(raw: Any) -> list[ChatMessage]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("'messages' must be a non-empty list")
        out = []
        for m in raw:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError("each message needs 'role' and 'content'")
            out.append(ChatMessage(str(m["role"]), str(m["content"])))
        return out

    def complete(self, payload: dict[str, Any]):
        """One chat-completion. Returns (status, body, stream_fn).

        ``stream_fn`` (when the request asked for streaming) yields SSE
        ``data:`` lines; the caller wires it to the HTTP response.
        """
        try:
            messages = self._parse_messages(payload.get("messages"))
        except ValueError as e:
            return 400, {"error": {"message": str(e), "type": "invalid_request_error"}}, None

        model_id = str(payload.get("model") or self._default_model_id())
        try:
            self._ensure_model(model_id)
        except KeyError as e:
            return 404, {"error": {"message": str(e), "type": "invalid_request_error"}}, None

        try:
            max_new = int(payload.get("max_tokens") or 64)
            temperature = float(payload.get("temperature", 0.7))
        except (TypeError, ValueError):
            return 400, {"error": {"message": "max_tokens/temperature must be numeric",
                                   "type": "invalid_request_error"}}, None
        max_new = min(max(8, max_new), 256)
        temperature = 0.0 if temperature is None else min(max(0.0, temperature), 2.0)

        use_tools = bool(payload.get("tools")) and payload.get("tool_choice") != "none"
        prompt_tokens = len(encode_for_model(self.tokenizer, list(messages),
                                             max_tokens=self.max_context))
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def finish(reply: str, tool_calls: list, reason: str, new_tokens: int):
            body = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": new_tokens,
                    "total_tokens": prompt_tokens + new_tokens,
                },
                "miko": {"tool_calls": tool_calls, "stopped_reason": reason},
            }
            return 200, body, None

        stream = bool(payload.get("stream"))

        if not use_tools:
            if stream:
                def stream_fn():
                    parts: list[str] = []

                    def on_token(tok_id: int) -> None:
                        text = self.tokenizer.decode([tok_id],
                                                     skip_special_tokens=False)
                        if text:
                            parts.append(text)
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0,
                                             "delta": {"content": text}}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    with self._lock:
                        gen = self._generate_fn(max_new, temperature,
                                                on_token=on_token)
                        if messages[0].role != "system":
                            msgs = [ChatMessage("system", DEFAULT_SYSTEM),
                                    *messages]
                        else:
                            msgs = list(messages)
                        gen(msgs)
                    yield "data: [DONE]\n\n"
                return 200, None, stream_fn
            with self._lock:
                gen = self._generate_fn(max_new, temperature)
                msgs = list(messages)
                if msgs[0].role != "system":
                    msgs = [ChatMessage("system", DEFAULT_SYSTEM), *msgs]
                reply = gen(msgs)
            return finish(reply, [], "stop", max_new)

        # tools: server-side agent loop (transparent), never streamed
        with self._lock:
            gen = self._generate_fn(max_new, temperature)
            res = run_agent(gen, self.registry, messages[-1].content,
                            AgentConfig(max_steps=4, max_tool_calls=4),
                            system=DEFAULT_SYSTEM)
        reason = "tool_calls" if res.tool_calls else "stop"
        return finish(res.answer, res.tool_calls, reason, max_new)

    def _default_model_id(self) -> str:
        models = self.list_models()
        return models[0]["id"] if models else "miko"

    # ------------------------------------------------------------------
    # OpenAI Responses API (POST /v1/responses) — the wire format Codex
    # 0.172+ requires (wire_api = "responses").
    # ------------------------------------------------------------------
    def _responses_messages(self, payload: dict[str, Any]) -> list[ChatMessage]:
        """Map Responses request -> ChatMessage list (instructions=system)."""
        msgs: list[ChatMessage] = []
        instr = payload.get("instructions")
        if isinstance(instr, str) and instr.strip():
            msgs.append(ChatMessage("system", instr))
        raw_input = payload.get("input")
        if isinstance(raw_input, str):
            msgs.append(ChatMessage("user", raw_input))
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, dict):
                    role = str(item.get("role", "user"))
                    content = item.get("content")
                    text = ""
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = "".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict))
                    if role not in ("system", "user", "assistant"):
                        role = "user"
                    if text.strip():
                        msgs.append(ChatMessage(role, text))
        if not msgs:
            raise ValueError("'input' must carry at least one message")
        return msgs

    def responses(self, payload: dict[str, Any]):
        """One Responses-API call. Returns (status, body, stream_fn)."""
        try:
            messages = self._responses_messages(payload)
        except ValueError as e:
            return 400, {"error": {"message": str(e),
                                   "type": "invalid_request_error"}}, None

        model_id = str(payload.get("model") or self._default_model_id())
        try:
            self._ensure_model(model_id)
        except KeyError as e:
            return 404, {"error": {"message": str(e),
                                   "type": "invalid_request_error"}}, None

        try:
            max_new = int(payload.get("max_output_tokens") or 64)
            temperature = payload.get("temperature")
            temperature = 0.7 if temperature is None else float(temperature)
        except (TypeError, ValueError):
            return 400, {"error": {"message": "max_output_tokens/temperature"
                                   " must be numeric",
                                   "type": "invalid_request_error"}}, None
        max_new = min(max(8, max_new), 256)
        temperature = min(max(0.0, temperature), 2.0)

        resp_id = f"resp-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        prompt_tokens = len(encode_for_model(self.tokenizer, list(messages),
                                             max_tokens=self.max_context))

        def final_body(text: str):
            return {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "status": "completed",
                "model": model_id,
                "output": [{
                    "type": "message",
                    "id": f"msg-{uuid.uuid4().hex[:12]}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                }],
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": max_new,
                    "total_tokens": prompt_tokens + max_new,
                },
            }

        stream = bool(payload.get("stream"))

        def _gen_reply() -> str:
            if messages[0].role != "system":
                msgs = [ChatMessage("system", DEFAULT_SYSTEM), *messages]
            else:
                msgs = list(messages)
            gen = self._generate_fn(max_new, temperature)
            return gen(msgs)

        if stream:
            def stream_fn():
                def sse(event: str, data: dict) -> str:
                    return (f"event: {event}\n"
                            f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
                msg_id = f"msg-{uuid.uuid4().hex[:12]}"
                yield sse("response.created",
                          {"type": "response.created",
                           "response": {"id": resp_id, "object": "response",
                                        "status": "in_progress",
                                        "model": model_id}})
                yield sse("response.output_item.added",
                          {"type": "response.output_item.added",
                           "output_index": 0,
                           "item": {"id": msg_id, "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress", "content": []}})
                yield sse("response.content_part.added",
                          {"type": "response.content_part.added",
                           "item_id": msg_id, "output_index": 0,
                           "content_index": 0,
                           "part": {"type": "output_text", "text": ""}})

                parts: list[str] = []

                def collect(tok_id: int) -> None:
                    text = self.tokenizer.decode([tok_id],
                                                 skip_special_tokens=False)
                    if text:
                        parts.append(text)

                with self._lock:
                    gen = self._generate_fn(max_new, temperature, on_token=collect)
                    if messages[0].role != "system":
                        msgs = [ChatMessage("system", DEFAULT_SYSTEM), *messages]
                    else:
                        msgs = list(messages)
                    gen(msgs)
                # Re-stream the finished text in protocol-shaped deltas
                # (burst at end; true incremental needs an async bridge).
                text = "".join(parts)
                for ch in text:
                    yield sse("response.output_text.delta",
                              {"type": "response.output_text.delta",
                               "item_id": msg_id, "output_index": 0,
                               "content_index": 0, "delta": ch})
                yield sse("response.output_text.done",
                          {"type": "response.output_text.done",
                           "item_id": msg_id, "output_index": 0,
                           "content_index": 0, "text": text})
                yield sse("response.content_part.done",
                          {"type": "response.content_part.done",
                           "item_id": msg_id, "output_index": 0,
                           "content_index": 0,
                           "part": {"type": "output_text", "text": text}})
                yield sse("response.output_item.done",
                          {"type": "response.output_item.done",
                           "output_index": 0,
                           "item": {"id": msg_id, "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text",
                                                 "text": text}]}})
                yield sse("response.completed",
                          {"type": "response.completed",
                           "response": final_body(text)})
            return 200, None, stream_fn

        with self._lock:
            reply = _gen_reply()
        return 200, final_body(reply), None

    @property
    def tokenizer(self) -> MikoTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("no model loaded")
        return self._tokenizer
