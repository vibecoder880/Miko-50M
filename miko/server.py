"""HTTP serving for Miko (plan 20261124-miko-production-e2e, phase 01).

A deliberately small product surface: stdlib ``http.server`` (no new
dependencies), the model loaded once, a lock serializing generation
(single CPU model), and three endpoints::

    GET  /health   -> {"status": "ok", "step": N}
    GET  /v1/info  -> {"name": "miko", "version": ..., "params": N,
                       "context_length": 512, "tools": ["calculator", "time"]}
    POST /v1/chat  -> {"reply": str, "tool_calls": [...],
                       "stopped_reason": str, "usage": {...}}

Run::

    python -m miko.server --bundle dist/miko-sft-v2 [--port 8000]
    python -m miko.server --checkpoint checkpoints_sft_v2/latest \
        --tokenizer tokenizer/real_v0.2/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import torch

from .agent import AgentConfig, encode_for_model, run_agent
from .openai_compat import OpenAICompatService
from .chat_template import ChatMessage
from .checkpoint import load_model
from .generation import GenerationConfig, generate
from .tokenizer import MikoTokenizer
from .tooling import make_default_registry
from .utils import get_logger, seed_everything

logger = get_logger("miko.server")

DEFAULT_SYSTEM = (
    "Bạn là Miko, một trợ lý nhỏ nói tiếng Việt. Trả lời ngắn gọn, rõ ràng. "
    "Khi cần tính toán chính xác, dùng công cụ calculator theo định dạng: "
    '<tool_call>\n{"name": "calculator", "arguments": {"expression": "..."}}\n</tool_call>'
)


class MikoService:
    """Model + tokenizer + registry, loaded once; generation serialized."""

    def __init__(self, checkpoint: str, tokenizer_path: str,
                 version: str = "dev", threads: int = 2) -> None:
        torch.set_num_threads(threads)
        seed_everything(42)
        self.model = load_model(checkpoint)
        self.model.eval()
        self.tokenizer = MikoTokenizer.load(tokenizer_path)
        self.registry = make_default_registry()
        self.version = version
        self.step = 0
        state_path = os.path.join(checkpoint, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as fh:
                self.step = int(json.load(fh).get("step", 0))
        self._lock = threading.Lock()
        self.max_context = 448

    # -- generation -----------------------------------------------------
    def _generate_fn(self, max_new_tokens: int, temperature: float):
        model, tok = self.model, self.tokenizer
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40, top_p=0.9, repetition_penalty=1.1,
            eos_token_id=tok.eos_id,
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

    # -- public API -----------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "name": "miko",
            "version": self.version,
            "step": self.step,
            "params": self.model.param_count(),
            "context_length": self.model.config.context_length,
            "tools": self.registry.names(),
        }

    def chat(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Returns (http_status, response_body)."""
        raw = payload.get("messages")
        if not isinstance(raw, list) or not raw:
            return 400, {"error": "'messages' must be a non-empty list"}
        messages: list[ChatMessage] = []
        for m in raw:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                return 400, {"error": "each message needs 'role' and 'content'"}
            try:
                messages.append(ChatMessage(str(m["role"]), str(m["content"])))
            except ValueError as e:
                return 400, {"error": str(e)}

        tools = bool(payload.get("tools", False))
        try:
            max_new = int(payload.get("max_new_tokens", 96))
            # temperature 0.0 = greedy (generate() maps <= 0 to greedy)
            temperature = float(payload.get("temperature", 0.7))
        except (TypeError, ValueError):
            return 400, {"error": "max_new_tokens/temperature must be numeric"}
        max_new = min(max(8, max_new), 256)
        temperature = min(max(0.0, temperature), 2.0)

        t0 = time.time()
        prompt_tokens = len(encode_for_model(self.tokenizer, list(messages),
                                             max_tokens=self.max_context))
        with self._lock:  # one model, one CPU box: serialize generation
            gen = self._generate_fn(max_new, temperature)
            try:
                if tools:
                    res = run_agent(gen, self.registry, messages[-1].content,
                                    AgentConfig(max_steps=4, max_tool_calls=4),
                                    system=DEFAULT_SYSTEM)
                    body = {
                        "reply": res.answer,
                        "tool_calls": res.tool_calls,
                        "stopped_reason": res.stopped_reason,
                    }
                else:
                    if messages[0].role != "system":
                        messages = [ChatMessage("system", DEFAULT_SYSTEM),
                                    *messages]
                    text = gen(messages)
                    body = {"reply": text, "tool_calls": [],
                            "stopped_reason": "final"}
            except RuntimeError as e:  # generation blew up (OOM etc.)
                logger.exception("generation failed")
                return 500, {"error": f"generation failed: {e}"}

        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "new_tokens": max_new,  # bound; exact count needs decode cost
            "elapsed_s": round(time.time() - t0, 2),
        }
        return 200, body


def make_handler(service: MikoService,
                 compat: OpenAICompatService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MikoHTTP/1"

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path == "/health":
                self._send(200, {"status": "ok", "step": service.step})
            elif self.path == "/v1/info":
                self._send(200, service.info())
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": compat.list_models()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/v1/chat/completions":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    self._send(400, {"error": {"message": "invalid JSON body",
                                               "type": "invalid_request_error"}})
                    return
                t0 = time.time()
                status, body, stream_fn = compat.complete(payload)
                logger.info("POST /v1/chat/completions -> %d (%.0f ms)",
                            status, (time.time() - t0) * 1000)
                if stream_fn is not None:
                    self.send_response(status)
                    self.send_header("Content-Type",
                                     "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for chunk in stream_fn():
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                    return
                self._send(status, body)
                return
            if self.path == "/v1/responses":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    self._send(400, {"error": {"message": "invalid JSON body",
                                               "type": "invalid_request_error"}})
                    return
                t0 = time.time()
                status, body, stream_fn = compat.responses(payload)
                logger.info("POST /v1/responses -> %d (%.0f ms)",
                            status, (time.time() - t0) * 1000)
                if stream_fn is not None:
                    self.send_response(status)
                    self.send_header("Content-Type",
                                     "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for chunk in stream_fn():
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                    return
                self._send(status, body)
                return
            if self.path != "/v1/chat":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "invalid JSON body"})
                return
            t0 = time.time()
            status, body = service.chat(payload)
            logger.info("%s %s -> %d (%.0f ms)", self.command, self.path,
                        status, (time.time() - t0) * 1000)
            self._send(status, body)

        def log_message(self, fmt: str, *args) -> None:  # quiet stdlib lines
            pass

    return Handler


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    """(checkpoint, tokenizer, version) from --bundle or explicit paths."""
    if args.bundle:
        ckpt = os.path.join(args.bundle, "checkpoint")
        tok = os.path.join(args.bundle, "tokenizer.json")
        ver_path = os.path.join(args.bundle, "VERSION")
        version = "dev"
        if os.path.exists(ver_path):
            with open(ver_path, encoding="utf-8") as fh:
                version = fh.read().strip().splitlines()[0]
        return ckpt, tok, version
    return (args.checkpoint or "checkpoints_sft_v2/latest",
            args.tokenizer or "tokenizer/real_v0.2/tokenizer.json",
            "dev")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Serve Miko over HTTP")
    ap.add_argument("--bundle", help="versioned model bundle dir")
    ap.add_argument("--checkpoint", help="checkpoint dir (if no --bundle)")
    ap.add_argument("--tokenizer", help="tokenizer.json (if no --bundle)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--bundles-root", default="dist",
                    help="dir holding serving bundles (OpenAI /v1/models)")
    args = ap.parse_args(argv)

    ckpt, tok, version = resolve_paths(args)
    if not os.path.exists(os.path.join(ckpt, "model.safetensors")):
        logger.error("no model at %s — export a bundle or pass --checkpoint", ckpt)
        return 1
    service = MikoService(ckpt, tok, version=version, threads=args.threads)
    compat = OpenAICompatService(bundles_root=args.bundles_root,
                                 threads=args.threads)
    httpd = ThreadingHTTPServer(
        (args.host, args.port), make_handler(service, compat))
    logger.info("miko %s (step %d) serving on http://%s:%d — /health /v1/info /v1/chat /v1/models /v1/chat/completions",
                version, service.step, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
