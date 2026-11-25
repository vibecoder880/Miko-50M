"""OpenAI-compatibility contract tests (plan 20261125-miko-cli phase 01/02).

Real HTTP against the ThreadingHTTPServer with a real bundle; SSE parsing
included. Skips pre-export like the base e2e suite.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from miko.openai_compat import OpenAICompatService
from miko.server import MikoService, make_handler

BUNDLE_ROOT = os.environ.get("MIKO_E2E_BUNDLE_ROOT", "dist")
CKPT = os.path.join(BUNDLE_ROOT, "miko-sft-v2")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "checkpoint", "model.safetensors")),
    reason=f"bundle {CKPT} not exported yet",
)


@pytest.fixture(scope="module")
def base_url():
    service = MikoService(os.path.join(CKPT, "checkpoint"),
                          os.path.join(CKPT, "tokenizer.json"),
                          version="e2e-openai", threads=2)
    compat = OpenAICompatService(bundles_root=BUNDLE_ROOT, threads=2)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, compat))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", compat
    httpd.shutdown()
    httpd.server_close()


def test_models_list(base_url):
    url, _ = base_url
    with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as r:
        body = json.loads(r.read().decode("utf-8"))
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "miko-sft-v2" in ids
    assert all(m["object"] == "model" for m in body["data"])


def test_completion_non_stream(base_url):
    url, _ = base_url
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps({
            "model": "miko-sft-v2",
            "messages": [{"role": "user", "content": "Xin chào!"}],
            "max_tokens": 16,
            "temperature": 0.0,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.headers["Content-Type"].startswith("application/json")
        body = json.loads(r.read().decode("utf-8"))
    assert body["object"] == "chat.completion"
    assert body["model"] == "miko-sft-v2"
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant" and isinstance(msg["content"], str)
    assert body["usage"]["total_tokens"] > 0


def test_completion_stream_sse(base_url):
    url, _ = base_url
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps({
            "model": "miko-sft-v2",
            "messages": [{"role": "user", "content": "2 cộng 2?"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": True,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        raw = r.read().decode("utf-8")
    events = [line[len("data: "):] for line in raw.splitlines()
              if line.startswith("data: ")]
    assert events and events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all("delta" in c["choices"][0] for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert isinstance(text, str)


def test_unknown_model_404(base_url):
    url, _ = base_url
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps({"model": "nope",
                         "messages": [{"role": "user", "content": "x"}]}
                        ).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 404


def test_bad_messages_400(base_url):
    url, _ = base_url
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps({"model": "miko-sft-v2", "messages": []}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 400


def test_responses_non_stream(base_url):
    url, _ = base_url
    req = urllib.request.Request(
        f"{url}/v1/responses",
        data=json.dumps({
            "model": "miko-sft-v2",
            "instructions": "Bạn là Miko.",
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text",
                                    "text": "Xin chào!"}]}],
            "max_output_tokens": 16,
            "temperature": 0.0,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode("utf-8"))
    assert body["object"] == "response" and body["status"] == "completed"
    msg = body["output"][0]
    assert msg["role"] == "assistant"
    assert msg["content"][0]["type"] == "output_text"
    assert isinstance(msg["content"][0]["text"], str)


def test_responses_stream_events(base_url):
    url, _ = base_url
    req = urllib.request.Request(
        f"{url}/v1/responses",
        data=json.dumps({
            "model": "miko-sft-v2",
            "input": "2 cộng 2?",
            "max_output_tokens": 16, "temperature": 0.0,
            "stream": True,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8")
    events = [line[len("event: "):] for line in raw.splitlines()
              if line.startswith("event: ")]
    assert events[0] == "response.created"
    assert "response.output_text.delta" in events
    assert events[-1] == "response.completed"
