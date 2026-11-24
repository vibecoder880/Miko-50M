"""E2E tests: real HTTP against a real Miko bundle (plan phase 03).

Starts the ThreadingHTTPServer in-process on an OS-assigned port with the
38M model loaded from ``dist/miko-sft-v2`` (exported by
scripts/export_bundle.py). Skips with a clear reason when the bundle is
not present yet, so CI-less local runs before phase 04 stay green.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from miko.server import MikoService, make_handler
from http.server import ThreadingHTTPServer

BUNDLE = os.environ.get("MIKO_E2E_BUNDLE", "dist/miko-sft-v2")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(BUNDLE, "checkpoint", "model.safetensors")),
    reason=f"bundle {BUNDLE} not exported yet — run scripts/export_bundle.py",
)


@pytest.fixture(scope="module")
def base_url():
    service = MikoService(os.path.join(BUNDLE, "checkpoint"),
                          os.path.join(BUNDLE, "tokenizer.json"),
                          version="e2e-test", threads=2)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def test_health(base_url):
    status, body = _get(f"{base_url}/health")
    assert status == 200 and body["status"] == "ok"
    assert isinstance(body["step"], int)


def test_info(base_url):
    status, body = _get(f"{base_url}/v1/info")
    assert status == 200
    assert body["name"] == "miko"
    assert body["params"] > 1_000_000
    assert "calculator" in body["tools"]


def test_chat_no_tools_contract(base_url):
    status, body = _post(f"{base_url}/v1/chat", {
        "messages": [{"role": "user", "content": "Xin chào!"}],
        "max_new_tokens": 24,
    })
    assert status == 200
    assert isinstance(body["reply"], str)
    assert body["tool_calls"] == []
    assert body["usage"]["elapsed_s"] >= 0


def test_chat_tools_contract(base_url):
    """Mechanical tier: shape + registry execution when a call is emitted.

    Model content quality is gated separately by the eval suites; here we
    only require the HTTP contract and, if the model emitted a parseable
    call, that the registry actually executed it.
    """
    status, body = _post(f"{base_url}/v1/chat", {
        "messages": [{"role": "user", "content": "123*456 bằng bao nhiêu?"}],
        "tools": True, "max_new_tokens": 64,
    })
    assert status == 200
    assert isinstance(body["tool_calls"], list)
    for call in body["tool_calls"]:
        assert {"name", "arguments", "result"} <= set(call)
        assert call["result"]  # registry executed and returned something
    assert isinstance(body["stopped_reason"], str)


def test_bad_json_is_400(base_url):
    req = urllib.request.Request(
        f"{base_url}/v1/chat", data=b"{not json", method="POST")
    try:
        urllib.request.urlopen(req, timeout=30)
        raised = False
    except urllib.error.HTTPError as e:
        raised = e.code == 400
    assert raised


def test_unknown_path_is_404(base_url):
    try:
        urllib.request.urlopen(f"{base_url}/nope", timeout=30)
        raised = False
    except urllib.error.HTTPError as e:
        raised = e.code == 404
    assert raised
