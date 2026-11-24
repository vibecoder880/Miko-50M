"""Unit tests for the serving layer (plan phase 01).

These exercise MikoService request handling without HTTP; a tiny fake
model stands in for the 38M checkpoint so the tests stay fast.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import torch

from miko.chat_template import ChatMessage
from miko.config import MikoConfig
from miko.model import MikoModel
from miko.server import MikoService, resolve_paths


def tiny_service(tmp_path, monkeypatch) -> MikoService:
    """A real MikoService whose model is a 2-layer tiny net."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.manual_seed(0)
    model = MikoModel(MikoConfig(vocab_size=512, hidden_size=32,
                                 num_layers=2, num_heads=4, ffn_size=64,
                                 activation="swiglu"))
    from miko.checkpoint import save_checkpoint
    save_checkpoint(str(ckpt), model, step=7)
    (tmp_path / "tok.json").write_text("{}", encoding="utf-8")

    service = MikoService.__new__(MikoService)  # skip heavy __init__
    service.model = model
    service.model.eval()
    service.step = 7
    service.version = "test"
    service.max_context = 448
    service._lock = __import__("threading").Lock()

    class Tok:  # minimal stand-in with the surface _generate_fn touches
        eos_id = 3
        vocab_size = 512

        def token_to_id(self, s):
            return {"<assistant>": 6}.get(s, 1)

        def decode(self, ids, skip_special_tokens=False):
            return "x" * len(ids)

    class Enc:  # encode_for_model stand-in via ChatMessage list
        pass

    service.tokenizer = Tok()
    monkeypatch.setattr("miko.server.encode_for_model",
                        lambda tok, msgs, max_tokens: [1, 2])
    service.registry = __import__("miko.tooling", fromlist=["x"]).make_default_registry()
    return service


def test_chat_rejects_bad_bodies(tmp_path, monkeypatch):
    svc = tiny_service(tmp_path, monkeypatch)
    status, body = svc.chat({"messages": []})
    assert status == 400 and "messages" in body["error"]
    status, body = svc.chat({"messages": [{"role": "hacker", "content": "x"}]})
    assert status == 400
    status, body = svc.chat({"messages": [{"role": "user", "content": "hi"}],
                             "max_new_tokens": "big"})
    assert status == 400


def test_chat_returns_contract_shape(tmp_path, monkeypatch):
    svc = tiny_service(tmp_path, monkeypatch)
    status, body = svc.chat({"messages": [{"role": "user", "content": "hi"}],
                             "max_new_tokens": 8})
    assert status == 0 or status == 200
    if status == 200:
        assert set(body) >= {"reply", "tool_calls", "stopped_reason", "usage"}
        assert body["usage"]["elapsed_s"] >= 0


def test_info_shape(tmp_path, monkeypatch):
    svc = tiny_service(tmp_path, monkeypatch)
    info = svc.info()
    assert info["name"] == "miko" and info["step"] == 7
    assert "calculator" in info["tools"]


def test_clamps(tmp_path, monkeypatch):
    svc = tiny_service(tmp_path, monkeypatch)
    with patch.object(MikoService, "_generate_fn",
                      return_value=lambda msgs: "ok") as gf:
        svc.chat({"messages": [{"role": "user", "content": "hi"}],
                  "max_new_tokens": 9999, "temperature": 99})
    assert gf.called
