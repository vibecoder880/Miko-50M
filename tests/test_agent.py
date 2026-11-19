"""Agent loop tests (plan §35, §53) — scripted generator, no live model."""

from __future__ import annotations

import json

import pytest

from miko.agent import AgentConfig, encode_for_model, run_agent
from miko.chat_template import ChatMessage
from miko.tooling import TOOL_CALL_CLOSE, TOOL_CALL_OPEN, make_default_registry


def _call(name: str, args: dict) -> str:
    payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"


@pytest.fixture()
def registry():
    return make_default_registry()


def test_plain_answer_no_tool(registry):
    def gen(messages):
        assert messages[-1].role == "user"
        return "Xin chào! Tôi là Miko."

    res = run_agent(gen, registry, "Xin chào!")
    assert res.answer == "Xin chào! Tôi là Miko."
    assert res.steps == 1 and res.tool_calls == []
    assert res.stopped_reason == "final"


def test_single_tool_call_then_answer(registry):
    replies = iter([
        _call("calculator", {"expression": "123*456"}),
        "Kết quả là 56088.",
    ])
    seen = []

    def gen(messages):
        seen.append(list(messages))
        return next(replies)

    res = run_agent(gen, registry, "123 nhân 456 bằng bao nhiêu?")
    assert res.answer == "Kết quả là 56088."
    assert res.steps == 2
    assert res.tool_calls == [{
        "name": "calculator",
        "arguments": {"expression": "123*456"},
        "result": "56088",
    }]
    assert res.stopped_reason == "final"
    # The second generation saw the tool result turn.
    assert any(m.role == "tool" and m.content == "56088" for m in seen[1])


def test_error_result_is_fed_back_and_model_recovers(registry):
    # Malformed call -> tool error string -> model retries correctly.
    replies = iter([
        f"{TOOL_CALL_OPEN}\n{{'name': bad json\n{TOOL_CALL_CLOSE}",
        _call("calculator", {"expression": "2+2"}),
        "Kết quả là 4.",
    ])

    def gen(messages):
        return next(replies)

    res = run_agent(gen, registry, "2+2?")
    assert res.answer == "Kết quả là 4."
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["result"] == "4"


def test_max_tool_calls_bound(registry):
    def gen(messages):
        return _call("calculator", {"expression": "1+1"})

    cfg = AgentConfig(max_steps=10, max_tool_calls=2)
    res = run_agent(gen, registry, "tính mãi đi", config=cfg)
    assert res.stopped_reason == "max_tool_calls"
    assert len(res.tool_calls) == 2
    assert "giới hạn" in res.answer


def test_max_steps_bound_forces_plain_answer(registry):
    # Model answers with a tool call once, then keeps calling; with
    # max_steps=2 the loop must terminate with a forced plain answer.
    state = {"n": 0}

    def gen(messages):
        state["n"] += 1
        if state["n"] <= 2:
            return _call("calculator", {"expression": "5*5"})
        return "25."

    cfg = AgentConfig(max_steps=2)
    res = run_agent(gen, registry, "tính 5*5", config=cfg)
    assert res.stopped_reason == "max_steps"
    assert res.steps == 2


def test_run_tool_never_raises_into_loop(registry):
    def gen(messages):
        return _call("calculator", {"expression": "1/0"})  # ZeroDivisionError

    # The loop keeps going; with a single generator the next reply is plain.
    replies = iter([_call("calculator", {"expression": "1/0"}), "Không chia được cho 0."])

    def gen2(messages):
        return next(replies)

    res = run_agent(gen2, registry, "1/0?")
    assert res.answer == "Không chia được cho 0."
    assert res.tool_calls[0]["result"].startswith("error:")


def test_encode_for_model_budget(tok=None):
    from miko.tokenizer import MikoTokenizer

    tok = MikoTokenizer.from_config(vocab_size=600)
    msgs = [ChatMessage("system", "S")] + [
        ChatMessage("user", f"câu hỏi dài thứ {i} " + "x" * 40)
        for i in range(10)
    ]
    ids = encode_for_model(tok, msgs, max_tokens=150)
    assert len(ids) <= 150
    # System turn is preserved while older user turns get dropped.
    assert msgs[0].role == "system"
