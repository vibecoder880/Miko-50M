"""Chat-format eval item tests (SFT tier, plan §27)."""

from __future__ import annotations

import pytest
import torch

from miko.config import MikoConfig
from miko.evaluation import EvalItem, evaluate_model, load_eval_sets, score_item
from miko.model import MikoModel
from miko.tokenizer import MikoTokenizer
from miko.utils import seed_everything


@pytest.fixture(scope="module")
def tiny_pipeline():
    seed_everything(11)
    tok = MikoTokenizer.from_config(vocab_size=600)
    cfg = MikoConfig(vocab_size=tok.vocab_size, hidden_size=32, num_layers=2,
                     num_heads=4, ffn_size=64, activation="swiglu",
                     context_length=64)
    model = MikoModel(cfg)
    return model, tok


def test_load_chat_format_items(tmp_path):
    path = tmp_path / "chat_sft.jsonl"
    path.write_text(
        '{"id": "a", "format": "chat", "system": "Bạn là Miko.", '
        '"prompt": "Bạn là ai?", "expected": ["miko"]}\n',
        encoding="utf-8",
    )
    items = load_eval_sets([str(path)])
    assert items[0].format == "chat"
    assert items[0].system == "Bạn là Miko."


def test_load_rejects_unknown_format(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"prompt": "x", "expected": ["y"], "format": "weird"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_eval_sets([str(path)])


def test_chat_format_generation_prompt_used(tiny_pipeline):
    """Chat items must route through encode_chat (ends with assistant header).

    We verify structurally: generation runs, results keep the item's shape,
    and identical runs agree (greedy determinism).
    """
    model, tok = tiny_pipeline
    items = [
        EvalItem("a", "Xin chào", ["chào"], format="chat", category="chat"),
        EvalItem("b", "1+1?", ["2"], format="chat", system="Bạn là Miko.",
                 category="chat"),
    ]
    r1 = evaluate_model(model, tok, items, max_new_tokens=6)
    r2 = evaluate_model(model, tok, items, max_new_tokens=6)
    assert r1 == r2
    assert set(r1) == {"chat"}
    for entry in r1["chat"]["items"]:
        assert "generated" in entry and "passed" in entry


# ---------------------------------------------------------------------------
# tool_call metric (plan §45)
# ---------------------------------------------------------------------------
TOOL_SYSTEM = (
    "Bạn là Miko. Khi cần tính chính xác, hãy dùng công cụ:\n"
    '<tool_call>\n{"name": "calculator", "arguments": {"expression": "..."}}\n</tool_call>'
)


def _tool_item(**kw):
    defaults = dict(id="t", prompt="Tính 123*456.", expected=[""],
                    metric="tool_call", format="chat", category="tool",
                    system=TOOL_SYSTEM, expected_tool="calculator",
                    expected_args={"expression": "123*456"})
    defaults.update(kw)
    return EvalItem(**defaults)


def test_tool_call_scores_correct_emission():
    item = _tool_item()
    good = ('<tool_call>\n{"name": "calculator", "arguments": '
            '{"expression": "123 * 456"}}\n</tool_call>')  # spaces tolerated
    assert score_item(item, good)


def test_tool_call_rejects_wrong_tool_or_expression():
    item = _tool_item()
    wrong_tool = ('<tool_call>\n{"name": "time", "arguments": {}}\n</tool_call>')
    assert not score_item(item, wrong_tool)
    wrong_expr = ('<tool_call>\n{"name": "calculator", "arguments": '
                  '{"expression": "12*45"}}\n</tool_call>')
    assert not score_item(item, wrong_expr)


def test_tool_call_rejects_malformed_json():
    item = _tool_item()
    assert not score_item(item, "<tool_call>{'name': calculator</tool_call>")
    assert not score_item(item, "Kết quả là 4.")  # no call at all


def test_tool_call_negative_items():
    item = _tool_item(expected_tool="__none__", expected_args=None)
    assert score_item(item, "Tôi là Miko, trợ lý nhỏ.")      # no call: pass
    assert not score_item(item, '<tool_call>{"name": "calculator", '
                                '"arguments": {"expression": "2+2"}}</tool_call>')


def test_tool_call_load_and_evaluate(tmp_path, tiny_pipeline):
    model, tok = tiny_pipeline
    path = tmp_path / "tool_test.jsonl"
    path.write_text(
        '{"id": "t1", "format": "chat", "metric": "tool_call", '
        '"prompt": "Tính 1*2.", "expected_tool": "calculator", '
        '"expected_args": {"expression": "1*2"}}\n',
        encoding="utf-8",
    )
    items = load_eval_sets([str(path)])
    assert items[0].metric == "tool_call"
    assert items[0].expected_tool == "calculator"
    res = evaluate_model(model, tok, items, max_new_tokens=8)
    assert "tool_test" in res  # category defaults to file stem
