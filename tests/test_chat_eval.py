"""Chat-format eval item tests (SFT tier, plan §27)."""

from __future__ import annotations

import pytest
import torch

from miko.config import MikoConfig
from miko.evaluation import EvalItem, evaluate_model, load_eval_sets
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
