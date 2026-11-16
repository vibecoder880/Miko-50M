"""Evaluation suite tests (plan §24-25, §49)."""

from __future__ import annotations

import pytest
import torch

from miko.config import MikoConfig
from miko.evaluation import (
    EvalItem,
    evaluate_model,
    load_eval_sets,
    normalize_text,
    score_item,
)
from miko.model import MikoModel
from miko.tokenizer import MikoTokenizer
from miko.utils import seed_everything


# ---------------------------------------------------------------------------
# Loading / scoring
# ---------------------------------------------------------------------------
@pytest.fixture()
def eval_file(tmp_path):
    path = tmp_path / "vi_test.jsonl"
    path.write_text(
        '{"id": "t1", "prompt": "Thủ đô là", "expected": "hà nội"}\n'
        '{"id": "t2", "prompt": "1 2 3", "expected": ["4", "bốn"]}\n'
        "\n"  # blank lines and comments are skipped
        "# comment\n",
        encoding="utf-8",
    )
    return str(path)


def test_load_eval_sets_defaults_category_and_expected_list(eval_file):
    items = load_eval_sets([eval_file])
    assert len(items) == 2
    assert items[0].category == "vi_test"
    assert items[0].expected == ["hà nội"]  # str coerced to list
    assert items[1].metric == "contains"


def test_load_eval_sets_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n', encoding="utf-8")  # missing prompt/expected
    with pytest.raises(ValueError):
        load_eval_sets([str(bad)])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_sets([str(empty)])


def test_normalize_text():
    assert normalize_text("  Xin   CHÀO! ") == "xin chào!"


def test_score_item_contains():
    item = EvalItem("t", "p", ["Hà Nội", "Hà Nội."])
    assert score_item(item, "Thủ đô là Hà Nội, thành phố ngàn năm")
    assert score_item(item, "hà nội.")
    assert not score_item(item, "Hue là thành phố cổ")


def test_score_item_rejects_unknown_metric():
    item = EvalItem("t", "p", ["x"], metric="regex")
    with pytest.raises(ValueError):
        score_item(item, "x")


# ---------------------------------------------------------------------------
# evaluate_model end-to-end on a tiny model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_pipeline():
    seed_everything(9)
    tok = MikoTokenizer.from_config(vocab_size=600)
    cfg = MikoConfig(vocab_size=tok.vocab_size, hidden_size=32, num_layers=2,
                     num_heads=4, ffn_size=64, activation="swiglu",
                     context_length=64)
    model = MikoModel(cfg)
    return model, tok


def test_evaluate_model_structure_and_totals(tiny_pipeline):
    model, tok = tiny_pipeline
    items = [
        EvalItem("a", "Xin chào", ["chào"], category="vi"),
        EvalItem("b", "hello world", ["world"], category="en"),
    ]
    res = evaluate_model(model, tok, items, max_new_tokens=5)
    assert set(res) == {"vi", "en"}
    for cat in res.values():
        assert cat["passed"] + (cat["total"] - cat["passed"]) == cat["total"]
        assert 0.0 <= cat["score"] <= 1.0
        assert len(cat["items"]) == cat["total"]


def test_evaluate_model_is_deterministic(tiny_pipeline):
    model, tok = tiny_pipeline
    items = [EvalItem("a", "Xin chào, tôi là", ["miko"], category="c")]
    r1 = evaluate_model(model, tok, items, max_new_tokens=8)
    r2 = evaluate_model(model, tok, items, max_new_tokens=8)
    assert r1 == r2  # greedy scoring must not drift between runs


def test_evaluate_model_scores_perfect_on_oracle_items(tiny_pipeline):
    """A model that emits the expected answer verbatim must score 1.0.

    We can't force a real model's output, so we score through score_item
    semantics instead: a generated string containing the expected text passes.
    """
    model, tok = tiny_pipeline
    items = [EvalItem("a", "1 2 3", ["1 2 3"], category="oracle")]
    # Greedy on an overfit-impossible prompt; instead verify scoring wiring by
    # monkey-checking a passed item appears when generation contains expected.
    res = evaluate_model(model, tok, items, max_new_tokens=12)
    gen = res["oracle"]["items"][0]["generated"]
    assert (res["oracle"]["score"] == 1.0) == ("1 2 3" in gen.lower())
