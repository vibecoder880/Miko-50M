"""Chat template tests (plan §26, §29 — M4 prep)."""

from __future__ import annotations

import pytest

from miko.chat_template import (
    IGNORE_INDEX,
    ChatMessage,
    encode_chat,
    encode_chat_masked,
    render_chat,
)
from miko.tokenizer import MikoTokenizer


@pytest.fixture(scope="module")
def tok():
    return MikoTokenizer.from_config(vocab_size=600)


def test_render_chat_format():
    text = render_chat([
        ChatMessage("system", "Bạn là Miko."),
        ChatMessage("user", "Xin chào"),
        ChatMessage("assistant", "Chào bạn!"),
    ])
    assert text == (
        "<system>\nBạn là Miko.\n"
        "<user>\nXin chào\n"
        "<assistant>\nChào bạn!"
    )


def test_render_generation_prompt_ends_with_assistant_header():
    text = render_chat([ChatMessage("user", "Hi")], add_generation_prompt=True)
    assert text.endswith("<assistant>\n")


def test_encode_chat_special_tokens_in_order(tok):
    ids = encode_chat(tok, [
        ChatMessage("user", "Xin chào"),
        ChatMessage("assistant", "Chào bạn!"),
    ])
    assert ids[0] == tok.bos_id
    assert ids[1] == tok.token_to_id("<user>")
    assert tok.eos_id in ids
    assert ids.count(tok.token_to_id("<assistant>")) == 1


def test_encode_chat_each_turn_closed_with_eos(tok):
    ids = encode_chat(tok, [
        ChatMessage("system", "S"),
        ChatMessage("user", "U"),
        ChatMessage("assistant", "A"),
    ])
    # 3 turns → exactly 3 <eos> after bos.
    assert ids.count(tok.eos_id) == 3


def test_encode_chat_content_survives_decode(tok):
    content = "Miko học tiếng Việt và code."
    ids = encode_chat(tok, [ChatMessage("assistant", content)])
    # Strip bos + role header + eos, decode the middle.
    body = ids[2:-1]
    assert tok.decode(body, skip_special_tokens=False) == content


def test_encode_chat_masked_labels_only_on_assistant(tok):
    ids, labels = encode_chat_masked(tok, [
        ChatMessage("system", "Bạn là Miko."),
        ChatMessage("user", "1+1?"),
        ChatMessage("assistant", "2"),
    ])
    assert len(ids) == len(labels)
    # Exactly one contiguous labeled span: the assistant content + <eos>.
    labeled = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
    assert len(labeled) >= 1
    assert labeled == list(range(labeled[0], labeled[-1] + 1))
    # The labeled span ends with <eos> (learn to stop).
    assert ids[labeled[-1]] == tok.eos_id
    # Labels equal ids inside the span.
    assert all(labels[i] == ids[i] for i in labeled)


def test_encode_chat_masked_multi_turn(tok):
    ids, labels = encode_chat_masked(tok, [
        ChatMessage("user", "A?"),
        ChatMessage("assistant", "B."),
        ChatMessage("user", "C?"),
        ChatMessage("assistant", "D."),
    ])
    labeled = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
    # Two separated spans (one per assistant turn).
    gaps = [b - a for a, b in zip(labeled, labeled[1:]) if b - a > 1]
    assert len(gaps) == 1


def test_encode_chat_masked_truncation_keeps_alignment(tok):
    long_user = "x" * 500
    ids, labels = encode_chat_masked(
        tok, [ChatMessage("user", long_user)], max_len=50
    )
    assert len(ids) == len(labels) == 50


def test_rejects_bad_roles_and_empty(tok):
    with pytest.raises(ValueError):
        ChatMessage("tool", "x")  # tool roles come with M5
    with pytest.raises(ValueError):
        render_chat([])
    with pytest.raises(ValueError):
        encode_chat(tok, [])
