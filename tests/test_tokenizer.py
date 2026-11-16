"""Tokenizer tests (plan §48 — Miko Tokenizer v0.1 acceptance)."""

from __future__ import annotations

import pytest

from miko.config import SPECIAL_TOKENS
from miko.tokenizer import MikoTokenizer, train_tokenizer

VI_EN_CODE = (
    "Xin chào! Tôi là Miko, một trợ lý tiếng Việt nhỏ.\n"
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
    'SELECT * FROM users WHERE name = "miko" LIMIT 10;\n'
    "{\"name\": \"miko\", \"version\": 0.1}\n"
    "The quick brown fox jumps over the lazy dog.\n"
    "$ curl -s https://api.example.com/v1/items | jq '.items[0]'\n"
)


@pytest.fixture(scope="module")
def tok(tmp_path_factory) -> MikoTokenizer:
    data = tmp_path_factory.mktemp("tokdata") / "corpus.txt"
    data.write_text((VI_EN_CODE + "\n") * 50, encoding="utf-8")
    return train_tokenizer([str(data)], vocab_size=800,
                           output_path=str(data.parent / "tokenizer.json"))


def test_special_tokens_reserved_in_order(tok):
    # Trainer reserves special ids first: <pad>=0, <unk>=1, ...
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i
    assert tok.pad_id == 0 and tok.bos_id == 2 and tok.eos_id == 3


def test_roundtrip_vietnamese_english_code(tok):
    assert tok.decode(tok.encode(VI_EN_CODE, add_special_tokens=False)) == VI_EN_CODE


def test_diacritics_preserved(tok):
    text = "Xin chào buổi sáng, đất nước Việt Nam tươi đẹp!"
    assert tok.decode(tok.encode(text, add_special_tokens=False)) == text


def test_add_special_tokens_wraps_bos_eos(tok):
    ids = tok.encode("hello")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_unknown_chars_are_byte_fallback(tok):
    text = "emoji 🌟 test"
    ids = tok.encode(text, add_special_tokens=False)
    assert tok.decode(ids) == text  # byte-level: never any <unk> loss


def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "t.json"
    tok.save(str(path))
    loaded = MikoTokenizer.load(str(path))
    sample = "Ngôn ngữ máy tính và tiếng Việt."
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.vocab_size == tok.vocab_size


def test_vocab_size_respected(tok):
    # BPE saturates below the requested size on small corpora (no more merges
    # pass min_frequency); it must never exceed the requested size and must
    # keep the byte alphabet + specials.
    assert 300 < tok.vocab_size <= 800


def test_from_config_provides_usable_tokenizer():
    t = MikoTokenizer.from_config(vocab_size=600)
    assert 300 < t.vocab_size <= 600
    text = "Xin chào"
    assert t.decode(t.encode(text, add_special_tokens=False)) == text


def test_train_rejects_vocab_too_small():
    with pytest.raises(ValueError):
        train_tokenizer([], vocab_size=100)
    with pytest.raises(ValueError):
        train_tokenizer([], vocab_size=800)
