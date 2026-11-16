"""Miko BPE tokenizer (Phase 4).

Byte-level BPE trained with Hugging Face ``tokenizers``. Byte-level handling
makes Vietnamese diacritics, code, JSON, URLs and shell text lossless: every
byte is representable, so ``decode(encode(x)) == x`` for any Unicode input.

All :data:`miko.config.SPECIAL_TOKENS` are registered *during training* so the
trainer assigns them deterministic ids first (``<pad>`` = 0, ``<unk>`` = 1, ...)
and never splits them during merges. Tool/chat tokens are reserved from day one
per the training plan, ready for later fine-tuning tiers.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Optional

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from .config import SPECIAL_TOKENS, MikoConfig

__all__ = ["MikoTokenizer", "train_tokenizer"]


def train_tokenizer(
    input_files: Iterable[str],
    vocab_size: int = 32000,
    output_path: Optional[str] = None,
    min_frequency: int = 2,
) -> "MikoTokenizer":
    """Train a byte-level BPE tokenizer on ``input_files`` and return it.

    ``vocab_size`` is the *total* vocabulary size including the reserved
    special tokens and the 256-byte base alphabet, so it must exceed
    ``len(SPECIAL_TOKENS) + 256``.

    On small corpora BPE may saturate *below* the requested size once every
    remaining merge fails ``min_frequency``; the resulting vocabulary is then
    smaller than requested. Always build the model with
    ``vocab_size=tokenizer.vocab_size`` to keep dimensions consistent.
    """
    min_required = len(SPECIAL_TOKENS) + 256
    if vocab_size < min_required:
        raise ValueError(
            f"vocab_size ({vocab_size}) must be >= {min_required} "
            "(special tokens + byte alphabet)"
        )

    files = [str(f) for f in input_files]
    if not files:
        raise ValueError("input_files must contain at least one file")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        # Reserve special-token ids first so they are deterministic and never
        # produced by merges.
        special_tokens=list(SPECIAL_TOKENS),
        # Keep the full byte alphabet (limit_alphabet=None) — byte-level
        # fallback is what makes Vietnamese/emoji/JSON lossless.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train(files, trainer)

    # <bos> ... text ... <eos> whenever special tokens are requested.
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    tok = MikoTokenizer(tokenizer)
    if output_path is not None:
        tok.save(output_path)
    return tok


class MikoTokenizer:
    """Thin wrapper adding Miko-specific special-token accessors."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        missing = [t for t in SPECIAL_TOKENS if tokenizer.token_to_id(t) is None]
        if missing:
            raise ValueError(f"tokenizer is missing special tokens: {missing}")

    # ------------------------------------------------------------------
    # Ids / sizes
    # ------------------------------------------------------------------
    @property
    def backend(self) -> Tokenizer:
        return self._tok

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    @property
    def special_tokens(self) -> list[str]:
        return list(SPECIAL_TOKENS)

    def token_to_id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise KeyError(f"unknown special token: {token!r}")
        return tid

    @property
    def pad_id(self) -> int:
        return self.token_to_id("<pad>")

    @property
    def unk_id(self) -> int:
        return self.token_to_id("<unk>")

    @property
    def bos_id(self) -> int:
        return self.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.token_to_id("<eos>")

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode ``text`` to ids.

        With ``add_special_tokens=True`` the output is wrapped as
        ``<bos> ... <eos>`` (post-processor). Pass ``False`` when packing raw
        documents that already carry their own delimiters.
        """
        enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return enc.ids

    def encode_batch(self, texts: Iterable[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [
            e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens)
        ]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def iter_encode_stream(
        self,
        text_stream: Iterator[str],
        add_special_tokens: bool = False,
        buffer_chars: int = 1_000_000,
    ) -> Iterator[list[int]]:
        """Encode a lazily-read stream of documents token-efficiently.

        Yields one id-list per document. Long streams never need to fit in RAM
        whole — each yielded batch is bounded by ``buffer_chars``.
        """
        buf: list[str] = []
        size = 0
        for chunk in text_stream:
            buf.append(chunk)
            size += len(chunk)
            if size >= buffer_chars:
                for ids in self.encode_batch(buf, add_special_tokens):
                    yield ids
                buf, size = [], 0
        if buf:
            for ids in self.encode_batch(buf, add_special_tokens):
                yield ids

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "MikoTokenizer":
        return cls(Tokenizer.from_file(path))

    @classmethod
    def from_config(cls, vocab_size: int) -> "MikoTokenizer":
        """A tokenizer sized for ``vocab_size``; convenience for tests.

        Trains on a tiny built-in Vietnamese/English/code corpus so that a
        functional tokenizer always exists without external data files.
        """
        import tempfile

        sample = (
            "Miko là một model ngôn ngữ nhỏ được huấn luyện từ con số 0. "
            "Xin chào! Tôi là Miko, một trợ lý ảo tiếng Việt.\n"
            "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
            "        yield a\n        a, b = b, a + b\n"
            'SELECT * FROM users WHERE name = "miko" LIMIT 10;\n'
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning models predict the next token.\n"
        ) * 20
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(sample)
            tmp = fh.name
        try:
            return train_tokenizer([tmp], vocab_size=vocab_size)
        finally:
            os.unlink(tmp)
