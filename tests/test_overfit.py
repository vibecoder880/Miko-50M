"""Tiny overfit test — **Gate A** of the training plan (plan §8, §44).

The gate: if the model cannot overfit a tiny corpus, it is *not allowed* to
train on a large dataset. A ~1M-param Miko must drive next-token loss far
below the uniform baseline ``ln(vocab)`` on a small memorised corpus.

Runs on CPU in well under a couple of minutes.
"""

from __future__ import annotations

import math

import torch

from miko.config import MikoConfig
from miko.model import MikoModel
from miko.tokenizer import MikoTokenizer
from miko.trainer import next_token_loss
from miko.utils import seed_everything

CORPUS = (
    "Miko là một model ngôn ngữ nhỏ được huấn luyện từ con số 0. "
    "Miko học tiếng Việt, tiếng Anh và code. "
    "Xin chào, tôi là Miko! "
    "def hello():\n    print('Xin chao Miko')\n"
) * 6


def build_pipeline(seq_len: int = 32):
    tok = MikoTokenizer.from_config(vocab_size=700)
    ids = tok.encode(CORPUS, add_special_tokens=False)
    # Tile the corpus into non-overlapping seq_len windows.
    windows = [
        torch.tensor(ids[i : i + seq_len], dtype=torch.long)
        for i in range(0, len(ids) - seq_len + 1, seq_len)
    ]
    cfg = MikoConfig(
        vocab_size=tok.vocab_size, hidden_size=64, num_layers=3, num_heads=8,
        ffn_size=256, activation="swiglu", context_length=seq_len,
    )
    model = MikoModel(cfg)
    return model, windows


def test_gate_a_tiny_overfit():
    seed_everything(42)
    model, windows = build_pipeline()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    ln_vocab = math.log(model.config.vocab_size)  # uniform baseline
    first_loss = None

    model.train()
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        losses = []
        for w in windows:
            # IMPORTANT: inputs and targets must be the next-token shift
            # (dataset does ids[:-1] / ids[1:]); feeding the same window to
            # both lets causal attention copy inputs and fake zero loss.
            logits = model(w[:-1].unsqueeze(0))
            losses.append(next_token_loss(logits, w[1:].unsqueeze(0)))
        loss = torch.stack(losses).mean()
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    final_loss = loss.item()

    # Sanity: initial loss sits around (or above) the uniform baseline.
    assert first_loss > ln_vocab - 1.5, (
        f"initial loss {first_loss:.3f} unexpectedly below ln(vocab)"
    )
    # The gate: memorise the corpus.
    assert final_loss < 0.5, (
        f"Gate A FAILED: loss {final_loss:.4f} not below 0.5 after overfit "
        f"(started {first_loss:.4f})"
    )
    # And loss actually went down.
    assert final_loss < first_loss * 0.25, (
        f"loss did not collapse: {first_loss:.3f} -> {final_loss:.3f}"
    )
