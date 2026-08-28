"""Tests for Miko's causal self-attention through ``MikoModel``.

Covers: output shapes, the causal mask (future tokens must not affect past
logits), deterministic output under fixed seeding, and gradient flow into the
token embedding.
"""

from __future__ import annotations

import torch

from miko.config import MikoConfig
from miko.model import MikoModel
from miko.utils import seed_everything


def _make_model() -> MikoModel:
    cfg = MikoConfig(hidden_size=64, num_layers=2, ffn_size=128, use_rope=True)
    return MikoModel(cfg).eval()


def test_output_shape():
    model = _make_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, model.config.vocab_size)


def test_causal_mask_isolation():
    """Changing only future tokens must not change logits at past positions."""
    model = _make_model()
    vocab = model.config.vocab_size

    # Two sequences identical in positions 0..2, differing only afterwards.
    base = [1, 2, 3]
    a = torch.tensor([base + [4, 5]])
    b = torch.tensor([base + [9, 9]])  # future tokens differ, past identical
    assert torch.equal(a[:, :3], b[:, :3])

    logits_a = model(a)
    logits_b = model(b)

    # Logits at past positions (0,1,2) must be bit-identical.
    assert torch.equal(logits_a[:, :3], logits_b[:, :3])

    # And at least one future position should differ (sanity that model reacts).
    assert not torch.equal(logits_a[:, 3:], logits_b[:, 3:])


def test_deterministic_seed():
    cfg = MikoConfig(hidden_size=64, num_layers=2, ffn_size=128)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))

    seed_everything(42)
    m1 = MikoModel(cfg)
    out1 = m1(input_ids)

    seed_everything(42)
    m2 = MikoModel(cfg)
    out2 = m2(input_ids)

    assert torch.equal(out1, out2)


def test_gradient_flow_to_embedding():
    model = _make_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    logits = model(input_ids)
    loss = logits[0, 0, 0].sum()
    loss.backward()

    grad = model.tok_emb.weight.grad
    assert grad is not None
    assert grad.shape == model.tok_emb.weight.shape
    assert torch.count_nonzero(grad) > 0
