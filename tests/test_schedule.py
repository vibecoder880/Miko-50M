"""LR schedule + loss/eval helper tests (plan §18-19)."""

from __future__ import annotations

import pytest
import torch

from miko.trainer import cosine_lr, next_token_loss


def test_warmup_ramps_linearly():
    peak, warm = 3e-4, 100
    lrs = [cosine_lr(s, 1000, warm, peak) for s in range(warm)]
    assert lrs[0] == pytest.approx(peak / warm)
    assert lrs[-1] == pytest.approx(peak)
    assert all(b > a for a, b in zip(lrs, lrs[1:]))  # strictly increasing


def test_cosine_decay_monotone_after_warmup():
    peak, warm, total = 3e-4, 100, 1000
    lrs = [cosine_lr(s, total, warm, peak) for s in range(warm, total)]
    assert all(b <= a for a, b in zip(lrs, lrs[1:]))
    assert lrs[0] == pytest.approx(peak)
    assert lrs[-1] == pytest.approx(peak * 0.1, rel=1e-3)


def test_cosine_is_half_at_midpoint():
    peak, warm, total = 1e-3, 100, 1000
    mid = cosine_lr((warm + total) // 2, total, warm, peak)
    assert mid == pytest.approx(peak * (0.1 + 0.9 * 0.5), rel=1e-3)


def test_next_token_loss_matches_manual_ce():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))
    manual = -torch.log_softmax(logits.reshape(-1, 10), dim=-1)[
        torch.arange(8), targets.reshape(-1)
    ].mean()
    assert float(next_token_loss(logits, targets)) == pytest.approx(
        float(manual), rel=1e-5
    )


def test_next_token_loss_skips_ignored_positions():
    logits = torch.randn(1, 4, 10)
    targets = torch.tensor([[2, -100, 5, -100]])
    expected = -torch.log_softmax(logits[0], dim=-1)[[0, 2], [2, 5]].mean()
    assert float(next_token_loss(logits, targets)) == pytest.approx(
        float(expected), rel=1e-5
    )
