"""Training-loop helpers shared by ``tools/train.py`` and the test suite.

Kept dependency-light and side-effect-free so schedule shape and evaluation
can be unit-tested without touching the CLI.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import ShardedTokenDataset
from .model import MikoModel

__all__ = ["cosine_lr", "evaluate", "next_token_loss"]


def cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay to ``peak_lr * min_lr_ratio``.

    ``step`` is 0-based; during warmup the LR ramps from ``peak_lr/warmup``
    toward ``peak_lr`` so the very first update is not a spike.
    """
    if max_steps <= 0:
        return peak_lr
    warmup_steps = max(0, min(warmup_steps, max_steps // 2))
    if warmup_steps and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def next_token_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy over all positions.

    ``logits``: (B, T, vocab); ``targets``: (B, T) with
    ``targets[:, t] == tokens[t+1]``.
    """
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab).float(),
        targets.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(
    model: MikoModel,
    dataset: ShardedTokenDataset,
    batches: int,
    batch_size: int = 1,
    max_batches: Optional[int] = None,
) -> float:
    """Mean loss over ``max_batches`` windows sampled *evenly* across the set.

    The first-N-windows sample is biased (shard 0 starts with whatever
    document was written first — e.g. one long Wikipedia article), so the
    sample is spread over the whole index range instead. Deterministic.
    """
    model.eval()
    n = batches if batches else max_batches
    if max_batches is not None:
        n = min(n, max_batches) if n else max_batches
    n = min(int(n), len(dataset))
    if n <= 0:
        raise ValueError("no evaluation batches available")

    # Evenly spaced window indices, independent of sharding.
    positions = np.linspace(0, len(dataset) - 1, n).astype(np.int64)
    total, count = 0.0, 0
    for i in range(0, n, batch_size):
        chunk = positions[i : i + batch_size]
        xs, ys = zip(*(dataset[int(j)] for j in chunk))
        x = torch.stack(xs)
        y = torch.stack(ys)
        logits = model(x)
        total += next_token_loss(logits, y).item()
        count += 1
    model.train()
    if count == 0:
        raise ValueError("no evaluation batches available")
    return total / count
