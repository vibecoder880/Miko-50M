"""Optimizer factory (plan §17): AdamW vs memory-saving Adafactor.

On the 1 GB target box the optimizer state is the largest memory block
(plan §2/§46): full AdamW stores two extra fp32 tensors per parameter
(~2x params ≈ 400 MB extra at 50M). Adafactor keeps *factored* second
moments (row/col), cutting state to ~1% of that, at some convergence
quality cost — exactly the trade the plan calls for.

Use ``adafactor`` in the train yaml when training the 50M model on the
1 GB box; ``adamw`` otherwise. Verified by ``scripts/benchmark_cpu.py
--optimizer`` and ``evals/benchmark_*.json``.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["build_optimizer", "optimizer_state_bytes", "SUPPORTED_OPTIMIZERS"]

SUPPORTED_OPTIMIZERS = ("adamw", "adafactor")


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on norms / embeddings (standard small-GPT recipe)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.dim() <= 1 or "emb" in name) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: torch.nn.Module,
    optimizer: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """Build the configured optimizer with the Miko decay groups."""
    name = optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups(model, weight_decay), lr=lr, betas=betas, eps=1e-8,
            fused=False,
        )
    if name == "adafactor":
        # torch >= 2.x Adafactor: factored second moment + factored momentum,
        # update size rho = min(lr, 1/sqrt(t)) and RMS(theta)-scaled alpha —
        # i.e. self-limiting warmup for free. Passing an explicit ``lr`` keeps
        # the warmup/cosine schedule in tools/train.py driving it like AdamW.
        return torch.optim.Adafactor(
            param_groups(model, weight_decay),
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(
        f"unsupported optimizer {optimizer!r}; choose from {SUPPORTED_OPTIMIZERS}"
    )


def optimizer_state_bytes(opt: torch.optim.Optimizer) -> int:
    """Total bytes held in optimizer state (the RAM bottleneck on 1 GB boxes)."""
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total
