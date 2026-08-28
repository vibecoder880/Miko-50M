"""Root-mean-square layer normalization (RMSNorm).

Miko uses RMSNorm throughout: no mean subtraction, no bias. Only a learned
per-channel ``weight`` vector scales the normalized activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """RMSNorm with a single learned ``weight`` vector and no bias.

    Computes ``x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps) * weight``.
    The variance is accumulated in float32 for numerical stability even when
    ``x`` is a reduced-precision (e.g. bfloat16) tensor, then cast back.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Work in float32 for the normalize/scale so low-precision training
        # stays stable; restore the input dtype for the affine weight.
        input_dtype = x.dtype
        x_fp = x.float()
        variance = x_fp.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed.to(input_dtype))
