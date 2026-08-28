"""Pre-layer-normalization Transformer decoder block.

Layout (Pre-LN):

    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))

Two RMSNorm instances (``norm1`` before attention, ``norm2`` before the MLP).
No biases anywhere; residual additions only.
"""

from __future__ import annotations

import torch.nn as nn

from .attention import CausalSelfAttention
from .config import MikoConfig
from .mlp import MLP
from .norm import RMSNorm

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    def __init__(self, config: MikoConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
