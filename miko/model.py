"""Miko decoder-only language model (v0.1 Base LM).

Stacks ``num_layers`` Pre-LN Transformer blocks on top of a token embedding,
applies a final RMSNorm, and projects to vocabulary logits via an (optionally
tied) ``lm_head``.

Parameter budget (no biases; RMSNorm = weight vector only):

    token embedding : vocab_size * hidden_size
    per layer       : 4*h^2 + (3|2)*h*ffn      (Q,K,V,O + SwiGLU|GELU MLP)
    norms           : (2*num_layers + 1)*hidden
    lm_head         : 0 if tied else hidden*vocab
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import MikoConfig
from .norm import RMSNorm
from .transformer import TransformerBlock

__all__ = ["MikoModel"]


class MikoModel(nn.Module):
    def __init__(self, config: MikoConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size

        self.tok_emb = nn.Embedding(config.vocab_size, h)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # Final RMSNorm — this is the "+1" norm counted in parameter_count().
        self.final_norm = RMSNorm(h, eps=config.norm_eps)

        if config.tie_embeddings:
            self.lm_head = nn.Linear(h, config.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight  # tie weights
        else:
            self.lm_head = nn.Linear(h, config.vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Parameter accounting
    # ------------------------------------------------------------------
    def param_count(self) -> int:
        """Exact parameter count, matching ``config.parameter_count()``."""
        return self.config.parameter_count()

    def num_parameters(self) -> int:
        """Sum of ``p.numel()`` over all parameters (cross-check)."""
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # Keep the forward deterministic; compute in float32.
        x = self.tok_emb(input_ids).float()

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
