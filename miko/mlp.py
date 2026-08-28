"""Feed-forward MLP block.

Two activation variants selectable via ``config.activation``:

* ``"swiglu"`` -> gate/up/down with a SiLU (swish) gating: ``down(silu(gate(x)) * up(x))``
* ``"gelu"``   -> up/down with a GELU: ``down(gelu(up(x)))``

No biases are used in any projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MikoConfig

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(self, config: MikoConfig) -> None:
        super().__init__()
        h = config.hidden_size
        f = config.ffn_size

        if config.activation == "swiglu":
            self.gate_proj = nn.Linear(h, f, bias=False)
            self.up_proj = nn.Linear(h, f, bias=False)
            self.down_proj = nn.Linear(f, h, bias=False)
            self.activation_kind = "swiglu"
        elif config.activation == "gelu":
            self.gate_proj = None  # not used in GELU mode
            self.up_proj = nn.Linear(h, f, bias=False)
            self.down_proj = nn.Linear(f, h, bias=False)
            self.activation_kind = "gelu"
        else:  # pragma: no cover - guarded by config.__post_init__
            raise ValueError(f"unsupported activation: {config.activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_kind == "swiglu":
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            return self.down_proj(F.silu(gate) * up)
        # GELU path
        return self.down_proj(F.gelu(self.up_proj(x)))
