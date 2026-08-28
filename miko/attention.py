"""Causal multi-head self-attention with optional RoPE.

Two positional schemes are supported:

* ``use_rope=True``  -> rotary positional embedding applied to Q and K.
* ``use_rope=False`` -> a learned absolute positional embedding table added to
  the input before attention.

Both are causal (token ``i`` attends only ``j <= i``) and scaled by
``1 / sqrt(head_dim)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import MikoConfig

__all__ = ["CausalSelfAttention"]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MikoConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Query, Key, Value, Output projections. No biases anywhere.
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        if config.use_rope:
            self.pos_emb = None
        else:
            # Learned absolute positional embedding table.
            self.pos_emb = nn.Parameter(
                torch.zeros(config.max_position_embeddings, h)
            )

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
        """Precompute the rotary cos/sin tables for positions 0..seq_len-1.

        Uses the standard 2D (GPT-NeoX style) formulation: frequencies are
        assigned to pairs of adjacent channels.
        """
        # shape (head_dim/2,)
        half = head_dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
        )
        # positions 0..seq_len-1
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        # (seq_len, head_dim/2)
        freqs = torch.outer(position, inv_freq)
        cos = torch.cos(freqs)  # (seq_len, head_dim/2)
        sin = torch.sin(freqs)
        return cos, sin

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` of shape (B, n_heads, T, head_dim).

        ``cos``/``sin`` have shape (T, head_dim/2) and are broadcast over the
        batch and head dimensions.
        """
        # Reshape last dim into pairs: (B, n_heads, T, head_dim/2, 2)
        x_pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x0 = x_pairs[..., 0]  # (B, n_heads, T, head_dim/2)
        x1 = x_pairs[..., 1]

        # cos/sin: (T, head_dim/2) -> (1, 1, T, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotated = torch.stack(
            [x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1
        )
        # flatten pairs back to (B, n_heads, T, head_dim)
        return rotated.reshape(*x.shape).to(x.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.pos_emb is not None:
            # Learned absolute positional embedding added to the input
            # representations before splitting into heads. Clamp positions to
            # the table size for safety when T exceeds max_position_embeddings.
            pos_idx = torch.arange(T, device=device).clamp_max(
                self.pos_emb.shape[0] - 1
            )
            pe = self.pos_emb[pos_idx]  # (T, h)
            q = q + pe
            k = k + pe

        # (B, T, h) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.pos_emb is None:
            # RoPE path
            cos, sin = self._rope_cos_sin(
                T, self.head_dim, self.config.rope_base, device
            )
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        # Scaled dot-product attention with a causal mask.
        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Lower-triangular causal mask: position i may attend j <= i only.
        causal_mask = torch.full((T, T), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        scores = scores + causal_mask

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)

        # Merge heads back: (B, n_heads, T, head_dim) -> (B, T, h)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)

        return self.o_proj(out)
