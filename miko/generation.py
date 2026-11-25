"""Text generation (plan §60-61): greedy, temperature, top-k, top-p sampling.

CPU-first: no KV cache in v0.1 (recomputing the forward over the growing
prefix keeps the code simple and memory tiny; a cached path can slot in later
without changing this API).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .model import MikoModel

__all__ = ["GenerationConfig", "generate"]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    greedy: bool = False
    eos_token_id: Optional[int] = None
    # Streaming hook: called with each newly chosen token id (batch 0 only).
    # Lets callers (SSE server, CLI) emit output token-by-token without
    # changing the sampling math.
    on_token: Optional[callable] = None


def _sample_logits(
    logits: torch.Tensor, cfg: GenerationConfig, generated: list[int]
) -> int:
    """Sample one token id from ``logits`` of shape ``[vocab]``."""
    logits = logits.float().clone()

    if cfg.repetition_penalty and cfg.repetition_penalty != 1.0 and generated:
        penalty = cfg.repetition_penalty
        for tok in set(generated):
            if logits[tok] > 0:
                logits[tok] /= penalty
            else:
                logits[tok] *= penalty

    if cfg.greedy or cfg.temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / cfg.temperature

    if cfg.top_k and cfg.top_k > 0:
        k = min(cfg.top_k, logits.shape[-1])
        thresh = torch.topk(logits, k).values[-1]
        logits[logits < thresh] = float("-inf")

    if cfg.top_p and 0.0 < cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Keep tokens until the cumulative mass first reaches top_p.
        cutoff = cum > cfg.top_p
        # Ensure at least the first token survives.
        cutoff[0] = False
        sorted_logits[cutoff] = float("-inf")
        logits = sorted_logits.gather(-1, sorted_idx.argsort(-1))

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: MikoModel,
    input_ids: torch.LongTensor,
    cfg: Optional[GenerationConfig] = None,
) -> torch.LongTensor:
    """Generate tokens after ``input_ids`` (shape ``[B, T]`` or ``[T]``).

    Returns the full sequence (prompt + continuation) with the same leading
    batch dimension as the input. Batches advance until every row hits
    ``max_new_tokens`` or EOS (when ``cfg.eos_token_id`` is set).
    """
    cfg = cfg or GenerationConfig()
    was_1d = input_ids.dim() == 1
    seq = input_ids if not was_1d else input_ids.unsqueeze(0)
    seq = seq.long()
    B = seq.shape[0]
    done = torch.zeros(B, dtype=torch.bool)

    for _ in range(cfg.max_new_tokens):
        logits = model(seq[:, -model.config.context_length:])  # (B, T, vocab)
        next_ids = []
        for b in range(B):
            if done[b]:
                next_ids.append(seq[b, -1].item())  # pad with a repeat; masked later
                continue
            tok = _sample_logits(logits[b, -1], cfg, generated=seq[b].tolist())
            next_ids.append(tok)
        nxt = torch.tensor(next_ids, dtype=torch.long, device=seq.device).unsqueeze(1)
        if cfg.on_token is not None and not done[0]:
            cfg.on_token(int(next_ids[0]))
        seq = torch.cat([seq, nxt], dim=1)
        if cfg.eos_token_id is not None:
            done |= nxt.squeeze(1) == cfg.eos_token_id
        if bool(done.all()):
            break

    return seq if not was_1d else seq.squeeze(0)
