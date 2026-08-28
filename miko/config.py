"""Miko model configuration.

This module is intentionally dependency-light (no torch import at module load)
so that the configuration contract can be imported by tooling, tests, and the
package root without pulling in heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Special tokens reserved from day one so future chat/tool/memory tiers can be
# fine-tuned without re-indexing the vocabulary.
SPECIAL_TOKENS: list[str] = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
    "<tool_call>",
    "<tool_result>",
]


@dataclass
class MikoConfig:
    """Configuration for the Miko decoder-only Transformer.

    All fields have safe defaults that together yield roughly a 50M-parameter
    model. Override via ``from_yaml`` / ``from_dict`` or by constructing with
    explicit keyword arguments.
    """

    # --- Tokenizer / vocabulary ---
    vocab_size: int = 32000
    # Number of BPE/Unigram merges the tokenizer trains with (informational).
    tokenizer_merges: int = 16000

    # --- Architecture ---
    hidden_size: int = 384
    num_layers: int = 13
    num_heads: int = 6
    ffn_size: int = 2048
    context_length: int = 512
    tie_embeddings: bool = True

    # --- Normalization / activations ---
    norm_eps: float = 1e-6
    activation: str = "swiglu"  # "swiglu" | "gelu"
    # Rotary positional embedding; falls back to learned absolute if disabled.
    use_rope: bool = True
    rope_base: float = 10000.0

    # --- Training-relevant defaults (overridable per run) ---
    max_position_embeddings: int = 512
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.context_length > self.max_position_embeddings:
            # Allow training with a shorter context than the model was sized for.
            self.max_position_embeddings = self.context_length
        if self.activation not in ("swiglu", "gelu"):
            raise ValueError(f"unsupported activation: {self.activation!r}")

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_kv_heads(self) -> int:
        # Miko v0.1 uses full attention (no GQA); kept as a property so future
        # tiers can switch to grouped-query attention without API churn.
        return self.num_heads
    def parameter_count(self) -> int:
        """Exact parameter count matching ``miko.model.MikoModel``.

        Layout (no biases; RMSNorm = weight vector only; tied embeddings):
          - token embedding : vocab_size * hidden_size
          - per layer       : 4 * h^2            (Q, K, V, O projections)
                              + 3 * h * ffn     (SwiGLU gate/up/down)
                              | 2 * h * ffn     (GELU up/down)
          - norms           : (2*num_layers + 1) * hidden
          - lm_head         : 0 when tied to token embedding
        """
        h = self.hidden_size
        f = self.ffn_size
        mlp_factor = 3 if self.activation == "swiglu" else 2
        per_layer = 4 * h * h + mlp_factor * h * f
        tok_emb = self.vocab_size * h
        norms = (2 * self.num_layers + 1) * h
        lm_head = 0 if self.tie_embeddings else h * self.vocab_size
        return tok_emb + per_layer * self.num_layers + norms + lm_head
    @property
    def approx_params_millions(self) -> float:
        return self.parameter_count() / 1_000_000.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MikoConfig":
        known = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "MikoConfig":
        import yaml  # local import keeps module torch/pyyaml-light

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)


def dataclass_fields(cls):  # pragma: no cover - tiny helper
    from dataclasses import fields

    return fields(cls)
