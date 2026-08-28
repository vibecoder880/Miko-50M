"""Miko 50M — a from-scratch ~50M-parameter CPU-trained language model.

This package root intentionally exposes only the stable, dependency-light
contract objects (configuration and utilities). Submodules (model, tokenizer,
dataset, training, generation) are imported explicitly by callers and by the
integration layer once all components land. Keeping the root import minimal
lets the model-agnostic parts (config, utils) be used without pulling in torch
or heavy submodules.
"""

from .config import MikoConfig
from .utils import seed_everything, get_logger

__version__ = "0.1.0"

__all__ = [
    "MikoConfig",
    "seed_everything",
    "get_logger",
]
