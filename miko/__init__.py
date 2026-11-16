"""Miko 50M — a from-scratch ~50M-parameter CPU-trained language model.

Root imports stay dependency-light (config + utils only, no torch) so
tooling can read the configuration contract without heavy deps. Import the
torch-backed submodules explicitly when needed::

    from miko.model import MikoModel
    from miko.tokenizer import MikoTokenizer
    from miko.dataset import ShardedTokenDataset
    from miko.checkpoint import save_checkpoint, load_model
    from miko.generation import generate
"""

from .config import SPECIAL_TOKENS, MikoConfig
from .utils import seed_everything, get_logger

__version__ = "0.1.0"

__all__ = [
    "SPECIAL_TOKENS",
    "MikoConfig",
    "seed_everything",
    "get_logger",
]
