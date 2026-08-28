"""Shared utilities: deterministic seeding and logging."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional


def seed_everything(seed: int = 42) -> None:
    """Make runs as deterministic as practical across the libraries Miko uses.

    Note: full bit-for-bit reproducibility on CPU with threaded BLAS is not
    guaranteed, but this fixes the major entropy sources (Python, numpy, torch
    RNGs and cuDNN/algorithm selection) so that the deterministic-output tests
    are stable.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        if torch.cuda.is_available():  # never expected on Miko's box
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_logger(name: str = "miko", level: Optional[int] = None) -> logging.Logger:
    """Return a minimally-configured logger.

    Avoids double-attaching handlers when called repeatedly (e.g. across
    modules) and defaults to INFO unless ``MIKO_LOG_LEVEL`` is set.
    """
    logger = logging.getLogger(name)
    if level is None:
        lvl = os.environ.get("MIKO_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, lvl, logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger
