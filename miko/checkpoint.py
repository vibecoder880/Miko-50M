"""Checkpoint system (plan §20/§22-23): save, load, resume.

Checkpoint layout mirrors the plan:

    checkpoints/step-000500/
    ├── model.safetensors     # weights only (inference-loadable)
    ├── optimizer.pt          # optimizer state (torch.save)
    ├── trainer_state.json    # step, metrics, seed, schedule position
    └── config.json           # MikoConfig snapshot

Tied weights (``lm_head.weight is tok_emb.weight``) share one storage; the
safetensors writer requires contiguous unshared tensors, so duplicates are
cloned at save time (only the *duplicate* tensors are cloned — the 50M model
peaks at ~250 MB here, acceptable on the target box).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .config import MikoConfig
from .model import MikoModel

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "load_model",
    "latest_checkpoint",
]

TRAINER_STATE = "trainer_state.json"
WEIGHTS = "model.safetensors"
OPTIMIZER = "optimizer.pt"
CONFIG = "config.json"


def _dedup_shared(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone tensors that share storage so safetensors can write them."""
    seen: dict[int, str] = {}
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        ptr = tensor.data_ptr()
        if ptr in seen:
            out[name] = tensor.detach().clone().contiguous()
        else:
            seen[ptr] = name
            out[name] = tensor.detach().contiguous()
    return out


def save_checkpoint(
    dir_path: str,
    model: MikoModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    metrics: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> str:
    """Write a complete checkpoint. Returns the directory written."""
    os.makedirs(dir_path, exist_ok=True)

    state = _dedup_shared(model.state_dict())
    save_file(state, os.path.join(dir_path, WEIGHTS))

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(dir_path, OPTIMIZER))

    np_state = np.random.get_state()  # ('MT19937', keys, pos, has_gauss, cached)
    py_state = random.getstate()  # (version, internal_tuple, gauss_next)
    trainer_state: dict[str, Any] = {
        "step": int(step),
        "metrics": metrics or {},
        "seed": seed,
        "rng_torch": torch.get_rng_state().tolist(),
        "rng_numpy": {"keys": np_state[1].tolist(), "pos": int(np_state[2])},
        "rng_python": {
            "version": py_state[0],
            "state": list(py_state[1]),
            "gauss": py_state[2],
        },
    }
    with open(os.path.join(dir_path, TRAINER_STATE), "w", encoding="utf-8") as fh:
        json.dump(trainer_state, fh, indent=2)

    if model.config is not None:
        with open(os.path.join(dir_path, CONFIG), "w", encoding="utf-8") as fh:
            json.dump(model.config.to_dict(), fh, indent=2)

    return dir_path


def load_checkpoint(
    dir_path: str,
    model: Optional[MikoModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore weights / optimizer / RNG in-place. Returns trainer state."""
    weights_path = os.path.join(dir_path, WEIGHTS)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"no {WEIGHTS} in {dir_path!r}")

    if model is not None:
        model.load_state_dict(load_file(weights_path))

    if optimizer is not None:
        opt_path = os.path.join(dir_path, OPTIMIZER)
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

    state: dict[str, Any] = {}
    state_path = os.path.join(dir_path, TRAINER_STATE)
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)

        if restore_rng:
            if "rng_torch" in state:
                torch.set_rng_state(
                    torch.tensor(state["rng_torch"], dtype=torch.uint8)
                )
            if "rng_numpy" in state:
                np_rng = state["rng_numpy"]
                np.random.set_state(
                    (
                        "MT19937",
                        np.array(np_rng["keys"], dtype=np.uint32),
                        int(np_rng["pos"]),
                        0,
                        0.0,
                    )
                )
            if "rng_python" in state:
                py_rng = state["rng_python"]
                random.setstate(
                    (
                        int(py_rng["version"]),
                        tuple(py_rng["state"]),
                        py_rng["gauss"],
                    )
                )

    return state


def load_model(dir_path: str, config: Optional[MikoConfig] = None) -> MikoModel:
    """Build a model from the checkpoint config (or ``config``) and load weights."""
    if config is None:
        cfg_path = os.path.join(dir_path, CONFIG)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"no {CONFIG} in {dir_path!r}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            config = MikoConfig.from_dict(json.load(fh))
    model = MikoModel(config)
    model.load_state_dict(load_file(os.path.join(dir_path, WEIGHTS)))
    model.eval()
    return model


def latest_checkpoint(root: str = "checkpoints") -> Optional[str]:
    """Path of the newest checkpoint under ``root``.

    Prefers the ``latest`` pointer dir (what launchers seed and maintain)
    and falls back to the newest ``step-*`` dir. Before this fix the
    pointer was skipped, so ``--resume auto`` saw a freshly-seeded
    lineage as empty and trained from random init — the root cause of
    the v0.4/v2/v3/v4 SFT cycles all starting at ~340 nats (2026-11-23).
    """
    latest = os.path.join(root, "latest")
    if os.path.isfile(os.path.join(latest, WEIGHTS)):
        return latest
    if not os.path.isdir(root):
        return None
    candidates = sorted(
        d for d in os.listdir(root)
        if d.startswith("step-") and os.path.isdir(os.path.join(root, d))
    )
    return os.path.join(root, candidates[-1]) if candidates else None
