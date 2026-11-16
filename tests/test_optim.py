"""Optimizer factory tests (plan §17 — memory-saving optimizer option)."""

from __future__ import annotations

import pytest
import torch

from miko.config import MikoConfig
from miko.model import MikoModel
from miko.optim import (
    SUPPORTED_OPTIMIZERS,
    build_optimizer,
    optimizer_state_bytes,
)
from miko.trainer import next_token_loss
from miko.utils import seed_everything


def tiny_model() -> MikoModel:
    seed_everything(3)
    return MikoModel(
        MikoConfig(vocab_size=512, hidden_size=32, num_layers=2, num_heads=4,
                   ffn_size=64, activation="swiglu")
    )


def test_factory_returns_expected_types():
    m = tiny_model()
    assert isinstance(build_optimizer(m, "adamw"), torch.optim.AdamW)
    assert isinstance(build_optimizer(m, "adafactor"), torch.optim.Adafactor)


def test_factory_rejects_unknown_optimizer():
    with pytest.raises(ValueError):
        build_optimizer(tiny_model(), "rmsprop")


def test_adafactor_state_is_far_smaller_than_adamw():
    m = tiny_model()
    adamw = build_optimizer(m, "adamw")
    state_bytes_adamw = 0
    # Take one step so AdamW materialises exp_avg/exp_avg_sq.
    x = torch.randint(0, 512, (2, 16))
    next_token_loss(m(x), x).backward()
    adamw.step()
    state_bytes_adamw = optimizer_state_bytes(adamw)

    m2 = tiny_model()
    adaf = build_optimizer(m2, "adafactor")
    next_token_loss(m2(x), x).backward()
    adaf.step()
    state_bytes_adaf = optimizer_state_bytes(adaf)

    # Factored second moments must be a small fraction of full AdamW state
    # (AdamW stores 2 fp32 tensors/param; Adafactor factors them).
    assert state_bytes_adaf < state_bytes_adamw / 4


def test_both_optimizers_reduce_loss_on_tiny_problem():
    x = torch.randint(0, 512, (4, 16))
    for name in SUPPORTED_OPTIMIZERS:
        seed_everything(5)
        m = tiny_model()
        opt = build_optimizer(m, name, lr=1e-3)
        first = None
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            loss = next_token_loss(m(x), x)  # type: ignore[arg-type]
            if first is None:
                first = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        assert loss.item() < first, f"{name} did not reduce loss"


def test_param_groups_exclude_norms_and_embeddings():
    m = tiny_model()
    opt = build_optimizer(m, "adamw", weight_decay=0.1)
    wd_by_param_id = {}
    for group in opt.param_groups:
        for p in group["params"]:
            wd_by_param_id[id(p)] = group["weight_decay"]
    assert wd_by_param_id[id(m.tok_emb.weight)] == 0.0
    assert wd_by_param_id[id(m.final_norm.weight)] == 0.0
    # A 2-D projection must carry the configured decay.
    assert wd_by_param_id[id(m.blocks[0].attn.q_proj.weight)] == 0.1
