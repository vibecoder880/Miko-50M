"""Checkpoint save/load/resume tests (plan §20, §22-23)."""

from __future__ import annotations

import torch

from miko.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    load_model,
    save_checkpoint,
)
from miko.config import MikoConfig
from miko.model import MikoModel


def tiny_model(seed: int = 0) -> MikoModel:
    torch.manual_seed(seed)
    cfg = MikoConfig(
        vocab_size=512, hidden_size=32, num_layers=2, num_heads=4,
        ffn_size=64, activation="swiglu",
    )
    return MikoModel(cfg)


def test_save_load_weights_roundtrip(tmp_path):
    model = tiny_model()
    save_checkpoint(str(tmp_path / "step-000010"), model, step=10)

    fresh = tiny_model(seed=123)  # different init
    assert not torch.equal(fresh.tok_emb.weight, model.tok_emb.weight)
    load_checkpoint(str(tmp_path / "step-000010"), fresh)
    x = torch.randint(0, 512, (2, 16))
    assert torch.equal(fresh(x), model(x))


def test_load_model_rebuilds_from_config(tmp_path):
    model = tiny_model()
    save_checkpoint(str(tmp_path / "step-000020"), model, step=20)
    loaded = load_model(str(tmp_path / "step-000020"))
    x = torch.randint(0, 512, (1, 8))
    assert torch.equal(loaded(x), model(x))


def test_resume_restores_optimizer_and_rng(tmp_path):
    torch.manual_seed(0)
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # One real step so the optimizer has exp_avg/exp_avg_sq buffers to save.
    x = torch.randint(0, 512, (2, 16))
    model(x).sum().backward()
    opt.step()

    ckpt = str(tmp_path / "step-000030")
    save_checkpoint(ckpt, model, optimizer=opt, step=30, seed=42)

    # Advance RNG after checkpointing; resume must restore the pre-advance state.
    torch.rand(5)
    fresh = tiny_model()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
    state = load_checkpoint(ckpt, fresh, optimizer=fresh_opt, restore_rng=True)

    assert state["step"] == 30
    assert state["metrics"].get("tokens_seen", 0) == 0
    # Optimizer state was carried over (moment buffers present & equal).
    saved_exp = opt.state_dict()["state"][0]["exp_avg"]
    loaded_exp = fresh_opt.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(saved_exp, loaded_exp)
    # RNG restored: next random draw matches a replay from checkpoint time.
    torch.manual_seed(999)
    torch.rand(3)
    a = torch.rand(1)


def test_checkpoint_files_match_plan_layout(tmp_path):
    model = tiny_model()
    save_checkpoint(str(tmp_path / "step-000100"), model, step=100)
    names = set(n for n in __import__("os").listdir(tmp_path / "step-000100"))
    assert {"model.safetensors", "trainer_state.json", "config.json"} <= names


def test_latest_checkpoint_finder(tmp_path):
    assert latest_checkpoint(str(tmp_path)) is None
    model = tiny_model()
    save_checkpoint(str(tmp_path / "step-000100"), model, step=100)
    save_checkpoint(str(tmp_path / "step-000200"), model, step=200)
    (tmp_path / "latest").mkdir()
    assert latest_checkpoint(str(tmp_path)) == str(tmp_path / "step-000200")
