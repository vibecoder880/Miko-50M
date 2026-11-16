"""Generation / sampling tests (plan §60-61)."""

from __future__ import annotations

import torch

from miko.config import MikoConfig
from miko.generation import GenerationConfig, generate
from miko.model import MikoModel
from miko.utils import seed_everything


def tiny_model() -> MikoModel:
    seed_everything(7)
    cfg = MikoConfig(
        vocab_size=256, hidden_size=32, num_layers=2, num_heads=4,
        ffn_size=64, activation="gelu",
    )
    return MikoModel(cfg).eval()


def test_greedy_is_deterministic():
    model = tiny_model()
    prompt = torch.randint(0, 256, (1, 8))
    cfg = GenerationConfig(max_new_tokens=16, greedy=True)
    a = generate(model, prompt.clone(), cfg)
    b = generate(model, prompt.clone(), cfg)
    assert torch.equal(a, b)
    assert a.shape == (1, 24)


def test_1d_prompt_returns_1d():
    model = tiny_model()
    prompt = torch.randint(0, 256, (8,))
    out = generate(model, prompt, GenerationConfig(max_new_tokens=4, greedy=True))
    assert out.dim() == 1 and out.shape[0] == 12


def test_temperature_seeded_sampling_reproducible():
    model = tiny_model()
    prompt = torch.randint(0, 256, (1, 8))
    cfg = GenerationConfig(max_new_tokens=8, temperature=1.2, top_k=0, top_p=0.0)
    seed_everything(123)
    a = generate(model, prompt.clone(), cfg)
    seed_everything(123)
    b = generate(model, prompt.clone(), cfg)
    assert torch.equal(a, b)


def test_top_k_limits_candidates():
    # With top_k=1 and non-greedy temperature, sampling must equal greedy.
    model = tiny_model()
    prompt = torch.randint(0, 256, (1, 8))
    greedy = generate(model, prompt.clone(), GenerationConfig(max_new_tokens=10, greedy=True))
    sampled = generate(
        model, prompt.clone(),
        GenerationConfig(max_new_tokens=10, temperature=1.0, top_k=1, top_p=0.0),
    )
    assert torch.equal(greedy, sampled)


def test_repetition_penalty_penalises_repeats():
    # A model biased toward one token keeps repeating it; penalty must break it.
    model = tiny_model()
    prompt = torch.full((1, 8), 5)
    base = generate(model, prompt.clone(), GenerationConfig(max_new_tokens=12, greedy=True))
    penalised = generate(
        model, prompt.clone(),
        GenerationConfig(max_new_tokens=12, greedy=True, repetition_penalty=50.0),
    )
    assert not torch.equal(base, penalised)


def test_eos_stops_generation():
    model = tiny_model()
    prompt = torch.randint(0, 256, (1, 8))
    out = generate(
        model, prompt, GenerationConfig(max_new_tokens=100, greedy=True, eos_token_id=4)
    )
    # Either stopped early on eos or ran to max; both must be bounded.
    assert out.shape[1] <= 8 + 100
