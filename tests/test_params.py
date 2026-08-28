"""Parameter-count accounting tests for ``MikoModel``.

These assert that ``MikoModel.param_count()`` matches the closed-form
``MikoConfig.parameter_count()`` exactly, that the summed ``num_parameters()``
agrees, and that the tied-vs-untied and small-config variants behave per the
documented formula.
"""

from __future__ import annotations

import torch

from miko.config import MikoConfig
from miko.model import MikoModel


def test_default_config_param_count():
    cfg = MikoConfig()
    model = MikoModel(cfg)
    assert model.param_count() == cfg.parameter_count()
    assert model.param_count() == model.num_parameters()


def test_non_tied_adds_hidden_times_vocab():
    cfg_tied = MikoConfig(tie_embeddings=True)
    cfg_free = MikoConfig(tie_embeddings=False)
    model_free = MikoModel(cfg_free)

    # Untied adds exactly one extra Linear(h, vocab) = h * vocab params.
    expected_extra = cfg_free.hidden_size * cfg_free.vocab_size
    assert cfg_free.parameter_count() - cfg_tied.parameter_count() == expected_extra
    assert model_free.param_count() == cfg_free.parameter_count()
    assert model_free.param_count() == model_free.num_parameters()


def test_small_config_matches_manual_formula():
    cfg = MikoConfig(hidden_size=64, num_layers=2, ffn_size=128, activation="swiglu")
    model = MikoModel(cfg)
    h, f, L = 64, 128, 2
    tok_emb = cfg.vocab_size * h
    per_layer = 4 * h * h + 3 * h * f  # SwiGLU
    norms = (2 * L + 1) * h
    lm_head = 0 if cfg.tie_embeddings else h * cfg.vocab_size
    manual = tok_emb + per_layer * L + norms + lm_head

    assert cfg.parameter_count() == manual
    assert model.param_count() == manual
    assert model.num_parameters() == manual


def test_params_dtype_and_tie_assignment():
    cfg = MikoConfig(hidden_size=64, num_layers=2, ffn_size=128)
    model = MikoModel(cfg)
    # Tied: lm_head.weight is the very same Parameter object as tok_emb.weight.
    assert model.lm_head.weight is model.tok_emb.weight
    # Untied: distinct Parameter objects.
    model2 = MikoModel(MikoConfig(hidden_size=64, num_layers=2, ffn_size=128,
                                  tie_embeddings=False))
    assert model2.lm_head.weight is not model2.tok_emb.weight
