"""CPU benchmark: measure training-step cost and peak RSS per model size.

Answers the two sizing questions before any long run (plan §13, §23):

1. tokens/sec on this box (so token budgets can be translated into hours);
2. peak RSS of a real optimizer step (so RAM-fit is verified *before* launch).

Usage::

    python scripts/benchmark_cpu.py --seq-len 128 --steps 5
    python scripts/benchmark_cpu.py --config configs/miko_50m.yaml --steps 1
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.config import MikoConfig  # noqa: E402
from miko.model import MikoModel  # noqa: E402
from miko.optim import build_optimizer, optimizer_state_bytes  # noqa: E402
from miko.trainer import next_token_loss  # noqa: E402
from miko.utils import get_logger, seed_everything  # noqa: E402

logger = get_logger("miko.bench")


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench(config: MikoConfig, seq_len: int, steps: int, grad_accum: int,
          optimizer: str = "adamw") -> dict:
    seed_everything(42)
    model = MikoModel(config)
    model.train()
    opt = build_optimizer(model, optimizer, lr=3e-4)

    x = torch.randint(0, config.vocab_size, (1, seq_len))
    y = torch.randint(0, config.vocab_size, (1, seq_len))

    # Warmup + timed steps.
    t_first = None
    times = []
    for i in range(steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            loss = next_token_loss(model(x), y)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.time() - t0
        if i == 0:
            t_first = dt
        else:
            times.append(dt)

    steady = sum(times) / max(len(times), 1)
    window_sec = steady / grad_accum
    return {
        "params": config.parameter_count(),
        "optimizer": optimizer,
        "optimizer_state_mb": round(optimizer_state_bytes(opt) / 1e6, 1),
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "first_step_sec": round(t_first, 3),
        "steady_step_sec": round(steady, 3),
        "window_sec": round(window_sec, 3),
        "tokens_per_sec": round(seq_len / window_sec, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Miko CPU training")
    ap.add_argument("--config", help="model config yaml (default: 50M defaults)")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"])
    ap.add_argument("--tiny", action="store_true", help="also benchmark tiny")
    ap.add_argument("--out", default="evals/benchmark.json")
    args = ap.parse_args()

    results = {}
    if args.tiny:
        cfg = MikoConfig(vocab_size=1160, hidden_size=96, num_layers=4,
                         num_heads=6, ffn_size=384, context_length=args.seq_len)
        results["tiny_0_7m"] = bench(cfg, args.seq_len, args.steps, args.grad_accum,
                                     args.optimizer)
        logger.info("tiny: %s", json.dumps(results["tiny_0_7m"]))

    if args.config:
        cfg = MikoConfig.from_yaml(args.config)
    else:
        cfg = MikoConfig()
    cfg.context_length = max(cfg.context_length, args.seq_len)
    results["target_50m" if cfg.parameter_count() > 10_000_000 else "model"] = \
        bench(cfg, args.seq_len, args.steps, args.grad_accum, args.optimizer)
    key = [k for k in results if k != "tiny_0_7m"][-1]
    logger.info("%s: %s", key, json.dumps(results[key]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("saved %s", args.out)


if __name__ == "__main__":
    main()
