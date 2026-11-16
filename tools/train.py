"""CPU pretraining loop for Miko (plan §16-21).

Features
--------
* next-token prediction only, cross-entropy loss;
* micro-batch 1 + gradient accumulation (RAM-first design);
* AdamW with linear warmup + cosine decay;
* gradient clipping + NaN detection (emergency checkpoint + abort);
* CSV training log (logs/train.csv) with plan §21 fields;
* periodic + milestone checkpoints, ``latest`` pointer, old-step pruning;
* ``--resume`` restarts exactly from the last checkpoint (weights, optimizer,
  schedule position, RNG states).

Example::

    python tools/train.py --config configs/train_cpu.yaml
    python tools/train.py --config configs/train_cpu.yaml --resume checkpoints/latest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.checkpoint import (  # noqa: E402
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from miko.config import MikoConfig  # noqa: E402
from miko.dataset import ShardedTokenDataset, find_shards, shard_stats  # noqa: E402
from miko.model import MikoModel  # noqa: E402
from miko.optim import build_optimizer  # noqa: E402
from miko.trainer import cosine_lr, next_token_loss  # noqa: E402
from miko.utils import get_logger, seed_everything  # noqa: E402

logger = get_logger("miko.train")

CSV_FIELDS = [
    "step",
    "split",
    "loss",
    "lr",
    "tokens_seen",
    "elapsed",
    "samples_per_sec",
    "tokens_per_sec",
]


def build_model(model_cfg: dict) -> MikoModel:
    return MikoModel(MikoConfig.from_dict(dict(model_cfg)))


def prune_old_checkpoints(out_dir: str, keep_last: int, milestones: set[str]) -> None:
    step_dirs = sorted(
        d for d in os.listdir(out_dir)
        if d.startswith("step-") and d not in milestones
    )
    while len(step_dirs) > keep_last:
        victim = os.path.join(out_dir, step_dirs.pop(0))
        shutil.rmtree(victim, ignore_errors=True)
        logger.info("pruned old checkpoint %s", victim)


def write_csv_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Miko on CPU")
    ap.add_argument("--config", required=True, help="path to train yaml")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="checkpoint dir to resume from ('auto' = latest)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override max_steps from the config")
    args = ap.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    tr = cfg.get("train", {})

    seq_len: int = tr.get("seq_len", 128)
    batch_size: int = tr.get("batch_size", 1)
    grad_accum: int = tr.get("grad_accum", 8)
    max_steps: int = args.max_steps or tr.get("max_steps", 1000)
    peak_lr: float = tr.get("lr", 3e-4)
    min_lr_ratio: float = tr.get("min_lr_ratio", 0.1)
    warmup_pct: float = tr.get("warmup_pct", 0.03)
    weight_decay: float = tr.get("weight_decay", 0.1)
    betas = tuple(tr.get("betas", [0.9, 0.95]))
    clip_grad: float = tr.get("clip_grad", 1.0)
    seed: int = tr.get("seed", 42)
    log_every: int = tr.get("log_every", 10)
    eval_every: int = tr.get("eval_every", 100)
    eval_batches: int = tr.get("eval_batches", 20)
    checkpoint_every: int = tr.get("checkpoint_every", 250)
    milestone_every: int = tr.get("milestone_every", 1000)
    keep_last: int = tr.get("keep_last", 2)
    out_dir: str = tr.get("out_dir", "checkpoints")
    log_csv: str = tr.get("log_csv", "logs/train.csv")

    seed_everything(seed)

    train_shards = find_shards(data_cfg.get("train_dir", "data/shards/train"))
    val_dir = data_cfg.get("val_dir")
    train_ds = ShardedTokenDataset(train_shards, seq_len, seed=seed)
    stats = shard_stats(train_shards)
    logger.info(
        "train: %d shards / %.2fM tokens / %d windows",
        stats.shards, stats.tokens / 1e6, len(train_ds),
    )
    val_ds = None
    if val_dir and os.path.isdir(val_dir):
        val_ds = ShardedTokenDataset(find_shards(val_dir), seq_len, seed=seed)
        logger.info("val: %.2fM tokens / %d windows",
                    val_ds.total_tokens / 1e6, len(val_ds))

    model = build_model(model_cfg)
    model.train()
    logger.info("model params: %d (%.2fM)", model.param_count(),
                model.param_count() / 1e6)

    opt = build_optimizer(model, tr.get("optimizer", "adamw"), peak_lr,
                          weight_decay, betas)

    warmup_steps = int(warmup_pct * max_steps)
    step = 0
    tokens_seen = 0

    resume_dir = args.resume
    if resume_dir == "auto":
        resume_dir = latest_checkpoint(out_dir)
    if resume_dir:
        if resume_dir == "latest" or os.path.basename(resume_dir) == "latest":
            resume_dir = os.path.join(out_dir, "latest")
        state = load_checkpoint(resume_dir, model, optimizer=opt, restore_rng=True)
        step = int(state.get("step", 0))
        tokens_seen = int(state.get("metrics", {}).get("tokens_seen", 0))
        logger.info("resumed from %s at step %d", resume_dir, step)
        # Align dataset shuffle epoch with the resumed step so order resumes.
        while train_ds.epoch < step * grad_accum // max(1, len(train_ds)):
            train_ds.advance_epoch()

    os.makedirs(out_dir, exist_ok=True)
    milestones: set[str] = set()
    # Counters *this process* is responsible for, so per-sec rates stay true
    # across --resume (tokens_seen itself keeps the cumulative value).
    tokens_at_start = tokens_seen
    steps_at_start = step

    def run_eval() -> float:
        if val_ds is None:
            return float("nan")
        from miko.trainer import evaluate

        return evaluate(model, val_ds, batches=eval_batches,
                        batch_size=batch_size, max_batches=eval_batches)

    model.train()
    t_start = time.time()
    accum_loss = 0.0
    accum_steps = 0
    opt.zero_grad(set_to_none=True)

    def emergency_stop(msg: str) -> None:
        ck = save_checkpoint(
            os.path.join(out_dir, "emergency"),
            model, opt, step=step,
            metrics={"tokens_seen": tokens_seen, "reason": msg},
            seed=seed,
        )
        logger.error("%s — emergency checkpoint saved to %s", msg, ck)
        raise RuntimeError(msg)

    batches = train_ds.iter_batches(batch_size=batch_size, shuffle=True)
    done = False
    while not done:
        for x, y in batches:
            logits = model(x)
            loss = next_token_loss(logits, y) / grad_accum
            if not torch.isfinite(loss):
                emergency_stop(f"non-finite loss at step {step}")
            loss.backward()
            accum_loss += loss.item()
            accum_steps += 1

            if accum_steps < grad_accum:
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if not torch.isfinite(gnorm):
                emergency_stop(f"non-finite grad norm at step {step}")

            lr = cosine_lr(step, max_steps, warmup_steps, peak_lr, min_lr_ratio)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)

            step += 1
            tokens_seen += batch_size * grad_accum * seq_len

            if step % log_every == 0:
                elapsed = time.time() - t_start
                samples = (step - steps_at_start) * batch_size * grad_accum
                tokens_done = tokens_seen - tokens_at_start
                sps = samples / max(elapsed, 1e-6)
                tps = tokens_done / max(elapsed, 1e-6)
                logger.info(
                    "step %5d | loss %.4f | lr %.2e | %.0f tok/s | %.1fs",
                    step, accum_loss / accum_steps, lr, tps, elapsed,
                )
                write_csv_row(log_csv, {
                    "step": step,
                    "split": "train",
                    "loss": round(accum_loss / accum_steps, 6),
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "elapsed": round(elapsed, 2),
                    "samples_per_sec": round(sps, 4),
                    "tokens_per_sec": round(tps, 1),
                })
                accum_loss, accum_steps = 0.0, 0

            if eval_every and step % eval_every == 0 and val_ds is not None:
                vl = run_eval()
                elapsed = time.time() - t_start
                logger.info("step %5d | val loss %.4f", step, vl)
                write_csv_row(log_csv, {
                    "step": step, "split": "val", "loss": round(vl, 6),
                    "lr": lr, "tokens_seen": tokens_seen,
                    "elapsed": round(elapsed, 2),
                })

            if checkpoint_every and step % checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(out_dir, f"step-{step:08d}"),
                    model, opt, step=step,
                    metrics={"tokens_seen": tokens_seen, "loss": accum_loss},
                    seed=seed,
                )
                # 'latest' pointer: plain copy for simple filesystem resume.
                latest = os.path.join(out_dir, "latest")
                save_checkpoint(
                    latest, model, opt, step=step,
                    metrics={"tokens_seen": tokens_seen}, seed=seed,
                )
                prune_old_checkpoints(out_dir, keep_last, milestones)

            if milestone_every and step % milestone_every == 0:
                mdir = os.path.join(out_dir, f"step-{step:08d}")
                milestones.add(f"step-{step:08d}")
                if not os.path.isdir(mdir):
                    save_checkpoint(
                        mdir, model, opt, step=step,
                        metrics={"tokens_seen": tokens_seen}, seed=seed,
                    )

            if step >= max_steps:
                done = True
                break
        if not done:
            train_ds.advance_epoch()
            batches = train_ds.iter_batches(batch_size=batch_size, shuffle=True)

    final = save_checkpoint(
        os.path.join(out_dir, "latest"), model, opt, step=step,
        metrics={"tokens_seen": tokens_seen}, seed=seed,
    )
    val_loss = run_eval() if val_ds is not None else float("nan")
    logger.info("training complete: step %d, val loss %.4f, checkpoint %s",
                step, val_loss, final)
    with open(os.path.join(out_dir, "final_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"step": step, "val_loss": val_loss,
                   "tokens_seen": tokens_seen}, fh, indent=2)


if __name__ == "__main__":
    main()
