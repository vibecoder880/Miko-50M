"""Run the fixed Miko eval suite + val loss/perplexity (plan §24-25, §49).

Examples::

    python tools/evaluate.py --checkpoint checkpoints/latest \
        --tokenizer tokenizer/tokenizer.json
    python tools/evaluate.py --checkpoint ckpt --data-dir data/shards/val \
        --out evals/results-miko-base-v0.1.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.checkpoint import load_model  # noqa: E402
from miko.dataset import ShardedTokenDataset, find_shards  # noqa: E402
from miko.evaluation import evaluate_model, load_eval_sets  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402
from miko.trainer import evaluate  # noqa: E402
from miko.utils import get_logger, seed_everything  # noqa: E402

logger = get_logger("miko.eval")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Miko checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--evals", nargs="*",
                    default=sorted(glob.glob("evals/*.jsonl")),
                    help="eval JSONL files (default: all under evals/)")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--data-dir", help="val shard dir for loss/perplexity")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="evals/results.json")
    args = ap.parse_args()

    seed_everything(args.seed)
    model = load_model(args.checkpoint)
    model.eval()

    tok = None
    if os.path.exists(args.tokenizer):
        tok = MikoTokenizer.load(args.tokenizer)
        if tok.vocab_size != model.config.vocab_size:
            logger.warning(
                "tokenizer vocab %d != model vocab %d — generation eval may "
                "be unreliable", tok.vocab_size, model.config.vocab_size,
            )

    results: dict = {"checkpoint": args.checkpoint, "categories": {}}

    if tok is not None and args.evals:
        items = load_eval_sets(args.evals)
        logger.info("loaded %d eval items from %d file(s)",
                    len(items), len(args.evals))
        cat_results = evaluate_model(model, tok, items,
                                     max_new_tokens=args.max_new_tokens)
        results["categories"] = cat_results
        for name, cat in sorted(cat_results.items()):
            logger.info("%-14s score %.3f (%d/%d)",
                        name, cat["score"], cat["passed"], cat["total"])
            for it in cat["items"]:
                if not it["passed"]:
                    logger.debug("FAIL %s | gen=%r", it["id"], it["generated"])

    if args.data_dir:
        ds = ShardedTokenDataset(find_shards(args.data_dir), args.seq_len,
                                 seed=args.seed)
        val_loss = evaluate(model, ds, batches=args.eval_batches,
                            max_batches=args.eval_batches)
        results["val_loss"] = round(val_loss, 4)
        results["perplexity"] = round(math.exp(min(val_loss, 20)), 2)
        logger.info("val loss %.4f | ppl %.2f", val_loss,
                    results["perplexity"])

    # Plan §49-style flat summary for quick version comparison.
    results["summary"] = {
        name: cat["score"] for name, cat in sorted(results["categories"].items())
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("saved %s", args.out)


if __name__ == "__main__":
    main()
