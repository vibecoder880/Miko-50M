"""Tokenize + pack raw text into binary training shards (plan §13-14).

Documents are separated by one or more blank lines. Each document is encoded
as ``<bos> ... <eos>`` and appended to the shard stream, then split into
train/val by document count.

Example::

    python tools/pack_dataset.py --input "data/cleaned/*.txt" \
        --tokenizer tokenizer/tokenizer.json \
        --out-train data/shards/train --out-val data/shards/val \
        --val-frac 0.02 --docs-per-shard 5000
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.dataset import PackedShardWriter, shard_stats  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402
from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.pack")


def iter_documents(patterns: list[str], min_chars: int, max_chars: int):
    """Yield documents from text files, split on blank lines, length-filtered."""
    n_files = 0
    for pattern in patterns:
        matched = sorted(glob.glob(pattern)) or (
            [pattern] if os.path.isfile(pattern) else []
        )
        for path in matched:
            n_files += 1
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                doc_lines: list[str] = []
                for line in fh:
                    if line.strip() == "":
                        if doc_lines:
                            yield "\n".join(doc_lines)
                            doc_lines = []
                    else:
                        doc_lines.append(line.rstrip())
                if doc_lines:
                    yield "\n".join(doc_lines)

    logger.info("read %d file(s)", n_files)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack text into token shards")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out-train", default="data/shards/train")
    ap.add_argument("--out-val", default="data/shards/val")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--docs-per-shard", type=int, default=5000)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = MikoTokenizer.load(args.tokenizer)
    bos, eos = tok.bos_id, tok.eos_id

    rng = random.Random(args.seed)
    os.makedirs(args.out_train, exist_ok=True)
    os.makedirs(args.out_val, exist_ok=True)
    val_every = round(1 / args.val_frac) if args.val_frac > 0 else 0

    stats = {"docs": 0, "tokens": 0, "skipped": 0}
    with PackedShardWriter(
        os.path.join(args.out_train, "part"), args.docs_per_shard
    ) as train_w, PackedShardWriter(
        os.path.join(args.out_val, "part"), args.docs_per_shard
    ) as val_w:
        for doc in iter_documents(args.input, args.min_chars, args.max_chars):
            if not (args.min_chars <= len(doc) <= args.max_chars):
                stats["skipped"] += 1
                continue
            ids = tok.encode(doc, add_special_tokens=False)
            if not ids:
                stats["skipped"] += 1
                continue
            # Route every val_every-th document to val (streaming split).
            if val_every and rng.randrange(val_every) == 0:
                writer = val_w
            else:
                writer = train_w
            writer.write_document(ids, bos, eos)
            stats["docs"] += 1
            stats["tokens"] += len(ids) + 2

    for name, directory in (("train", args.out_train), ("val", args.out_val)):
        shards = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(".bin")
        )
        if shards:
            s = shard_stats(shards)
            logger.info("%s: %d shard(s), %.3fM tokens, %d docs",
                        name, s.shards, s.tokens / 1e6, s.documents)

    logger.info(
        "packed %d docs / %.3fM tokens (%d skipped)", stats["docs"],
        stats["tokens"] / 1e6, stats["skipped"],
    )


if __name__ == "__main__":
    main()
