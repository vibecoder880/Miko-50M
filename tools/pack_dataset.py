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
import json
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.dataset import PackedShardWriter, shard_stats  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402
from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.pack")


def iter_documents(patterns: list[str], min_chars: int, max_chars: int,
                   yield_path: bool = False):
    """Yield ``(path, document)`` from text files, split on blank lines."""
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
                            doc = "\n".join(doc_lines)
                            yield (path, doc) if yield_path else doc
                            doc_lines = []
                    else:
                        doc_lines.append(line.rstrip())
                if doc_lines:
                    doc = "\n".join(doc_lines)
                    yield (path, doc) if yield_path else doc

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
    ap.add_argument("--report",
                    default="data/manifests/dataset-packed.json",
                    help="mixture manifest (per-source docs/tokens, plan §11)")
    args = ap.parse_args()

    tok = MikoTokenizer.load(args.tokenizer)
    bos, eos = tok.bos_id, tok.eos_id

    rng = random.Random(args.seed)
    os.makedirs(args.out_train, exist_ok=True)
    os.makedirs(args.out_val, exist_ok=True)
    val_every = round(1 / args.val_frac) if args.val_frac > 0 else 0

    stats = {"docs": 0, "tokens": 0, "skipped": 0}
    per_source: dict[str, dict[str, int]] = {}  # plan §11 mixture accounting
    current_source = {"path": ""}
    with PackedShardWriter(
        os.path.join(args.out_train, "part"), args.docs_per_shard
    ) as train_w, PackedShardWriter(
        os.path.join(args.out_val, "part"), args.docs_per_shard
    ) as val_w:
        for path, doc in iter_documents(args.input, args.min_chars,
                                        args.max_chars, yield_path=True):
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
            src = os.path.basename(path)
            acc = per_source.setdefault(src, {"docs": 0, "tokens": 0})
            acc["docs"] += 1
            acc["tokens"] += len(ids) + 2

    for name, directory in (("train", args.out_train), ("val", args.out_val)):
        shards = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(".bin")
        )
        if shards:
            s = shard_stats(shards)
            logger.info("%s: %d shard(s), %.3fM tokens, %d docs",
                        name, s.shards, s.tokens / 1e6, s.documents)

    total_tokens = stats["tokens"]
    mixture = {
        os.path.splitext(src)[0]: {
            "docs": acc["docs"],
            "tokens": acc["tokens"],
            "share": round(acc["tokens"] / total_tokens, 4),
        }
        for src, acc in sorted(per_source.items(), key=lambda kv: -kv[1]["tokens"])
    }
    report = {
        "packed_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": args.tokenizer,
        "inputs": list(args.input),
        "docs": stats["docs"],
        "tokens": total_tokens,
        "skipped": stats["skipped"],
        "mixture": mixture,
        "train_dir": args.out_train,
        "val_dir": args.out_val,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info(
        "packed %d docs / %.3fM tokens (%d skipped) | report: %s",
        stats["docs"], total_tokens / 1e6, stats["skipped"], args.report,
    )
    for src, m in mixture.items():
        logger.info("mixture %-16s %.1f%% (%.2fM tokens, %d docs)",
                    src, m["share"] * 100, m["tokens"] / 1e6, m["docs"])


if __name__ == "__main__":
    main()
