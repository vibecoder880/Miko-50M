"""Train the Miko BPE tokenizer on a set of text files (plan §9, §48).

Example::

    python tools/train_tokenizer.py \
        --input data/raw/*.txt --vocab-size 32000 \
        --output tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.config import SPECIAL_TOKENS  # noqa: E402
from miko.tokenizer import MikoTokenizer, train_tokenizer  # noqa: E402
from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.tokenize")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Miko BPE tokenizer")
    ap.add_argument("--input", nargs="+", required=True,
                    help="text files or glob patterns")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--output", default="tokenizer/tokenizer.json")
    args = ap.parse_args()

    files: list[str] = []
    for pattern in args.input:
        matched = sorted(glob.glob(pattern))
        if matched:
            files.extend(matched)
        elif os.path.isfile(pattern):
            files.append(pattern)
        else:
            logger.warning("no match for --input pattern: %s", pattern)
    if not files:
        raise SystemExit("no input files found")

    total_bytes = sum(os.path.getsize(f) for f in files)
    logger.info("training on %d file(s), %.2f MB, vocab_size=%d",
                len(files), total_bytes / 1e6, args.vocab_size)

    tok = train_tokenizer(
        files,
        vocab_size=args.vocab_size,
        output_path=args.output,
        min_frequency=args.min_frequency,
    )

    # Sanity checks: special-token ids + Vietnamese roundtrip.
    ids = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
    sample = "Xin chào! Tôi là Miko. def add(a, b): return a + b"
    rt = tok.decode(tok.encode(sample, add_special_tokens=False))
    assert rt == sample, f"roundtrip mismatch: {rt!r}"

    summary = {
        "output": args.output,
        "vocab_size": tok.vocab_size,
        "special_tokens": ids,
        "sample_tokens": len(tok.encode(sample, add_special_tokens=False)),
        "bytes_trained": total_bytes,
    }
    with open(os.path.splitext(args.output)[0] + ".summary.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("vocab=%d | sample tokens=%d | saved %s",
                tok.vocab_size, summary["sample_tokens"], args.output)


if __name__ == "__main__":
    main()
