"""Pack instruction/chat JSONL into SFT shards (plan §26-27, M4).

Input format (one per line)::

    {"messages": [
        {"role": "system", "content": "Bạn là Miko."},
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn!"}
    ]}

Default mode writes **masked-label SFT windows** (``SFTShardWriter``):
targets equal ids inside assistant spans and ``-100`` elsewhere, so
``tools/train.py --sft`` trains only on assistant responses.
``--full-lm`` keeps the legacy pretraining-style stream packing instead.

Examples::

    python tools/pack_sft_dataset.py --input "data/sft/*.jsonl" \
        --tokenizer tokenizer/real_v0.2/tokenizer.json \
        --out-train data/shards_sft/train --out-val data/shards_sft/val
    python tools/pack_sft_dataset.py --input ... --full-lm
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

from miko.chat_template import ChatMessage, encode_chat, encode_chat_masked  # noqa: E402
from miko.dataset import (  # noqa: E402
    PackedShardWriter,
    SFTShardWriter,
    shard_stats,
)
from miko.tokenizer import MikoTokenizer  # noqa: E402
from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.pack_sft")


def load_conversations(paths: list[str]) -> list[list[ChatMessage]]:
    convs: list[list[ChatMessage]] = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or ([pattern] if os.path.isfile(pattern) else []):
            with open(path, "r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
                    msgs = row.get("messages") or []
                    if not msgs:
                        raise ValueError(f"{path}:{line_no}: missing 'messages'")
                    conv = [ChatMessage(str(m["role"]), str(m["content"])) for m in msgs]
                    # Contract: every conversation must end with an assistant
                    # turn, otherwise there is nothing to learn to answer.
                    if conv[-1].role != "assistant":
                        raise ValueError(
                            f"{path}:{line_no}: last message must be assistant"
                        )
                    convs.append(conv)
    if not convs:
        raise ValueError("no conversations loaded")
    return convs


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack chat JSONL into SFT shards")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/real_v0.2/tokenizer.json")
    ap.add_argument("--out-train", default="data/shards_sft/train")
    ap.add_argument("--out-val", default="data/shards_sft/val")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=6,
                    help="drop conversations whose total text is shorter")
    ap.add_argument("--seq-len", type=int, default=128,
                    help="window size for masked SFT shards")
    ap.add_argument("--full-lm", action="store_true",
                    help="legacy mode: pretraining-style token streams "
                    "(no label masking)")
    ap.add_argument("--report", default="data/manifests/dataset-sft.json")
    args = ap.parse_args()

    tok = MikoTokenizer.load(args.tokenizer)
    bos, eos = tok.bos_id, tok.eos_id
    convs = load_conversations(args.input)
    rng = random.Random(42)
    rng.shuffle(convs)  # mix sources before the streaming val split

    os.makedirs(args.out_train, exist_ok=True)
    os.makedirs(args.out_val, exist_ok=True)
    val_every = round(1 / args.val_frac) if args.val_frac > 0 else 0

    stats = {"convs": 0, "tokens": 0, "skipped": 0}
    if args.full_lm:
        with PackedShardWriter(os.path.join(args.out_train, "part")) as tw, \
             PackedShardWriter(os.path.join(args.out_val, "part")) as vw:
            for conv in convs:
                total_chars = sum(len(m.content) for m in conv)
                if total_chars < args.min_chars:
                    stats["skipped"] += 1
                    continue
                ids = encode_chat(tok, conv)
                writer = vw if (val_every and rng.randrange(val_every) == 0) else tw
                writer.write_ids([bos, *ids, eos], documents=1)
                stats["convs"] += 1
                stats["tokens"] += len(ids) + 2
        fmt = "full-LM chat streams (bos + turns with eos per turn)"
    else:
        with SFTShardWriter(os.path.join(args.out_train, "part"), args.seq_len,
                            tok.pad_id) as tw, \
             SFTShardWriter(os.path.join(args.out_val, "part"), args.seq_len,
                            tok.pad_id) as vw:
            for conv in convs:
                total_chars = sum(len(m.content) for m in conv)
                if total_chars < args.min_chars:
                    stats["skipped"] += 1
                    continue
                ids, labels = encode_chat_masked(tok, conv)
                writer = vw if (val_every and rng.randrange(val_every) == 0) else tw
                writer.write_example([bos, *ids, eos],
                                     [-100, *labels, eos])
                stats["convs"] += 1
                stats["tokens"] += len(ids) + 2
        fmt = (f"masked-label SFT windows (seq_len={args.seq_len}, "
               "targets carry -100 outside assistant spans)")

    for name, directory in (("train", args.out_train), ("val", args.out_val)):
        shards = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(".bin")
        )
        if shards:
            if args.full_lm:
                s = shard_stats(shards)
                logger.info("%s: %d shard(s), %.3fM tokens", name, s.shards,
                            s.tokens / 1e6)
            else:
                shards = [s for s in shards if s.endswith(".ids.bin")]
                # Count real windows: rows of seq_len ids (2 bytes each).
                n_windows = sum(
                    os.path.getsize(s) // (args.seq_len * 2) for s in shards
                )
                logger.info("%s: %d shard(s), %d windows", name, len(shards),
                            n_windows)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump({
            "packed_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer": args.tokenizer,
            "inputs": list(args.input),
            "conversations": stats["convs"],
            "tokens": stats["tokens"],
            "skipped": stats["skipped"],
            "format": fmt,
            "train_dir": args.out_train,
            "val_dir": args.out_val,
        }, fh, indent=2, ensure_ascii=False)
    logger.info("packed %d conversations / %.3fM tokens (%d skipped)",
                stats["convs"], stats["tokens"] / 1e6, stats["skipped"])


if __name__ == "__main__":
    main()
