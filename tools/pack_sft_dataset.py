"""Pack instruction/chat JSONL into token shards for SFT (plan §26-27, M4).

Input format (one per line)::

    {"messages": [
        {"role": "system", "content": "Bạn là Miko."},
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn!"}
    ]}

Each conversation is rendered with the reserved special tokens
(``miko.chat_template``), wrapped as ``<bos> ... <eos>`` and packed into the
same uint16 shard format used for pretraining — so the existing training
loop trains on chat streams unchanged (full-LM SFT; masked-label SFT is a
later refinement, tracked for M4 follow-up).

Example::

    python tools/pack_sft_dataset.py --input data/sft/chat_v0.1.jsonl \
        --tokenizer tokenizer/real_v0.2/tokenizer.json \
        --out-train data/shards_sft/train --out-val data/shards_sft/val
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

from miko.chat_template import ChatMessage, encode_chat  # noqa: E402
from miko.dataset import PackedShardWriter, shard_stats  # noqa: E402
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

    for name, directory in (("train", args.out_train), ("val", args.out_val)):
        shards = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(".bin")
        )
        if shards:
            s = shard_stats(shards)
            logger.info("%s: %d shard(s), %.3fM tokens", name, s.shards,
                        s.tokens / 1e6)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump({
            "packed_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer": args.tokenizer,
            "inputs": list(args.input),
            "conversations": stats["convs"],
            "tokens": stats["tokens"],
            "skipped": stats["skipped"],
            "format": "full-LM chat streams (bos + turns with eos per turn)",
            "train_dir": args.out_train,
            "val_dir": args.out_val,
        }, fh, indent=2, ensure_ascii=False)
    logger.info("packed %d conversations / %.3fM tokens (%d skipped)",
                stats["convs"], stats["tokens"] / 1e6, stats["skipped"])


if __name__ == "__main__":
    main()
