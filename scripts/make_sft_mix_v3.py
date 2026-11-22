"""Build the v3 mixed SFT dataset (copy-gym cycle).

vs v2 (70/30 mix): 50/50 SFT/pretraining and a copy-gym component.
- copy gym (7999 convs): echo / display->compact / transcribe / restate —
  dedicated gradient on the number-copy mechanism that two cycles proved
  missing, in the exact chat format + system prompt of the tool calls.
- tool seed v0.4 (5736 convs, restate 70%) + chat seed v0.2 (618).
- LM share 50% at full-label windows (v2's 30% failed to prevent
  forgetting: base-val 6.93 -> 31.96).
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from miko.chat_template import ChatMessage, encode_chat_masked  # noqa: E402
from miko.dataset import SFTShardWriter  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402
from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.sft_mix_v3")

SEQ_LEN = 128
PAD_ID = 0
LM_FRACTION = 0.50

SFT_INPUTS = [
    "data/sft/chat_seed_v0.2.jsonl",
    "data/sft/tool_seed_v0.4.jsonl",
    "data/sft/copy_gym_v0.1.jsonl",
]


def load_convs(path: str) -> list[list[ChatMessage]]:
    convs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            row = json.loads(line)
            convs.append([ChatMessage(m["role"], m["content"])
                          for m in row["messages"]])
    return convs


def pretraining_chunks(directory: str, tokens_needed: int,
                       rng: random.Random) -> list[list[int]]:
    paths = sorted(p for p in glob.glob(os.path.join(directory, "*.bin"))
                   if not p.endswith(".ids.bin") and not p.endswith(".lbl.bin"))
    if not paths:
        raise FileNotFoundError(f"no packed shards in {directory!r}")
    streams = []
    for p in paths:
        raw = np.fromfile(p, dtype=np.uint16)
        if raw.size > 200:
            streams.append(raw)
    chunks, drawn = [], 0
    while drawn < tokens_needed:
        s = streams[rng.randrange(len(streams))]
        start = rng.randrange(0, max(1, s.size - 4000))
        take = min(4000, s.size - start)
        chunks.append(s[start:start + take].tolist())
        drawn += take
    return chunks


def main() -> None:
    rng = random.Random(20261122)
    tok = MikoTokenizer.load("tokenizer/real_v0.2/tokenizer.json")

    convs: list[list[ChatMessage]] = []
    for path in SFT_INPUTS:
        part = load_convs(path)
        logger.info("%s: %d convs", path, len(part))
        convs.extend(part)
    rng.shuffle(convs)

    sft_tokens = 0
    for c in convs:
        ids, _ = encode_chat_masked(tok, c)
        sft_tokens += len(ids)
    lm_tokens = int(sft_tokens * LM_FRACTION / (1 - LM_FRACTION))
    logger.info("SFT tokens ~%d -> LM tokens %d (50/50)", sft_tokens, lm_tokens)

    lm_chunks = pretraining_chunks("data/shards_real/train", lm_tokens, rng)

    out_train, out_val = "data/shards_sft_v3/train", "data/shards_sft_v3/val"
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_val, exist_ok=True)

    val_every = 20
    with SFTShardWriter(os.path.join(out_train, "part"), SEQ_LEN, PAD_ID) as tw, \
         SFTShardWriter(os.path.join(out_val, "part"), SEQ_LEN, PAD_ID) as vw:
        for i, conv in enumerate(convs):
            ids, labels = encode_chat_masked(tok, conv)
            writer = vw if (i % val_every == 0) else tw
            writer.write_example(ids, labels)
        for j, chunk in enumerate(lm_chunks):
            writer = vw if (j % val_every == 0) else tw
            writer.write_example(chunk, chunk)  # full-LM windows

    logger.info("wrote shards -> %s", out_train)


if __name__ == "__main__":
    main()
