"""Build the v0.3 mixed SFT dataset (phase-03 playbook mitigation).

Mitigations baked in:
1. **Restate-then-call** tool conversations (seed v0.3): the assistant
   preface restates the expression so the JSON copies adjacent context.
2. **70/30 SFT / pretraining mix**: pretraining token streams become
   full-LM SFT windows (labels = ids) so ordinary-text statistics keep
   receiving gradient — the tool-only cycle regressed every other tier
   (base-val 6.93 -> 37.8, catastrophic forgetting).

Output: data/shards_sft_v2/{train,val} in SFTShardWriter format, readable
by the standard ``tools/train.py --sft`` path.
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

logger = get_logger("miko.sft_mix")

SEQ_LEN = 128
PAD_ID = 0  # <pad>
LM_FRACTION = 0.30
TOKENS_PER_WINDOW = SEQ_LEN


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


def pretraining_stream_chunks(directory: str, tokens_needed: int,
                              rng: random.Random) -> list[list[int]]:
    """Sample raw token chunks from packed uint16 pretraining shards."""
    paths = sorted(p for p in glob.glob(os.path.join(directory, "*.bin"))
                   if not p.endswith(".ids.bin") and not p.endswith(".lbl.bin"))
    if not paths:
        raise FileNotFoundError(f"no packed shards in {directory!r}")
    streams = []
    for p in paths:
        raw = np.fromfile(p, dtype=np.uint16)
        if raw.size > 200:
            streams.append(raw)
    total = sum(s.size for s in streams)
    logger.info("pretraining pool: %d shards / %.2fM tokens", len(streams),
                total / 1e6)
    chunks = []
    drawn = 0
    while drawn < tokens_needed:
        s = streams[rng.randrange(len(streams))]
        start = rng.randrange(0, max(1, s.size - 4000))
        take = min(4000, s.size - start)
        chunks.append(s[start:start + take].tolist())
        drawn += take
    return chunks


def main() -> None:
    rng = random.Random(20261121)
    tok = MikoTokenizer.load("tokenizer/real_v0.2/tokenizer.json")

    chat = load_convs("data/sft/chat_seed_v0.2.jsonl")
    tool = load_convs("data/sft/tool_seed_v0.3.jsonl")
    logger.info("conversations: chat %d + tool %d", len(chat), len(tool))

    # Estimate SFT token volume to size the LM share.
    def conv_tokens(conv: list[ChatMessage]) -> int:
        ids, _ = encode_chat_masked(tok, conv)
        return len(ids)

    sft_tokens = sum(conv_tokens(c) for c in chat + tool)
    lm_tokens = int(sft_tokens * LM_FRACTION / (1 - LM_FRACTION))
    logger.info("SFT tokens ~%d -> LM windows target %d tokens (%.0f%% mix)",
                sft_tokens, lm_tokens, 100 * LM_FRACTION)

    lm_chunks = pretraining_stream_chunks("data/shards_real/train",
                                          lm_tokens, rng)

    out_train = "data/shards_sft_v2/train"
    out_val = "data/shards_sft_v2/val"
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_val, exist_ok=True)

    # Streaming 5% val split for conversations; LM chunks: one chunk -> val.
    val_every = 20
    with SFTShardWriter(os.path.join(out_train, "part"), SEQ_LEN, PAD_ID) as tw, \
         SFTShardWriter(os.path.join(out_val, "part"), SEQ_LEN, PAD_ID) as vw:
        convs = [(c, "sft") for c in chat] + [(c, "sft") for c in tool]
        rng.shuffle(convs)
        for i, (conv, _) in enumerate(convs):
            ids, labels = encode_chat_masked(tok, conv)
            writer = vw if (i % val_every == 0) else tw
            writer.write_example(ids, labels)
        for j, chunk in enumerate(lm_chunks):
            writer = vw if (j % val_every == 0) else tw
            writer.write_example(chunk, chunk)  # full-LM: targets = ids

    n_train = len(glob.glob(os.path.join(out_train, "*.ids.bin")))
    n_val = len(glob.glob(os.path.join(out_val, "*.ids.bin")))
    logger.info("wrote %d train shards / %d val shards -> %s", n_train, n_val,
                out_train)


if __name__ == "__main__":
    main()
