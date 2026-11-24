"""Export a versioned Miko bundle for serving (plan phase 02).

Bundle layout (flat, server-readable via ``--bundle``)::

    dist/<name>/
    ├── checkpoint/model.safetensors, config.json, trainer_state.json
    ├── tokenizer.json
    ├── VERSION        (first line: <name>-<date>-step<N>)
    └── MODEL_CARD.md  (copied when present)

Example::

    python scripts/export_bundle.py --checkpoint checkpoints_sft_v2/latest \
        --out dist/miko-sft-v2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.export")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a serving bundle")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/real_v0.2/tokenizer.json")
    ap.add_argument("--out", required=True, help="e.g. dist/miko-sft-v2")
    args = ap.parse_args()

    need = ["model.safetensors", "config.json"]
    for f in need:
        if not os.path.exists(os.path.join(args.checkpoint, f)):
            logger.error("checkpoint %s missing %s", args.checkpoint, f)
            return 1

    ckpt_out = os.path.join(args.out, "checkpoint")
    os.makedirs(ckpt_out, exist_ok=True)
    for f in os.listdir(args.checkpoint):
        if f == "optimizer.pt":  # serving never resumes training
            continue
        shutil.copy2(os.path.join(args.checkpoint, f), os.path.join(ckpt_out, f))
    shutil.copy2(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    if os.path.exists("MODEL_CARD.md"):
        shutil.copy2("MODEL_CARD.md", os.path.join(args.out, "MODEL_CARD.md"))

    step = 0
    state_path = os.path.join(args.checkpoint, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as fh:
            step = int(json.load(fh).get("step", 0))
    name = os.path.basename(args.out.rstrip("/"))
    version = f"{name}-{date.today().isoformat()}-step{step}"
    with open(os.path.join(args.out, "VERSION"), "w", encoding="utf-8") as fh:
        fh.write(version + "\n" + json.dumps(
            {"name": name, "step": step, "exported": date.today().isoformat()},
            ensure_ascii=False) + "\n")

    logger.info("bundle %s -> %s (version %s)", args.checkpoint, args.out,
                version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
