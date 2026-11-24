"""Greedy copy/behavior probe for an SFT checkpoint (cycle diagnostics).

Usage::

    python scripts/probe_copy.py checkpoints_sft_v4/latest [--max-new 56]

Prints one line per probe with the specials-visible generation, so the
<tool_call> marker is visible in the output.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.chat_template import ChatMessage, encode_chat  # noqa: E402
from miko.checkpoint import load_model  # noqa: E402
from miko.generation import GenerationConfig, generate  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402
from miko.utils import seed_everything  # noqa: E402

TOOL_SYSTEM = (
    "Bạn là Miko. Khi cần tính toán chính xác, hãy dùng công cụ:\n"
    "<tool_call>\n"
    '{"name": "calculator", "arguments": {"expression": "..."}}\n'
    "</tool_call>"
)
CHAT_SYSTEM = "Bạn là Miko, một trợ lý nhỏ nói tiếng Việt."

# (label, system, user prompt)
PROBES = [
    ("echo     ", TOOL_SYSTEM, "Nhắc lại biểu thức: 4787*2458"),
    ("disp2cmp ", TOOL_SYSTEM, "Viết biểu thức 4787 × 2458 dưới dạng gọn."),
    ("tool-new ", TOOL_SYSTEM, "123 nhân 456 bằng bao nhiêu?"),
    ("tool-sum ", TOOL_SYSTEM, "Tính tổng 45678 và 12345."),
    ("no-tool  ", TOOL_SYSTEM, "Bạn có thích nhạc không?"),
    ("greet    ", CHAT_SYSTEM, "Xin chào!"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", default="tokenizer/real_v0.2/tokenizer.json")
    ap.add_argument("--max-new", type=int, default=56)
    ap.add_argument("--only", nargs="*", help="run just these labels")
    args = ap.parse_args()

    seed_everything(5)
    model = load_model(args.checkpoint)
    model.eval()
    tok = MikoTokenizer.load(args.tokenizer)
    cfg = GenerationConfig(max_new_tokens=args.max_new, greedy=True,
                           eos_token_id=tok.eos_id)

    for label, system, user in PROBES:
        if args.only and label.strip() not in args.only:
            continue
        ids = encode_chat(tok, [ChatMessage("system", system),
                                ChatMessage("user", user)],
                          add_generation_prompt=True)
        out = generate(model, torch.tensor([ids]), cfg=cfg)
        new = out[0, len(ids):].tolist()
        text = tok.decode(new, skip_special_tokens=False)
        print(f"{label}: {text!r}")
        print(f"           eos={tok.eos_id in new} tool_call_id8={8 in new}")


if __name__ == "__main__":
    main()
