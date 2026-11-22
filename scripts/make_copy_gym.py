"""Copy-gym SFT seed (v3 cycle) — teach the number-copy mechanism directly.

Two SFT cycles proved the 38M model learns every surface pattern (preface,
JSON scaffold, restate-then-call) but never binds the question's digits
into the slot it emits. That copy mechanism gets no dedicated gradient in
either seed, so v3 trains it explicitly, in the SAME chat format and with
the SAME system prompt the tool calls use — maximal transfer to the
expression slot.

Task families (all assistant content is verified against the user numbers
by construction):
  1. echo        — repeat the expression/number exactly
  2. disp2compact— "4787 × 2458" (display) -> "4787*2458" (compact): the
                   exact mapping the tool JSON needs
  3. transcribe  — repeat 2-3 numbers in order
  4. restate     — "Để tính 4787 × 2458 chính xác, Miko dùng máy tính."
                   WITHOUT the tool call: the exact prefix of the v0.3/v0.4
                   tool conversations, pure copy-slot gradient
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.copy_gym")

SYSTEM = (
    "Bạn là Miko. Khi cần tính toán chính xác, hãy dùng công cụ:\n"
    "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"...\"}}\n</tool_call>"
)

OPS = [("*", " × ", "nhân"), ("+", " + ", "cộng"), ("-", " − ", "trừ"),
       ("/", " ÷ ", "chia")]


def _num(rng: random.Random) -> int:
    """2-6 digit operand — spans the eval range (123, 45678, 987654...)."""
    digits = rng.choice([2, 3, 3, 4, 4, 5, 6])
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    return rng.randint(lo, hi)


def _expr(rng: random.Random) -> tuple[int, str, int, str]:
    a = _num(rng)
    b = _num(rng)
    op, disp, word = rng.choice(OPS)
    if op == "/":
        b = rng.randint(2, 99)
        a = b * rng.randint(11, 9999)  # clean division
    return a, op, b, disp


def build(rng: random.Random, n_each: int) -> list[dict]:
    convs: list[dict] = []

    def add(u: str, a: str) -> None:
        convs.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]})

    for _ in range(n_each):
        a, op, b, disp = _expr(rng)
        compact = f"{a}{op}{b}"
        display = f"{a}{disp}{b}"
        # 1. echo
        q = rng.choice([
            f"Nhắc lại biểu thức: {compact}",
            f"Viết lại đúng biểu thức {compact}.",
        ])
        add(q, compact)
        # 2. display -> compact (the exact JSON-slot mapping)
        add(f"Viết biểu thức {display} dưới dạng gọn.", compact)
        # 3. transcribe (2-3 numbers)
        nums = [_num(rng) for _ in range(rng.choice([2, 2, 3]))]
        add("Nhắc lại các số: " + ", ".join(str(n) for n in nums),
            ", ".join(str(n) for n in nums))
        # 4. restate-only (exact prefix of tool convs)
        add(f"Tính {a} {rng.choice(['nhân', 'cộng với', 'trừ', 'chia'])} {b}.",
            f"Để tính {display} chính xác, Miko dùng máy tính.")
        # 4b. restate with the same phrasings as tool questions
        a2, op2, b2, disp2 = _expr(rng)
        add(f"{a2} {rng.choice(['nhân', 'cộng với'])} {b2} bằng bao nhiêu?",
            f"Để tính {a2}{disp2}{b2} chính xác, Miko dùng máy tính.")
    return convs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft/copy_gym_v0.1.jsonl")
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--n-each", type=int, default=1600,
                    help="convs per task family (4 families -> ~8K total)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    convs = build(rng, args.n_each)

    # Verify by construction: every assistant turn carries digits that
    # appear in the user turn (echo/compact/restate all satisfy this).
    for c in convs:
        u_digits = {ch for m in c["messages"] if m["role"] == "user"
                    for ch in m["content"] if ch.isdigit()}
        a_digits = {ch for m in c["messages"] if m["role"] == "assistant"
                    for ch in m["content"] if ch.isdigit()}
        assert a_digits and a_digits <= u_digits, \
            f"assistant invented digits: {c['messages'][1]['content'][:60]}"

    # Dedup
    seen: set[str] = set()
    unique = []
    for c in convs:
        k = json.dumps(c["messages"], ensure_ascii=False)
        if k not in seen:
            seen.add(k)
            unique.append(c)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for c in unique:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("wrote %d copy-gym conversations -> %s", len(unique), args.out)


if __name__ == "__main__":
    main()
