"""Generate verified tool-calling SFT data (plan §29, §33 — M5 seed).

Every conversation follows the plan §33 shape, with answers COMPUTED by the
real ``safe_calc`` tool at generation time — never invented::

    user:      "123 * 456 bằng bao nhiêu?"
    assistant: <tool_call>{"name": "calculator", "arguments":
               {"expression": "123*456"}}</tool_call>
    tool:      56088
    assistant: "Kết quả là 56088."

Numbers are deliberately multi-digit so the 38M model must delegate to the
tool rather than memorize arithmetic. Also includes time-tool examples and
error-recovery examples (bad JSON / unknown tool -> error result -> retry),
per plan §36 (repair path).

Usage::

    PYTHONPATH=scripts python scripts/make_tool_seed.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from miko.tooling import (  # noqa: E402
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
    make_default_registry,
    run_tool,
    safe_calc,
    ToolCall,
)
from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.tool_seed")

SYSTEM = (
    "Bạn là Miko. Khi cần tính toán chính xác, hãy dùng công cụ:\n"
    "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"...\"}}\n</tool_call>"
)


def _call_block(expr: str) -> str:
    payload = json.dumps(
        {"name": "calculator", "arguments": {"expression": expr}},
        ensure_ascii=False,
    )
    return f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"


def _result_block(result: str) -> str:
    return f"{TOOL_RESULT_OPEN}\n{result}\n{TOOL_RESULT_CLOSE}"


def _mul_question(rng: random.Random) -> tuple[str, str]:
    """A multiplication too large to memorize: a x b, 3-4 digit operands."""
    a = rng.randint(100, 9999)
    b = rng.randint(100, 9999)
    expr = f"{a}*{b}"
    q = rng.choice([
        f"{a} nhân {b} bằng bao nhiêu?",
        f"Tính {a} × {b}.",
        f"{a} * {b} = ?",
        f"Cho mình biết kết quả của {a} nhân {b}.",
    ])
    return q, expr


def _sum_question(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(1000, 999999)
    b = rng.randint(1000, 999999)
    q = rng.choice([
        f"Tính tổng {a} và {b}.",
        f"{a} + {b} bằng mấy?",
        f"{a} cộng với {b} được bao nhiêu?",
    ])
    return q, f"{a}+{b}"


def _sub_question(rng: random.Random) -> tuple[str, str]:
    """Subtraction with a positive result, 4-6 digit operands."""
    a = rng.randint(100000, 999999)
    b = rng.randint(1000, a - 1)
    q = rng.choice([
        f"Tính hiệu của {a} và {b}.",
        f"{a} trừ {b} bằng mấy?",
        f"{a} - {b} = ?",
    ])
    return q, f"{a}-{b}"


def _div_question(rng: random.Random) -> tuple[str, str]:
    """Clean division: quotient stays integer (answer reads naturally)."""
    b = rng.randint(2, 99)
    q = rng.randint(11, 9999)
    a = b * q
    q_text = rng.choice([
        f"Tính thương của {a} và {b}.",
        f"{a} chia {b} bằng mấy?",
        f"{a} / {b} = ?",
    ])
    return q_text, f"{a}/{b}"


def _paren_question(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(2, 99)
    b = rng.randint(2, 99)
    c = rng.randint(2, 20)
    expr = f"({a}+{b})*{c}"
    q = f"Tính ({a} + {b}) × {c}."
    return q, expr


def _power_question(rng: random.Random) -> tuple[str, str]:
    base = rng.randint(2, 30)
    exp = rng.randint(2, 5)
    q = rng.choice([
        f"{base} mũ {exp} bằng bao nhiêu?",
        f"Tính {base}^{exp}.",
    ])
    return q, f"{base}**{exp}"


def _display_expr(expr: str) -> str:
    """Human-readable operator forms for the restate-then-call preface."""
    return (expr.replace("**", "^").replace("*", " × ")
                .replace("/", " ÷ ").replace("+", " + ").replace("-", " − "))


def build_conversations(rng: random.Random, core_count: int = 240,
                        restate_pct: float = 0.0) -> list[dict]:
    registry = make_default_registry()
    convs: list[dict] = []

    def add(msgs: list[dict]) -> None:
        convs.append({"messages": msgs})

    generators = [
        _mul_question, _sum_question, _sub_question, _div_question,
        _paren_question, _power_question,
    ]

    # Core pattern: question -> tool_call -> tool_result -> final answer.
    # The copy-binding gap (phase-3 eval: expression always '7') needs volume
    # AND operator coverage — v0.1 trained mul/sum/paren/pow only, yet the
    # eval suite also tests subtraction and division.
    for _ in range(core_count):
        gen = rng.choice(generators)
        question, expr = gen(rng)
        call = ToolCall(name="calculator", arguments={"expression": expr})
        result = run_tool(registry, call)
        if result.startswith("error:"):
            continue  # skip degenerate draws (e.g. division by zero patterns)

        assistant_call = _call_block(expr)
        if rng.random() < restate_pct:
            # Restate-then-call scaffold: the preface carries the numbers so
            # the JSON expression copies from adjacent, just-emitted context
            # instead of the distant question (v0.2 showed the direct
            # question->expression copy never formed at this scale).
            assistant_call = (f"Để tính {_display_expr(expr)} chính xác, "
                              "Miko dùng máy tính.\n" + assistant_call)
        elif rng.random() < 0.3:
            # With a short generic "let me compute" preamble.
            assistant_call = "Để tính chính xác, Miko dùng máy tính.\n" + assistant_call

        add([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_call},
            {"role": "tool", "content": result},
            {"role": "assistant",
             "content": rng.choice([
                 f"Kết quả là {result}.",
                 f"{expr.replace('*', ' × ').replace('+', ' + ')} = {result}.",
                 f"Đáp số: {result}.",
             ])},
        ])

    # Multi-step tool chains (two calls in one conversation).
    for _ in range(40):
        q1, e1 = _mul_question(rng)
        q2, e2 = _sum_question(rng)
        r1 = run_tool(registry, ToolCall("calculator", {"expression": e1}))
        r2 = run_tool(registry, ToolCall("calculator", {"expression": e2}))
        add([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q1},
            {"role": "assistant", "content": _call_block(e1)},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": f"Kết quả là {r1}."},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": _call_block(e2)},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": f"Kết quả là {r2}."},
        ])

    # Error-recovery: malformed JSON -> tool error -> corrected retry.
    for a, b in [(123, 456), (789, 12), (555, 77)]:
        expected = safe_calc(f"{a}*{b}")
        broken = '{"name": "calculator", "arguments": {"expr"'  # truncated JSON
        add([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{a} nhân {b} bằng bao nhiêu?"},
            {"role": "assistant",
             "content": f"{TOOL_CALL_OPEN}\n{broken}\n{TOOL_CALL_CLOSE}"},
            {"role": "tool",
             "content": "error: invalid JSON: Expecting ',' delimiter"},
            {"role": "assistant",
             "content": "Miko gửi lại đúng định dạng.\n" + _call_block(f"{a}*{b}")},
            {"role": "tool", "content": expected},
            {"role": "assistant", "content": f"Kết quả là {expected}. Miko xin lỗi vì lần đầu sai định dạng."},
        ])

    # Unknown-argument recovery: wrong arg name -> error -> fix.
    for a, b in [(321, 654), (111, 222)]:
        expected = safe_calc(f"{a}*{b}")
        wrong = json.dumps({"name": "calculator",
                            "arguments": {"expr": f"{a}*{b}"}},
                           ensure_ascii=False)
        add([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Tính {a} × {b}."},
            {"role": "assistant",
             "content": f"{TOOL_CALL_OPEN}\n{wrong}\n{TOOL_CALL_CLOSE}"},
            {"role": "tool",
             "content": "error: missing required argument: 'expression'"},
            {"role": "assistant", "content": _call_block(f"{a}*{b}")},
            {"role": "tool", "content": expected},
            {"role": "assistant", "content": f"Kết quả là {expected}."},
        ])

    # Time tool examples.
    for fmt in ["%Y-%m-%d", "%H:%M:%S", "%d/%m/%Y"]:
        call = ToolCall("time", {"format": fmt})
        result = run_tool(registry, call)
        add([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": rng.choice([
                "Bây giờ là mấy giờ?",
                "Hôm nay ngày mấy?",
            ])},
            {"role": "assistant",
             "content": "Để xem giờ chính xác, Miko dùng công cụ time.\n"
             + f"{TOOL_CALL_OPEN}\n{json.dumps({'name': 'time', 'arguments': {'format': fmt}}, ensure_ascii=False)}\n{TOOL_CALL_CLOSE}"},
            {"role": "tool", "content": result},
            {"role": "assistant", "content": f"Theo đồng hồ máy chủ: {result}."},
        ])

    # Plain no-tool answers (model must NOT call when unnecessary).
    for q, a in [
        ("Bạn là ai?", "Tôi là Miko, một trợ lý nhỏ nói tiếng Việt."),
        ("2 + 2 bằng mấy?", "2 + 2 bằng 4."),
        ("Cảm ơn nhé!", "Không có gì!"),
    ]:
        add([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ])

    return convs


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate verified tool SFT data")
    ap.add_argument("--out", default="data/sft/tool_seed_v0.1.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--core-count", type=int, default=240,
                    help="number of core Q->call conversations (v0.2 uses 1200)")
    ap.add_argument("--restate-pct", type=float, default=0.0,
                    help="fraction of core convs using the restate-then-call "
                    "preface (v0.3: 0.7)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    convs = build_conversations(rng, core_count=args.core_count,
                                restate_pct=args.restate_pct)

    seen: set[str] = set()
    unique = []
    for c in convs:
        key = json.dumps(c["messages"], ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    # Final verification pass: re-execute every *parseable* calculator call
    # in the data and require the tool-result block to match. Intentionally
    # malformed blocks (error-recovery examples) are skipped when their tool
    # reply is an error string.
    reg = make_default_registry()
    for c in unique:
        for i, m in enumerate(c["messages"]):
            if m["role"] == "assistant" and TOOL_CALL_OPEN in m["content"]:
                result_msg = c["messages"][i + 1]
                assert result_msg["role"] == "tool"
                if "error:" in result_msg["content"]:
                    continue  # error-recovery example — expected mismatch
                parsed = run_tool(reg, _parse_or_fail(m["content"]))
                assert parsed == result_msg["content"], (
                    f"data/tool mismatch: {parsed!r} vs {result_msg['content']!r}"
                )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for c in unique:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("wrote %d tool conversations -> %s", len(unique), args.out)


def _parse_or_fail(text: str):
    from miko.tooling import parse_tool_call

    parsed = parse_tool_call(text)
    assert parsed and "call" in parsed, f"unparseable call in data: {text[:80]}"
    return parsed["call"]


if __name__ == "__main__":
    main()
