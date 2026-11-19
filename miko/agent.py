"""Bounded agent loop (plan §34-36, §53 — M8 foundation).

The controller — not the model — owns safety: max steps, max tool calls,
context budget, and fail-safe tool execution (plan §36 checks 1-6 run in
``miko.tooling.run_tool``; timeouts/permissions are policy knobs here).

Loop::

    while step < max_steps:
        out = generate(...)
        call = parse_tool_call(out)
        if call is None:            return final answer
        if calls >= max_tool_calls:  return forced stop
        result = run_tool(call)
        append (assistant=out, tool=result) turns
        truncate to the context budget

Generation is injected (``generate_fn``), so the loop is unit-testable
without a live model and reusable for any checkpoint later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .chat_template import ChatMessage, encode_chat
from .tooling import ToolRegistry, parse_tool_call, run_tool

__all__ = ["AgentConfig", "AgentResult", "run_agent"]

GenerateFn = Callable[[list[ChatMessage]], str]
"""Renders the conversation, generates, and returns the assistant text."""


def _strip_call_block(text: str) -> str:
    """Remove tool-call markup when surfacing leftover text as an answer."""
    from .tooling import _CALL_RE

    return _CALL_RE.sub("", text).strip() or "(không có câu trả lời)"


@dataclass
class AgentConfig:
    max_steps: int = 4
    max_tool_calls: int = 4
    max_context_tokens: int = 768      # plan §52-style context budget
    forced_stop_message: str = (
        "Miko đã đạt giới hạn số bước. Đây là phần trả lời tốt nhất:"
    )


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_calls: list[dict] = field(default_factory=list)
    stopped_reason: str = "final"      # final | no_tool | max_tool_calls | max_steps


def _truncate(messages: list[ChatMessage], keep_system: bool = True) -> None:
    """Cheap in-place context trim: drop the oldest non-system turns first."""
    # The token-precise budget lives with the real generator (it owns the
    # tokenizer); the loop only guards runaway growth structurally.
    while len(messages) > 24:
        for i, m in enumerate(messages):
            if m.role != "system":
                del messages[i]
                break
        else:
            break


def run_agent(
    generate_fn: GenerateFn,
    registry: ToolRegistry,
    user_message: str,
    config: Optional[AgentConfig] = None,
    system: Optional[str] = None,
) -> AgentResult:
    """Run one bounded user request to a final answer.

    ``generate_fn(messages)`` must return the *new* assistant text (it may
    contain a ``<tool_call>`` block); the loop appends turns and re-invokes
    until the model answers without a call or a bound trips.
    """
    cfg = config or AgentConfig()
    messages: list[ChatMessage] = []
    if system:
        messages.append(ChatMessage("system", system))
    messages.append(ChatMessage("user", user_message))

    tool_calls: list[dict] = []
    steps = 0
    last_text = ""
    consecutive_errors = 0

    while steps < cfg.max_steps:
        steps += 1
        text = generate_fn(list(messages))
        messages.append(ChatMessage("assistant", text))
        last_text = text

        parsed = parse_tool_call(text)
        if parsed is None:
            # No tool-call block at all: this is the final answer.
            return AgentResult(text, steps, tool_calls, "final")

        if "error" in parsed:
            # Malformed call (plan §36 repair path): feed the error back so
            # the model can retry with a corrected block, but bound retries.
            consecutive_errors += 1
            if consecutive_errors > 2:
                return AgentResult(
                    f"Miko không gọi được công cụ ({parsed['error']}). "
                    "Đây là câu trả lời tốt nhất:\n" + _strip_call_block(text),
                    steps, tool_calls, "max_tool_calls",
                )
            messages.append(ChatMessage("tool", f"error: {parsed['error']}"))
            _truncate(messages)
            continue

        if len(tool_calls) >= cfg.max_tool_calls:
            return AgentResult(
                f"{cfg.forced_stop_message}\n{text}",
                steps, tool_calls, "max_tool_calls",
            )

        consecutive_errors = 0
        call = parsed["call"]
        result = run_tool(registry, call)  # never raises (plan §36)
        tool_calls.append({
            "name": call.name,
            "arguments": call.arguments,
            "result": result,
        })
        messages.append(ChatMessage("tool", result))
        _truncate(messages)

    # Bound tripped mid-loop: force a plain answer from what we have.
    forced = generate_fn(list(messages) + [ChatMessage("user", "Trả lời ngắn gọn.")])
    return AgentResult(forced, steps, tool_calls, "max_steps")


def encode_for_model(tokenizer, messages: list[ChatMessage],
                     max_tokens: int) -> list[int]:
    """Tokenize a conversation under a budget (oldest non-system turns drop)."""
    ids = encode_chat(tokenizer, messages)
    while len(ids) > max_tokens:
        for i, m in enumerate(messages):
            if m.role != "system":
                del messages[i]
                break
        else:
            break
        ids = encode_chat(tokenizer, messages)
    return ids
