"""Tool-calling infrastructure (plan §28-29, §36 — M5).

The model *learns the format*; the runtime executes. Contract:

* assistant content may contain one call::

      <tool_call>
      {"name": "calculator", "arguments": {"expression": "123*456"}}
      </tool_call>

* the runtime parses it, validates against the registry (existence, JSON,
  schema, required args — plan §36 checks 1-4), executes, and feeds back::

      <tool_result>
      56088
      </tool_result>

Validation-first: a 38M model *will* emit malformed JSON sometimes; the
runtime must fail safe (report an error result) rather than crash.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import time as _time
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "ToolCall",
    "Tool",
    "ToolRegistry",
    "safe_calc",
    "parse_tool_call",
    "run_tool",
    "make_default_registry",
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"

_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, type]          # arg name -> python type
    required: tuple[str, ...] = ()
    func: Optional[Any] = None           # callable(**arguments) -> str

    def schema_prompt(self) -> str:
        """Compact schema block for the model context (plan §28)."""
        params = ", ".join(
            f"{n}: {t.__name__}" + ("" if n in self.required else " (optional)")
            for n, t in self.parameters.items()
        )
        return f"- {self.name}({params}): {self.description}"


class ToolRegistry:
    """Registry with plan §36 validation (steps 1-4; timeout/permission are
    runtime-controller concerns layered outside the model)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name!r}")
        if tool.func is None:
            raise ValueError(f"tool {tool.name!r} needs a func")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def to_prompt(self) -> str:
        """Render all tool schemas as text for the model context."""
        return "\n".join(t.schema_prompt() for t in
                         sorted(self._tools.values(), key=lambda t: t.name))

    def validate_call(self, name: str, arguments: Any) -> Optional[str]:
        """Return an error string, or None when the call is valid."""
        tool = self._tools.get(name)                      # 1. exists?
        if tool is None:
            return f"unknown tool: {name!r}"
        if not isinstance(arguments, dict):               # 2. shape
            return "arguments must be a JSON object"
        for arg in tool.required:                         # 4. required args
            if arg not in arguments:
                return f"missing required argument: {arg!r}"
        for arg, value in arguments.items():              # 3. schema types
            expected = tool.parameters.get(arg)
            if expected is None:
                return f"unexpected argument: {arg!r}"
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                continue  # ints are acceptable where a number is expected
            if not isinstance(value, expected):
                return (f"argument {arg!r} must be {expected.__name__}, "
                        f"got {type(value).__name__}")
        return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    """Extract and validate the first ``<tool_call>`` block in ``text``.

    Returns a dict::

        {"call": ToolCall}                       on success
        {"error": "<reason>", "raw": "<block>"}  on any failure

    and ``None`` when the text contains no tool-call block (plain answer).
    """
    match = _CALL_RE.search(text)
    if match is None:
        return None
    raw = match.group(0)
    body = match.group(1)

    try:                                                  # 2. JSON validity
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON: {e}", "raw": raw}
    if not isinstance(data, dict):
        return {"error": "tool call must be a JSON object", "raw": raw}

    name = data.get("name")
    arguments = data.get("arguments", {})
    if not isinstance(name, str) or not name:
        return {"error": "missing 'name'", "raw": raw}

    return {"call": ToolCall(name=name, arguments=arguments, raw=raw)}


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_calc(expression: str) -> str:
    """Evaluate a pure-arithmetic expression safely (no eval of names).

    Accepts numbers and ``+ - * / // % **`` with parentheses; anything else
    (names, calls, attributes) raises ``ValueError``.
    """
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:60]}")

    try:
        tree = ast.parse(str(expression).strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e}") from e
    result = _eval(tree.body)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def _tool_calculator(expression: str) -> str:
    try:
        return safe_calc(expression)
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return f"error: {e}"


def _tool_time(format: str = "%Y-%m-%d %H:%M:%S", tz_offset_hours: float = 0.0) -> str:
    """Local (or offset) wall-clock time — the runtime box is the clock."""
    t = _time.time() + tz_offset_hours * 3600
    return _time.strftime(format, _time.localtime(t))


def make_default_registry() -> ToolRegistry:
    """Plan §29 initial toolset: calculator + time (search/memory come with
    their own adapters in M6/M7)."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="calculator",
        description="Tính biểu thức số học. Hỗ trợ + - * / // % ** và dấu ngoặc.",
        parameters={"expression": str},
        required=("expression",),
        func=_tool_calculator,
    ))
    reg.register(Tool(
        name="time",
        description="Lấy thời gian hiện tại của máy chủ.",
        parameters={"format": str, "tz_offset_hours": float},
        required=(),
        func=_tool_time,
    ))
    return reg


def run_tool(registry: ToolRegistry, call: ToolCall) -> str:
    """Validate then execute; failures return an error *string* for the
    model to read (never raises into the agent loop)."""
    error = registry.validate_call(call.name, call.arguments)
    if error:
        return f"error: {error}"
    tool = registry.get(call.name)
    try:
        return str(tool.func(**call.arguments))
    except Exception as e:  # noqa: BLE001 — the loop must keep running
        return f"error: {type(e).__name__}: {e}"
