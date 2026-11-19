"""Tool-calling infrastructure tests (plan §28-29, §36 — M5)."""

from __future__ import annotations

import pytest

from miko.tooling import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    Tool,
    ToolCall,
    ToolRegistry,
    make_default_registry,
    parse_tool_call,
    run_tool,
    safe_calc,
)


# ---------------------------------------------------------------------------
# safe_calc — the security boundary
# ---------------------------------------------------------------------------
def test_safe_calc_basic_and_precedence():
    assert safe_calc("2+2") == "4"
    assert safe_calc("123*456") == "56088"
    assert safe_calc("2**10") == "1024"
    assert safe_calc("(2+3)*4") == "20"
    assert safe_calc("10/4") == "2.5"
    assert safe_calc("10//4") == "2"
    assert safe_calc("-5+3") == "-2"
    assert safe_calc("7%3") == "1"


def test_safe_calc_rejects_names_and_calls():
    for evil in ["__import__('os').system('x')", "open('f')", "a+b",
                 "eval('1')", "print(1)", "1 if True else 2", "'a'+'b'"]:
        with pytest.raises(ValueError):
            safe_calc(evil)


def test_safe_calc_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        safe_calc("1/0")


# ---------------------------------------------------------------------------
# Registry validation (plan §36 steps 1-4)
# ---------------------------------------------------------------------------
@pytest.fixture()
def registry():
    return make_default_registry()


def test_registry_lists_tools_and_prompt(registry):
    assert registry.names() == ["calculator", "time"]
    prompt = registry.to_prompt()
    assert "calculator(expression: str)" in prompt
    assert "time(" in prompt


def test_registry_validate_unknown_tool(registry):
    err = registry.validate_call("nonexistent", {})
    assert err and "unknown tool" in err


def test_registry_validate_missing_required(registry):
    err = registry.validate_call("calculator", {})
    assert err and "missing required argument" in err


def test_registry_validate_wrong_type(registry):
    err = registry.validate_call("calculator", {"expression": 123})
    assert err and "must be str" in err


def test_registry_validate_unexpected_arg(registry):
    err = registry.validate_call("calculator", {"expression": "1", "x": 1})
    assert err and "unexpected argument" in err


def test_registry_validate_accepts_int_for_float(registry):
    assert registry.validate_call("time", {"tz_offset_hours": 7}) is None


def test_register_rejects_missing_func():
    with pytest.raises(ValueError):
        reg = ToolRegistry()
        reg.register(Tool(name="t", description="d", parameters={}))  # type: ignore


# ---------------------------------------------------------------------------
# parse_tool_call — malformed input must fail safe (plan §36)
# ---------------------------------------------------------------------------
def test_parse_returns_none_for_plain_text():
    assert parse_tool_call("Kết quả là 4.") is None


def test_parse_valid_call():
    text = f'{TOOL_CALL_OPEN}\n{{"name": "calculator", "arguments": {{"expression": "2+2"}}}}\n{TOOL_CALL_CLOSE}'
    out = parse_tool_call(text)
    assert out and "call" in out
    assert out["call"].name == "calculator"
    assert out["call"].arguments == {"expression": "2+2"}


def test_parse_invalid_json_reports_error():
    text = f"{TOOL_CALL_OPEN}\n{{'name': bad\n{TOOL_CALL_CLOSE}"
    out = parse_tool_call(text)
    assert out and "error" in out and "invalid JSON" in out["error"]


def test_parse_missing_name_reports_error():
    text = f'{TOOL_CALL_OPEN}\n{{"arguments": {{}}}}\n{TOOL_CALL_CLOSE}'
    out = parse_tool_call(text)
    assert out and "missing 'name'" in out["error"]


def test_parse_embedded_in_assistant_text():
    text = 'Để tính chính xác:\n<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'
    out = parse_tool_call(text)
    assert out and out["call"].arguments["expression"] == "6*7"


# ---------------------------------------------------------------------------
# run_tool — end-to-end
# ---------------------------------------------------------------------------
def test_run_tool_calculator(registry):
    result = run_tool(registry, ToolCall("calculator", {"expression": "123*456"}))
    assert result == "56088"


def test_run_tool_error_is_string_not_exception(registry):
    result = run_tool(registry, ToolCall("calculator", {"expression": "1/0"}))
    assert result.startswith("error:")


def test_run_tool_unknown_tool_returns_error_string(registry):
    result = run_tool(registry, ToolCall("ghost", {}))
    assert "unknown tool" in result


def test_run_tool_time_format(registry):
    result = run_tool(registry, ToolCall("time", {"format": "%Y"}))
    assert result.isdigit() and len(result) == 4
