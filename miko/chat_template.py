"""Chat template for instruction tuning (plan §26, §29, §33 — M4/M5).

Renders ``system`` / ``user`` / ``assistant`` / ``tool`` messages using the
special tokens reserved since day one::

    <system>
    Bạn là Miko.
    <eos>
    <user>
    Xin chào
    <eos>
    <assistant>
    <tool_call>
    {"name": "calculator", "arguments": {"expression": "2+2"}}
    </tool_call>
    <eos>
    <tool_result>
    4
    </tool_result>
    <eos>
    <assistant>
    Kết quả là 4.
    <eos>

The ``tool`` role carries the runtime's result and renders with the
``<tool_result>`` header (plan §33). Assistant content may embed
``<tool_call>{json}</tool_call>`` text — parsed and executed by the runtime
(``miko.tooling``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = ["ChatMessage", "render_chat", "encode_chat", "encode_chat_masked"]

SUPPORTED_ROLES = ("system", "user", "assistant", "tool")
IGNORE_INDEX = -100

# Role -> header token (tool results use the <tool_result> token).
ROLE_HEADERS = {"system": "system", "user": "user", "assistant": "assistant",
                "tool": "tool_result"}


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in SUPPORTED_ROLES:
            raise ValueError(
                f"unsupported role {self.role!r}; expected one of "
                f"{SUPPORTED_ROLES}"
            )
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")


def render_chat(messages: Sequence[ChatMessage], add_generation_prompt: bool = False) -> str:
    """Render the chat as plain text with special-token headers.

    With ``add_generation_prompt=True`` the string ends with the assistant
    header (no trailing newline content), ready for the model to continue.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<{ROLE_HEADERS[msg.role]}>\n{msg.content}")
    if add_generation_prompt:
        parts.append("<assistant>\n")
    return "\n".join(parts)


def encode_chat(
    tokenizer,
    messages: Sequence[ChatMessage],
    add_generation_prompt: bool = False,
) -> list[int]:
    """Token ids for the rendered chat, each turn closed with ``<eos>``.

    Encoding is done turn-by-turn so ``<eos>`` lands after every message
    (including the system preamble), which is what the base model's
    next-token objective needs at SFT time.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    eos = tokenizer.eos_id
    ids: list[int] = [tokenizer.bos_id]
    for msg in messages:
        ids.append(tokenizer.token_to_id(f"<{ROLE_HEADERS[msg.role]}>"))
        ids.extend(tokenizer.encode(msg.content, add_special_tokens=False))
        ids.append(eos)
    if add_generation_prompt:
        ids.append(tokenizer.token_to_id("<assistant>"))
        ids.append(tokenizer.backend.encode("\n", add_special_tokens=False).ids[0])
    return ids


def encode_chat_masked(
    tokenizer,
    messages: Sequence[ChatMessage],
    max_len: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    """ ``(ids, labels)`` for SFT: labels equal ids inside assistant turns
    (including the closing ``<eos>``) and ``IGNORE_INDEX`` elsewhere.

    With ``max_len``, the tail is truncated; truncation always keeps whole
    tokens (no partial assistant mask fixing needed — a truncated tail
    simply loses its labels, which is the safe direction).
    """
    if not messages:
        raise ValueError("messages must not be empty")
    eos = tokenizer.eos_id

    ids: list[int] = [tokenizer.bos_id]
    labels: list[int] = [IGNORE_INDEX]
    for msg in messages:
        role_id = tokenizer.token_to_id(f"<{ROLE_HEADERS[msg.role]}>")
        turn_ids = [role_id, *tokenizer.encode(msg.content, add_special_tokens=False), eos]
        ids.extend(turn_ids)
        # Labels stay aligned 1:1 with ids. For assistant turns the role
        # header is masked (it is prompt structure) while content + the
        # closing <eos> carry labels so the model learns to stop; every
        # non-assistant turn is fully masked.
        if msg.role == "assistant":
            labels.extend([IGNORE_INDEX, *turn_ids[1:]])
        else:
            labels.extend([IGNORE_INDEX] * len(turn_ids))

    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len]
        labels = labels[:max_len]
    return ids, labels
