"""Chat template for instruction tuning (plan §26, §29 — M4 prep).

Renders ``system`` / ``user`` / ``assistant`` messages using the special
tokens reserved since day one (ids 4-6 in the v0.1 tokenizer)::

    <system>
    Bạn là Miko.
    <eos>
    <user>
    Xin chào
    <eos>
    <assistant>
    Chào bạn!
    <eos>

``encode_chat`` additionally produces SFT labels: every token outside an
assistant turn is masked to ``-100`` so instruction tuning trains only on
assistant responses (plan §26-27). Tool roles arrive with M5 and are
rejected here for now to keep the v0.2 format contract tight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = ["ChatMessage", "render_chat", "encode_chat", "encode_chat_masked"]

SUPPORTED_ROLES = ("system", "user", "assistant")
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in SUPPORTED_ROLES:
            raise ValueError(
                f"unsupported role {self.role!r}; expected one of "
                f"{SUPPORTED_ROLES} (tool roles come with M5)"
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
        parts.append(f"<{msg.role}>\n{msg.content}")
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
        ids.append(tokenizer.token_to_id(f"<{msg.role}>"))
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
        role_id = tokenizer.token_to_id(f"<{msg.role}>")
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
