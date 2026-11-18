"""Fixed evaluation suite (plan §24-25, §49).

Every significant checkpoint runs the *same* benchmark files so versions are
comparable over time (``Miko 0.1`` vs ``Miko 0.2`` ...). For the base LM the
task is completion: greedy-generate after a prompt and check whether one of
the accepted answers appears.

Categories ship as JSONL files in ``evals/``::

    vi_basic.jsonl    Vietnamese completion / grammar / basic knowledge
    code_basic.jsonl  Python / SQL / shell continuations
    chat_basic.jsonl  conversational continuation
    tool_basic.jsonl  tool-call format (added at v0.3)
    regression.jsonl  regressions vs the previous release

Each item::

    {"id": "vi_cap_001", "prompt": "Thủ đô của Việt Nam là",
     "expected": ["hà nội"], "metric": "contains"}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch

from .chat_template import ChatMessage, encode_chat
from .generation import GenerationConfig, generate
from .tokenizer import MikoTokenizer

__all__ = [
    "EvalItem",
    "load_eval_sets",
    "normalize_text",
    "score_item",
    "evaluate_model",
]


@dataclass
class EvalItem:
    id: str
    prompt: str
    expected: list[str]
    metric: str = "contains"
    category: str = "misc"
    format: str = "raw"  # "raw" = plain continuation | "chat" = chat template
    system: Optional[str] = None  # optional system turn for format=chat


def load_eval_sets(paths: Iterable[str]) -> list[EvalItem]:
    """Load JSONL eval files. Category defaults to the file stem."""
    items: list[EvalItem] = []
    for path in paths:
        category = path.rsplit("/", 1)[-1].removesuffix(".jsonl")
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:  # noqa: PERF203
                    raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
                expected = row.get("expected")
                if isinstance(expected, str):
                    expected = [expected]
                if not row.get("prompt") or not expected:
                    raise ValueError(
                        f"{path}:{line_no}: needs 'prompt' and 'expected'"
                    )
                fmt = str(row.get("format", "raw"))
                if fmt not in ("raw", "chat"):
                    raise ValueError(f"{path}:{line_no}: unsupported format {fmt!r}")
                items.append(
                    EvalItem(
                        id=str(row.get("id", f"{category}-{line_no}")),
                        prompt=str(row["prompt"]),
                        expected=[str(e) for e in expected],
                        metric=str(row.get("metric", "contains")),
                        category=str(row.get("category", category)),
                        format=fmt,
                        system=(str(row["system"]) if row.get("system") else None),
                    )
                )
    if not items:
        raise ValueError("no eval items loaded")
    return items


_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace (diacritics kept — they matter)."""
    return _WS.sub(" ", text.strip().lower())


def score_item(item: EvalItem, generated: str) -> bool:
    """Score one item. Only ``contains`` is defined for the base LM v0.1."""
    gen = normalize_text(generated)
    if item.metric == "contains":
        return any(normalize_text(e) in gen for e in item.expected)
    raise ValueError(f"unsupported metric: {item.metric!r}")


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer: MikoTokenizer,
    items: list[EvalItem],
    max_new_tokens: int = 24,
    gen_cfg: Optional[GenerationConfig] = None,
) -> dict:
    """Run all items and aggregate per category.

    Returns::

        {"vi_basic": {"score": 0.4, "passed": 4, "total": 10,
                      "items": [{"id": ..., "passed": bool, "generated": ...},
                                ...]},
         ...}
    """
    cfg = gen_cfg or GenerationConfig(
        max_new_tokens=max_new_tokens,
        greedy=True,  # deterministic scoring: no sampling noise
        eos_token_id=tokenizer.eos_id,
    )
    model.eval()

    categories: dict[str, dict] = {}
    for item in items:
        if item.format == "chat":
            msgs: list[ChatMessage] = []
            if item.system:
                msgs.append(ChatMessage("system", item.system))
            msgs.append(ChatMessage("user", item.prompt))
            ids = encode_chat(tokenizer, msgs, add_generation_prompt=True)
        else:
            ids = tokenizer.encode(item.prompt, add_special_tokens=False)
        out = generate(model, torch.tensor([ids], dtype=torch.long), cfg=cfg)
        new_ids = out[0, len(ids):].tolist()
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)
        passed = score_item(item, generated)

        cat = categories.setdefault(
            item.category,
            {"score": 0.0, "passed": 0, "total": 0, "items": []},
        )
        cat["total"] += 1
        cat["passed"] += int(passed)
        cat["items"].append(
            {"id": item.id, "passed": passed, "generated": generated[:120]}
        )

    for cat in categories.values():
        cat["score"] = round(cat["passed"] / cat["total"], 4)
    return categories
