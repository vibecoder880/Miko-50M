"""Interactive chat CLI with tool calling (plan §59 — 'miko chat').

Loads a (SFT) checkpoint, wraps ``miko.generation.generate`` into the
agent-loop ``generate_fn``, and runs an interactive REPL where calculator/
time calls are executed for real by the runtime.

Examples::

    python tools/chat.py --checkpoint checkpoints_sft_38m/latest \
        --tokenizer tokenizer/real_v0.2/tokenizer.json
    python tools/chat.py --checkpoint ... --tools      # enable tool loop
    python tools/chat.py --checkpoint ... --prompt "123*456?" --tools
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.agent import AgentConfig, encode_for_model, run_agent  # noqa: E402
from miko.chat_template import ChatMessage, encode_chat  # noqa: E402
from miko.checkpoint import load_model  # noqa: E402
from miko.generation import GenerationConfig, generate  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402
from miko.tooling import TOOL_CALL_OPEN, make_default_registry  # noqa: E402
from miko.utils import get_logger, seed_everything  # noqa: E402

logger = get_logger("miko.chat")

DEFAULT_SYSTEM = (
    "Bạn là Miko, một trợ lý nhỏ nói tiếng Việt. Trả lời ngắn gọn, rõ ràng. "
    "Khi cần tính toán chính xác, dùng công cụ calculator theo định dạng: "
    '<tool_call>\n{"name": "calculator", "arguments": {"expression": "..."}}\n</tool_call>'
)


def make_generate_fn(model, tokenizer: MikoTokenizer,
                     gen_cfg: GenerationConfig, max_context: int):
    """Adapt model+tokenizer into the agent loop's generate_fn."""

    def generate_fn(messages: list[ChatMessage]) -> str:
        ids = encode_for_model(tokenizer, list(messages), max_tokens=max_context)
        # End with an assistant generation prompt — header only, matching
        # the SFT data (encode_chat_masked has no newline after the header).
        ids = ids + [tokenizer.token_to_id("<assistant>")]
        input_ids = torch.tensor([ids], dtype=torch.long)[..., -model.config.context_length:]
        out = generate(model, input_ids, cfg=gen_cfg)
        new_ids = out[0, input_ids.shape[1]:].tolist()
        return tokenizer.decode(new_ids, skip_special_tokens=False)

    return generate_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with Miko")
    ap.add_argument("--checkpoint", default="checkpoints_sft_38m/latest")
    ap.add_argument("--tokenizer", default="tokenizer/real_v0.2/tokenizer.json")
    ap.add_argument("--prompt", help="one-shot mode")
    ap.add_argument("--tools", action="store_true", help="enable the tool loop")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-context", type=int, default=448,
                    help="token budget for the conversation (below 512)")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-tool-calls", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    model = load_model(args.checkpoint)
    model.eval()
    tokenizer = MikoTokenizer.load(args.tokenizer)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_id,
    )
    generate_fn = make_generate_fn(model, tokenizer, gen_cfg, args.max_context)
    registry = make_default_registry()
    agent_cfg = AgentConfig(max_steps=args.max_steps,
                            max_tool_calls=args.max_tool_calls)

    def ask(question: str) -> None:
        if args.tools:
            res = run_agent(generate_fn, registry, question, agent_cfg,
                            system=DEFAULT_SYSTEM)
            for call in res.tool_calls:
                print(f"  [tool] {call['name']}({call['arguments']}) -> {call['result']}")
            print(f"Miko: {res.answer}")
        else:
            messages = [ChatMessage("system", DEFAULT_SYSTEM),
                        ChatMessage("user", question)]
            text = generate_fn(messages)
            print(f"Miko: {text}")

    if args.prompt is not None:
        ask(args.prompt)
        return

    logger.info("Miko chat (tools=%s) — Ctrl-C to exit", args.tools)
    history: list[ChatMessage] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        ask(question)


if __name__ == "__main__":
    main()
