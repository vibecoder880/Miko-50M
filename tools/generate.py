"""Interactive / one-shot generation from a checkpoint (plan Phase 1 DoD).

Examples::

    python tools/generate.py --checkpoint checkpoints/latest --prompt "Xin chào"
    python tools/generate.py --checkpoint checkpoints/latest --interactive
    python tools/generate.py --checkpoint ckpt --prompt "Xin chào" --debug
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.checkpoint import load_model  # noqa: E402
from miko.generation import GenerationConfig, generate  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402
from miko.utils import get_logger, seed_everything  # noqa: E402

logger = get_logger("miko.generate")

DEFAULT_TOKENIZER = "tokenizer/tokenizer.json"


def generate_text(
    model,
    tok: MikoTokenizer,
    prompt: str,
    gen_cfg: GenerationConfig,
    debug: bool = False,
) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    if debug:
        logger.info("prompt: %d tokens | context limit %d",
                    len(ids), model.config.context_length)
    t0 = time.time()
    out = generate(
        model,
        torch.tensor([ids], dtype=torch.long),
        cfg=gen_cfg,
    )
    dt = time.time() - t0
    new_ids = out[0, len(ids):].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    if debug:
        logger.info("generated %d tokens in %.2fs (%.0f tok/s)",
                    len(new_ids), dt, len(new_ids) / max(dt, 1e-6))
        topk = torch.topk(
            model(torch.tensor([ids[-1:]]))[:, -1], k=5
        )
        logger.info(
            "last-step top-5: %s",
            ", ".join(
                f"{tok.backend.decode([i])!r}:{p:.2f}"
                for i, p in zip(topk.indices[0].tolist(), topk.values[0].tolist())
            ),
        )
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text with Miko")
    ap.add_argument("--checkpoint", default="checkpoints/latest")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--prompt", help="one-shot prompt (omit for --interactive)")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    model = load_model(args.checkpoint)
    model.eval()

    tok = MikoTokenizer.load(args.tokenizer) if os.path.exists(args.tokenizer) \
        else MikoTokenizer.from_config(model.config.vocab_size)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy,
        eos_token_id=tok.eos_id,
    )

    if args.interactive:
        logger.info("Miko interactive — Ctrl-C to exit")
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            print(f"Miko: {generate_text(model, tok, prompt, gen_cfg, args.debug)}")
    elif args.prompt is not None:
        print(generate_text(model, tok, args.prompt, gen_cfg, args.debug))
    else:
        ap.error("provide --prompt or --interactive")


if __name__ == "__main__":
    main()
