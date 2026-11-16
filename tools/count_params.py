"""Print the exact parameter count for a Miko config (plan Phase 1 DoD).

Examples::

    python tools/count_params.py                        # built-in 50M defaults
    python tools/count_params.py --config configs/miko_50m.yaml
    python tools/count_params.py --hidden 384 --layers 13 --ffn 2048
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.config import MikoConfig  # noqa: E402
from miko.model import MikoModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Count Miko parameters")
    ap.add_argument("--config", help="yaml model config")
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--ffn", type=int)
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--context", type=int)
    ap.add_argument("--no-tie", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if args.config:
        cfg = MikoConfig.from_yaml(args.config)
    else:
        cfg = MikoConfig()

    overrides = {
        "hidden_size": args.hidden, "num_layers": args.layers,
        "num_heads": args.heads, "ffn_size": args.ffn,
        "vocab_size": args.vocab, "context_length": args.context,
        "tie_embeddings": False if args.no_tie else None,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    cfg.__post_init__()

    model = MikoModel(cfg)
    closed_form = cfg.parameter_count()
    summed = model.num_parameters()

    info = {
        "config": cfg.to_dict(),
        "parameters_closed_form": closed_form,
        "parameters_summed": summed,
        "parameters_millions": round(closed_form / 1e6, 3),
        "match": closed_form == summed,
        "embeddings_share_of_total": round(
            cfg.vocab_size * cfg.hidden_size / closed_form * 100, 1
        ),
    }
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        kind = "OK" if info["match"] else "MISMATCH"
        print(f"parameters : {closed_form:,} ({info['parameters_millions']}M) [{kind}]")
        print(f"hidden/layers/heads/ffn : {cfg.hidden_size}/{cfg.num_layers}"
              f"/{cfg.num_heads}/{cfg.ffn_size}")
        print(f"vocab/context            : {cfg.vocab_size}/{cfg.context_length}")
        print(f"tied embeddings          : {cfg.tie_embeddings}")
        print(f"embedding share          : {info['embeddings_share_of_total']}%")
    if not info["match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
