"""Embedding geometry of a checkpoint, and how the SFT seeds relate to base.

Answers, from the safetensors files alone (no model build):
  - how anisotropic is the embedding (norm of the mean row vs row norms)?
  - did the SFT seeds actually start from the mean-init specials
    (seed specials ~ identical to each other / to the base mean row), or
    from the base rows (cos(seed, base) ~ 1)?

Usage::

    python scripts/diag_embeddings.py checkpoints_38m/latest checkpoints_sft_v4/latest
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECIALS = [0, 1, 4, 5, 6, 7, 8, 9]


def emb(ckpt):
    sd = load_file(os.path.join(ROOT, ckpt, "model.safetensors"))
    same = torch.equal(sd["lm_head.weight"], sd["tok_emb.weight"])
    return sd["tok_emb.weight"].float(), same


def main() -> None:
    base, same_b = emb(sys.argv[1])
    mean_row = base[10:].mean(0)
    resid = base[10:] - mean_row
    print(f"base: |mean row| {mean_row.norm():.3f} | row norm mean "
          f"{base.norm(dim=1).mean():.3f} | row-mean residual norm mean "
          f"{resid.norm(dim=1).mean():.3f} | lm_head==tok_emb in file: {same_b}")
    cs = F.cosine_similarity(base[10:], mean_row.expand_as(base[10:]), dim=1)
    print(f"      cos(row, mean row): mean {cs.mean():.3f} min {cs.min():.3f}"
          "   (high = rows share one dominant direction)")

    for ckpt in sys.argv[2:]:
        e, same = emb(ckpt)
        print(f"\n{ckpt} (lm_head==tok_emb in file: {same})")
        print("   id | base norm | ckpt norm | cos(base,ckpt) | cos(ckpt,base mean) | cos(ckpt, ckpt mean-of-10+)")
        m2 = e[10:].mean(0)
        for i in range(10):
            print(f"   {i:2d} |   {base[i].norm():6.3f}   |  {e[i].norm():6.3f}   |"
                  f"     {F.cosine_similarity(base[i], e[i], dim=0):6.3f}     |"
                  f"       {F.cosine_similarity(e[i], mean_row, dim=0):6.3f}         |"
                  f"       {F.cosine_similarity(e[i], m2, dim=0):6.3f}")
        s = F.normalize(e[SPECIALS], dim=1)
        g = s @ s.T
        n = len(SPECIALS)
        print(f"   mean pairwise cos among the 8 mean-init ids: "
              f"{((g.sum() - g.diag().sum()) / (n * n - n)):.4f}")
        # how far did ordinary rows move during SFT?
        d = (e[10:] - base[10:]).norm(dim=1) / base[10:].norm(dim=1)
        print(f"   ordinary rows relative change |ckpt-base|/|base|: mean {d.mean():.4f}"
              f" max {d.max():.4f}")


if __name__ == "__main__":
    main()
