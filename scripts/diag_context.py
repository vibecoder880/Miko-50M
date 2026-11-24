"""Does the model use its context at all? Two black-box measurements.

  A. information flow — change one token d positions before the end and
     measure how much the logits at the last position move (same document
     only). ~0 at every d = the trained network is context-blind.
  B. n-gram baselines — unigram / bigram / trigram NLL of the pretraining
     val shards, to place the model's val loss on a scale (a model near the
     unigram entropy has learned no context use).

Usage::

    python scripts/diag_context.py                 # both, base model
    python scripts/diag_context.py --skip-ngram checkpoints_sft_v4/latest
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from miko.checkpoint import load_model  # noqa: E402
from miko.dataset import ShardedTokenDataset, find_shards  # noqa: E402

V = 16000
EOS = 3


def info_flow(ckpt: str) -> None:
    model = load_model(os.path.join(ROOT, ckpt))
    model.eval()
    ds = ShardedTokenDataset(find_shards(os.path.join(ROOT, "data/shards/val")),
                             128, seed=42)
    T = 128
    windows = []
    for i in range(40):
        w = ds[i][0][:T]
        if int((w[T - 72:] == EOS).sum()) == 0:  # last 72 tokens = one document
            windows.append(w)
        if len(windows) == 6:
            break

    def last_logits(x):
        with torch.no_grad():
            return model(x.unsqueeze(0))[0, -1]

    print(f"  [A] information flow, {len(windows)} same-document windows "
          f"(mean |dlogit| at last position; max over vocab in brackets)")
    for d in (1, 2, 4, 8, 16, 32, 64):
        mean_d, max_d = [], []
        for w in windows:
            ref = last_logits(w)
            pos = T - 1 - d
            for step in (7919, 104729):
                w2 = w.clone()
                w2[pos] = (int(w[pos]) + step) % V
                dl = (last_logits(w2) - ref).abs()
                mean_d.append(dl.mean().item())
                max_d.append(dl.max().item())
        print(f"      d={d:<3} mean {np.mean(mean_d):.4f}   max {np.mean(max_d):.3f}")
    del model


def _codes(a, b, c=None):
    if c is None:
        return a.astype(np.int64) * V + b
    return (a.astype(np.int64) * V + b) * V + c


class Lookup:
    def __init__(self, codes):
        uniq, cnt = np.unique(codes, return_counts=True)
        self.keys, self.counts = uniq, cnt

    def __call__(self, codes):
        idx = np.searchsorted(self.keys, codes)
        idx = np.clip(idx, 0, len(self.keys) - 1)
        hit = self.keys[idx] == codes
        return np.where(hit, self.counts[idx], 0)


def ngram_baselines() -> None:
    train = sorted(glob.glob(os.path.join(ROOT, "data/shards/train/*.bin")))
    train = [p for p in train if not p.endswith((".ids.bin", ".lbl.bin"))]
    val = sorted(glob.glob(os.path.join(ROOT, "data/shards/val/*.bin")))
    val = [p for p in val if not p.endswith((".ids.bin", ".lbl.bin"))]
    tr = np.concatenate([np.fromfile(p, dtype=np.uint16).astype(np.int64)
                         for p in train[:14]])
    va = np.concatenate([np.fromfile(p, dtype=np.uint16).astype(np.int64)
                         for p in val])[:400_000]
    print(f"  [B] n-gram counts on {len(tr)/1e6:.1f}M train tokens, "
          f"scored on {len(va)/1e3:.0f}K val tokens")

    uni = np.bincount(tr, minlength=V).astype(np.float64)
    uni_a = (uni + 0.5) / (uni.sum() + 0.5 * V)
    nll_uni = -np.log(uni_a[va[1:]]).mean()

    bi_c, bi_l = Lookup(_codes(tr[:-1], tr[1:])), Lookup(tr[:-1])
    a, b, c = va[:-2], va[1:-1], va[2:]
    alpha = 20.0
    p_bi = (bi_c(_codes(a, b)) + alpha * uni_a[b]) / (bi_l(a) + alpha)
    nll_bi = -np.log(p_bi).mean()

    tri_c, tri_l = Lookup(_codes(tr[:-2], tr[1:-1], tr[2:])), Lookup(_codes(tr[:-2], tr[1:-1]))
    seen = tri_l(_codes(a, b))
    print(f"      unigram  NLL {nll_uni:.3f}   (token identity only)")
    print(f"      bigram   NLL {nll_bi:.3f}   (one previous token)")
    print(f"      val trigram contexts seen in train: {100*seen.mean():.1f}%")
    for k in (0, 3, 10):
        p_tri = np.where(seen >= max(k, 1), tri_c(_codes(a, b, c)) / np.maximum(seen, 1), p_bi)
        print(f"      trigram  NLL {-np.log(p_tri).mean():.3f}   (min count {k})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", default=["checkpoints_38m/latest"])
    ap.add_argument("--skip-ngram", action="store_true")
    args = ap.parse_args()
    if not args.skip_ngram:
        ngram_baselines()
    for ckpt in args.checkpoints:
        print(f"\n=== {ckpt} ===")
        info_flow(ckpt)


if __name__ == "__main__":
    main()
