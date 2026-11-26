"""Benchmark a bigger Miko config on this box (75M candidate).

Runs N training steps of the real trainer loop (AdamW, seq 128, grad_accum 8)
on the real corpus shards and reports tok/s + RSS — the honest number to
decide whether a 75M pretrain is worth launching.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resource as _res
import torch

from miko.config import MikoConfig
from miko.dataset import SFTShardedDataset, find_sft_shards
from miko.model import MikoModel
from miko.optim import build_optimizer
from miko.trainer import next_token_loss

torch.set_num_threads(3)

CFG = MikoConfig(
    vocab_size=16000, hidden_size=512, num_layers=16, num_heads=8,
    ffn_size=2048, context_length=512, activation="swiglu",
    use_rope=True, tie_embeddings=True,
)
print(f"params: {CFG.parameter_count()/1e6:.1f}M")

model = MikoModel(CFG)
model.train()
opt = build_optimizer(model, "adamw", lr=1e-4, weight_decay=0.1)

ds = SFTShardedDataset(find_sft_shards("data/shards_sft_v3/train"), 128, seed=42)
batches = ds.iter_batches(batch_size=1, shuffle=True)

# warmup + measure
N_MEASURE = 12
tokens = 0
t0 = None
for i, (x, y) in enumerate(batches):
    ce = next_token_loss(model(x), y)
    (ce / 8).backward()
    if (i + 1) % 8 == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i + 1 == 8 * 2:  # skip first optimizer steps (lazy allocs)
            t0, tokens = time.time(), 0
            continue
        if t0 is not None:
            tokens += 8 * 128
            n_steps = (i + 1) // 8 - 2
            if n_steps >= N_MEASURE:
                break

dt = time.time() - t0
rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1e6
print(f"measured: {tokens/dt:.1f} tok/s | RSS {rss:.2f} GB | "
      f"loss {float(ce):.2f}")
print(f"→ 3 epochs of 67.7M-token corpus ≈ {203e6/(tokens/dt)/86400:.1f} ngày "
      f"(box đơn độc, chưa tính tranh chấp)")
