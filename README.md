# Miko 50M

A from-scratch ~50M-parameter decoder-only Transformer language model, trained
primarily on CPU, designed to evolve from a base LM into a mini agent with tool
calling, web/search, and memory.

This repository currently implements **Miko v0.1 — Base LM**:

- Decoder-only Transformer (GPT-style) with RMSNorm, RoPE, SwiGLU, causal attention.
- Trainable BPE/Unigram tokenizer (special tokens reserved for future tool/chat tiers).
- Streaming / mmap dataset shards with sequence packing.
- CPU-friendly pretraining loop: gradient accumulation, clipping, LR schedule,
  frequent checkpoints, CSV training log.
- Tiny-overfit and shape/causal-mask/deterministic/gradient-flow test suite.

## Roadmap

```
Miko Base (v0.1) -> Miko Chat (v0.2) -> Miko Tool (v0.3)
              -> Miko Web (v0.4) -> Miko Memory (v0.5) -> Miko Agent (v1.0)
```

The model learns *how to think and call tools + format*; the runtime executes
tools. The model weights do not embed web/search/memory directly.

## Constraints

Trained on a small CPU box (≈1 GB RAM, 1–2 cores). Therefore:

- Disk-backed streaming dataset, never fully loaded into RAM.
- Small batch size (1) with gradient accumulation.
- Frequent checkpoints; modest context length (128 → 512 staged).
- Optimizer state is the main memory bottleneck; configs are tuned to fit.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Quick start

```bash
python tools/count_params.py                 # verify ~50M params
python tools/tokenize.py --input data/raw --output miko_tokenizer
python tools/pack_dataset.py --input data/raw --output data/tokenized
python tools/train.py --config configs/train_cpu.yaml
python tools/generate.py --prompt "Xin chào"
```

## Repository layout

See `configs/`, `miko/` (library), `tools/` (entry points), `data/`,
`checkpoints/`, `logs/`, `evals/`, `scripts/`.

## License

MIT — see `LICENSE`.
