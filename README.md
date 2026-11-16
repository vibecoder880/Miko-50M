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
# 1. Verify architecture sizes
python tools/count_params.py --config configs/miko_50m.yaml   # ~50.6M params
python tools/count_params.py --config configs/miko_tiny.yaml  # ~1M params

# 2. (smoke) generate a placeholder corpus, train the tokenizer, pack shards
python scripts/make_seed_corpus.py --docs 600
python tools/train_tokenizer.py --input "data/raw/seed/*.txt" \
    --vocab-size 4096 --output tokenizer/tokenizer.json
python tools/pack_dataset.py --input "data/raw/seed/*.txt" \
    --tokenizer tokenizer/tokenizer.json \
    --out-train data/shards/train --out-val data/shards/val --val-frac 0.05

# 3. Sanity-train Miko Tiny (~1M) end-to-end
python tools/train.py --config configs/train_tiny_cpu.yaml
python tools/train.py --config configs/train_tiny_cpu.yaml \
    --resume auto                       # resume from checkpoints_tiny/latest

# 4. Generate
python tools/generate.py --checkpoint checkpoints_tiny/latest \
    --tokenizer tokenizer/tokenizer.json --prompt "Xin chào" --greedy

# 5. The real run (50M, staged context 128 -> 512)
python tools/train.py --config configs/train_cpu.yaml
```

Notes:

- The tokenizer's BPE may saturate below `--vocab-size` on small corpora;
  keep `model.vocab_size` in the train config equal to the trained
  tokenizer's actual vocab (`tokenizer/tokenizer.summary.json`).
- `logs/train.csv` records step / split / loss / lr / tokens_seen / elapsed /
  samples_per_sec / tokens_per_sec (plan §21).
- Tests include the mandatory Gate A tiny-overfit: a ~1M model must memorise
  a tiny corpus (loss < 0.5) before large-scale training is allowed.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

42 tests: attention shape/causal-mask/determinism/grad-flow, tokenizer
roundtrips (Vietnamese / code / byte-fallback), shard writer & mmap dataset,
checkpoint save/load/resume (incl. RNG restore), sampling behaviour, LR
schedule, and the Gate A tiny-overfit.

## Repository layout

See `configs/`, `miko/` (library), `tools/` (entry points), `data/`,
`checkpoints/`, `logs/`, `evals/`, `scripts/`. See `MODEL_CARD.md` for the
model card.

## License

MIT — see `LICENSE`.
