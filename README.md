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

# 5. Real corpus (dataset-v0.2, 48M tokens; provenance in data/manifests/)
python scripts/fetch_corpus.py --source vi-wiki --docs 7000 --out data/cleaned_v2
python scripts/fetch_corpus.py --source en-wiki --docs 2100 --out data/cleaned_v2
python scripts/fetch_corpus.py --source py-code --docs 15000 --out data/cleaned_v2
python tools/train_tokenizer.py --input "data/cleaned_v2/*.txt" \
    --vocab-size 16000 --output tokenizer/real_v0.2/tokenizer.json
python tools/pack_dataset.py --input "data/cleaned_v2/*.txt" \
    --tokenizer tokenizer/real_v0.2/tokenizer.json \
    --out-train data/shards_v2/train --out-val data/shards_v2/val \
    --val-frac 0.02 --docs-per-shard 2000 \
    --report data/manifests/dataset-v0.2.json

# 6. Dataset probe (tiny, ~30 min) then the real 38M run (multi-day)
python tools/train.py --config configs/train_tiny_v2.yaml
python tools/train.py --config configs/train_38m_v2.yaml   # adafactor, ~800MB

# 7. The 50M run on boxes with >= 1.3 GB free (staged context 128 -> 512)
python tools/train.py --config configs/train_cpu.yaml
```

### Optimizer choice for a 1 GB box

The optimizer state is the largest RAM block (plan §17/§46). Measured on this
box (seq 128, micro-batch 1, grad-accum 1):

| Model | Optimizer | Peak RSS | tok/s |
|---|---|---|---|
| 50M (32K vocab) | AdamW | 1375 MB | 46.8 |
| 50M (32K vocab) | Adafactor | **937 MB** | 46.7 |
| 38M (16K vocab) | Adafactor | **798 MB** | 58.0 |

- `configs/train_cpu.yaml` (50M) uses AdamW — fits this 6 GB dev box.
- On the strict 1 GB target box: switch the train config to
  `optimizer: adafactor` and use `configs/miko_38m_1gb.yaml`
  (38M params, 16K vocab) — ~798 MB RSS leaves headroom for OS + SSH.
  Adafactor keeps factored second moments (~0.7 MB state vs ~405 MB) with
  self-limiting step size (`min(lr, 1/sqrt(t))`).

## Evaluation

Fixed benchmark suite (plan §49) — every checkpoint runs the same files:

```bash
python tools/evaluate.py --checkpoint checkpoints_tiny/latest \
    --tokenizer tokenizer/tokenizer.json \
    --data-dir data/shards_real/val \
    --out evals/results.json
```

- `evals/vi_basic.jsonl` (16 items), `code_basic.jsonl` (12),
  `chat_basic.jsonl` (8): greedy completion scored by accepted-answer
  containment, aggregated per category.
- `--data-dir` adds val loss + perplexity from packed shards.
- Output JSON includes a plan §49-style flat summary for version diffs:
  `{"vi_basic": 0.12, "code_basic": 0.17, "chat_basic": 0.12, ...}`.


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

55 tests: attention shape/causal-mask/determinism/grad-flow, tokenizer
roundtrips (Vietnamese / code / byte-fallback), shard writer & mmap dataset,
checkpoint save/load/resume (incl. RNG restore), sampling behaviour, LR
schedule, optimizer factory (AdamW vs memory-saving Adafactor), the eval
harness, and the Gate A tiny-overfit.

## Repository layout

See `configs/`, `miko/` (library), `tools/` (entry points), `data/`,
`checkpoints/`, `logs/`, `evals/`, `scripts/`. See `MODEL_CARD.md` for the
model card.

## License

MIT — see `LICENSE`.
