# MODEL CARD — Miko 50M (v0.1 Base)

## Model details

| Field | Value |
|---|---|
| Model name | Miko-50M-Base-v0.1 |
| Architecture | Decoder-only Transformer (GPT-style, Pre-LN) |
| Parameters | 50,636,928 (~50.6M) |
| Hidden size | 384 |
| Layers | 13 |
| Attention heads | 6 (head_dim 64) |
| FFN | SwiGLU, intermediate 2048 |
| Context length | 512 |
| Positional encoding | RoPE (base 10000) |
| Norm | RMSNorm (eps 1e-6) |
| Embedding tying | Tied (input embeddings = lm_head) |
| Vocab | 32,000 byte-level BPE |
| Precision | FP32 (CPU training) |

## Tokenizer

- Byte-level BPE (Hugging Face `tokenizers`), lossless roundtrip on any Unicode
  input (Vietnamese diacritics, emoji, JSON, URLs, shell text).
- 10 special tokens reserved from day one: `<pad> <unk> <bos> <eos> <system>
  <user> <assistant> <tool> <tool_call> <tool_result>` (ids 0-9).

## Training data

Categories and target mixture (plan §11):

```
Vietnamese text      50%   (Wikipedia-like, public-domain, cleaned)
English text         20%   (documentation, public-domain books)
Code                 25%   (Python, JS/TS, Go, SQL, HTML/CSS, Bash, JSON/YAML)
Markdown/JSON         5%
```

- Cleaning: dedup, Unicode normalization, language filter, length filter,
  garbage removal. Packed as `<bos> doc <eos>` streams into uint16 shards.
- Data provenance and per-source license tracking live in `data/manifests/`.
- Dataset versions: `dataset-v0.1` (tiny) → `dataset-v0.2` (small) → ...

## Training compute

- Hardware: CPU-only (1–3 cores), ~1–2 GB usable RAM.
- Objective: next-token prediction (cross-entropy).
- Optimizer: AdamW (β = 0.9 / 0.95, wd 0.1 on 2-D params), LR 3e-4,
  3% linear warmup + cosine decay to 10%.
- Batch: micro-batch 1 × grad accumulation 16 (effective 16) at seq 128,
  staged up to seq 512.
- Checkpoints: `step-*` + `latest`; resume restores weights, optimizer,
  schedule position, and RNG states.

## Evaluation

Fixed benchmark suite (runs on every major checkpoint):

```
evals/vi_basic.jsonl     Vietnamese completion / grammar / QA
evals/code_basic.jsonl   Python / JSON / SQL continuation
evals/chat_basic.jsonl   hello / simple QA / rewrite
evals/tool_basic.jsonl   tool-call format (v0.3+)
evals/regression.jsonl   regressions vs previous release
```

Metrics: train loss, val loss, perplexity, tokens/sec, generation latency.

## Intended use

- Vietnamese-lightweight assistant experiments on tiny CPU boxes.
- Research into from-scratch small-LM training pipelines.
- Seed architecture for the Miko family (50M → 100M → 300M → 1B) and the
  tool-calling mini-agent runtime.

## Non-intended use

- Not a frontier reasoning model; not long-context; not a knowledge base.
- Must not be used unsupervised in high-stakes or safety-critical settings.
- Web search, memory, and tool execution live in the external runtime, not in
  the weights — do not treat model outputs as current web facts.

## Known limitations

- Trained from scratch on a small corpus: high hallucination rate expected;
  factual recall is shallow.
- Small context (512) limits coherence over long inputs.
- Repetition can occur during sampling; mitigate with temperature /
  repetition penalty (see plan §61).

## Version

| Version | Date | Notes |
|---|---|---|
| v0.1 | 2026-11-16 | Base LM skeleton: model, tokenizer, dataset, training, checkpoint, generation. Pretraining in progress. |
