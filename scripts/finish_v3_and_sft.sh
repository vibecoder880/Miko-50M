#!/usr/bin/env bash
# One-command finish line (run AFTER v3 reaches step 60000):
#   1. milestone eval of the v3 base (vi/code/chat/tool tiers + val ppl)
#   2. SFT run (chat+tool, 400 steps, guarded)
#   3. post-SFT eval of the chat tier (chat_sft + tool_basic scores)
# Safe to re-run: evals overwrite their JSON, SFT resumes its own lineage.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
TOK=tokenizer/real_v0.2/tokenizer.json

step_now() {
  "$PY" - <<'EOF'
import json
try:
    print(json.load(open("checkpoints_38m/latest/trainer_state.json"))["step"])
except Exception:
    print(0)
EOF
}

echo "== v3 progress: step $(step_now)/60000 =="
if pgrep -f "configs/train_38m_v3.yaml" > /dev/null; then
  echo "v3 still training — wait for it to finish first."
  exit 1
fi

echo "== 1/3 milestone eval (v3 base) =="
nice -n 10 "$PY" tools/evaluate.py \
  --checkpoint checkpoints_38m/latest --tokenizer "$TOK" \
  --data-dir data/shards_v3/val --eval-batches 48 \
  --out evals/results_38m_step60000.json 2>&1 | grep -E "score|val loss" || true

echo "== 2/3 SFT run (chat + tool) =="
./scripts/train_sft_38m.sh

echo "== 3/3 post-SFT eval =="
nice -n 10 "$PY" tools/evaluate.py \
  --checkpoint checkpoints_sft_38m/latest --tokenizer "$TOK" \
  --data-dir data/shards_sft/val --eval-batches 16 \
  --out evals/results_sft_38m.json 2>&1 | grep -E "score|val loss" || true

echo ""
echo "== baselines to beat: chat_sft 0/14 (pre-SFT), tool_basic 0/16 =="
echo "== chat with tools: .venv/bin/python tools/chat.py \\"
echo "    --checkpoint checkpoints_sft_38m/latest --tools"
