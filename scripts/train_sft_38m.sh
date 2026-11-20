#!/usr/bin/env bash
# First real SFT run (M4): base checkpoint -> chat tier, into its own
# checkpoints_sft_38m/ directory. Guards against any pretraining run
# still holding the box.
#
# The first launch seeds the SFT checkpoint from the base weights:
#   cp -r checkpoints_38m/latest checkpoints_sft_38m_latest_base
# then resumes with --config configs/train_sft_38m.yaml --sft.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || { echo "missing $PY — create it first"; exit 1; }

if pgrep -f "configs/train_38m_v[23].yaml" > /dev/null; then
  echo "a pretraining run is still active — wait for it first:"
  pgrep -af "configs/train_38m_v[23].yaml" || true
  tail -1 logs/train_38m_v3.csv 2>/dev/null || true
  exit 1
fi

mkdir -p logs checkpoints_sft_38m

# Seed SFT lineage from the base checkpoint on first run only.
# The base runs Adafactor; SFT uses AdamW — the saved optimizer state is
# incompatible (col_var/row_var + beta2_decay param group pollute AdamW),
# so it is dropped and the fine-tune starts with a fresh optimizer.
if [ ! -f checkpoints_sft_38m/latest/model.safetensors ]; then
  echo "seeding SFT checkpoint from checkpoints_38m/latest ..."
  rm -rf checkpoints_sft_38m/latest
  cp -r checkpoints_38m/latest checkpoints_sft_38m/latest
  rm -f checkpoints_sft_38m/latest/optimizer.pt
  "$PY" - <<'EOF'
import json
p = "checkpoints_sft_38m/latest/trainer_state.json"
with open(p) as fh:
    state = json.load(fh)
state["step"] = 0  # SFT counts its own steps from zero
state["metrics"] = {"seeded_from": "checkpoints_38m/latest"}
with open(p, "w") as fh:
    json.dump(state, fh, indent=2)
print("SFT lineage seeded at step 0 from base weights (fresh optimizer)")
EOF
fi

(
  attempts=0
  while [ "$attempts" -lt 30 ]; do
    nice -n 10 "$PY" tools/train.py \
      --config configs/train_sft_38m.yaml \
      --sft \
      --resume auto \
      >> logs/train_sft_38m.out 2>&1 && break
    attempts=$((attempts + 1))
    echo "$(date '+%F %T') SFT exited (attempt $attempts) — restart in 60s" \
      >> logs/train_sft_38m.out
    sleep 60
  done
) &

echo "started SFT run (auto-restart, nice 10) — logs: logs/train_sft_38m.out"
echo "monitor: tail -f logs/train_sft_38m.csv"
echo "after:   .venv/bin/python tools/evaluate.py --checkpoint checkpoints_sft_38m/latest"
