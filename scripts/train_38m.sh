#!/usr/bin/env bash
# Launch the real 38M pretraining run in the background (plan §59 step 15).
# nice +10 keeps the box responsive for interactive work; checkpoints every
# 500 steps so the run survives restarts via --resume auto.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || { echo "missing $PY — create it first"; exit 1; }

mkdir -p logs
if pgrep -f "tools/train.py --config configs/train_38m_v2.yaml" > /dev/null; then
  echo "38M training already running:"
  pgrep -af "tools/train.py --config configs/train_38m_v2.yaml"
  exit 0
fi

nohup nice -n 10 "$PY" tools/train.py \
  --config configs/train_38m_v2.yaml \
  --resume auto \
  >> logs/train_38m.out 2>&1 &

echo "started 38M run (pid $!, nice 10) — logs: logs/train_38m.out"
echo "monitor: tail -f logs/train_38m.csv"
echo "stop:    pkill -f 'train_38m_v2'   (progress kept up to last checkpoint)"
