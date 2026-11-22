#!/usr/bin/env bash
# Continue the 38M run on dataset-v0.3 (67.7M tokens, vi 56.6%) after the
# v2 base reaches step 40000. Same run lineage + tokenizer, so the
# checkpoint resumes directly (train_38m_v3.yaml: max_steps 60000,
# LR re-anchored on the 60K-step horizon).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || { echo "missing $PY — create it first"; exit 1; }

mkdir -p logs
if pgrep -f "configs/train_38m_v3.yaml" > /dev/null; then
  echo "v3 continuation already running:"
  pgrep -af "configs/train_38m_v3.yaml"
  exit 0
fi

# Any other trainer (SFT cycles included) holds the box — refuse.
if pgrep -f "tools/train[.]py" > /dev/null; then
  echo "another training run is still active — wait for it first:"
  pgrep -af "tools/train[.]py"
  exit 1
fi

# If the v2 run is still going, refuse (both write checkpoints_38m/).
if pgrep -f "configs/train_38m_v2.yaml" > /dev/null; then
  echo "v2 base run is still active — wait for step 40000 first:"
  tail -1 logs/train_38m.csv
  exit 1
fi

(
  attempts=0
  while [ "$attempts" -lt 10 ]; do
    # Disk guard: a checkpoint write on a full disk killed the first v3
    # attempt (OSError 28 at step 56000). Need ~700MB free for one
    # checkpoint pair (model.safetensors + optimizer.pt at 38M).
    free_mb=$(df -Pm . | awk 'NR==2 {print $4}')
    if [ "$free_mb" -lt 700 ]; then
      echo "$(date '+%F %T') low disk (${free_mb}MB free) — waiting 5 min" \
        >> logs/train_38m_v3.out
      sleep 300
      continue
    fi
    nice -n 10 "$PY" tools/train.py \
      --config configs/train_38m_v3.yaml \
      --resume auto \
      >> logs/train_38m_v3.out 2>&1 && break
    attempts=$((attempts + 1))
    echo "$(date '+%F %T') v3 training exited (attempt $attempts) — restart in 60s" \
      >> logs/train_38m_v3.out
    sleep 60
  done
) &

echo "started v3 continuation (auto-restart, nice 10) — logs: logs/train_38m_v3.out"
echo "monitor: tail -f logs/train_38m_v3.csv"
echo "stop:    pkill -f 'train_38m_v3'  (progress kept to last 500-step checkpoint)"
