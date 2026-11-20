#!/usr/bin/env bash
# Tool-only SFT continuation (phase-03 playbook) — resumes from
# checkpoints_sft_38m/latest (step 2000) on the expanded tool seed v0.2.
# Auto-restart + disk guard, mirroring scripts/train_sft_38m.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || { echo "missing $PY — create it first"; exit 1; }

if pgrep -f "tools/train[.]py" > /dev/null; then
  echo "a training run is still active:"
  pgrep -af "tools/train[.]py" || true
  exit 1
fi

mkdir -p logs checkpoints_sft_38m

(
  attempts=0
  while [ "$attempts" -lt 30 ]; do
    # Disk guard: checkpoint writes need ~500MB headroom.
    free_mb=$(df -Pm . | awk 'NR==2 {print $4}')
    if [ "$free_mb" -lt 700 ]; then
      echo "$(date '+%F %T') disk low (${free_mb}MB) — wait 5 min" \
        >> logs/train_sft_tool.out
      sleep 300
      continue
    fi
    nice -n 10 "$PY" tools/train.py \
      --config configs/train_sft_tool_v02.yaml \
      --sft \
      --resume auto \
      --max-steps 2800 \
      >> logs/train_sft_tool.out 2>&1 && break
    attempts=$((attempts + 1))
    echo "$(date '+%F %T') tool-SFT exited (attempt $attempts) — restart in 60s" \
      >> logs/train_sft_tool.out
    sleep 60
  done
) &

echo "started tool-only SFT continuation (auto-restart, nice 10)"
echo "monitor: tail -f logs/train_sft_tool.csv"
