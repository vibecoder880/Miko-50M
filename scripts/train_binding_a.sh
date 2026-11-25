#!/usr/bin/env bash
# Binding epochs launcher (plan 20261125-binding-and-base60k phase 01).
# CONTINUATION: seeds checkpoints_binding_a/latest ONCE from the gym
# lineage checkpoints_sft_v3/latest (step 3000) — no optimizer drop here
# (same AdamW schedule family), then --resume auto must report step 3000.
# Acceptance guard from the plan: if the log says "resumed ... at step 0"
# the seeding went wrong and the run must be stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || { echo "missing $PY — create it first"; exit 1; }

if pgrep -f "tools/train[.]py" > /dev/null; then
  echo "a training run is still active:"
  pgrep -af "tools/train[.]py" || true
  exit 1
fi

mkdir -p logs checkpoints_binding_a

if [ ! -f checkpoints_binding_a/latest/model.safetensors ]; then
  echo "continuation seed: checkpoints_sft_v3/latest -> checkpoints_binding_a/latest"
  cp -r checkpoints_sft_v3/latest checkpoints_binding_a/latest
  # keep optimizer.pt: same optimizer family, cosine continues at ~2e-5
  "$PY" - <<'EOF'
import json

p = "checkpoints_binding_a/latest/trainer_state.json"
with open(p, encoding="utf-8") as fh:
    state = json.load(fh)
state.setdefault("metrics", {})["cycle"] = "binding-a-continuation"
with open(p, "w", encoding="utf-8") as fh:
    json.dump(state, fh, indent=2)
print("continuation lineage seeded at step", state["step"])
EOF
fi

(
  attempts=0
  while [ "$attempts" -lt 30 ]; do
    free_mb=$(df -Pm . | awk 'NR==2 {print $4}')
    if [ "$free_mb" -lt 700 ]; then
      echo "$(date '+%F %T') disk low (${free_mb}MB) — wait 5 min" \
        >> logs/train_binding_a.out
      sleep 300
      continue
    fi
    nice -n 10 "$PY" tools/train.py \
      --config configs/train_binding_a.yaml \
      --sft \
      --resume auto \
      >> logs/train_binding_a.out 2>&1 && break
    attempts=$((attempts + 1))
    echo "$(date '+%F %T') binding-a exited (attempt $attempts) — restart in 60s" \
      >> logs/train_binding_a.out
    sleep 60
  done
) &

echo "started binding-a continuation (auto-restart, nice 10)"
echo "monitor: tail -f logs/train_binding_a.csv"
