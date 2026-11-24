#!/usr/bin/env bash
# SFT cycle v2 launcher — fresh lineage from the base checkpoint into
# checkpoints_sft_v2/, mixed 70/30 data, mean-init special embeddings.
# Auto-restart + disk guard (mirrors scripts/train_sft_38m.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || { echo "missing $PY — create it first"; exit 1; }

if pgrep -f "tools/train[.]py" > /dev/null; then
  echo "a training run is still active:"
  pgrep -af "tools/train[.]py" || true
  exit 1
fi

mkdir -p logs checkpoints_sft_v2

# Seed the v2 lineage from the base weights on first run only:
# drop the base optimizer (Adafactor) — SFT uses AdamW — and mean-init
# the never-trained special-token embeddings (ids 0,1,4-9; bos/eos were
# trained during pretraining document packing).
if [ ! -f checkpoints_sft_v2/latest/model.safetensors ]; then
  echo "seeding SFT v2 checkpoint from checkpoints_38m/latest ..."
  cp -r checkpoints_38m/latest checkpoints_sft_v2/latest
  rm -f checkpoints_sft_v2/latest/optimizer.pt
  "$PY" - <<'EOF'
import json

import torch
from safetensors.torch import load_file, save_file

w_path = "checkpoints_sft_v2/latest/model.safetensors"
sd = load_file(w_path)
emb = sd["tok_emb.weight"]
never_trained = [0, 1, 4, 5, 6, 7, 8, 9]
mean_row = emb[10:].mean(dim=0)
# Norm-matched mean-init: the plain mean vector has norm ~0.7 while
# ordinary rows sit at ~19.4 — invisible in the residual stream, the
# model routed around the specials (never emitted <tool_call>). Scale
# the mean direction up to the median ordinary norm instead.
target_norm = emb[10:].norm(dim=1).median()
mean_row = mean_row * (target_norm / mean_row.norm())
for i in never_trained:
    emb[i] = mean_row
sd["tok_emb.weight"] = emb
sd["lm_head.weight"] = emb.clone()
save_file(sd, w_path)

p = "checkpoints_sft_v2/latest/trainer_state.json"
with open(p) as fh:
    state = json.load(fh)
state["step"] = 0
state["metrics"] = {"seeded_from": "checkpoints_38m/latest",
                    "special_emb_mean_init": never_trained,
                    "cycle": "v2-restate-mix7030"}
with open(p, "w") as fh:
    json.dump(state, fh, indent=2)
print("SFT v2 lineage seeded: fresh optimizer + mean-init specials")
EOF
fi

(
  attempts=0
  while [ "$attempts" -lt 30 ]; do
    free_mb=$(df -Pm . | awk 'NR==2 {print $4}')
    if [ "$free_mb" -lt 700 ]; then
      echo "$(date '+%F %T') disk low (${free_mb}MB) — wait 5 min" \
        >> logs/train_sft_v2.out
      sleep 300
      continue
    fi
    nice -n 10 "$PY" tools/train.py \
      --config configs/train_sft_v2.yaml \
      --sft \
      --resume auto \
      >> logs/train_sft_v2.out 2>&1 && break
    attempts=$((attempts + 1))
    echo "$(date '+%F %T') SFT v2 exited (attempt $attempts) — restart in 60s" \
      >> logs/train_sft_v2.out
    sleep 60
  done
) &

echo "started SFT v2 (auto-restart, nice 10) — logs: logs/train_sft_v2.out"
echo "monitor: tail -f logs/train_sft_v2.csv"
