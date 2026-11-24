#!/usr/bin/env bash
# SFT cycle v3 launcher (copy-gym, 50/50 mix) — fresh lineage from the base checkpoint into
# checkpoints_sft_v3/, mixed 50/50 data, mean-init special embeddings.
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

mkdir -p logs checkpoints_sft_v3

# Seed the v3 lineage from the base weights on first run only:
# drop the base optimizer (Adafactor) — SFT uses AdamW — and mean-init
# the never-trained special-token embeddings (ids 0,1,4-9; bos/eos were
# trained during pretraining document packing).
if [ ! -f checkpoints_sft_v3/latest/model.safetensors ]; then
  echo "seeding SFT v3 checkpoint from checkpoints_38m/latest ..."
  cp -r checkpoints_38m/latest checkpoints_sft_v3/latest
  rm -f checkpoints_sft_v3/latest/optimizer.pt
  "$PY" - <<'EOF'
import json

import torch
from safetensors.torch import load_file, save_file

w_path = "checkpoints_sft_v3/latest/model.safetensors"
sd = load_file(w_path)
emb = sd["tok_emb.weight"]
never_trained = [0, 1, 4, 5, 6, 7, 8, 9]
mean_row = emb[10:].mean(dim=0)
# Random-direction norm-matched init: (1) plain mean vector norm ~0.7 vs
# ~19.4 ordinary -> invisible, model routed around specials; (2) sharing
# ONE direction for all 8 specials made them indistinguishable — the
# model emitted <pad> where <tool_call> belongs. Give each special its
# own random unit direction scaled to the median ordinary norm.
target_norm = emb[10:].norm(dim=1).median()
gen = torch.Generator().manual_seed(7)
for i in never_trained:
    d = torch.randn(emb.shape[1], generator=gen)
    emb[i] = d * (target_norm / d.norm())
sd["tok_emb.weight"] = emb
sd["lm_head.weight"] = emb.clone()
save_file(sd, w_path)

p = "checkpoints_sft_v3/latest/trainer_state.json"
with open(p) as fh:
    state = json.load(fh)
state["step"] = 0
state["metrics"] = {"seeded_from": "checkpoints_38m/latest",
                    "special_emb_mean_init": never_trained,
                    "cycle": "v3-copygym-mix5050"}
with open(p, "w") as fh:
    json.dump(state, fh, indent=2)
print("SFT v3 lineage seeded: fresh optimizer + mean-init specials")
EOF
fi

(
  attempts=0
  while [ "$attempts" -lt 30 ]; do
    free_mb=$(df -Pm . | awk 'NR==2 {print $4}')
    if [ "$free_mb" -lt 700 ]; then
      echo "$(date '+%F %T') disk low (${free_mb}MB) — wait 5 min" \
        >> logs/train_sft_v3.out
      sleep 300
      continue
    fi
    nice -n 10 "$PY" tools/train.py \
      --config configs/train_sft_v3.yaml \
      --sft \
      --resume auto \
      >> logs/train_sft_v3.out 2>&1 && break
    attempts=$((attempts + 1))
    echo "$(date '+%F %T') SFT v2 exited (attempt $attempts) — restart in 60s" \
      >> logs/train_sft_v3.out
    sleep 60
  done
) &

echo "started SFT v3 (auto-restart, nice 10) — logs: logs/train_sft_v3.out"
echo "monitor: tail -f logs/train_sft_v3.csv"
