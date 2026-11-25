#!/usr/bin/env bash
# Phase-04 finalize for the v3 gym cycle (plan 20261125-miko-cli).
# One command: probe -> gates -> export final bundle -> codex demo ->
# transcript. Run AFTER "training complete" for checkpoints_sft_v3.
# Heavy steps are sequential and CPU-polite; rerun-safe.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
BUNDLE=dist/miko-sft-gym
PORT="${MIKO_PORT:-8890}"
export OMP_NUM_THREADS=8

step() { echo; echo "=== $1 ==="; }

# 0. Preflight: training must be complete.
grep -q "training complete" logs/train_sft_v3.out || {
  echo "training not complete yet — aborting"; exit 1; }

step "1/5 copy probe (greedy, frozen final ckpt)"
OMP_NUM_THREADS=2 nice -n 15 timeout 400 "$PY" scripts/probe_copy.py \
  checkpoints_sft_v3/latest --max-new 56 | tee evals/probe_gym_final.txt

step "2/5 eval suites"
"$PY" tools/evaluate.py --checkpoint checkpoints_sft_v3/latest \
  --tokenizer tokenizer/real_v0.2/tokenizer.json --max-new-tokens 56 \
  --evals evals/tool_basic.jsonl \
  --out evals/results_gym_tool.json 2>&1 | grep -E "score"
"$PY" tools/evaluate.py --checkpoint checkpoints_sft_v3/latest \
  --tokenizer tokenizer/real_v0.2/tokenizer.json --max-new-tokens 56 \
  --evals evals/chat_sft.jsonl evals/chat_basic.jsonl evals/regression.jsonl \
  --out evals/results_gym_chat.json 2>&1 | grep -E "score"

step "3/5 export final bundle"
"$PY" scripts/export_bundle.py --checkpoint checkpoints_sft_v3/latest \
  --out "$BUNDLE"

step "4/5 codex e2e demo (greedy serving)"
./miko-cli exec --skip-git-repo-check -s read-only \
  "Nói xin chào bằng tiếng Việt" > evals/codex_gym_final.txt 2>&1
grep -A2 "^codex$" evals/codex_gym_final.txt | head -3
echo "protocol errors: $(grep -c ERROR evals/codex_gym_final.txt || true)"

step "5/5 gate summary"
"$PY" - <<'EOF'
import json
def sc(path, cat):
    try:
        d = json.load(open(path))["categories"][cat]
        return d["passed"], d["total"]
    except Exception:
        return None, None
for path, cats in [("evals/results_gym_tool.json", ["tool_basic"]),
                   ("evals/results_gym_chat.json",
                    ["chat_sft", "chat_basic", "regression"])]:
    for cat in cats:
        p, t = sc(path, cat)
        print(f"{cat}: {p}/{t}")
d = json.load(open("evals/results_gym_tool.json"))["categories"]["tool_basic"]
pos = [i for i in d["items"] if not i["id"].startswith("tool_neg")]
print(f"tool positives: {sum(i['passed'] for i in pos)}/{len(pos)}")
print("checkpoints_sft_v3 final val:", json.load(
    open("checkpoints_sft_v3/final_metrics.json"))["val_loss"])
EOF
echo
echo "finalize steps done — MODEL_CARD v0.7 + journal + push remain"
