#!/usr/bin/env bash
# One-shot eval-gates runner for the SFT v3 (copy-gym) cycle.
# Usage: ./scripts/eval_gates_v3.sh [step-label]
# Writes evals/results_sft_v3_*.json + evals/probe_v3_<label>.txt and
# prints a GATE SUMMARY block. CPU-polite: OMP 2 threads, nice 15.
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-latest}"
CKPT_DIR="checkpoints_sft_v3/${CKPT}"
LABEL=$(date '+%H%M')
PY=.venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN="nice -n 15 timeout 560 $PY"

[ -f "$CKPT_DIR/model.safetensors" ] || { echo "no checkpoint: $CKPT_DIR"; exit 1; }

echo "=== COPY PROBE (greedy) ==="
"$PY" - <<EOF > evals/probe_v3_${LABEL}.txt 2>&1
import sys
sys.path.insert(0, '.')
from miko.checkpoint import load_model
from miko.tokenizer import MikoTokenizer
from miko.chat_template import ChatMessage, encode_chat
from miko.generation import GenerationConfig, generate
from miko.utils import seed_everything
import torch

seed_everything(5)
model = load_model('${CKPT_DIR}'); model.eval()
tok = MikoTokenizer.load('tokenizer/real_v0.2/tokenizer.json')
SYSTEM = ("Bạn là Miko. Khi cần tính toán chính xác, hãy dùng công cụ:\n"
          "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"...\"}}\n</tool_call>")
cfg = GenerationConfig(max_new_tokens=44, greedy=True, eos_token_id=tok.eos_id)

def ask(user):
    ids = encode_chat(tok, [ChatMessage('system', SYSTEM), ChatMessage('user', user)],
                      add_generation_prompt=True)
    out = generate(model, torch.tensor([ids]), cfg=cfg)
    return tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=False)

print('ECHO    :', repr(ask('Nhắc lại biểu thức: 4787*2458'))[:200])
print('DISP2CMP:', repr(ask('Viết biểu thức 4787 × 2458 dưới dạng gọn.'))[:200])
print('RESTATE :', repr(ask('Tính 4787 nhân 2458.'))[:240])
print('TOOLNEW :', repr(ask('123 nhân 456 bằng bao nhiêu?'))[:300])
EOF
cat evals/probe_v3_${LABEL}.txt

echo "=== SUITE EVALS (56-token budget) ==="
$RUN "$PY" tools/evaluate.py --checkpoint "$CKPT_DIR" \
  --tokenizer tokenizer/real_v0.2/tokenizer.json --max-new-tokens 56 \
  --evals evals/chat_sft.jsonl evals/chat_basic.jsonl \
  --out evals/results_sft_v3_part1.json 2>&1 | grep -E "score"
$RUN "$PY" tools/evaluate.py --checkpoint "$CKPT_DIR" \
  --tokenizer tokenizer/real_v0.2/tokenizer.json --max-new-tokens 56 \
  --evals evals/tool_basic.jsonl \
  --out evals/results_sft_v3_part2.json 2>&1 | grep -E "score"
$RUN "$PY" tools/evaluate.py --checkpoint "$CKPT_DIR" \
  --tokenizer tokenizer/real_v0.2/tokenizer.json --max-new-tokens 56 \
  --evals evals/vi_basic.jsonl evals/code_basic.jsonl evals/regression.jsonl \
  --out evals/results_sft_v3_part3.json 2>&1 | grep -E "score"

echo "=== TOOL SMOKE (agent loop) ==="
$RUN "$PY" tools/chat.py --checkpoint "$CKPT_DIR" \
  --tokenizer tokenizer/real_v0.2/tokenizer.json --tools \
  --prompt "123*456 bằng bao nhiêu?" --max-new-tokens 96 2>&1 | head -c 400
echo

echo "=== GATE SUMMARY ==="
$PY - <<'EOF'
import json
def score(path, cat):
    try:
        d = json.load(open(path))
        return d["categories"][cat]["passed"], d["categories"][cat]["total"]
    except Exception:
        return None, None
gates = []
p, t = score("evals/results_sft_v3_part1.json", "chat_sft")
gates.append(("chat_sft > 0", p is not None and p > 0, f"{p}/{t}"))
p, t = score("evals/results_sft_v3_part2.json", "tool_basic")
d = json.load(open("evals/results_sft_v3_part2.json"))["categories"]["tool_basic"]
pos = [it for it in d["items"] if not it["id"].startswith("tool_neg")]
tp = sum(it["passed"] for it in pos)
gates.append(("tool positives > 0/12", tp > 0, f"{tp}/{len(pos)}"))
for cat, path in [("vi_basic", "evals/results_sft_v3_part3.json"),
                  ("code_basic", "evals/results_sft_v3_part3.json"),
                  ("regression", "evals/results_sft_v3_part3.json")]:
    p, t = score(path, cat)
    gates.append((f"{cat} score", p is not None and t and p / t >= 0.5,
                  f"{p}/{t}"))
for name, ok, detail in gates:
    print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}")
EOF
