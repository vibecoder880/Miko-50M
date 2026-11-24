#!/usr/bin/env bash
# E2E demo: export bundle -> serve -> 3 requests -> transcript artifact.
# Usage: ./scripts/e2e_demo.sh [bundle-dir]   (default dist/miko-sft-v2)
set -euo pipefail
cd "$(dirname "$0")/.."

BUNDLE="${1:-dist/miko-sft-v2}"
OUT=evals/e2e_demo_transcript.txt
PORT="${MIKO_E2E_PORT:-8765}"
PY=.venv/bin/python

# 1. Export the bundle from the current SFT lineage (idempotent).
if [ ! -f "$BUNDLE/checkpoint/model.safetensors" ]; then
  "$PY" scripts/export_bundle.py --checkpoint checkpoints_sft_v2/latest \
      --out "$BUNDLE"
fi

# 2. Serve in the background.
OMP_NUM_THREADS=2 nice -n 10 "$PY" -m miko.server --bundle "$BUNDLE" \
    --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done

req() { # method path [json-body]
  local method="$1" path="$2" body="${3:-}"
  echo "### $method $path"
  if [ -n "$body" ]; then
    curl -s -m 180 -X POST "http://127.0.0.1:$PORT$path" \
      -H 'Content-Type: application/json' -d "$body"
  else
    curl -s -m 30 "http://127.0.0.1:$PORT$path"
  fi
  echo
}

{
  echo "# Miko E2E demo transcript — $(date '+%F %T')"
  echo "# bundle: $BUNDLE ($(cat "$BUNDLE/VERSION" 2>/dev/null | head -1))"
  req GET /health
  req GET /v1/info
  req POST /v1/chat '{"messages":[{"role":"user","content":"Xin chào, bạn là ai?"}],"max_new_tokens":48}'
  req POST /v1/chat '{"messages":[{"role":"user","content":"123*456 bằng bao nhiêu?"}],"tools":true,"max_new_tokens":96}'
} > "$OUT" 2>&1

echo "transcript -> $OUT"
cat "$OUT"
