#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${NANO_VLLM_JAX_SERVER_CONFIG:-$ROOT/configs/server/gpu_optimal.yaml}"
MODEL_HOST="${NANO_VLLM_JAX_HOST:-127.0.0.1}"
MODEL_PORT="${NANO_VLLM_JAX_PORT:-6791}"
UI_HOST="${NANO_VLLM_JAX_CHAT_UI_HOST:-127.0.0.1}"
UI_PORT="${NANO_VLLM_JAX_CHAT_UI_PORT:-6789}"
LOG_DIR="${NANO_VLLM_JAX_RUN_LOG_DIR:-/mountpoint/.exp/run_logs}"
SKIP_COMPILE="${NANO_VLLM_JAX_SKIP_COMPILE_STARTUP:-0}"
STARTUP_TIMEOUT_SECONDS="${NANO_VLLM_JAX_STARTUP_TIMEOUT_SECONDS:-1200}"

mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/nvj_model_server_${MODEL_PORT}.log"
UI_LOG="$LOG_DIR/nvj_chat_ui_${UI_PORT}.log"

server_args=(
  "$ROOT/server.py"
  --config "$CONFIG"
  --host "$MODEL_HOST"
  --port "$MODEL_PORT"
)
if [[ "$SKIP_COMPILE" == "1" || "$SKIP_COMPILE" == "true" ]]; then
  server_args+=(--skip-compile)
fi

setsid "$PYTHON" -u "${server_args[@]}" > "$SERVER_LOG" 2>&1 < /dev/null &
server_pid=$!

for _ in $(seq 1 "$STARTUP_TIMEOUT_SECONDS"); do
  if grep -q "server_ready=http://$MODEL_HOST:$MODEL_PORT" "$SERVER_LOG"; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "model server exited while starting; see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 1
done

if ! grep -q "server_ready=http://$MODEL_HOST:$MODEL_PORT" "$SERVER_LOG"; then
  echo "timed out waiting for model server; see $SERVER_LOG" >&2
  kill "$server_pid" 2>/dev/null || true
  exit 1
fi

setsid "$PYTHON" -u "$ROOT/tools/chat_ui_server.py" \
  --host "$UI_HOST" \
  --port "$UI_PORT" \
  --backend-url "http://$MODEL_HOST:$MODEL_PORT" \
  > "$UI_LOG" 2>&1 < /dev/null &
ui_pid=$!

for _ in $(seq 1 10); do
  if grep -q "chat_ui_ready=http://$UI_HOST:$UI_PORT" "$UI_LOG"; then
    break
  fi
  if ! kill -0 "$ui_pid" 2>/dev/null; then
    echo "chat UI exited while starting; see $UI_LOG" >&2
    exit 1
  fi
  sleep 1
done

if ! grep -q "chat_ui_ready=http://$UI_HOST:$UI_PORT" "$UI_LOG"; then
  echo "timed out waiting for chat UI; see $UI_LOG" >&2
  kill "$ui_pid" 2>/dev/null || true
  exit 1
fi

echo "model_server=http://$MODEL_HOST:$MODEL_PORT pid=$server_pid log=$SERVER_LOG"
echo "chat_ui=http://$UI_HOST:$UI_PORT pid=$ui_pid log=$UI_LOG"
