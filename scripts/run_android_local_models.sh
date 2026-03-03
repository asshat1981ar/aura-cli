#!/usr/bin/env bash
set -euo pipefail

DEFAULT_LLAMA_SERVER_BIN="llama-server"
if ! command -v "$DEFAULT_LLAMA_SERVER_BIN" >/dev/null 2>&1; then
  if [ -x "$HOME/src/llama.cpp/build/bin/llama-server" ]; then
    DEFAULT_LLAMA_SERVER_BIN="$HOME/src/llama.cpp/build/bin/llama-server"
  elif [ -x "$HOME/.cache/aura/llama.cpp/build/bin/llama-server" ]; then
    DEFAULT_LLAMA_SERVER_BIN="$HOME/.cache/aura/llama.cpp/build/bin/llama-server"
  fi
fi

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$DEFAULT_LLAMA_SERVER_BIN}"
AURA_ANDROID_HOST="${AURA_ANDROID_HOST:-127.0.0.1}"
AURA_ANDROID_THREADS="${AURA_ANDROID_THREADS:-4}"

AURA_ANDROID_CODER_MODEL="${AURA_ANDROID_CODER_MODEL:-}"
AURA_ANDROID_PLANNER_MODEL="${AURA_ANDROID_PLANNER_MODEL:-}"

AURA_ANDROID_CODER_PORT="${AURA_ANDROID_CODER_PORT:-8080}"
AURA_ANDROID_PLANNER_PORT="${AURA_ANDROID_PLANNER_PORT:-8081}"

AURA_ANDROID_CODER_CTX="${AURA_ANDROID_CODER_CTX:-4096}"
AURA_ANDROID_PLANNER_CTX="${AURA_ANDROID_PLANNER_CTX:-4096}"

LOG_DIR="${AURA_ANDROID_LOG_DIR:-logs/local_models}"
CODER_LOG="${LOG_DIR}/coder.log"
PLANNER_LOG="${LOG_DIR}/planner.log"

if ! command -v "$LLAMA_SERVER_BIN" >/dev/null 2>&1; then
  if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "error: could not find llama.cpp server binary: $LLAMA_SERVER_BIN" >&2
  echo "hint: run scripts/setup_llama_cpp_termux.sh or set LLAMA_SERVER_BIN explicitly" >&2
  exit 1
  fi
fi

if [ -z "$AURA_ANDROID_CODER_MODEL" ]; then
  echo "error: AURA_ANDROID_CODER_MODEL is required" >&2
  exit 1
fi

if [ -z "$AURA_ANDROID_PLANNER_MODEL" ]; then
  echo "error: AURA_ANDROID_PLANNER_MODEL is required" >&2
  exit 1
fi

if [ ! -f "$AURA_ANDROID_CODER_MODEL" ]; then
  echo "error: coder model not found: $AURA_ANDROID_CODER_MODEL" >&2
  exit 1
fi

if [ ! -f "$AURA_ANDROID_PLANNER_MODEL" ]; then
  echo "error: planner model not found: $AURA_ANDROID_PLANNER_MODEL" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

start_server() {
  local model_path="$1"
  local port="$2"
  local ctx="$3"
  local log_path="$4"

  "$LLAMA_SERVER_BIN" \
    -m "$model_path" \
    --host "$AURA_ANDROID_HOST" \
    --port "$port" \
    --ctx-size "$ctx" \
    --threads "$AURA_ANDROID_THREADS" \
    >"$log_path" 2>&1 &
  echo $!
}

cleanup() {
  if [ -n "${CODER_PID:-}" ] && kill -0 "$CODER_PID" 2>/dev/null; then
    kill "$CODER_PID" 2>/dev/null || true
  fi
  if [ -n "${PLANNER_PID:-}" ] && kill -0 "$PLANNER_PID" 2>/dev/null; then
    kill "$PLANNER_PID" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

echo "Starting coder model on http://${AURA_ANDROID_HOST}:${AURA_ANDROID_CODER_PORT}/v1"
CODER_PID="$(start_server "$AURA_ANDROID_CODER_MODEL" "$AURA_ANDROID_CODER_PORT" "$AURA_ANDROID_CODER_CTX" "$CODER_LOG")"

echo "Starting planner model on http://${AURA_ANDROID_HOST}:${AURA_ANDROID_PLANNER_PORT}/v1"
PLANNER_PID="$(start_server "$AURA_ANDROID_PLANNER_MODEL" "$AURA_ANDROID_PLANNER_PORT" "$AURA_ANDROID_PLANNER_CTX" "$PLANNER_LOG")"

echo "coder pid: $CODER_PID log: $CODER_LOG"
echo "planner pid: $PLANNER_PID log: $PLANNER_LOG"
echo "Press Ctrl+C to stop both local model servers."

wait "$CODER_PID" "$PLANNER_PID"
