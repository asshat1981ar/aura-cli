#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

echo "Preparing llama.cpp in $LLAMA_CPP_DIR"

if command -v pkg >/dev/null 2>&1; then
  echo "Installing Termux build dependencies..."
  pkg install -y git cmake make clang
else
  echo "pkg not found; skipping dependency install. Ensure git, cmake, make, and clang are available."
fi

if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
else
  echo "Existing llama.cpp checkout found. Pulling latest changes..."
  git -C "$LLAMA_CPP_DIR" pull --ff-only
fi

cmake -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" -DBUILD_SHARED_LIBS=OFF
cmake --build "$BUILD_DIR" --config Release -j "$JOBS" --target llama-server

echo
echo "llama-server built successfully:"
echo "  $BUILD_DIR/bin/llama-server"
echo
echo "Suggested next commands:"
echo "  export LLAMA_SERVER_BIN=\"$BUILD_DIR/bin/llama-server\""
echo "  export AURA_ANDROID_CODER_MODEL=\"\$HOME/models/qwen2.5-coder-3b-instruct-q4.gguf\""
echo "  export AURA_ANDROID_PLANNER_MODEL=\"\$HOME/models/phi-4-mini-instruct-q4.gguf\""
echo "  export AURA_ANDROID_EMBED_MODEL=\"\$HOME/models/bge-small-en-v1.5-q8_0.gguf\""
echo "  bash scripts/run_android_local_models.sh"
