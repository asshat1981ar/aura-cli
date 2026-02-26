#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DISABLE_SAFE_GETCWD=1
export PYTHONPATH="/data/data/com.termux/files/usr/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python -S -m pytest "$@"
