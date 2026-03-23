#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${COPILOT_MCP_CONFIG_PATH:-$HOME/.config/github-copilot/mcp.json}"
TOKEN_MODE="${COPILOT_MCP_TOKEN_MODE:-placeholder}"

banner() { echo ">> $*"; }

resolve_host() {
  if [[ -n "${COPILOT_MCP_HOST:-}" ]]; then
    printf '%s\n' "${COPILOT_MCP_HOST}"
    return
  fi
  if [[ -n "${COPILOT_MCP_BASE_URL:-}" ]]; then
    COPILOT_MCP_BASE_URL="${COPILOT_MCP_BASE_URL}" python3 - <<'PY'
import os
from urllib.parse import urlparse

value = os.environ["COPILOT_MCP_BASE_URL"].strip()
parsed = urlparse(value if "://" in value else f"http://{value}")
print(parsed.hostname or value.rstrip("/"))
PY
    return
  fi
  printf '%s\n' "${AURA_COPILOT_MCP_HOST:-${AURA_MCP_HOST:-127.0.0.1}}"
}

HOST="$(resolve_host)"
mkdir -p "$(dirname "$OUT")"

python3 "$ROOT_DIR/scripts/write_copilot_mcp_config.py" \
  --output "$OUT" \
  --host "$HOST" \
  --token-mode "$TOKEN_MODE"

banner "Wrote Copilot MCP config to $OUT"

if [[ "$TOKEN_MODE" == "placeholder" ]]; then
  banner "Auth headers use \${env:...} placeholders. Export the referenced token env vars before using the config."
fi
