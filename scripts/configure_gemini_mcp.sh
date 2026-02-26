#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${MCP_SERVER_URL:-http://localhost:8001}
TOKEN=${MCP_API_TOKEN:-dev-mcp-token}
OUT=${GEMINI_SETTINGS_PATH:-$HOME/.gemini/settings.json}

banner() { echo ">> $*"; }

if [[ -z "${MCP_API_TOKEN:-}" ]]; then
  banner "MCP_API_TOKEN not set; using fake token 'dev-mcp-token' (local dev only)."
fi

mkdir -p "$(dirname "$OUT")"
cat > "$OUT" <<EOF
{
  "mcp": {
    "allowed": ["aura-mcp"]
  },
  "mcpServers": {
    "aura-mcp": {
      "httpUrl": "$BASE_URL",
      "headers": {
        "Authorization": "Bearer $TOKEN"
      },
      "timeout": 120000,
      "trust": false,
      "description": "AURA MCP server"
    }
  }
}
EOF
banner "Wrote $OUT pointing to $BASE_URL"
banner "Tip: run 'gemini mcp reload' (or restart gemini-cli) to pick up the new config."
