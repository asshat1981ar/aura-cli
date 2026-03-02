#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${MCP_SERVER_URL:-http://localhost}
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
    "allowed": ["aura-dev", "aura-skills", "aura-control", "aura-loop", "aura-copilot"]
  },
  "mcpServers": {
    "aura-dev": {
      "httpUrl": "${BASE_URL}:8001",
      "headers": { "Authorization": "Bearer $TOKEN" },
      "timeout": 120000,
      "trust": true
    },
    "aura-skills": {
      "httpUrl": "${BASE_URL}:8002",
      "headers": { "Authorization": "Bearer $TOKEN" },
      "timeout": 120000,
      "trust": true
    },
    "aura-control": {
      "httpUrl": "${BASE_URL}:8003",
      "headers": { "Authorization": "Bearer $TOKEN" },
      "timeout": 120000,
      "trust": true
    },
    "aura-loop": {
      "httpUrl": "${BASE_URL}:8006",
      "headers": { "Authorization": "Bearer $TOKEN" },
      "timeout": 120000,
      "trust": true
    },
    "aura-copilot": {
      "httpUrl": "${BASE_URL}:8007",
      "headers": { "Authorization": "Bearer $TOKEN" },
      "timeout": 120000,
      "trust": true
    }
  }
}
EOF
banner "Wrote $OUT with 5 AURA MCP servers pointing to $BASE_URL"
banner "Tip: run 'gemini mcp reload' (or restart gemini-cli) to pick up the new config."
