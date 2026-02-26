#!/data/data/com.termux/files/usr/bin/bash
# Start the GitHub MCP HTTP bridge for AURA.
# Requires: GITHUB_PERSONAL_ACCESS_TOKEN set in env or .env file

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Use gh CLI token as fallback if GITHUB_PERSONAL_ACCESS_TOKEN not set
if [ -z "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    GH_TOKEN=$(gh auth token 2>/dev/null || true)
    if [ -n "$GH_TOKEN" ]; then
        export GITHUB_PERSONAL_ACCESS_TOKEN="$GH_TOKEN"
        echo "[mcp] Using token from gh CLI"
    else
        echo "[mcp] ERROR: GITHUB_PERSONAL_ACCESS_TOKEN not set and gh auth not available"
        exit 1
    fi
fi

PORT="${MCP_SERVER_PORT:-8001}"
echo "[mcp] Starting GitHub MCP bridge on http://localhost:$PORT"

cd "$PROJECT_ROOT"
exec python3 -m aura_cli.mcp_github_bridge
