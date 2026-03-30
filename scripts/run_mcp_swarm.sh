#!/usr/bin/env bash
# Start the Swarm MCP HTTP bridge for AURA.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${MCP_SERVER_PORT:-8050}"
echo "[mcp] Starting Swarm MCP bridge on http://localhost:$PORT"

cd "$PROJECT_ROOT"
exec python3 -m aura_cli.mcp_swarm_bridge
