#!/usr/bin/env bash
set -euo pipefail
PORT=${MCP_PORT:-8001}
HOST=${MCP_HOST:-0.0.0.0}
MOD=${MCP_APP_MOD:-tools.mcp_server:app}
echo "Starting MCP server on $HOST:$PORT (module=$MOD)"
uvicorn "$MOD" --host "$HOST" --port "$PORT"
