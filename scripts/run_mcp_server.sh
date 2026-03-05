#!/usr/bin/env bash
set -euo pipefail
PORT=${MCP_PORT:-8001}
HOST=${MCP_HOST:-0.0.0.0}
MOD=${MCP_APP_MOD:-tools.mcp_server:app}
PYTHON_BIN=${PYTHON_BIN:-python3}
echo "Starting MCP server on $HOST:$PORT (module=$MOD)"
"$PYTHON_BIN" -m uvicorn "$MOD" --host "$HOST" --port "$PORT"
