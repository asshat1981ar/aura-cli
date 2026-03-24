#!/usr/bin/env bash
# Validates the isolated Claude Code MCP environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$PROJECT_ROOT/environments/workspaces/claude"

echo "=== Validating Claude Code Environment ==="

if [ ! -d "$WORKSPACE" ]; then
    echo "ERROR: Claude workspace not found at $WORKSPACE."
    echo "Run ./scripts/bootstrap_claude_env.sh first."
    exit 1
fi

REQUIRED_DIRS=("config" "logs" "temp" "secrets" "deps")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$WORKSPACE/$dir" ]; then
        echo "ERROR: Missing required directory: $WORKSPACE/$dir"
        exit 1
    fi
    echo "✓ Directory '$dir' exists"
done

if [ ! -f "$WORKSPACE/config/mcp.json" ]; then
    echo "ERROR: Missing MCP configuration: $WORKSPACE/config/mcp.json"
    exit 1
fi
echo "✓ MCP config (mcp.json) exists"

if [ ! -f "$WORKSPACE/secrets/.env.template" ]; then
    echo "WARN: secrets/.env.template not found. You might need to set up secrets."
else
    echo "✓ secrets/.env.template exists"
fi

echo "=== Validation Successful ==="
exit 0
