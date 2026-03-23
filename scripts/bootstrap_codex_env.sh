#!/usr/bin/env bash
# Bootstrap Codex CLI environment for AURA multi-environment MCP architecture.
#
# Usage:
#   ./scripts/bootstrap_codex_env.sh [--start-servers]
#
# Creates an isolated workspace at environments/workspaces/codex/ and
# generates a codex.mcp.config.json pointing to all AURA MCP servers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$PROJECT_ROOT/environments/workspaces/codex"
START_SERVERS="${1:-}"

echo "=== AURA Codex CLI Environment Bootstrap ==="
echo "Project root: $PROJECT_ROOT"
echo "Workspace:    $WORKSPACE"
echo ""

# --- Prerequisites ---
echo "[1/5] Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
command -v node >/dev/null 2>&1 || echo "WARN: node not found (optional for stdio MCP servers)"
echo "  python3: $(python3 --version 2>&1)"

# --- Create workspace ---
echo "[2/5] Creating workspace directories..."
mkdir -p "$WORKSPACE"/{config,logs,temp,deps}
mkdir -p "$WORKSPACE/secrets" && chmod 700 "$WORKSPACE/secrets"
echo "  Created: config, logs, temp, secrets (0700), deps"

# --- Generate config ---
echo "[3/5] Generating environment config..."
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from pathlib import Path
from environments.config import EnvironmentConfig
from environments.bootstrap import bootstrap_codex

env = EnvironmentConfig.from_name(
    name='codex',
    cli_type='codex-cli',
    base_dir=Path('$PROJECT_ROOT/environments/workspaces'),
    port_range=(8040, 8049),
)
path = bootstrap_codex(env, host='127.0.0.1')
print(f'  Generated: {path}')
" 2>&1 || {
    echo "WARN: Python bootstrap failed, generating minimal config..."
    cat > "$WORKSPACE/config/codex.mcp.config.json" << 'JSONEOF'
{
  "mcpServers": {
    "aura-hub": {"type": "http", "url": "http://127.0.0.1:8010"},
    "aura-dev_tools": {"type": "http", "url": "http://127.0.0.1:8001"},
    "aura-skills": {"type": "http", "url": "http://127.0.0.1:8002"},
    "aura-control": {"type": "http", "url": "http://127.0.0.1:8003"}
  }
}
JSONEOF
    echo "  Generated: $WORKSPACE/config/codex.mcp.config.json (minimal)"
}

# --- Write env template ---
echo "[4/5] Writing secrets template..."
if [ ! -f "$WORKSPACE/secrets/.env.template" ]; then
    cat > "$WORKSPACE/secrets/.env.template" << 'ENVEOF'
# Codex CLI environment — replace placeholders before use
MCP_API_TOKEN=changeme
OPENAI_API_KEY=changeme
HUB_TOKEN=changeme
ENVEOF
    echo "  Created: secrets/.env.template"
else
    echo "  Skipped: secrets/.env.template already exists"
fi

# --- Optionally start servers ---
echo "[5/5] Server startup..."
if [ "$START_SERVERS" = "--start-servers" ]; then
    echo "  Starting AURA MCP servers..."
    if [ -f "$PROJECT_ROOT/scripts/mcp_server_setup.sh" ]; then
        bash "$PROJECT_ROOT/scripts/mcp_server_setup.sh"
    else
        echo "  WARN: mcp_server_setup.sh not found"
    fi
else
    echo "  Skipped (pass --start-servers to auto-start)"
fi

echo ""
echo "=== Codex CLI environment ready ==="
echo "Workspace: $WORKSPACE"
echo "Config:    $WORKSPACE/config/codex.mcp.config.json"
echo ""
echo "To use with Codex CLI:"
echo "  export CODEX_MCP_CONFIG=$WORKSPACE/config/codex.mcp.config.json"
