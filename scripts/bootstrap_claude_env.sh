#!/usr/bin/env bash
# Bootstrap Claude Code environment for AURA multi-environment MCP architecture.
#
# Usage:
#   ./scripts/bootstrap_claude_env.sh [--start-servers]
#
# Creates an isolated workspace at environments/workspaces/claude/ and
# generates a .mcp.json pointing to all AURA MCP servers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$PROJECT_ROOT/environments/workspaces/claude"
START_SERVERS="${1:-}"

echo "=== AURA Claude Code Environment Bootstrap ==="
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
from environments.bootstrap import bootstrap_claude

env = EnvironmentConfig.from_name(
    name='claude',
    cli_type='claude-code',
    base_dir=Path('$PROJECT_ROOT/environments/workspaces'),
    port_range=(8030, 8039),
)
path = bootstrap_claude(env, host='127.0.0.1', project_root=Path('$PROJECT_ROOT'))
print(f'  Generated: {path}')
" 2>&1 || {
    echo "WARN: Python bootstrap failed, generating minimal config..."
    cat > "$WORKSPACE/config/mcp.json" << 'JSONEOF'
{
  "mcpServers": {
    "aura-hub": {"type": "http", "url": "http://127.0.0.1:8010"},
    "aura-dev_tools": {"type": "http", "url": "http://127.0.0.1:8001"},
    "aura-skills": {"type": "http", "url": "http://127.0.0.1:8002"},
    "aura-control": {"type": "http", "url": "http://127.0.0.1:8003"},
    "filesystem": {"type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
    "git": {"type": "stdio", "command": "npx", "args": ["-y", "@anthropic/mcp-server-git"]}
  }
}
JSONEOF
    echo "  Generated: $WORKSPACE/config/mcp.json (minimal)"
}

# --- Write env template ---
echo "[4/5] Writing secrets template..."
if [ ! -f "$WORKSPACE/secrets/.env.template" ]; then
    cat > "$WORKSPACE/secrets/.env.template" << 'ENVEOF'
# Claude Code environment — replace placeholders before use
MCP_API_TOKEN=changeme
ANTHROPIC_API_KEY=changeme
HUB_TOKEN=changeme
GITHUB_PAT=changeme
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
echo "=== Claude Code environment ready ==="
echo "Workspace: $WORKSPACE"
echo "Config:    $WORKSPACE/config/mcp.json"
echo ""
echo "To use with Claude Code:"
echo "  cp $WORKSPACE/config/mcp.json $PROJECT_ROOT/.mcp.json"
