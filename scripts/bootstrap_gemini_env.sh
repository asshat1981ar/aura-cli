#!/usr/bin/env bash
# Bootstrap Gemini CLI environment for AURA multi-environment MCP architecture.
#
# Usage:
#   ./scripts/bootstrap_gemini_env.sh [--start-servers]
#
# Creates an isolated workspace at environments/workspaces/gemini/ with
# config, logs, temp, secrets, and deps directories. Generates a Gemini
# CLI settings.json pointing to all AURA MCP servers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$PROJECT_ROOT/environments/workspaces/gemini"
START_SERVERS="${1:-}"

echo "=== AURA Gemini CLI Environment Bootstrap ==="
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
from environments.config import EnvironmentConfig
from environments.bootstrap import bootstrap_gemini

env = EnvironmentConfig.from_name(
    name='gemini',
    cli_type='gemini-cli',
    base_dir='$PROJECT_ROOT/environments/workspaces',
    port_range=(8020, 8029),
)
path = bootstrap_gemini(env, host='127.0.0.1')
print(f'  Generated: {path}')
" 2>&1 || {
    echo "WARN: Python bootstrap failed, generating minimal config..."
    cat > "$WORKSPACE/config/gemini_settings.json" << 'JSONEOF'
{
  "mcpServers": {
    "aura-hub": {"url": "http://127.0.0.1:8010"},
    "aura-dev_tools": {"url": "http://127.0.0.1:8001"},
    "aura-skills": {"url": "http://127.0.0.1:8002"},
    "aura-control": {"url": "http://127.0.0.1:8003"},
    "aura-agentic_loop": {"url": "http://127.0.0.1:8006"},
    "aura-copilot": {"url": "http://127.0.0.1:8007"}
  }
}
JSONEOF
    echo "  Generated: $WORKSPACE/config/gemini_settings.json (minimal)"
}

# --- Write env template ---
echo "[4/5] Writing secrets template..."
if [ ! -f "$WORKSPACE/secrets/.env.template" ]; then
    cat > "$WORKSPACE/secrets/.env.template" << 'ENVEOF'
# Gemini CLI environment — replace placeholders before use
MCP_API_TOKEN=changeme
GOOGLE_API_KEY=changeme
GEMINI_API_KEY=changeme
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
echo "=== Gemini environment ready ==="
echo "Workspace: $WORKSPACE"
echo "Config:    $WORKSPACE/config/gemini_settings.json"
echo ""
echo "To use with Gemini CLI:"
echo "  cp $WORKSPACE/config/gemini_settings.json ~/.gemini/settings.json"
echo "  # Or set GEMINI_MCP_CONFIG=$WORKSPACE/config/gemini_settings.json"
