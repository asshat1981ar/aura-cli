#!/usr/bin/env bash
# Start all AURA HTTP MCP servers after sourcing env vars.
# Usage: bash scripts/start-mcp-servers.sh [--foreground]
#
# By default all servers run as background daemons.
# Pass --foreground to block on the primary server (port 8001) for local dev.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Source env files when present
for envfile in ".env" ".env.n8n"; do
    if [[ -f "$envfile" ]]; then
        # shellcheck source=/dev/null
        set -a
        source "$envfile"
        set +a
        echo "[start-mcp-servers] sourced $envfile"
    else
        echo "[start-mcp-servers] warning: $envfile not found — some tokens may be missing"
    fi
done

# Verify critical API key is present
if [[ -z "${AURA_API_KEY:-}${OPENROUTER_API_KEY:-}" ]]; then
    echo "[start-mcp-servers] ERROR: AURA_API_KEY / OPENROUTER_API_KEY not set — LLM calls will fail"
    echo "  Set the variable in .env or export it before running this script."
    exit 1
fi

FOREGROUND=false
if [[ "${1:-}" == "--foreground" ]]; then
    FOREGROUND=true
fi

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

start_server() {
    local name="$1"
    local port="$2"
    local module="$3"
    local logfile="$LOG_DIR/${name}.log"

    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "[start-mcp-servers] $name (port $port) already running — skipping"
        return 0
    fi

    echo "[start-mcp-servers] starting $name on port $port → $logfile"
    nohup uvicorn "${module}:app" \
        --host 127.0.0.1 \
        --port "$port" \
        --log-level info \
        >>"$logfile" 2>&1 &
    echo $! >"$LOG_DIR/${name}.pid"
    echo "[start-mcp-servers] $name PID $! started"
}

# -- AURA HTTP MCP servers --
start_server "aura-dev-tools"    8001 "aura_cli.server"
start_server "aura-skills"       8002 "tools.aura_mcp_skills_server"
start_server "aura-control"      8003 "tools.aura_control_mcp"
start_server "aura-agentic-loop" 8006 "tools.agentic_loop_mcp"
start_server "aura-copilot"      8007 "tools.github_copilot_mcp"

if [[ "$FOREGROUND" == "true" ]]; then
    echo "[start-mcp-servers] --foreground: tailing aura-dev-tools log (Ctrl-C to stop)"
    tail -f "$LOG_DIR/aura-dev-tools.log"
else
    # Give servers a moment to bind, then do a health check
    sleep 2
    echo ""
    echo "[start-mcp-servers] health check:"
    for pair in "8001:aura-dev-tools" "8002:aura-skills" "8003:aura-control" "8006:aura-agentic-loop" "8007:aura-copilot"; do
        port="${pair%%:*}"
        name="${pair##*:}"
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "  ✅ $name (port $port)"
        else
            echo "  ❌ $name (port $port) — not yet responding (check logs/$name.log)"
        fi
    done
    echo ""
    echo "[start-mcp-servers] done. PIDs in logs/*.pid, logs in logs/*.log"
fi
