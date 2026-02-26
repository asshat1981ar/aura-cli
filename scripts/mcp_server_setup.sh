#!/usr/bin/env bash
# Setup helper for MCP servers (aura-mcp + mcpcodeserver + stdio helpers).
# - Validates configs and auto-corrects common issues (port drift, missing entries).
# - Starts servers only when their ports are free.
# - Adds light diagnostics and error reporting.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MCP_SETUP_LOG_DIR:-$ROOT_DIR/tmp_out}"
LOG_FILE="$LOG_DIR/mcp_server_setup.log"

AURA_PORT="${AURA_PORT:-8001}"
MCP_CODE_PORT="${MCP_CODE_PORT:-3000}"   # mcpcodeserver ignores mcp.json port; 3000 is stable.
MCP_CONFIG="${MCP_CONFIG:-$HOME/mcp.json}"
CODEX_CONFIG="${CODEX_CONFIG:-$HOME/codex.mcp.config.json}"
GEMINI_CONFIG="${GEMINI_CONFIG:-$HOME/.gemini/settings.json}"

NO_START="${MCP_SETUP_NO_START:-0}"

mkdir -p "$LOG_DIR"

banner() { printf '\n== %s ==\n' "$*"; }
warn()   { printf '[warn] %s\n' "$*" >&2; }
info()   { printf '[info] %s\n' "$*"; }

trap 'warn "Setup hit an error. See $LOG_FILE for details."' ERR

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    warn "Missing command '$1'. Install it and re-run."
    exit 1
  fi
}

is_listening() {
  local port=$1
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$port )" | grep -q LISTEN
  elif command -v lsof >/dev/null 2>&1; then
    lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    warn "Neither ss nor lsof found; cannot check port $port."
    return 1
  fi
}

print_port_owner() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -i ":$port" -sTCP:LISTEN || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp "( sport = :$port )" || true
  fi
}

patch_json() {
  # Args: file var_name python_snippet
  local file=$1
  local label=$2
  local py=$3
  python3 - <<'PY' "$file" "$label" "$py"
import json, os, sys

path, label, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
data = {}
if os.path.exists(path):
    with open(path, "r") as fh:
        try:
            data = json.load(fh)
        except Exception as exc:
            print(f"[warn] {path} unreadable ({exc}); replacing with fresh config.")
            data = {}

globals_dict = {"os": os, "json": json}
locals_dict = {"data": data}
env = dict(globals_dict)
env.update(locals_dict)
exec(snippet, env, env)
data = env["data"]

with open(path, "w") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
print(f"[info] ensured {label} at {path}")
PY
}

ensure_mcp_config() {
  patch_json "$MCP_CONFIG" "mcp.json" "$(cat <<'PY'
port = int(os.environ.get("MCP_CODE_PORT", "3000"))
data.setdefault("mcpServers", {})
data["mcpServers"].setdefault("fs", {
    "command": "npx",
    "args": ["@modelcontextprotocol/server-filesystem", os.path.expanduser("~")],
    "stdio": True
})
data["mcpServers"].setdefault("git", {
    "command": "npx",
    "args": ["@cyanheads/git-mcp-server"],
    "stdio": True
})
data["mcpServers"].setdefault("gh", {
    "command": "npx",
    "args": ["@modelcontextprotocol/server-github"],
    "stdio": True,
    "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
})
data["http"] = {"enabled": True, "port": port}
PY
)"
}

ensure_codex_config() {
  patch_json "$CODEX_CONFIG" "codex.mcp.config.json" "$(cat <<'PY'
data.setdefault("rmcpClient", True)
data.setdefault("mcpServers", {})
port = int(os.environ.get("MCP_CODE_PORT", "3000"))
aura_port = int(os.environ.get("AURA_PORT", "8001"))
def ensure_server(name, payload):
    data["mcpServers"].setdefault(name, {}).update(payload)
ensure_server("aura-mcp", {
    "enabled": True,
    "http": {
        "baseUrl": f"http://localhost:{aura_port}",
        "headers": {"Authorization": f"Bearer {os.environ.get('MCP_API_TOKEN', 'dev-mcp-token')}"}
    },
    "timeout": 120
})
ensure_server("mcpcodeserver", {
    "enabled": True,
    "http": {"baseUrl": f"http://localhost:{port}/mcp"},
    "timeout": 120
})
ensure_server("gh", {
    "enabled": True,
    "command": "npx",
    "args": ["@modelcontextprotocol/server-github"],
    "stdio": True,
    "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
})
PY
)"
}

ensure_gemini_config() {
  patch_json "$GEMINI_CONFIG" "gemini settings" "$(cat <<'PY'
allowed = set(data.get("mcp", {}).get("allowed", []))
allowed.update(["aura-mcp", "mcpcodeserver"])
data.setdefault("mcp", {})["allowed"] = sorted(allowed)
data.setdefault("mcpServers", {})
port = int(os.environ.get("MCP_CODE_PORT", "3000"))
aura_port = int(os.environ.get("AURA_PORT", "8001"))
data["mcpServers"]["aura-mcp"] = {
    "httpUrl": f"http://localhost:{aura_port}",
    "headers": {"Authorization": f"Bearer {os.environ.get('MCP_API_TOKEN', 'dev-mcp-token')}"},
    "timeout": 120000,
    "trust": False,
    "description": "AURA MCP server"
}
data["mcpServers"]["mcpcodeserver"] = {
    "httpUrl": f"http://localhost:{port}/mcp",
    "timeout": 120000,
    "trust": False,
    "description": "mcpcodeserver (fs/git/gh)"
}
PY
)"
}

start_aura_mcp() {
  if is_listening "$AURA_PORT"; then
    info "aura-mcp already listening on :$AURA_PORT"
    print_port_owner "$AURA_PORT"
    return
  fi
  if [[ "$NO_START" == "1" ]]; then
    warn "Skipping aura-mcp start (NO_START=1)."
    return
  fi
  banner "Starting aura-mcp on :$AURA_PORT"
  MCP_PORT="$AURA_PORT" MCP_HOST="127.0.0.1" \
    nohup "$ROOT_DIR/scripts/run_mcp_server.sh" >>"$LOG_FILE" 2>&1 &
  info "aura-mcp pid $! (logs: $LOG_FILE)"
}

start_mcpcodeserver() {
  if is_listening "$MCP_CODE_PORT"; then
    info "mcpcodeserver already listening on :$MCP_CODE_PORT"
    print_port_owner "$MCP_CODE_PORT"
    return
  fi
  if [[ "$NO_START" == "1" ]]; then
    warn "Skipping mcpcodeserver start (NO_START=1)."
    return
  fi
  banner "Starting mcpcodeserver on :$MCP_CODE_PORT"
  (
    cd "$HOME" || exit 1
    PORT="$MCP_CODE_PORT" nohup npx mcpcodeserver --http --config "$MCP_CONFIG" >>"$LOG_FILE" 2>&1
  ) &
  info "mcpcodeserver pid $! (logs: $LOG_FILE)"
}

discovery_probe() {
  banner "Discovery / sanity checks"
  info "Validating ports"
  for p in "$AURA_PORT" "$MCP_CODE_PORT"; do
    if is_listening "$p"; then
      info "Port $p OK"
    else
      warn "Port $p not listening yet (may still be starting)"
    fi
  done
}

main() {
  banner "MCP server setup"
  require_cmd python3
  require_cmd npx
  require_cmd uvicorn
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    warn "GITHUB_TOKEN is not set; github MCP calls will fail until you export it."
  fi

  ensure_mcp_config
  ensure_codex_config
  ensure_gemini_config

  start_aura_mcp
  start_mcpcodeserver

  discovery_probe

  banner "Done"
  info "Configs:"
  info "- $MCP_CONFIG"
  info "- $CODEX_CONFIG"
  info "- $GEMINI_CONFIG"
  info "Log: $LOG_FILE"
  info "Set MCP_SETUP_NO_START=1 to only patch configs."
}

main "$@"
