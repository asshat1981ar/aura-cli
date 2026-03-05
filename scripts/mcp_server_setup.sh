#!/usr/bin/env bash
# Setup helper for MCP servers (aura-mcp + mcpcodeserver + stdio helpers).
# - Validates configs and auto-corrects common issues (port drift, missing entries).
# - Starts servers only when their ports are free.
# - Runs active readiness probes and reports actionable diagnostics.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MCP_SETUP_LOG_DIR:-$ROOT_DIR/tmp_out}"
LOG_FILE="$LOG_DIR/mcp_server_setup.log"

AURA_PORT="${AURA_PORT:-8001}"
MCP_CODE_PORT="${MCP_CODE_PORT:-3000}"   # mcpcodeserver ignores mcp.json port; 3000 is stable.
MCP_CONFIG="${MCP_CONFIG:-$HOME/mcp.json}"
CODEX_CONFIG="${CODEX_CONFIG:-$HOME/codex.mcp.config.json}"
GEMINI_CONFIG="${GEMINI_CONFIG:-$HOME/.gemini/settings.json}"
CURL_BIN="${CURL_BIN:-curl}"
WAIT_TRIES="${WAIT_TRIES:-20}"
WAIT_SLEEP="${WAIT_SLEEP:-1}"

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

require_python_modules() {
  if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util
missing = [m for m in ("uvicorn", "fastapi") if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
PY
  then
    warn "python3 is missing required modules: uvicorn and/or fastapi."
    warn "Install with: python3 -m pip install --user 'fastapi<0.100' 'pydantic<2' 'uvicorn<0.25'"
    exit 1
  fi
}

is_listening() {
  local port=$1
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$port )" | grep -q LISTEN
  elif command -v lsof >/dev/null 2>&1; then
    lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | grep -q ":$port "
  else
    warn "No port-check tool found (ss/lsof/netstat); cannot check port $port."
    return 1
  fi
}

print_port_owner() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -i ":$port" -sTCP:LISTEN || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp "( sport = :$port )" || true
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | grep ":$port " || true
  fi
}

wait_http_get() {
  local label=$1
  local url=$2
  local i
  for i in $(seq 1 "$WAIT_TRIES"); do
    if "$CURL_BIN" -sS -m 3 "$url" >/dev/null 2>&1; then
      info "$label reachable: $url"
      return 0
    fi
    sleep "$WAIT_SLEEP"
  done
  warn "$label not reachable after ${WAIT_TRIES} tries: $url"
  return 1
}

probe_aura_mcp() {
  local base="http://127.0.0.1:${AURA_PORT}"
  if ! wait_http_get "aura-mcp /health" "$base/health"; then
    return 1
  fi
  if "$CURL_BIN" -sS -m 3 "$base/tools" >/dev/null 2>&1; then
    info "aura-mcp /tools reachable"
  else
    warn "aura-mcp /tools probe failed"
  fi
  if "$CURL_BIN" -sS -m 3 -X POST "$base/call" \
    -H "content-type: application/json" \
    --data '{"tool_name":"limits","args":{}}' >/dev/null 2>&1; then
    info "aura-mcp /call limits probe passed"
  else
    warn "aura-mcp /call limits probe failed"
  fi
}

probe_mcpcodeserver() {
  local url="http://127.0.0.1:${MCP_CODE_PORT}/mcp"
  local init_payload='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"mcp-setup","version":"1.0"}}}'
  local tools_payload='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
  local tools_resp=""
  local i
  for i in $(seq 1 "$WAIT_TRIES"); do
    if "$CURL_BIN" -sS -m 4 -X POST "$url" \
      -H "content-type: application/json" \
      --data "$init_payload" >/dev/null 2>&1; then
      info "mcpcodeserver /mcp initialize probe passed"
      break
    fi
    sleep "$WAIT_SLEEP"
  done
  if [[ "$i" -gt "$WAIT_TRIES" ]]; then
    warn "mcpcodeserver /mcp not reachable after ${WAIT_TRIES} tries"
    return 1
  fi
  for i in $(seq 1 "$WAIT_TRIES"); do
    tools_resp=$("$CURL_BIN" -sS -m 4 -X POST "$url" \
      -H "content-type: application/json" \
      --data "$tools_payload" || true)
    if [[ -n "$tools_resp" ]]; then
      break
    fi
    sleep "$WAIT_SLEEP"
  done
  if [[ -z "$tools_resp" ]]; then
    warn "mcpcodeserver tools/list probe failed"
  elif printf '%s' "$tools_resp" | grep -q '"error"'; then
    warn "mcpcodeserver tools/list returned JSON-RPC error, but endpoint is reachable"
  else
    info "mcpcodeserver tools/list probe passed"
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
# Normalize npx servers to non-interactive mode so startup cannot block on prompts.
for server_cfg in data.get("mcpServers", {}).values():
    if not isinstance(server_cfg, dict):
        continue
    if server_cfg.get("command") != "npx":
        continue
    args = server_cfg.get("args") or []
    if not args:
        server_cfg["args"] = ["-y"]
    elif args[0] != "-y":
        server_cfg["args"] = ["-y"] + list(args)
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
  (
    cd "$ROOT_DIR" || exit 1
    MCP_PORT="$AURA_PORT" MCP_HOST="127.0.0.1" \
      nohup "$ROOT_DIR/scripts/run_mcp_server.sh" >>"$LOG_FILE" 2>&1
  ) &
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
    PORT="$MCP_CODE_PORT" nohup npx -y mcpcodeserver --http --config "$MCP_CONFIG" >>"$LOG_FILE" 2>&1
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
  if [[ "$NO_START" != "1" ]]; then
    probe_aura_mcp || true
    probe_mcpcodeserver || true
  fi
}

main() {
  banner "MCP server setup"
  require_cmd python3
  require_cmd npx
  require_cmd "$CURL_BIN"
  require_python_modules
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
  info "Check: $ROOT_DIR/scripts/mcp_server_check.sh"
  info "Set MCP_SETUP_NO_START=1 to only patch configs."
}

main "$@"
