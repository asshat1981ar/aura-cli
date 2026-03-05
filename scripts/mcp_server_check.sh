#!/usr/bin/env bash
# Active readiness check for local MCP servers.
# - Verifies aura-mcp endpoints (/health, /tools, /call limits)
# - Verifies mcpcodeserver /mcp initialize and tools/list behavior

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${MCP_CHECK_OUT_DIR:-$ROOT_DIR/tmp_out}"
mkdir -p "$OUT_DIR"

CURL_BIN="${CURL_BIN:-curl}"
AURA_PORT="${AURA_PORT:-8001}"
MCP_CODE_PORT="${MCP_CODE_PORT:-3000}"
TRIES="${MCP_CHECK_TRIES:-12}"
SLEEP_SECS="${MCP_CHECK_SLEEP:-1}"
TOKEN="${MCP_API_TOKEN:-}"
RETRY_HTTP_CODE="000"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

banner() { printf "\n== %s ==\n" "$*"; }
pass()   { printf "[pass] %s\n" "$*"; PASS_COUNT=$((PASS_COUNT + 1)); }
warn()   { printf "[warn] %s\n" "$*"; WARN_COUNT=$((WARN_COUNT + 1)); }
fail()   { printf "[fail] %s\n" "$*" >&2; FAIL_COUNT=$((FAIL_COUNT + 1)); }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[fail] Missing command '$1'" >&2
    exit 1
  fi
}

request_http_code() {
  # Args: method url body_file data(optional) header(optional)
  local method=$1
  local url=$2
  local body_file=$3
  local data=${4:-}
  local auth_header=${5:-}

  if [[ -n "$data" ]]; then
    if [[ -n "$auth_header" ]]; then
      "$CURL_BIN" -sS -m 4 -o "$body_file" -w "%{http_code}" \
        -X "$method" "$url" \
        -H "content-type: application/json" \
        -H "$auth_header" \
        --data "$data"
    else
      "$CURL_BIN" -sS -m 4 -o "$body_file" -w "%{http_code}" \
        -X "$method" "$url" \
        -H "content-type: application/json" \
        --data "$data"
    fi
  else
    if [[ -n "$auth_header" ]]; then
      "$CURL_BIN" -sS -m 4 -o "$body_file" -w "%{http_code}" \
        -X "$method" "$url" \
        -H "$auth_header"
    else
      "$CURL_BIN" -sS -m 4 -o "$body_file" -w "%{http_code}" \
        -X "$method" "$url"
    fi
  fi
}

retry_request() {
  # Args: label method url out_file data(optional) auth_header(optional)
  local label=$1
  local method=$2
  local url=$3
  local out_file=$4
  local data=${5:-}
  local auth_header=${6:-}
  local code="000"
  local i

  for i in $(seq 1 "$TRIES"); do
    code=$(request_http_code "$method" "$url" "$out_file" "$data" "$auth_header" 2>/dev/null || true)
    if [[ "$code" != "000" ]]; then
      RETRY_HTTP_CODE="$code"
      return 0
    fi
    sleep "$SLEEP_SECS"
  done

  fail "$label unreachable after ${TRIES} tries ($url)"
  RETRY_HTTP_CODE="000"
  return 1
}

check_aura() {
  banner "aura-mcp"
  local base="http://127.0.0.1:${AURA_PORT}"
  local auth_header=""
  if [[ -n "$TOKEN" ]]; then
    auth_header="Authorization: Bearer $TOKEN"
  fi

  local health_file="$OUT_DIR/check_aura_health.json"
  local tools_file="$OUT_DIR/check_aura_tools.json"
  local call_file="$OUT_DIR/check_aura_call.json"
  local code

  retry_request "aura /health" "GET" "$base/health" "$health_file" || true
  code="$RETRY_HTTP_CODE"
  if [[ "$code" == "200" ]]; then
    pass "aura /health OK"
  elif [[ "$code" != "000" ]]; then
    warn "aura /health returned HTTP $code"
  fi

  retry_request "aura /tools" "GET" "$base/tools" "$tools_file" "" "$auth_header" || true
  code="$RETRY_HTTP_CODE"
  if [[ "$code" == "200" ]]; then
    pass "aura /tools OK"
  elif [[ "$code" == "401" || "$code" == "403" ]]; then
    fail "aura /tools auth failed (set MCP_API_TOKEN)"
  elif [[ "$code" != "000" ]]; then
    warn "aura /tools returned HTTP $code"
  fi

  retry_request \
    "aura /call limits" "POST" "$base/call" "$call_file" \
    '{"tool_name":"limits","args":{}}' "$auth_header" || true
  code="$RETRY_HTTP_CODE"
  if [[ "$code" == "200" ]]; then
    pass "aura /call limits OK"
  elif [[ "$code" == "401" || "$code" == "403" ]]; then
    fail "aura /call auth failed (set MCP_API_TOKEN)"
  elif [[ "$code" != "000" ]]; then
    warn "aura /call returned HTTP $code"
  fi
}

check_mcpcodeserver() {
  banner "mcpcodeserver"
  local url="http://127.0.0.1:${MCP_CODE_PORT}/mcp"
  local init_file="$OUT_DIR/check_mcpc_init.json"
  local tools_file="$OUT_DIR/check_mcpc_tools.json"
  local init_payload='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"mcp-check","version":"1.0"}}}'
  local tools_payload='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
  local code

  retry_request "mcpc initialize" "POST" "$url" "$init_file" "$init_payload" || true
  code="$RETRY_HTTP_CODE"
  if [[ "$code" == "200" ]]; then
    if grep -q '"error"' "$init_file"; then
      warn "mcpc initialize returned JSON-RPC error but endpoint is reachable"
    else
      pass "mcpc initialize OK"
    fi
  elif [[ "$code" != "000" ]]; then
    warn "mcpc initialize returned HTTP $code"
  fi

  retry_request "mcpc tools/list" "POST" "$url" "$tools_file" "$tools_payload" || true
  code="$RETRY_HTTP_CODE"
  if [[ "$code" == "200" ]]; then
    if grep -q '"error"' "$tools_file"; then
      warn "mcpc tools/list returned JSON-RPC error"
    else
      pass "mcpc tools/list OK"
    fi
  elif [[ "$code" != "000" ]]; then
    warn "mcpc tools/list returned HTTP $code"
  fi
}

main() {
  require_cmd "$CURL_BIN"

  banner "MCP check"
  check_aura
  check_mcpcodeserver

  banner "Summary"
  printf "[info] pass=%s warn=%s fail=%s\n" "$PASS_COUNT" "$WARN_COUNT" "$FAIL_COUNT"
  printf "[info] output files: %s\n" "$OUT_DIR"

  if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
