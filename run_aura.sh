#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PYTHON_BIN="${AURA_PYTHON_BIN:-python3}"

print_help() {
  cat <<'EOF'
Usage:
  ./run_aura.sh
  ./run_aura.sh help
  ./run_aura.sh run [goal-run flags...]
  ./run_aura.sh once "<goal>" [flags...]
  ./run_aura.sh add "<goal>"
  ./run_aura.sh status
  ./run_aura.sh <canonical aura command...>

Wrapper aliases:
  help, --help, -h   Show this wrapper help
  run                Forward to `python3 main.py goal run`
  once               Forward to `python3 main.py goal once`
  add                Forward to `python3 main.py goal add`
  status             Forward to `python3 main.py goal status`
  agents             Forward to `python3 main.py agent list`
  interactive        Start the interactive CLI

Pass-through:
  Canonical commands are forwarded unchanged, for example:
    ./run_aura.sh goal run --dry-run
    ./run_aura.sh contract-report --check  [EXPERIMENTAL]

Experimental commands (forwarded unchanged, stability not guaranteed):
  studio             Launch the rich real-time dashboard  [EXPERIMENTAL]
  watch              Launch the AuraStudio terminal UI    [EXPERIMENTAL]
  workflow run       Run an orchestrated workflow goal    [EXPERIMENTAL]
  mcp tools          List MCP tools from repo-local config [EXPERIMENTAL]
  mcp call           Invoke a repo-local MCP tool         [EXPERIMENTAL]
  memory search      Semantic memory search               [EXPERIMENTAL]
  memory reindex     Rebuild semantic memory embeddings   [EXPERIMENTAL]
  bootstrap          Bootstrap local config files         [EXPERIMENTAL]
  sadd               Sub-Agent Driven Development         [EXPERIMENTAL]
  diag               MCP diagnostics snapshot             [EXPERIMENTAL]

Convenience:
  If the first argument is a flag such as `--dry-run`, the wrapper assumes
  `goal run` so `./run_aura.sh --dry-run` still works.

Tip:
  Set `AURA_SKIP_CHDIR=1` when running locally or in tests to keep the
  current working directory unchanged.

  MCP servers need auth tokens exported before starting. Source the env file:
    source .env.n8n
  Then start servers:
    uvicorn aura_cli.server:app --port 8001 &
    uvicorn tools.aura_mcp_skills_server:app --port 8002 &
    uvicorn tools.github_copilot_mcp:app --port 8007 &

Android local model gate:
  Set `AURA_REQUIRE_LOCAL_MODEL_HEALTH=1` to verify configured local
  coder/planner/embedding endpoints before runtime commands start.
EOF
}

maybe_check_local_models() {
  if [ "${AURA_REQUIRE_LOCAL_MODEL_HEALTH:-0}" != "1" ]; then
    return 0
  fi

  local config_path="${AURA_ANDROID_CONFIG_PATH:-${ROOT_DIR}/aura.config.json}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_android_local_models.py" --config "${config_path}"
}

exec_main() {
  if [ "${AURA_WRAPPER_ECHO:-0}" = "1" ]; then
    printf '%s\n' "${PYTHON_BIN}" "${ROOT_DIR}/main.py" "$@"
    return 0
  fi
  exec "${PYTHON_BIN}" "${ROOT_DIR}/main.py" "$@"
}

exec_main_checked() {
  maybe_check_local_models
  exec_main "$@"
}

if [ "$#" -eq 0 ]; then
  exec_main_checked
fi

case "$1" in
  help|--help|-h)
    print_help
    ;;
  interactive)
    shift
    exec_main_checked "$@"
    ;;
  add)
    shift
    exec_main goal add "$@"
    ;;
  once)
    shift
    exec_main_checked goal once "$@"
    ;;
  resume)
    shift
    exec_main goal resume "$@"
    ;;
  run)
    shift
    exec_main_checked goal run "$@"
    ;;
  status)
    shift
    exec_main goal status "$@"
    ;;
  agents)
    shift
    exec_main agent list "$@"
    ;;
  --*)
    exec_main_checked goal run "$@"
    ;;
  *)
    case "$1" in
      goal|workflow|watch|studio|evolve)
        exec_main_checked "$@"
        ;;
      *)
        exec_main "$@"
        ;;
    esac
    ;;
esac
