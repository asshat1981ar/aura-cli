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
  interactive        Start the interactive CLI

Pass-through:
  Canonical commands are forwarded unchanged, for example:
    ./run_aura.sh goal run --dry-run
    ./run_aura.sh contract-report --check

Convenience:
  If the first argument is a flag such as `--dry-run`, the wrapper assumes
  `goal run` so `./run_aura.sh --dry-run` still works.

Tip:
  Set `AURA_SKIP_CHDIR=1` when running locally or in tests to keep the
  current working directory unchanged.

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
