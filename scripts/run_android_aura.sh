#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
WRAPPER="${AURA_ANDROID_WRAPPER:-${ROOT_DIR}/run_aura.sh}"

print_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_android_aura.sh
  bash scripts/run_android_aura.sh watch [watch flags...]
  bash scripts/run_android_aura.sh studio [studio flags...]

Behavior:
  - defaults to `watch`
  - enables `AURA_REQUIRE_LOCAL_MODEL_HEALTH=1` unless already set
  - forwards all remaining args to `run_aura.sh`

Examples:
  bash scripts/run_android_aura.sh
  bash scripts/run_android_aura.sh watch --autonomous
  bash scripts/run_android_aura.sh studio --autonomous

Optional env:
  AURA_ANDROID_CONFIG_PATH   Override config file used by local model health checks
  AURA_ANDROID_WRAPPER       Override wrapper path for advanced setups
EOF
}

if [ "${1:-}" = "help" ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  print_help
  exit 0
fi

MODE="${1:-watch}"
case "$MODE" in
  watch|studio)
    shift || true
    ;;
  --*)
    MODE="watch"
    ;;
  "")
    MODE="watch"
    ;;
  *)
    echo "error: expected \`watch\` or \`studio\`, got: $MODE" >&2
    echo "hint: run \`bash scripts/run_android_aura.sh --help\`" >&2
    exit 2
    ;;
esac

if [ ! -f "$WRAPPER" ]; then
  echo "error: wrapper not found: $WRAPPER" >&2
  exit 1
fi

export AURA_REQUIRE_LOCAL_MODEL_HEALTH="${AURA_REQUIRE_LOCAL_MODEL_HEALTH:-1}"

exec bash "$WRAPPER" "$MODE" "$@"
