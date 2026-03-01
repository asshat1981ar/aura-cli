# AURA CLI

Developer entry points:

- CLI reference (generated): `docs/CLI_REFERENCE.md`
- Integration map: `docs/INTEGRATION_MAP.md`
- Primary entrypoint: `main.py`
- Shell wrapper: `run_aura.sh`

## Wrapper Usage

`run_aura.sh` is a convenience wrapper around `python3 main.py`.

- `./run_aura.sh` starts the interactive CLI.
- `./run_aura.sh run --dry-run` forwards to `python3 main.py goal run --dry-run`.
- `./run_aura.sh once "Summarize repo" --max-cycles 1` forwards to `python3 main.py goal once ...`.
- `./run_aura.sh goal status` passes canonical commands through unchanged.
- `./run_aura.sh --help` shows wrapper-specific usage and alias help.

## CLI Maintenance

When changing CLI commands, help text, parsing, or JSON output contracts:

1. Regenerate the CLI reference:
   - `python3 scripts/generate_cli_reference.py`
2. Run CLI docs/snapshot checks:
   - `python3 -m pytest -q tests/test_cli_docs_generator.py tests/test_cli_help_snapshots.py tests/test_cli_error_snapshots.py tests/test_cli_main_dispatch.py -k snapshot`
3. If output changes intentionally, update the affected files in `tests/snapshots/`.
4. Verify docs are current:
   - `python3 scripts/generate_cli_reference.py --check`

CI enforces the generated CLI docs and snapshot contracts via `.github/workflows/ci.yml`.

## Autonomous Apply Safety

Autonomous code-apply paths (queue loop, orchestrator, hybrid loop, mutator, and atomic apply) enforce an explicit overwrite safety policy for stale-snippet mismatches.

- Stale snippet mismatch + `overwrite_file=true` is blocked by default.
- Intentional full-file replacement must use:
  - `overwrite_file=true`
  - `old_code=""` (empty string)
- Policy-block failures are logged as `old_code_mismatch_overwrite_blocked` with policy `explicit_overwrite_file_required`.

This policy is centralized in `core/file_tools.py` via `allow_mismatch_overwrite_for_change(...)` and `apply_change_with_explicit_overwrite_policy(...)`.

## Capability Bootstrap

AURA can now expand its skill set for a cycle when a goal clearly implies extra tooling, and it can optionally provision known MCP server config using the existing setup script.

- Skill augmentation is enabled by default via `auto_add_capabilities=true`.
- Missing skills can be turned into queued self-development goals via `auto_queue_missing_capabilities=true`.
- Those self-development goals are now pushed to the front of the remaining queue so AURA can close capability gaps right after the current goal completes.
- MCP provisioning is opt-in via `AURA_AUTO_PROVISION_MCP=true`.
- Starting MCP servers as part of that provisioning is separately opt-in via `AURA_AUTO_START_MCP_SERVERS=true`.
- Provisioning decisions are recorded in cycle `phase_outputs` as `capability_plan` and `capability_provisioning`.
- Queued self-development follow-ups are recorded as `capability_goal_queue`.
- `goal status` and `doctor` now surface the last matched capability rules plus pending/running MCP bootstrap actions.
