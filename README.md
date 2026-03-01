# AURA CLI

Developer entry points:

- CLI reference (generated): `docs/CLI_REFERENCE.md`
- Integration map: `docs/INTEGRATION_MAP.md`
- Primary entrypoint: `main.py`

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
