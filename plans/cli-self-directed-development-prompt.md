# Self-Directed CLI Development Prompt

Use this prompt when continuing the AURA CLI hardening work autonomously.

```text
Continue the CLI hardening work in `/data/data/com.termux/files/home/aura_cli/aura-cli`.

Current state (already implemented):
- Spec-driven command metadata + help schema in `aura_cli/options.py`
- Spec-driven parser tree + customizers in `aura_cli/cli_options.py`
- Shared action mapping used by parser normalization and dispatch
- Registry-backed dispatch in `aura_cli/cli_main.py`
- Legacy flag compatibility + canonical subcommands
- Help schema metadata/versioning (`version`, `generated_by`, `deterministic`) + parity tests (`tests/test_cli_contract.py`)
- Structured legacy warning records (`ParsedCLIArgs.warning_records`) and JSON warning serialization (`cli_warnings`)
- Help + error snapshots (`tests/test_cli_help_snapshots.py`, `tests/test_cli_error_snapshots.py`)
- Legacy JSON dispatch snapshots for warning injection paths in `tests/test_cli_main_dispatch.py` (`--mcp-tools`, `--diag`, `--mcp-call`, `--status --json`, `--workflow-goal ...`)
- CLI docs generator (`scripts/generate_cli_reference.py`) and generated docs (`docs/CLI_REFERENCE.md`)
- CI docs/help contract job in `.github/workflows/ci.yml`

Primary objective for this pass:
1. Pick the next highest-leverage CLI maintainability improvement.
2. Implement it end-to-end (code + tests + docs/schema updates if needed).
3. Keep canonical and legacy behavior compatible unless explicitly deprecating.
4. Preserve clean JSON behavior for `--json-help`, `help`, and parse errors.

Recommended next targets (choose one or combine if small):
- Group `docs/CLI_REFERENCE.md` by top-level command and add a generated-table-of-contents section.
- Add snapshots for canonical JSON command outputs (not just legacy paths) to lock parity with legacy output shape.
- Extend CI CLI contract job to run a targeted dispatch snapshot subset (or a fast no-network subset of `tests/test_cli_main_dispatch.py`).
- Emit `cli_warnings` in additional JSON outputs if new CLI handlers are added (keep text output unchanged).
- Add a small contributor guide section describing how to update snapshots and regenerate docs after CLI changes.

Constraints:
- Do not touch unrelated dirty files.
- Preserve legacy flags: `--add-goal`, `--run-goals`, `--status`, `--goal`, `--workflow-goal`, `--mcp-tools`, `--mcp-call`, `--diag`, `--bootstrap`, `--scaffold`, `--evolve`.
- Keep `watch` and `logs` working.
- Do not reintroduce placeholder `aura_cli/options.py/*` or `aura_cli/cli_options.py/*` junk paths.

Files likely relevant:
- `aura_cli/options.py`
- `aura_cli/cli_options.py`
- `aura_cli/cli_main.py`
- `main.py`
- `tests/test_cli_options.py`
- `tests/test_cli_contract.py`
- `tests/test_cli_main_dispatch.py`
- `tests/test_cli_help_snapshots.py`
- `tests/test_cli_error_snapshots.py`
- `tests/snapshots/*`
- `scripts/generate_cli_reference.py`
- `docs/CLI_REFERENCE.md`
- `.github/workflows/ci.yml`

Working style:
- Prefer small, targeted patches.
- Update tests first or alongside code changes.
- Regenerate `docs/CLI_REFERENCE.md` when schema/help changes.
- If snapshot output changes intentionally, update snapshots in the same patch.

Validation (run before finishing):
- `python3 -m py_compile aura_cli/options.py aura_cli/cli_options.py aura_cli/cli_main.py main.py scripts/generate_cli_reference.py`
- `python3 -m pytest -q tests/test_cli_help_snapshots.py tests/test_cli_error_snapshots.py tests/test_cli_contract.py tests/test_cli_options.py tests/test_cli_main_dispatch.py tests/test_server_api.py`
- `python3 scripts/generate_cli_reference.py --check`

Output requirements:
- Summarize what changed and why.
- Include file references with line numbers.
- State any behavior changes (or explicitly state none).
- List exact validation commands and results.
```
