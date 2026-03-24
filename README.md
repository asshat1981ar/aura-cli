# AURA CLI

Developer entry points:

- CLI reference (generated): `docs/CLI_REFERENCE.md`
- Integration map: `docs/INTEGRATION_MAP.md`
- Operator prompt: `docs/AURA_OPERATOR_PROMPT.md`
- Iterative workflow: `docs/AURA_ITERATIVE_WORKFLOW.md`
- Multi-agent workflow: `docs/AURA_MULTI_AGENT_WORKFLOW.md`
- Sweep templates: `docs/AURA_SWEEP_TEMPLATES.md`
- Active sweep status: `docs/ACTIVE_SWEEP_STATUS.md`
- PR reviewer summary template: `docs/PR_REVIEWER_SUMMARY_TEMPLATE.md`
- Active PR reviewer summary: `docs/ACTIVE_PR_REVIEWER_SUMMARY.md`
- Sweep artifact generator: `python3 scripts/generate_active_sweep_artifacts.py --pr <number>`
- Sweep artifact drift check: `python3 scripts/generate_active_sweep_artifacts.py --pr <number> --check`
- Canonical runtime entrypoint: installed `aura` console script → `aura_cli.cli_main:main`
- Developer shim: `main.py` (lightweight wrapper that delegates to `aura_cli.cli_main.main()`)
- Shell wrapper: `run_aura.sh`

## Sweep Artifact Generator

`scripts/generate_active_sweep_artifacts.py` can populate or check the live sweep docs for the current branch.

Defaults and inferred inputs:

- `--pr` overrides the active PR number.
- If `--pr` is omitted, the script checks `AURA_ACTIVE_PR`, `GITHUB_PR_NUMBER`, then `PR_NUMBER`.
- `--ci-checks` overrides the reviewer-summary check list.
- If `--ci-checks` is omitted, the script checks `AURA_CI_CHECKS`, then `GITHUB_CHECKS`.
- `--reviewer-complete` overrides reviewer completion state.
- If `--reviewer-complete` is omitted, the script checks `AURA_REVIEWER_COMPLETE`.
- `--check` validates the current artifact files instead of rewriting them.

Examples:

- `python3 scripts/generate_active_sweep_artifacts.py --pr 219`
- `python3 scripts/generate_active_sweep_artifacts.py --check`
- `AURA_CI_CHECKS="Python CI,Claude Code Review" AURA_REVIEWER_COMPLETE=true python3 scripts/generate_active_sweep_artifacts.py --check`
- `python3 scripts/generate_active_sweep_artifacts.py --config docs/ACTIVE_SWEEP_CONFIG.example.json`

Optional config file:

- Use `--config <path>` or `AURA_SWEEP_CONFIG=<path>` to load branch-specific defaults from JSON.
- See `docs/ACTIVE_SWEEP_CONFIG.example.json` for the supported keys.

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

## GitHub Copilot CLI in this repo

This repository already ships repo-specific Copilot guidance and MCP-compatible HTTP servers.

Use these setup assets for a safe local configuration:

- MCP example: `.vscode/mcp.json.example`
- Copilot MCP config helper: `bash scripts/configure_copilot_mcp.sh`
- Repo LSP config: `.github/lsp.json`
- Primary repo instructions: `.github/copilot-instructions.md`
- Focused instruction shards: `.github/instructions/copilot/`

Typical local flow:

1. Start the AURA servers you want to expose to Copilot:
   - `uvicorn aura_cli.server:app --port 8001`
   - `uvicorn tools.aura_mcp_skills_server:app --port 8002`
   - optional: `uvicorn tools.github_copilot_mcp:app --port 8007`
2. Generate your local Copilot MCP config:
   - `bash scripts/configure_copilot_mcp.sh`
3. Launch `copilot` in the repo and verify:
   - `/mcp` shows the configured servers
   - `/lsp` picks up the repo-local Python LSP
   - `/instructions` shows the repo guidance

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
