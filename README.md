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

## GitHub Automation

This repo now has a governed GitHub automation baseline:

- `.github/CODEOWNERS` defines review ownership for sensitive paths.
- `.github/ISSUE_TEMPLATE/` contains structured forms for bugs, features, and agent tasks.
- `.github/pull_request_template.md` standardizes risk, testing, and agent-disclosure notes.
- `.github/workflows/pr-review-orchestrator.yml` posts one synthesized PR review summary instead of multiple public bot comments.
- `.github/workflows/merge-readiness.yml` is the final merge gate that evaluates approvals, required checks, CODEOWNERS state, and the synthesized review verdict before applying `merge-ready`.
- `.github/workflows/issue-intake.yml` converts new issues into structured planning artifacts.
- `.github/workflows/issue-comment-commands.yml` handles `/plan`, `/queue aura`, and `/review <provider>` comment commands.
- `.github/workflows/agent-task-dispatch.yml` produces provider/profile dispatch plans for coding-agent work.
- `.github/workflows/nightly-repo-health.yml` publishes a nightly repo-health issue with stale PRs, flaky workflows, recurring failures, and hotspot summaries.

`ci.yml` also runs on `merge_group`, which is required before enabling GitHub merge queue on `main`.

Legacy provider-specific PR review workflows remain available behind `AURA_LEGACY_PROVIDER_REVIEWS=1` until the unified orchestrator is fully adopted.
The older issue queue workflow remains available behind `AURA_LEGACY_ISSUE_QUEUE=1` for controlled migration.
Coding-agent dispatch is branch-only by policy, and the GitHub MCP bridge now reports that branch policy in `/health` and `/metrics`.
Protected-path PRs can now receive an `escalate` verdict from the review orchestrator without failing that check outright; merge readiness becomes the authoritative gate that waits for human approval instead.
The scheduled AURA loop is now asynchronous maintenance only: if it generates code changes, it opens a branch and PR instead of pushing to `main`.

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
