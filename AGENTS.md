# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: primary CLI entrypoint.
- `run_aura.sh`: convenience wrapper for running the CLI.
- `aura_cli/`: CLI wiring and command handlers (e.g., `cli_main.py`).
- `core/`: orchestration, loops, queues, config, git tooling.
- `agents/`: specialized agent modules used by the loop.
- `memory/`: runtime state and persistence (history, DBs, queues).
- `tests/`: unit/integration tests.
- `docs/`: architecture notes (see `docs/INTEGRATION_MAP.md`).

## Autonomous Development Workflow
AURA runs an autonomous loop that processes queued goals with multiple agents.

Typical flow:
1. Start the CLI.
2. Add goals to the queue.
3. Run the loop to process goals.
4. Review logs and `memory/` artifacts.

Examples:
- `python3 main.py --add-goal "Refactor goal queue" --run-goals`
- `./run_aura.sh --dry-run`

Tip: set `AURA_SKIP_CHDIR=1` to keep the current working directory when running locally or in tests.

## Agent & Loop Overview
- Loop orchestrator: `core/hybrid_loop.py`.
- Model interface: `core/model_adapter.py`.
- Goal queue/archive: `core/goal_queue.py`, `core/goal_archive.py`.
- Agents: see `agents/` (planner, debugger, and others).

The loop selects and coordinates agents per goal. Agent behavior evolves as the loop iterates.

## Build, Test, and Development Commands
- `python3 main.py --help`: show CLI options.
- `./run_aura.sh --help`: wrapper help and usage.
- `python3 -m pytest`: run the test suite.

Note: `package.json` exists but no npm scripts are defined.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- No formatter config is present; match surrounding file style.

## Testing Guidelines
- Tests use `unittest`-style conventions and run with `pytest`.
- Naming: `test_*.py`, `Test*` classes, `test_*` methods.
- Tests live in `tests/` and some core tests in `core/`.

## Commit & Pull Request Guidelines

### Commit messages
- Use the **imperative mood** and **sentence case**: `Fix broken imports in model_adapter` not `fixed broken imports` or `Fixes broken imports`.
- Keep the subject line under **72 characters**.
- Do **not** end the subject line with a period.
- If a commit needs more context, leave a blank line after the subject and add a body (wrap at 72 characters).
- Reference related issues or PRs in the body when relevant: `Closes #42` or `See #69`.

Good examples:
```
Add lazy import guards for requests and numpy

Guards heavy optional deps behind try/except ImportError so the
cli_docs_and_help_contracts CI job can collect tests without a full
requirements.txt install.

Closes #69
```

```
Fix OldCodeNotFoundError when old_code is an empty string
```

### Pull requests
- Fill in the `.github/PULL_REQUEST_TEMPLATE.md` sections: Summary, Changes, Motivation, Testing, and Checklist.
- Include the specific files changed and a one-line description of each change.
- List every test command run and confirm all tests pass.
- Screenshots are only needed for UI changes (rare in this repo).

## Security & Configuration Tips
- `aura.config.json` contains an API key placeholder; do not commit real secrets.
- `.gitignore` excludes `.env`, `memory/*.db`, `memory/*.json`, and `.aura_history`.
