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
- Loop orchestrator: `core/orchestrator.py` (`LoopOrchestrator`).
- Legacy wrapper: `core/hybrid_loop.py` (`HybridClosedLoop`) — prefer `LoopOrchestrator` directly.
- Model interface: `core/model_adapter.py`.
- Goal queue/archive: `core/goal_queue.py`, `core/goal_archive.py`.
- Agents: see `agents/` (planner, debugger, and others).

**Orchestration pipeline** — each `run_cycle()` executes these phases in order:
1. `ingest` — gathers project context and memory hints
2. `plan` — `PlannerAgent` produces a list of steps
3. `critique` — `CriticAgent` flags issues in the plan
4. `synthesize` — `SynthesizerAgent` builds a `task_bundle` from plan + critique
5. `act` — `CoderAgent` generates code; on schema failure, `DebuggerAgent` retries once
6. `verify` — `VerifierAgent` checks the change set
7. `reflect` — `ReflectorAgent` records a cycle summary to `MemoryStore`

**Agent registry** (`agents/registry.py::default_agents()`): wires agents into the orchestrator. Agents are wrapped in adapters (`PlannerAdapter`, `CriticAdapter`, `ActAdapter`) to normalize I/O.

## Build, Test, and Development Commands

```bash
# Run all tests
python3 -m pytest

# Run a single test file
python3 -m pytest tests/test_file_tools.py

# Start the CLI
python3 main.py --help

# Add a goal and run it non-interactively
python3 main.py --add-goal "Fix the goal queue" --run-goals

# Run a one-off goal (bypasses queue)
python3 main.py --goal "Refactor core/model_adapter.py"

# Dry run (no file writes, no memory writes)
./run_aura.sh --dry-run

# Bootstrap config
python3 main.py --bootstrap

# Start the HTTP API server (FastAPI on port 8001)
uvicorn aura_cli.server:app --port 8001

# Start the MCP Skills Server (all 33 skills as HTTP tools, port 8002)
uvicorn tools.aura_mcp_skills_server:app --port 8002

# Regenerate the CLI reference doc after changing CLI commands or help text
python3 scripts/generate_cli_reference.py
# Update tests/snapshots/ if output changes intentionally, then verify:
python3 scripts/generate_cli_reference.py --check
```

Note: `package.json` exists but no npm scripts are defined.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- No formatter config is present; match surrounding file style.

## Testing Guidelines
- Tests use `unittest`-style conventions and run with `pytest`.
- Naming: `test_*.py`, `Test*` classes, `test_*` methods.
- Tests live in `tests/` and some core tests in `core/`.
- Set `AURA_SKIP_CHDIR=1` in tests to prevent `cli_main.main()` from calling `os.chdir()`.
- `core/test_goal_queue.py` is outdated (tests old SQLite-backed queue); CI ignores it.

## Key Conventions

**Config resolution** (`core/config_manager.py`):
Priority: runtime overrides > `AURA_*` env vars > `aura.config.json` > defaults. All config keys have an `AURA_<KEY>` env var equivalent (e.g., `AURA_DRY_RUN=1`).

**CoderAgent file targeting**:
The first line of a generated code block must be `# AURA_TARGET: path/to/file.py`. If absent, `ActAdapter` scores candidate files by keyword overlap with the task and falls back to generating a new file in `core/`.

**Code application** (`core/file_tools.py::_safe_apply_change()`):
Uses `{file_path, old_code, new_code, overwrite_file}`. When `old_code` is empty and `overwrite_file=True`, the file is overwritten. Raises `OldCodeNotFoundError` if `old_code` is not found.

**Logging**:
All log output uses `core/logging_utils.log_json(level, event_name, **kwargs)` which emits structured JSON to stdout. Never use bare `print()` or standard `logging` calls in production paths.

**Schema validation**:
`core/schema.py::validate_phase_output(phase_name, output)` validates each phase output. With `strict_schema=True` the cycle aborts on validation failure; default is lenient (logs error, continues).

## Memory Stores

| Store | Location | Contents |
|-------|----------|----------|
| `Brain` | `memory/brain_v2.db` (SQLite) | General memories, weaknesses, vector embeddings, response cache |
| `VectorStore` | `Brain`'s `vector_store_data` table | Semantic retrieval |
| `MemoryStore` | `memory/store/` (files) | Per-cycle summaries written by `ReflectorAgent` |
| `GoalQueue` | `memory/goal_queue_v2.json` | Pending and in-progress goals |

## Environment Variables Reference

All AURA variables are prefixed `AURA_`:

| Variable | Description |
|----------|-------------|
| `AURA_API_KEY` | OpenRouter API key |
| `AURA_DRY_RUN=1` | No file or memory writes |
| `AURA_SKIP_CHDIR=1` | Don't call `os.chdir()` (required in tests) |
| `AURA_STRICT_SCHEMA=1` | Abort cycle on schema validation failure |
| `AURA_MODEL_ROUTING_<SUBKEY>` | Override individual model routing entries |
| `AGENT_API_ENABLE_RUN=1` | Enable `/run` endpoint on HTTP API |
| `AGENT_API_TOKEN` | Auth token for HTTP API (port 8001) |
| `MCP_API_TOKEN` | Auth token for MCP Skills Server (port 8002) |
| `GEMINI_CLI_PATH` | Path to `gemini` binary (model fallback) |

## Commit & Pull Request Guidelines
- Commit messages follow imperative, sentence-case summaries (see `git log`).
- PRs should include a clear description, rationale, and test results.
- Screenshots are only needed for UI changes (rare in this repo).

## Security & Configuration Tips
- `aura.config.json` contains an API key placeholder; do not commit real secrets.
- `.gitignore` excludes `.env`, `memory/*.db`, `memory/*.json`, and `.aura_history`.
