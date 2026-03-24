# AURA CLI — Copilot Onboarding Guide

AURA is an autonomous AI development loop. Goals enter a queue and are processed through a fixed multi-agent pipeline each cycle. This file is the top-level Copilot guide for the repository.

## Quick Copilot setup

Prefer the repo-local setup assets instead of copying ad hoc JSON by hand:

- MCP example: `.vscode/mcp.json.example`
- MCP config helper: `bash scripts/configure_copilot_mcp.sh`
- Repo LSP config: `.github/lsp.json`
- Focused instruction shards: `.github/instructions/copilot/`

Recommended local startup:

```bash
uvicorn aura_cli.server:app --port 8001
uvicorn tools.aura_mcp_skills_server:app --port 8002
uvicorn tools.github_copilot_mcp:app --port 8007
bash scripts/configure_copilot_mcp.sh
```

Then launch `copilot` and verify `/mcp`, `/lsp`, and `/instructions`.

## Commands

```bash
# Run all tests
python3 -m pytest

# Run a single test file
python3 -m pytest tests/test_file_tools.py

# Run a single test
python3 -m pytest tests/test_file_tools.py::TestClassName::test_method_name

# Start the CLI
python3 main.py --help
aura help

# Add a goal and run it non-interactively
python3 main.py goal add "Fix the goal queue" --run

# Run a one-off goal (bypasses queue)
python3 main.py goal once "Refactor core/model_adapter.py"

# Dry run (no file writes, no memory writes)
./run_aura.sh --dry-run

# Bootstrap config
python3 main.py --bootstrap

# Start the HTTP API server (FastAPI on port 8001)
uvicorn aura_cli.server:app --port 8001

# Start the MCP Skills Server (all registered skills as HTTP tools, port 8002)
uvicorn tools.aura_mcp_skills_server:app --port 8002
# Or directly:
python3 tools/aura_mcp_skills_server.py

# Start the GitHub Copilot MCP server (port 8007 by config default)
uvicorn tools.github_copilot_mcp:app --port 8007
```

## Architecture

**Runtime initialization** (`aura_cli/cli_main.py::create_runtime()`):
Constructs all shared objects — `GoalQueue`, `ModelAdapter`, `Brain`, `VectorStore`, `RouterAgent`, `DebuggerAgent`, `PlannerAgent`, `LoopOrchestrator`, `GitTools` — and returns them as a dict.

**Orchestration pipeline** (`core/orchestrator.py::LoopOrchestrator`):
Each `run_cycle()` call executes the current canonical pipeline in order, with schema validation after phase boundaries:

1. `ingest` — gathers project context and memory hints
2. `skill dispatch` — adaptive static-analysis and capability routing
3. `plan` — `PlannerAgent` produces a list of steps
4. `critique` — `CriticAgent` flags issues in the plan
5. `synthesize` — `SynthesizerAgent` builds a `task_bundle` from plan + critique
6. `act` — `CoderAgent` generates code; on schema failure, `DebuggerAgent` retries once
7. `sandbox` — generated code is exercised in isolation before apply
8. `apply` — validated file changes are written atomically
9. `verify` — `VerifierAgent` checks the applied change set
10. `reflect` — `ReflectorAgent` records a cycle summary to `MemoryStore`

`HybridClosedLoop` (`core/hybrid_loop.py`) is a **legacy wrapper** around `LoopOrchestrator`; prefer `LoopOrchestrator` directly.

**Agent registry** (`agents/registry.py::default_agents()`):
Wires agents into the orchestrator. Agents are wrapped in adapters (`PlannerAdapter`, `CriticAdapter`, `ActAdapter`) to normalize I/O.

**Code application** (`core/file_tools.py::_safe_apply_change()`):
Applies changes using `{file_path, old_code, new_code, overwrite_file}`. When `old_code` is empty and `overwrite_file=True`, the file is overwritten. Raises `OldCodeNotFoundError` if `old_code` is not found.

**Model layer** (`core/model_adapter.py`):
- Default model: `google/gemini-2.0-flash-exp:free` via OpenRouter (`api_key` in `aura.config.json`)
- Falls back to the `gemini` CLI binary if `GEMINI_CLI_PATH` is set and executable
- Prompt-response cache stored in `memory/brain_v2.db` (`response_cache` table, 1hr TTL)
- `RouterAgent` does EMA-ranked model routing; attached via `model_adapter.set_router()`

## Memory

| Store | Location | Contents |
|-------|----------|----------|
| `Brain` | `memory/brain_v2.db` (SQLite) | General memories, weaknesses, vector embeddings, response cache |
| `VectorStore` | `Brain`'s `vector_store_data` table | Semantic retrieval |
| `MemoryStore` | `memory/store/` (files) | Per-cycle summaries written by `ReflectorAgent` |
| `GoalQueue` | `memory/goal_queue.json` | Pending and in-progress goals |

## HTTP API

**File:** `aura_cli/server.py` — FastAPI on port **8001**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools` | GET | List available tools |
| `/call` | POST | Invoke a tool |
| `/health` | GET | Health check |
| `/metrics` | GET | Runtime metrics |
| `/run` | POST | Trigger a goal cycle (requires `AGENT_API_ENABLE_RUN=1`) |

SSE streaming is supported on `/run`. Protect all endpoints with `AGENT_API_TOKEN`.

## MCP Skills Server

**File:** `tools/aura_mcp_skills_server.py` — FastAPI on port **8002**

Exposes all registered skills as MCP-compatible HTTP tools.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools` | GET | List all skill tools |
| `/call` | POST | Invoke a skill by name |
| `/skill/{name}` | GET | Describe a single skill |
| `/health` | GET | Health check |

Auth: set `MCP_API_TOKEN` env var.

**Add to Copilot MCP config** using `.vscode/mcp.json.example` or generate a local config with `bash scripts/configure_copilot_mcp.sh`.

Minimal example:

```json
{
  "mcpServers": {
    "aura-skills": {
      "type": "http",
      "url": "http://localhost:8002"
    }
  }
}
```

## Key Conventions

**Config resolution** (`core/config_manager.py`):
Priority: runtime overrides > `AURA_*` env vars > `aura.config.json` > defaults. All config keys have an `AURA_<KEY>` env var equivalent (e.g., `AURA_DRY_RUN=1`).

**CoderAgent file targeting**:
The first line of a generated code block must be `# AURA_TARGET: path/to/file.py`. If absent, `ActAdapter` scores candidate files by keyword overlap with the task and falls back to generating a new file in `core/`.

**Logging**:
All log output uses `core/logging_utils.log_json(level, event_name, **kwargs)` which emits structured JSON to stdout. Never use bare `print()` or standard `logging` calls in production paths.

**Schema validation**:
`core/schema.py::validate_phase_output(phase_name, output)` validates each phase output. With `strict_schema=True` the cycle aborts on validation failure; default is lenient (logs error, continues).

**Tests and `AURA_SKIP_CHDIR`**:
Set `AURA_SKIP_CHDIR=1` in tests to prevent `cli_main.main()` from calling `os.chdir()`. Tests live in `tests/` and some in `core/`.

**`core/test_goal_queue.py` is outdated** — it tests the old SQLite-backed `GoalQueue`; CI ignores it. The current implementation uses JSON.

**Never commit secrets**: `aura.config.json` holds `api_key`; the real value must come from the `AURA_API_KEY` env var. `memory/*.db`, `memory/*.json`, and `.aura_history` are gitignored.

## Model Routing Config

Add to `aura.config.json` under `"model_routing"`:

```json
"model_routing": {
  "code_generation": "google/gemini-2.0-flash-exp:free",
  "planning":        "google/gemini-2.0-flash-exp:free",
  "analysis":        "google/gemini-2.0-flash-exp:free",
  "critique":        "google/gemini-2.0-flash-exp:free",
  "embedding":       "openai/text-embedding-3-small",
  "fast":            "google/gemini-2.0-flash-exp:free",
  "quality":         "google/gemini-2.5-pro"
}
```

Or via env vars: `AURA_MODEL_ROUTING_CODE_GENERATION=...`, `AURA_MODEL_ROUTING_PLANNING=...`, etc.

`RouterAgent` (`agents/router.py`) tracks per-model EMA scores and routes requests to the best model for each task type. Attach it via `model_adapter.set_router(router)`.

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

## Skills System

Registered skill modules live in `agents/skills/`. Each has `run(input_data: dict) -> dict` and **never raises** — errors are returned as `{"error": "..."}`.

**Invoke any skill:**

```python
from agents.skills.registry import all_skills
skills = all_skills()
result = skills["security_scanner"].run({"project_root": "."})
```

**Common skills:**

| # | Name | Input keys | Primary output keys |
|---|------|-----------|-------------------|
| 1 | `dependency_analyzer` | `project_root` | `packages`, `conflicts`, `vulnerabilities` |
| 2 | `architecture_validator` | `project_root` | `circular_deps`, `coupling_score` |
| 3 | `complexity_scorer` | `code`/`project_root` | `functions`, `high_risk_count` |
| 4 | `test_coverage_analyzer` | `project_root` | `coverage_pct`, `meets_target` |
| 5 | `doc_generator` | `code`/`project_root` | `generated_docstrings`, `undocumented_count` |
| 6 | `performance_profiler` | `code` | `hotspots`, `antipatterns` |
| 7 | `refactoring_advisor` | `code`/`project_root` | `suggestions`, `smell_count` |
| 8 | `schema_validator` | `schema`, `instance`, `code` | `valid`, `errors`, `pydantic_models` |
| 9 | `security_scanner` | `code`/`project_root` | `findings`, `critical_count` |
| 10 | `type_checker` | `project_root`/`file_path` | `type_errors`, `annotation_coverage_pct` |
| 11 | `linter_enforcer` | `project_root`/`file_path` | `violations`, `naming_violations` |
| 12 | `incremental_differ` | `old_code`, `new_code` | `diff_summary`, `added_symbols`, `removed_symbols` |
| 13 | `tech_debt_quantifier` | `project_root` | `debt_score`, `debt_items` |
| 14 | `api_contract_validator` | `code`/`project_root` | `endpoints`, `breaking_changes` |
| 15 | `generation_quality_checker` | `task`, `generated_code` | `quality_score`, `issues`, `intent_match_score` |
| 16 | `git_history_analyzer` | `project_root` | `hotspot_files`, `patterns` |
| 17 | `skill_composer` | `goal` | `workflow`, `goal_category` |
| 18 | `error_pattern_matcher` | `current_error`, `error_history?` | `matched_pattern`, `suggested_fix`, `fix_steps` |
| 19 | `code_clone_detector` | `project_root` | `exact_clones`, `near_duplicates` |
| 20 | `adaptive_strategy_selector` | `goal`, `record_result?` | `recommended_strategy`, `confidence` |
| 21 | `web_fetcher` | `url` or `query` | `text`, `title`, `source`, `truncated` |
| 22 | `symbol_indexer` | `project_root` | `symbols`, `symbol_count`, `name_index`, `import_graph` |
| 23 | `multi_file_editor` | `goal`, `project_root?`, `symbol_map?` | `change_plan`, `affected_count`, `warnings` |

**Adding a new skill:** Create `agents/skills/your_skill.py` extending `SkillBase`, set `name`, implement `_run()`. Register in `agents/skills/registry.py`.

**Run skill tests:** `python3 -m pytest tests/test_skills.py -v`
