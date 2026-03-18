# CLAUDE.md — aura-cli

## Project Overview

Autonomous coding agent with a 10-phase orchestration pipeline. Ingests a goal, routes it through planning, coding, testing, critique, and self-healing phases, then applies the result. Built on skill-based dispatch with circuit breakers, a BEADS bridge for structured reasoning, and recursive self-improvement (RSI) loops.

## Quick Start

```bash
# Install runtime + dev deps (ruff installed via `pkg install ruff` on Termux)
pip install fastapi==0.129.0 uvicorn==0.40.0 requests==2.32.3 pydantic==2.12.5 \
    "python-dotenv>=1.0" numpy gitpython rich textblob networkx \
    "pytest>=7.0" "httpx>=0.24" "anyio>=4.0" "mypy>=1.10" "pytest-cov>=5.0" pytest-timeout

# Run CLI
python -m aura_cli.cli_main

# Run API server
uvicorn aura_cli.server:app
```

## Test Commands

```bash
# Fast unit tests (what pre-commit runs)
python3 -m pytest tests/ -x -q --timeout=30 -k "not integration"

# Full test suite
python3 -m pytest tests/ -q

# With coverage
python3 -m pytest tests/ --cov=core --cov=aura_cli --cov-report=term-missing

# Single test file
python3 -m pytest tests/test_orchestrator.py -x -q
```

## Lint / Format / Type-check

```bash
ruff check .                # lint
ruff check . --fix          # lint + autofix
ruff format .               # format
ruff format --check .       # format check only
mypy core/ aura_cli/ --ignore-missing-imports
```

## Architecture

| Directory | Purpose |
|-----------|---------|
| `aura_cli/` | CLI entry point, server, commands, doctor |
| `core/` | Orchestrator, config, circuit breaker, BEADS bridge, goal queue, model adapter, skills |
| `agents/` | Agent implementations (planner, coder, critic, debugger, refactor, etc.) |
| `tools/` | MCP servers and tool integrations |
| `memory/` | Brain, memory store, cache adapters |
| `conductor/` | Track-based project management and RSI evolution |
| `task_queue/` | Task queue for async work |
| `cli/` | Legacy CLI module |
| `tests/` | 1,656+ tests |
| `plans/` | Development plans and tracking docs |

## Key Design Patterns

- **Skill-based dispatch:** `core/skill_dispatcher.py` routes goals to registered skills by type
- **Circuit breaker:** `core/circuit_breaker.py` prevents cascading failures in agent calls
- **Config types:** `core/config_types.py` + `core/config_manager.py` for typed configuration
- **BEADS bridge:** `core/beads_bridge.py` structured reasoning protocol integration
- **10-phase pipeline:** Ingest > Plan > Code > Test > Critique > Refactor > Apply > Observe > Heal > Archive
- **HealthMonitor:** Dispatches health skills and generates remediation goals

## Termux / Android Notes

- Platform: Termux on Android (aarch64), Python 3.13
- `ruff` must be installed via `pkg install ruff` (no pip wheel for aarch64)
- Editable install (`pip install -e .`) requires fixing maturin linkage; install deps directly instead
- `pyproject.toml` uses `setuptools.build_meta` backend with explicit package discovery

## Config Files

- `pyproject.toml` — build config, deps, pytest/ruff/mypy settings (single source of truth)
- `.pre-commit-config.yaml` — ruff, mypy, trailing-whitespace, fast-test hooks
- `.env` — runtime environment variables (not committed)
