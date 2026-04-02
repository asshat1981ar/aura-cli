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
- `python3 main.py goal add "Refactor goal queue" --run`
- `python3 main.py goal once "Summarize repo" --dry-run`
- `./run_aura.sh run --dry-run`

Tip: set `AURA_SKIP_CHDIR=1` to keep the current working directory when running locally or in tests.

## Agent & Loop Overview
- Canonical loop orchestrator: `core/orchestrator.py` (`LoopOrchestrator`).
- Deprecated legacy loop: `core/hybrid_loop.py` (`HybridClosedLoop`).
- Model interface: `core/model_adapter.py`.
- Goal queue/archive: `core/goal_queue.py`, `core/goal_archive.py`.
- Agents: see `agents/` (planner, debugger, and others).

The loop selects and coordinates agents per goal. Agent behavior evolves as the loop iterates.

## Async Orchestration & Typed Registry (V2)
AURA has migrated to a high-performance asynchronous orchestration engine and a modern typed registry.

- **Async Pipeline:** Core phases in `LoopOrchestrator` now support non-blocking execution via `_dispatch_task`.
- **Typed Registry:** Agents are now defined as `AgentSpec` objects in `core/mcp_agent_registry.py`, allowing for capability-based resolution.
- **MCP Integration:** The registry autonomously discovers and provisions MCP-backed agents from configured servers.
- **Observability:** Shadow mode and detailed latency/retry logging are built into the transport layer.

Feature flags (enabled by default):
- `AURA_ENABLE_NEW_ORCHESTRATOR=true`
- `AURA_ENABLE_MCP_REGISTRY=true`

Emergency bypass: `AURA_FORCE_LEGACY_ORCHESTRATOR=true`

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
- Commit messages follow imperative, sentence-case summaries (see `git log`).
- PRs should include a clear description, rationale, and test results.
- Screenshots are only needed for UI changes (rare in this repo).

## Agent SDK Meta-Controller (Issue #378)
The Agent SDK meta-controller provides Claude-powered orchestration with production resilience:

- **Controller:** `core/agent_sdk/controller.py` - Main entry point (`AuraController`)
- **Tools:** `core/agent_sdk/tool_registry.py` - MCP tool wrappers for AURA subsystems
- **Resilience:** `core/agent_sdk/resilience.py` - Circuit breakers, retries, health monitoring
- **Config:** `core/agent_sdk/config.py` - AgentSDKConfig with env overrides

### Resilience Patterns (Production Hardening)
The resilience module provides enterprise-grade fault tolerance:

- **Circuit Breaker:** Prevents cascading failures by rejecting requests to failing MCP servers
  - States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)
  - Configurable thresholds and recovery timeouts
  
- **Retry with Exponential Backoff:** Automatic retry for transient failures
  - Configurable max attempts, base delay, jitter
  - Retryable exception filtering
  
- **Health Monitoring:** Background health checks for all MCP servers
  - Async health check loop with configurable intervals
  - Tracks response times and consecutive failures
  
- **Resilient MCP Client:** Combines all patterns for robust tool invocation
  - Usage: `ResilientMCPClient(health_monitor=monitor).invoke(server, tool, args)`

### Health Check API
```python
from core.agent_sdk.resilience import get_health_monitor

monitor = get_health_monitor()
monitor.register_server("dev_tools", "http://localhost:8001")
await monitor.start()  # Background health checks

# Check server health
health = monitor.get_health("dev_tools")
print(health.healthy, health.response_time_ms)
```

## Security & Configuration Tips
- `aura.config.json` contains an API key placeholder; do not commit real secrets.
- `.gitignore` excludes `.env`, `memory/*.db`, `memory/*.json`, and `.aura_history`.
- MCP servers should use authentication in production (not yet implemented).
