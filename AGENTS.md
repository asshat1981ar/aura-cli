# AURA Agent SDK Guide

> This file doubles as the AI-assistant repository guide **and** the developer-facing
> Agent SDK reference.  The original repository guidelines appear after the SDK sections.

---

## 1. Overview

AURA runs goals through a **10-phase pipeline**.  Each phase is owned by a dedicated
*agent* — a Python class that accepts a dict, does work, and returns a dict.  The
`LoopOrchestrator` (`core/orchestrator.py`) drives the loop; the `TypedAgentRegistry`
(`core/mcp_agent_registry.py`) resolves the right agent for each phase at runtime.

```
ingest → plan → critique → synthesize → act → sandbox → verify → reflect → apply → [repeat]
```

Agents are registered in `agents/registry.py` and wired to the orchestrator via
`default_agents()`.  *Skills* (`agents/skills/`) are lightweight helpers called from
within agents — they are not pipeline phases.

---

## 2. Agent Interface

All agents extend `agents.base.Agent`:

```python
# agents/base.py
from abc import ABC, abstractmethod
from typing import Dict

class Agent(ABC):
    name: str                  # unique identifier, matches registry key
    capabilities: list[str] = []  # semantic tags; first entry = primary capability

    @abstractmethod
    def run(self, input_data: Dict) -> Dict:
        """Return a JSON-serialisable phase output dict."""
        raise NotImplementedError
```

### Required / optional keys per phase

| Phase | Required input keys | Expected output keys |
|-------|---------------------|----------------------|
| `ingest` | `goal` | `context`, `memory_hints` |
| `plan` | `goal`, `context` | `steps`, `structured_output` |
| `critique` | `goal`, `plan` | `feedback`, `score` |
| `synthesize` | `goal`, `plan`, `feedback` | `task_bundle` |
| `act` | `goal`, `task_bundle` | `change_set` |
| `sandbox` | `change_set` | `sandbox_result`, `passed` |
| `verify` | `change_set`, `sandbox_result` | `verification_result`, `passed` |
| `reflect` | `goal`, `verification_result` | `reflection`, `skill_updates` |
| `apply` | `change_set` | `applied_files` |

The orchestrator validates output against `core/schema.py` when `strict_schema=True`.

---

## 3. Built-in Agents

| Registry key | Class | Module | Primary capability |
|---|---|---|---|
| `ingest` | `IngestAgent` | `agents.ingest` | context_gathering |
| `plan` | `PlannerAdapter` | `agents.registry` | planning |
| `critique` | `CriticAdapter` | `agents.registry` | critique |
| `synthesize` | `SynthesizerAgent` | `agents.synthesizer` | synthesis |
| `act` | `ActAdapter` | `agents.registry` | code_generation |
| `sandbox` | `SandboxAdapter` | `agents.registry` | sandbox |
| `verify` | `VerifierAgent` | `agents.verifier` | testing |
| `reflect` | `ReflectorAgent` | `agents.reflector` | reflection |
| `python_agent` | `PythonAgent` | `agents.python_agent` | python |
| `typescript_agent` | `TypeScriptAgent` | `agents.typescript_agent` | typescript |
| `debugging` | `DebuggerAgent` | `agents.debugger` | debugging |
| `self_correction` | `SelfCorrectionAgent` | `agents.self_correction_agent` | self_correction |
| `monitoring` | `MonitoringAgent` | `agents.monitoring_agent` | monitoring |
| `notification` | `NotificationAgent` | `agents.notification_agent` | notification |
| `telemetry` | `TelemetryAgent` | `agents.telemetry_agent` | telemetry |
| `mcp_discovery` | `MCPDiscoveryAgent` | `agents.mcp_discovery_agent` | mcp_discovery |
| `code_search` | `CodeSearchAgent` | `agents.code_search_agent` | code_search |
| `investigation` | `InvestigationAgent` | `agents.investigation_agent` | investigation |
| `external_llm` | `ExternalLLMAgent` | `agents.external_llm_agent` | routing |
| `innovation_swarm` | `InnovationSwarm` | `agents.innovation_swarm` | innovation |

Full capability declarations live in `agents/registry.py::FALLBACK_CAPABILITIES`.

---

## 4. Skills System

*Skills* are stateless helpers called from inside agents.  They share the same dict-in/
dict-out contract but are **not** pipeline phases and are not registered in the
orchestrator.

```python
# agents/skills/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from core.logging_utils import log_json

class SkillBase(ABC):
    name: str = "base_skill"

    def __init__(self, brain=None, model=None):
        self.brain = brain
        self.model = model

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Public entry point — wraps _run() with automatic error handling."""
        try:
            return self._run(input_data)
        except Exception as exc:
            log_json("ERROR", f"{self.name}_failed", details={"error": str(exc)})
            return {"error": str(exc), "skill": self.name}

    @abstractmethod
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement skill logic here."""
```

Skills live in `agents/skills/`.  Notable examples:

| Skill | Module | Purpose |
|---|---|---|
| `ASTAnalyzerSkill` | `agents.skills.ast_analyzer` | Parse + query Python ASTs |
| `ComplexityScorerSkill` | `agents.skills.complexity_scorer` | McCabe complexity |
| `DocGeneratorSkill` | `agents.skills.doc_generator` | Auto-docstring generation |
| `GitHistoryAnalyzerSkill` | `agents.skills.git_history_analyzer` | Blame / log queries |
| `IncrementalDifferSkill` | `agents.skills.incremental_differ` | Minimal diff computation |
| `EvalOptimizerSkill` | `agents.skills.eval_optimizer` | Prompt / eval improvement |

---

## 5. Worked Example — Custom `SentimentAnalyzerAgent`

### Step 1 — Create the agent class

```python
# agents/sentiment_analyzer.py
from __future__ import annotations
from typing import Dict
from agents.base import Agent
from core.logging_utils import log_json


class SentimentAnalyzerAgent(Agent):
    name = "sentiment"
    capabilities = ["sentiment_analysis", "text_classification"]

    def __init__(self, model=None):
        self.model = model

    def run(self, input_data: Dict) -> Dict:
        goal = input_data.get("goal", "")
        log_json("INFO", "sentiment_start", goal=goal)

        # --- your logic here ---
        score = self._score(goal)
        label = "positive" if score > 0 else "negative" if score < 0 else "neutral"

        log_json("INFO", "sentiment_done", goal=goal, details={"label": label, "score": score})
        return {"sentiment_label": label, "sentiment_score": score}

    def _score(self, text: str) -> float:
        positive = {"good", "great", "fix", "improve"}
        negative = {"bug", "broken", "fail", "error"}
        words = set(text.lower().split())
        return len(words & positive) - len(words & negative)
```

### Step 2 — Register it

Add an entry to `_AGENT_MODULE_MAP` in `agents/registry.py`:

```python
# agents/registry.py  (inside _AGENT_MODULE_MAP)
"sentiment": ("agents.sentiment_analyzer", "SentimentAnalyzerAgent"),
```

Add a fallback capability entry:

```python
# agents/registry.py  (inside FALLBACK_CAPABILITIES)
"sentiment": ["sentiment_analysis", "text_classification"],
```

### Step 3 — Wire into the orchestrator

Pass the agent via `default_agents()` or inject directly:

```python
from agents.sentiment_analyzer import SentimentAnalyzerAgent
from core.orchestrator import LoopOrchestrator

agents = {
    # ... existing agents ...
    "sentiment": SentimentAnalyzerAgent(model=my_model),
}
orchestrator = LoopOrchestrator(agents=agents, brain=brain, model=my_model)
```

Or use capability-based resolution (preferred):

```python
from core.mcp_agent_registry import TypedAgentRegistry

registry = TypedAgentRegistry()
agent = registry.resolve_by_capability("sentiment_analysis")
result = agent.run({"goal": "This PR fixes a broken build"})
```

---

## 6. Handler Package

`agents/handlers/` provides a thin dispatch layer on top of the core agents:

```
agents/handlers/
  planner.py          # low-level:  handle(task, context) -> dict
  planner_handler.py  # high-level: run_planner_phase(context, **kwargs) -> dict
  coder.py / coder_handler.py
  critic.py / critic_handler.py
  debugger.py / debugger_handler.py
  reflector.py / reflector_handler.py
  applicator.py / applicator_handler.py
  __init__.py         # exports HANDLER_MAP and PHASE_MAP
```

**`HANDLER_MAP`** — low-level `handle(task, context) -> dict`:

```python
from agents.handlers import HANDLER_MAP

result = HANDLER_MAP["plan"](task={"goal": "..."}, context={})
```

**`PHASE_MAP`** — high-level `run_<name>_phase(context, **kwargs) -> dict` (preferred in `aura_cli/dispatch.py`):

```python
from agents.handlers import PHASE_MAP

result = PHASE_MAP["planner"](context, goal="Add retry logic")
```

To add a new handler:
1. Create `agents/handlers/<name>.py` exposing `handle(task, context) -> dict`.
2. Create `agents/handlers/<name>_handler.py` exposing `run_<name>_phase(context, **kwargs) -> dict`.
3. Register both in `HANDLER_MAP` and `PHASE_MAP` inside `agents/handlers/__init__.py`.

---

## 7. Telemetry

Import and call `log_json` from any agent or skill:

```python
from core.logging_utils import log_json

log_json(
    level="INFO",            # "INFO" | "WARN" | "ERROR"
    event="my_agent_start",  # short snake_case label
    goal=input_data.get("goal"),
    details={"key": "value"},          # any JSON-serialisable dict
    correlation_id=None,               # auto-filled from context if omitted
)
```

`log_json` outputs a single-line JSON record to stderr, automatically:
- Masks secrets via `core.redaction.mask_secrets`.
- Injects UTC timestamp (`ts`) and correlation ID for distributed tracing.
- Forwards to the telemetry DB when `AURA_TELEMETRY=true`.

Recommended events to emit:
- `<agent>_start` — log the incoming `goal`.
- `<agent>_done` — log output summary (token count, confidence, etc.).
- `<agent>_error` — log exception details (already done by `SkillBase.run()`).

---

## 8. Testing Agents

Use `unittest` / `pytest` with stub inputs.  Do **not** call real LLM APIs in unit tests.

```python
# tests/test_sentiment_analyzer.py
import pytest
from agents.sentiment_analyzer import SentimentAnalyzerAgent


@pytest.fixture
def agent():
    return SentimentAnalyzerAgent(model=None)


def test_positive_goal(agent):
    result = agent.run({"goal": "improve performance and fix bug"})
    assert "sentiment_label" in result
    assert "sentiment_score" in result
    assert result["sentiment_label"] in ("positive", "negative", "neutral")


def test_neutral_goal(agent):
    result = agent.run({"goal": "rename variable"})
    assert result["sentiment_label"] == "neutral"


def test_missing_goal_key(agent):
    # Agents must handle missing keys gracefully
    result = agent.run({})
    assert "sentiment_label" in result
```

**Recommended test patterns:**
- Stub `input_data` with minimum required keys; assert all expected output keys exist.
- Use `monkeypatch` or dependency injection to replace `self.model` with a mock.
- Assert output is JSON-serialisable: `import json; json.dumps(result)`.
- For handler tests, test `HANDLER_MAP["<phase>"](task, context)` directly.

---

---

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
