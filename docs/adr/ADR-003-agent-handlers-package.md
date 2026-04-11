# ADR-003: Agent Handlers Package

**Status:** Accepted  
**Date:** 2025-07-11  
**Deciders:** AURA Architecture Team  
**Tags:** agents, dispatch, interface, extensibility

---

## Context

AURA's `agents/registry.py` contains both *adapter classes* (which bridge the
`run(input_data: dict) -> dict` pipeline contract) and the `default_agents()`
factory that wires the entire pipeline together.  This coupling has two
consequences:

1. **Testing a single agent in isolation** requires instantiating (or mocking)
   the full adapter class tree.
2. **Adding a new agent** means editing `registry.py` — a high-churn module —
   increasing the risk of regressions.

Additionally, the orchestrator (`core/orchestrator.py`) calls agents
exclusively through their `run(input_data)` method, but the underlying agent
classes (`PlannerAgent`, `CoderAgent`, etc.) expose domain-specific entry
points (`plan()`, `implement()`, `critique_plan()`, …).  Mapping between the
two interfaces is currently duplicated across the adapter classes inside
`registry.py`.

---

## Decision

Introduce `agents/handlers/` — a thin, **function-based** dispatch layer that:

- Exposes a single public function per module: `handle(task: dict, context: dict) -> dict`
- Wraps the domain-specific agent API behind this uniform surface
- Defers agent imports and construction to call-time (lazy imports)
- Captures and returns all exceptions as `{"error": "<message>"}` rather than
  raising (consistent with orchestrator error-handling conventions)

### Handler Interface Contract

```python
def handle(task: dict, context: dict) -> dict:
    """Standard handler interface for agent dispatch.

    Args:
        task:    Payload describing what the agent should do.
                 Keys are agent-specific; see each handler's module docstring.
        context: Execution context.  At minimum must contain either:
                   - "agent": a pre-initialised agent instance, OR
                   - "brain" + "model": used to construct the agent on demand.
                 Some agents (e.g. ReflectorAgent) need neither — the handler
                 falls back to a module-level shared singleton.

    Returns:
        dict — agent result on success, or {"error": "<message>"} on failure.
        Handlers NEVER raise.
    """
```

### Package Layout

```
agents/
  handlers/
    __init__.py       # exports all handlers + HANDLER_MAP
    planner.py        # wraps PlannerAgent.run()
    coder.py          # wraps CoderAgent.implement()
    critic.py         # wraps CriticAgent.critique_plan/code/validate_mutation()
    debugger.py       # wraps DebuggerAgent.diagnose()
    reflector.py      # wraps ReflectorAgent.run()
    applicator.py     # wraps ApplicatorAgent.apply() / .rollback()
```

### HANDLER_MAP

`agents/handlers/__init__.py` exposes a `HANDLER_MAP` dict for dynamic
dispatch:

```python
from agents.handlers import HANDLER_MAP

result = HANDLER_MAP["plan"](task={"goal": "..."}, context={"brain": b, "model": m})
```

Logical aliases (`"plan"` → `planner.handle`, `"planner"` → `planner.handle`)
ensure that callers using either convention work without changes.

---

## How to Add a New Agent

1. **Create** `agents/handlers/<newagent>.py` with a `handle(task, context) -> dict` function.
2. **Register** it in `agents/handlers/__init__.py`:
   - Import the module.
   - Add entries to `HANDLER_MAP` for all logical names.
   - Add the module name to `__all__`.
3. **Optionally** wire it into `agents/registry.py` via `_AGENT_MODULE_MAP` /
   `default_agents()` so the orchestrator can use it as a pipeline phase.

No changes to any existing handler or adapter are required.

### Minimal new handler template

```python
# agents/handlers/myagent.py
from __future__ import annotations
from core.logging_utils import log_json

def handle(task: dict, context: dict) -> dict:
    try:
        agent = _resolve_agent(context)
        result = agent.do_something(task.get("input", ""))
        return {"result": result}
    except Exception as exc:
        log_json("ERROR", "handler_myagent_failed", details={"error": str(exc)})
        return {"error": str(exc)}

def _resolve_agent(context: dict):
    if a := context.get("agent"):
        return a
    from agents.myagent import MyAgent
    brain, model = context["brain"], context["model"]
    return MyAgent(brain=brain, model=model)
```

---

## Migration Path

`registry.py` can adopt handlers incrementally — **no big-bang rewrite needed**:

1. **Adapter thin-wrapper:** Existing adapter classes (`PlannerAdapter`, etc.)
   can delegate to the handler with one line:
   ```python
   def run(self, input_data: dict) -> dict:
       from agents.handlers.planner import handle
       return handle(task=input_data, context={"agent": self.agent})
   ```
2. **Direct orchestrator wiring (future):** Once all phases have handlers,
   `LoopOrchestrator` can call `HANDLER_MAP[phase](task, ctx)` directly,
   removing the adapter layer entirely.

---

## Alternatives Considered

| Option | Rationale for rejection |
|---|---|
| Subclass-based handlers (class + `handle()` method) | Adds boilerplate with no benefit; function is sufficient since handlers are stateless |
| Extend adapters in registry.py | Increases coupling in an already-large module |
| Protocol / ABC enforcement | Overkill for a 6-module package; docstring contract + tests provide adequate safety |

---

## Consequences

**Positive:**
- New agents can be added in isolation without touching `registry.py`.
- Handlers are trivially unit-testable: inject a mock agent via `context["agent"]`.
- The `HANDLER_MAP` enables config-driven dispatch without `if/elif` chains.
- Deferred imports keep startup cost proportional to what is actually used.

**Negative / Trade-offs:**
- An extra indirection layer exists for the six wrapped agents.
- `context` dict is weakly typed; incorrect keys produce `ValueError` at runtime,
  not at import time.  Mitigated by clear error messages from each `_resolve_agent`.

---

## References

- `agents/registry.py` — existing adapter + factory module
- `core/orchestrator.py` — primary consumer of the agent pipeline
- `core/phase_dispatcher.py` — thin delegation layer between orchestrator and agents
- `agents/handlers/__init__.py` — canonical `HANDLER_MAP`
