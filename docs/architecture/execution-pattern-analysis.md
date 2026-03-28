# Execution Pattern Analysis — orchestrator.py vs workflow_engine.py

**Status:** Reference document — no code changes required in this sprint.
**Relates to:** Issue #331 (consolidate execution patterns)
**Date:** 2026-03-28

---

## Overview

Two modules manage goal-driven execution loops in AURA:

| Module | Class | Primary concern |
|--------|-------|-----------------|
| `core/orchestrator.py` | `LoopOrchestrator` | Single-goal, multi-phase autonomous coding loop |
| `core/workflow_engine.py` | `WorkflowEngine` / `AgenticLoop` | Named multi-step workflows and goal-driven cycle loops |

Both were developed independently and share several structural patterns. This document identifies the overlap and proposes a future `ExecutionFramework` base class.

---

## Common Patterns Identified

### 1. Phase/Step execution with retry

**`LoopOrchestrator`** (`_run_act_loop`, `_run_sandbox_loop`):
- `while attempt < max_attempts` loop
- Exponential-style sleep between retries (`time.sleep(min(2 ** (act_attempt - 2), 16))`)
- Failure classification → routing decision (act / plan / skip)
- Re-runs code generation with accumulated `fix_hints`

**`WorkflowEngine`** (`_execute_step`):
- `for attempt in range(max(1, step.retry.max_attempts))` loop
- `RetryPolicy.sleep_for(attempt)` with configurable backoff cap
- Error captured as `last_error`; on exhaustion returns `StepResult(status="failed")`
- No failure-type routing — all retries are homogeneous

**Overlap:** Both implement a bounded retry loop with backoff. The retry policy dataclass (`RetryPolicy`) in `workflow_engine.py` is already the cleaner form and could be reused.

---

### 2. State accumulation dict

**`LoopOrchestrator`:** `phase_outputs: Dict` — built up across all phases in a cycle; passed into every helper method; persisted to memory at cycle end.

**`WorkflowEngine`:** `step_outputs: Dict[str, Dict]` — accumulated across steps; used by `_wire_inputs` to resolve `inputs_from` references between steps.

**Overlap:** Both use a mutable accumulator dict that is threaded through the entire execution. The wiring pattern in `_wire_inputs` (resolving `"step_name.key"` references) is more explicit than the ad-hoc key lookups in `phase_outputs`.

---

### 3. Execution context / project root

**`LoopOrchestrator`:** `self.project_root: Path` passed into every phase input dict as `"project_root": str(self.project_root)`.

**`WorkflowEngine`:** `initial_inputs: Dict` passed to every step (including `project_root` by convention).

**Overlap:** Both propagate a context dict to every unit of work. `_wire_inputs` formalises this as `{**initial_inputs, **step.static_inputs, **resolved_wiring}`.

---

### 4. SQLite / memory persistence

**`LoopOrchestrator`:** Calls `memory_controller.persistent_store.append_log(entry)` at cycle end; also stores structured summaries in tiered memory.

**`WorkflowEngine`:** Uses a dedicated SQLite journal (`workflow_engine.db`) via `_open_db()` / `_db()` context manager; writes execution rows on status change.

**Overlap:** Both serialize execution state to SQLite. They use different schemas and connection helpers; unifying them would require a shared persistence layer.

---

### 5. Lifecycle notifications / hooks

**`LoopOrchestrator`:** `_notify_ui(method_name, *args)` broadcasts to `_ui_callbacks`; `hook_engine.run_pre_hooks` / `run_post_hooks` enforces shell-level hooks.

**`WorkflowEngine`:** No hook/callback mechanism — step transitions are fire-and-forget.

**Overlap:** Partial. `WorkflowEngine` would benefit from pre/post step hooks if it needs audit trails or blocking pre-conditions.

---

### 6. Early-exit / stop conditions

**`LoopOrchestrator`:** `Policy.should_stop(entry)` checked after each cycle; also `strict_schema` exits, BEADS gate blocks, and human-gate blocks.

**`WorkflowEngine._run_agentic_loop`:** `stop_reason` field on cycle result; `policy.should_stop()` called after each `run_cycle` result.

**Overlap:** Both call the same `Policy` interface. The `AgenticLoop` in `workflow_engine.py` actually delegates directly to `LoopOrchestrator.run_cycle()`, making it a wrapper rather than a parallel implementation.

---

## Proposed `ExecutionFramework` Base Class Interface

The following interface would extract the shared concerns. This is a **proposal for a future sprint** — do not implement now.

```python
# core/execution_framework.py  (PROPOSED — not yet implemented)

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetryPolicy:
    """Unified retry policy (currently duplicated in workflow_engine.py)."""
    max_attempts: int = 3
    backoff_base: float = 0.5
    max_backoff: float = 30.0

    def sleep_for(self, attempt: int) -> float:
        import time
        return min(self.backoff_base * (2 ** attempt), self.max_backoff)


@dataclass
class ExecutionState:
    """Accumulated outputs threaded through all steps/phases."""
    outputs: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    stop_reason: Optional[str] = None

    def record(self, key: str, value: Any) -> None:
        self.outputs[key] = value


class ExecutionFramework(ABC):
    """Shared base for LoopOrchestrator and WorkflowEngine.

    Responsibilities delegated to subclasses:
    - Unit of work definition (phase vs workflow step)
    - Failure classification and routing
    - Persistence (memory store vs SQLite journal)
    - Lifecycle notifications (UI callbacks vs none)
    """

    @abstractmethod
    def execute_unit(self, name: str, input_data: Dict) -> Dict:
        """Execute a single unit of work (phase or step) and return its output."""
        ...

    @abstractmethod
    def classify_failure(self, error: str, output: Dict) -> str:
        """Return a routing decision string for a failed unit."""
        ...

    @abstractmethod
    def persist_state(self, state: ExecutionState) -> None:
        """Persist the accumulated execution state."""
        ...

    def run_with_retry(
        self,
        name: str,
        input_data: Dict,
        policy: RetryPolicy,
        state: ExecutionState,
    ) -> Dict:
        """Execute *name* with retry/backoff, updating *state* on each attempt."""
        import time

        last_output: Dict = {}
        for attempt in range(max(1, policy.max_attempts)):
            last_output = self.execute_unit(name, input_data)
            if not last_output.get("error"):
                return last_output
            if attempt < policy.max_attempts - 1:
                time.sleep(policy.sleep_for(attempt))
        return last_output
```

---

## Migration Plan (Future Sprint)

**Prerequisite:** `PhaseDispatcher` extraction (Issue #330, this sprint) must be merged first.

| Step | Action | Effort |
|------|--------|--------|
| S1 | Move `RetryPolicy` from `workflow_engine.py` to `core/execution_framework.py` | XS |
| S2 | Move `ExecutionState` concept (replaces `phase_outputs` dict pattern) | S |
| S3 | Extract `ExecutionFramework` ABC; make `WorkflowEngine` inherit | M |
| S4 | Make `LoopOrchestrator` conform to the ABC interface | L — high regression risk |
| S5 | Unify SQLite persistence behind a shared journal helper | M |
| S6 | Port `WorkflowEngine` hook support using `HookEngine` | S |

**Risk notes:**
- Step S4 is the highest-risk change. `LoopOrchestrator` has deep state (BEADS gate, capability manager, confidence router, CASPA-W) that doesn't map cleanly to a generic framework. Consider keeping `LoopOrchestrator` as a standalone class that *uses* shared utilities rather than inheriting from the base.
- `AgenticLoop` in `workflow_engine.py` already delegates to `LoopOrchestrator.run_cycle()` — this is the correct layering. The base class should not attempt to re-unify them at the cycle level.

---

## Conclusion

The two modules share retry, state accumulation, and policy evaluation patterns. A full unification is feasible but carries meaningful regression risk in `LoopOrchestrator`. The recommended approach is:

1. **This sprint:** Extract `PhaseDispatcher` (Issue #330) — done.
2. **Next sprint:** Extract shared `RetryPolicy` and `ExecutionState` as standalone dataclasses (Steps S1–S2 above).
3. **Later sprint:** Introduce `ExecutionFramework` ABC if a third execution-loop consumer appears.
