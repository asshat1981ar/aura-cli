# AURA Self-Healing Infrastructure: Master Development Plan

**Date:** 2026-03-05
**Status:** Authoritative — supersedes individual plan files
**Scope:** Close all open plan gaps and deliver a complete self-healing feedback loop

---

## 1. Unified Vision

All six plans converge on a single architectural idea: **AURA must observe itself, detect failure, and recover without human intervention.**

The closed feedback loop looks like this:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                     AURA Run Cycle                          │
  │                                                             │
  │  [Ingest] ─► [Skill Dispatch] ─► [Plan] ─► [Critique]      │
  │       │            │                                        │
  │       │       lint_skill                                    │
  │       │       test_and_observe_skill      ◄──┐              │
  │       │                                      │              │
  │  [Act] ─► [Sandbox] ─► [Apply] ─► [Verify] ─┘              │
  │                                      │                      │
  │                              HealthMonitor                  │
  │                                      │                      │
  │                          handle_failure (retry/CB)          │
  │                                      │                      │
  │                          [Reflect] ─► [EvolutionLoop]       │
  └─────────────────────────────────────────────────────────────┘
```

The five subsystems that enable this loop:

| Subsystem | Plan Source | Current State |
|---|---|---|
| `HealthMonitor` | add-more-health-checks + self-healing-infra | **Missing** — `test_health_monitor.py` was deleted |
| Retry + Circuit Breaker in `_route_failure` | improve-self-healing | **Stub** — `_consecutive_fails` tracked but no logic |
| `LintSkill` (SkillBase) | develop-a-new-lint-skill | **Partial** — script exists, not registered as SkillBase |
| `TestAndObserveSkill` (full parsers) | implement-test_and_observe | **Partial** — only Python traceback parser, 3 missing |
| Skill dispatcher wiring for new skills | add-new-skills | **Missing** — neither skill appears in `SKILL_MAP` |

---

## 2. Dependency Order

The plans are not independent. Implementation must follow this dependency chain:

```
Phase A: TestAndObserveSkill (complete it)
    └─► Phase B: LintSkill (convert to SkillBase, register)
            └─► Phase C: HealthMonitor (build, wire into orchestrator)
                    └─► Phase D: Retry + Circuit Breaker (in _route_failure)
                            └─► Phase E: Skill Dispatcher Wiring (SKILL_MAP updates)
                                    └─► Phase F: Integration Tests
```

Phases A and B can be parallelized. Phase C depends on both. Phases D–F are strictly sequential.

---

## 3. Phase A — Complete TestAndObserveSkill

**File:** `agents/skills/test_and_observe.py`
**Status:** Exists. Missing 3 of 4 planned parser types.

### 3.1 Gap Analysis

The plan called for parsers: Python traceback, Node.js stack trace, pytest failures, flake8/linter output. Only `parse_python_traceback` exists.

### 3.2 Parser: pytest failures

```python
def parse_pytest_output(text: str) -> List[Diagnostic]:
    """Extract FAILED lines from pytest -v output."""
    diagnostics = []
    # Match: FAILED path/to/test.py::TestClass::test_method - ExceptionType: msg
    pattern = re.compile(
        r"FAILED\s+([\w/.\-]+\.py)::(\w+)(?:::\w+)?\s+-\s+(.+)"
    )
    for match in pattern.finditer(text):
        file_path, _, message = match.groups()
        # Find line number from following "line N" context if present
        diagnostics.append(Diagnostic(
            severity="error",
            kind="pytest_failure",
            message=message.strip(),
            primary_location=DiagnosticLocation(file=file_path, line=0),
            suggested_next_commands=[["pytest", file_path, "-v", "--tb=short"]],
        ))
    return diagnostics
```

### 3.3 Parser: Node.js stack traces

```python
def parse_node_stacktrace(text: str) -> List[Diagnostic]:
    """Parse Node.js error output: 'at <fn> (<file>:<line>:<col>)'."""
    diagnostics = []
    # Match the error message line (before the stack)
    err_match = re.search(r"^(\w+Error|Error): (.+)$", text, re.MULTILINE)
    if not err_match:
        return diagnostics
    error_message = f"{err_match.group(1)}: {err_match.group(2)}"
    # Find the innermost (first) non-Node-internal frame
    frame_pattern = re.compile(
        r"at .+? \((?!node:)(.+?):(\d+):(\d+)\)"
    )
    for frame in frame_pattern.finditer(text):
        file_path, line, col = frame.groups()
        diagnostics.append(Diagnostic(
            severity="error",
            kind="node_stacktrace",
            message=error_message,
            primary_location=DiagnosticLocation(
                file=file_path, line=int(line), col=int(col)
            ),
        ))
        break  # Only innermost frame
    return diagnostics
```

### 3.4 Parser: flake8/lint output

```python
def parse_flake8_output(text: str) -> List[Diagnostic]:
    """Parse flake8 output: 'file.py:line:col: Exxxx message'."""
    diagnostics = []
    pattern = re.compile(r"^(.+\.py):(\d+):(\d+):\s+(E\d+|W\d+|F\d+)\s+(.+)$", re.MULTILINE)
    for match in pattern.finditer(text):
        file_path, line, col, code, message = match.groups()
        severity = "warning" if code.startswith("W") else "error"
        diagnostics.append(Diagnostic(
            severity=severity,
            kind="lint_violation",
            message=f"{code}: {message}",
            primary_location=DiagnosticLocation(
                file=file_path, line=int(line), col=int(col)
            ),
            suggested_next_commands=[["flake8", "--select", code, file_path]],
        ))
    return diagnostics
```

### 3.5 Wire parsers into registry

In `TestAndObserveSkill.__init__`, update `self.parsers`:

```python
self.parsers = {
    "python_traceback": parse_python_traceback,
    "pytest_failure":   parse_pytest_output,
    "node_stacktrace":  parse_node_stacktrace,
    "lint_violation":   parse_flake8_output,
}
```

### 3.6 Populate `suggested_next_commands`

`parse_python_traceback` currently leaves `suggested_next_commands` empty. Update it:

```python
# After appending diagnostic, add repair hint:
diag.suggested_next_commands = [
    ["python3", "-m", "pytest", file_path, "-v", "--tb=long"],
    ["python3", file_path],
]
```

### 3.7 Tests to add in `tests/test_test_and_observe_skill.py`

- `test_parse_pytest_output_extracts_failed_test()`
- `test_parse_node_stacktrace_extracts_innermost_frame()`
- `test_parse_flake8_output_maps_codes_to_severity()`
- `test_skill_auto_selects_correct_parser_by_content()`
- `test_timeout_kills_process_group()`  — important safety test

---

## 4. Phase B — Convert LintSkill to SkillBase

**File:** `agents/skills/lint.py`
**Status:** Exists as a standalone script. Not a SkillBase. Not in registry.

### 4.1 Current problem

`lint_staged_files()` in `lint.py` is a plain function that calls `sys.exit()` on failure — incompatible with orchestrator pipeline which expects `SkillBase.run()` returning a dict.

### 4.2 Rewrite as SkillBase

```python
"""Lint skill: runs flake8 on a configurable set of files."""
from __future__ import annotations
import subprocess
from typing import Any, Dict, List

from agents.skills.base import SkillBase
from agents.skills.test_and_observe import parse_flake8_output


class LintSkill(SkillBase):
    """Run flake8 over a list of files and return structured diagnostics."""

    name = "lint"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        files: List[str] = input_data.get("files") or self._get_staged_files()
        config: str = input_data.get("config", "")

        if not files:
            return {"status": "success", "violations": [], "files_checked": 0}

        cmd = ["flake8"] + ([f"--config={config}"] if config else []) + files
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
        except FileNotFoundError:
            return {
                "status": "error",
                "error": "flake8 not installed",
                "violations": [],
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "lint timed out", "violations": []}

        violations = parse_flake8_output(proc.stdout + proc.stderr)

        return {
            "status": "success" if proc.returncode == 0 else "violations_found",
            "files_checked": len(files),
            "violation_count": len(violations),
            "violations": [v.__dict__ for v in violations],
            "raw": proc.stdout,
        }

    def _get_staged_files(self) -> List[str]:
        """Fall back to git staged Python files when no explicit list given."""
        try:
            out = subprocess.check_output(
                ["git", "diff", "--name-only", "--cached"],
                timeout=10,
            ).decode()
            return [f for f in out.splitlines() if f.endswith(".py")]
        except Exception:
            return []
```

### 4.3 Register in `agents/skills/registry.py`

Add to `all_skills()`:

```python
from agents.skills.lint import LintSkill

# In the return dict:
"lint": LintSkill(brain=brain, model=model),
```

### 4.4 Tests to create: `tests/test_lint_skill.py`

- `test_lint_skill_clean_file_returns_success()`
- `test_lint_skill_violation_returns_structured_dict()`
- `test_lint_skill_falls_back_to_staged_files()`
- `test_lint_skill_handles_missing_flake8_gracefully()`
- `test_lint_skill_registered_in_all_skills()`

---

## 5. Phase C — Build HealthMonitor

**File:** `core/health_monitor.py`
**Status:** Missing (deleted or never created). Referenced by two plans and `core/orchestrator.py` docstrings.

### 5.1 Design

The `HealthMonitor` is a composable checker. Each check is independent and returns a typed result. The orchestrator calls `run_all()` at the start of every cycle and at each `handle_failure`.

```
HealthMonitor
├── check_brain_db()          → SQLite connectivity + schema version
├── check_memory_controller() → MemoryTier availability
├── check_model_adapter()     → LLM endpoint reachability (fast probe)
├── check_skill_registry()    → All skills load without import error
├── check_goal_queue()        → Queue file readable + valid JSON
└── check_vector_store()      → VectorStore index accessible
```

### 5.2 Implementation

```python
"""System health monitor for AURA.

Runs lightweight checks against each major subsystem and returns
a structured HealthReport. Designed to be called at cycle start
and after failures without adding significant latency.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.logging_utils import log_json


@dataclass
class CheckResult:
    name: str
    ok: bool
    latency_ms: float
    detail: str = ""
    error: str = ""


@dataclass
class HealthReport:
    timestamp: float
    all_ok: bool
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "all_ok": self.all_ok,
            "checks": [c.__dict__ for c in self.checks],
            "failed_count": len(self.failed),
        }


class HealthMonitor:
    """Lightweight health checker for AURA subsystems."""

    def __init__(self, brain=None, model=None, goal_queue=None,
                 memory_controller=None, vector_store=None):
        self._brain = brain
        self._model = model
        self._goal_queue = goal_queue
        self._mc = memory_controller
        self._vs = vector_store
        self._checks: List[Callable[[], CheckResult]] = [
            self.check_brain_db,
            self.check_memory_controller,
            self.check_model_adapter,
            self.check_skill_registry,
            self.check_goal_queue,
        ]
        if vector_store is not None:
            self._checks.append(self.check_vector_store)

    def _timed(self, name: str, fn: Callable) -> CheckResult:
        t0 = time.monotonic()
        try:
            detail = fn()
            return CheckResult(
                name=name,
                ok=True,
                latency_ms=(time.monotonic() - t0) * 1000,
                detail=detail or "",
            )
        except Exception as exc:
            return CheckResult(
                name=name,
                ok=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                error=str(exc),
            )

    def check_brain_db(self) -> CheckResult:
        def _check():
            if self._brain is None:
                raise RuntimeError("Brain not configured")
            self._brain.db.execute("SELECT 1").fetchone()
            return f"schema_v{self._brain.SCHEMA_VERSION}"
        return self._timed("brain_db", _check)

    def check_memory_controller(self) -> CheckResult:
        def _check():
            if self._mc is None:
                raise RuntimeError("MemoryController not configured")
            # Lightweight: just check the store attribute exists
            _ = self._mc.persistent_store
            return "ok"
        return self._timed("memory_controller", _check)

    def check_model_adapter(self) -> CheckResult:
        def _check():
            if self._model is None:
                raise RuntimeError("ModelAdapter not configured")
            # Check the adapter has a callable respond method
            if not callable(getattr(self._model, "respond", None)):
                raise RuntimeError("ModelAdapter missing .respond()")
            provider = getattr(self._model, "provider", "unknown")
            return f"provider={provider}"
        return self._timed("model_adapter", _check)

    def check_skill_registry(self) -> CheckResult:
        def _check():
            from agents.skills.registry import all_skills
            skills = all_skills()
            return f"{len(skills)} skills loaded"
        return self._timed("skill_registry", _check)

    def check_goal_queue(self) -> CheckResult:
        def _check():
            if self._goal_queue is None:
                return "not configured"
            # GoalQueue exposes .queue list
            count = len(getattr(self._goal_queue, "queue", []))
            return f"{count} pending goals"
        return self._timed("goal_queue", _check)

    def check_vector_store(self) -> CheckResult:
        def _check():
            if self._vs is None:
                raise RuntimeError("VectorStore not configured")
            # Check index is accessible
            _ = getattr(self._vs, "index", None)
            return "index accessible"
        return self._timed("vector_store", _check)

    def run_all(self) -> HealthReport:
        results = [check() for check in self._checks]
        report = HealthReport(
            timestamp=time.time(),
            all_ok=all(r.ok for r in results),
            checks=results,
        )
        level = "INFO" if report.all_ok else "WARN"
        log_json(level, "health_check_complete", details=report.as_dict())
        return report
```

### 5.3 Integrate into LoopOrchestrator

In `core/orchestrator.py`, `__init__`:

```python
from core.health_monitor import HealthMonitor

self.health_monitor = HealthMonitor(
    brain=self.brain,
    model=self.model,
    goal_queue=self.goal_queue,
    memory_controller=self.memory_controller,
)
```

In `run_cycle()`, at the top (after ingest, before skill dispatch):

```python
health = self.health_monitor.run_all()
phase_outputs["health"] = health.as_dict()
if not health.all_ok:
    log_json("WARN", "cycle_starting_degraded",
             details={"failed": [c.name for c in health.failed]})
```

### 5.4 Surface in `doctor` command

In `aura_cli/commands.py` `_handle_doctor()`:
- Call `orchestrator.health_monitor.run_all()`
- Print each check with pass/fail status
- Return exit code 1 if any check fails

### 5.5 Tests: `tests/test_health_monitor.py`

This file was deleted — recreate it:

- `test_all_checks_pass_with_valid_mocks()`
- `test_brain_db_check_fails_on_closed_connection()`
- `test_model_adapter_check_fails_when_respond_missing()`
- `test_skill_registry_check_counts_loaded_skills()`
- `test_run_all_returns_failed_list()`
- `test_report_as_dict_structure()`

---

## 6. Phase D — Retry + Circuit Breaker in `_route_failure`

**File:** `core/orchestrator.py`
**Status:** `_consecutive_fails` is tracked. `_route_failure` is documented but circuit-break logic is absent.

### 6.1 Circuit Breaker design

```
States: CLOSED → OPEN → HALF_OPEN → CLOSED
         (normal)  (failing)  (testing)

Thresholds:
  open_threshold:    3 consecutive failures
  reset_timeout_s:  60 seconds (configurable)
  half_open_probes: 1 success closes; 1 failure re-opens
```

### 6.2 CircuitBreaker class (new file: `core/circuit_breaker.py`)

```python
"""Lightweight circuit breaker for orchestrator failure routing."""
from __future__ import annotations
import time
from enum import Enum
from typing import Optional
from core.logging_utils import log_json


class CBState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, open_threshold: int = 3, reset_timeout_s: float = 60.0):
        self.open_threshold = open_threshold
        self.reset_timeout_s = reset_timeout_s
        self._state = CBState.CLOSED
        self._fail_count = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CBState:
        if self._state == CBState.OPEN:
            if time.monotonic() - self._opened_at >= self.reset_timeout_s:
                self._state = CBState.HALF_OPEN
                log_json("INFO", "circuit_breaker_half_open")
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CBState.OPEN

    def record_failure(self) -> None:
        self._fail_count += 1
        if self._fail_count >= self.open_threshold and self._state == CBState.CLOSED:
            self._state = CBState.OPEN
            self._opened_at = time.monotonic()
            log_json("WARN", "circuit_breaker_opened",
                     details={"fail_count": self._fail_count})

    def record_success(self) -> None:
        if self._state == CBState.HALF_OPEN:
            self._state = CBState.CLOSED
            self._fail_count = 0
            log_json("INFO", "circuit_breaker_closed")
        elif self._state == CBState.CLOSED:
            self._fail_count = max(0, self._fail_count - 1)

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "fail_count": self._fail_count,
        }
```

### 6.3 Retry with exponential backoff (new file: `core/retry.py`)

```python
"""Retry utility with exponential backoff and jitter."""
from __future__ import annotations
import time
import random
from typing import Callable, Optional, Type, Tuple
from core.logging_utils import log_json


def retry_with_backoff(
    fn: Callable,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = "operation",
):
    """Call fn up to max_attempts times with exponential backoff.

    Returns the result of the first successful call.
    Raises the last exception if all attempts fail.
    """
    last_exc: Optional[Exception] = None
    delay = base_delay_s

    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            if attempt > 1:
                log_json("INFO", f"{label}_retry_succeeded",
                         details={"attempt": attempt})
            return result
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            wait = delay + (random.uniform(0, 0.5) if jitter else 0)
            log_json("WARN", f"{label}_retry_backoff",
                     details={"attempt": attempt, "wait_s": round(wait, 2),
                               "error": str(exc)})
            time.sleep(wait)
            delay = min(delay * backoff_factor, max_delay_s)

    raise last_exc
```

### 6.4 Wire into `_route_failure` in orchestrator

```python
# In LoopOrchestrator.__init__, add:
from core.circuit_breaker import CircuitBreaker
self._circuit_breaker = CircuitBreaker(
    open_threshold=self.policy.get("cb_open_threshold", 3),
    reset_timeout_s=self.policy.get("cb_reset_timeout_s", 60.0),
)

# Replace/augment existing _route_failure:
def _route_failure(self, phase: str, error: Exception,
                   attempt: int, max_attempts: int) -> str:
    """Route a phase failure to: retry | replan | skip | abort.

    Returns one of: 'retry', 'replan', 'skip', 'abort'.
    """
    self._circuit_breaker.record_failure()
    self._consecutive_fails += 1

    if self._circuit_breaker.is_open:
        log_json("ERROR", "circuit_breaker_abort",
                 details={"phase": phase, "error": str(error)})
        return "abort"

    is_env_error = isinstance(error, (OSError, TimeoutError))
    if is_env_error:
        log_json("WARN", "routing_skip_env_error",
                 details={"phase": phase, "error": str(error)})
        return "skip"

    if attempt < max_attempts:
        return "retry"

    if phase in ("act", "sandbox"):
        return "replan"

    return "skip"
```

### 6.5 Record success path

In `run_cycle`, after successful verify:

```python
self._circuit_breaker.record_success()
self._consecutive_fails = 0
```

### 6.6 Tests: `tests/test_orchestrator_phases.py` additions

- `test_circuit_breaker_opens_after_threshold()`
- `test_circuit_breaker_transitions_half_open_after_timeout()`
- `test_route_failure_returns_abort_when_open()`
- `test_route_failure_returns_skip_for_env_errors()`
- `test_retry_with_backoff_succeeds_on_second_attempt()`
- `test_retry_with_backoff_raises_after_max_attempts()`

---

## 7. Phase E — Skill Dispatcher Wiring

**File:** `core/skill_dispatcher.py`
**Status:** `SKILL_MAP` has no entries for `lint` or `test_and_observe`.

### 7.1 Updates to `SKILL_MAP`

```python
SKILL_MAP: Dict[str, list[str]] = {
    "bug_fix": [
        "symbol_indexer",
        "error_pattern_matcher",
        "test_and_observe",      # ← NEW: run failing tests before planning
        "git_history_analyzer",
        "type_checker",
        "linter_enforcer",
        "lint",                  # ← NEW: catch pre-existing violations
    ],
    "feature": [
        "symbol_indexer",
        "architecture_validator",
        "api_contract_validator",
        "complexity_scorer",
        "dependency_analyzer",
        "lint",                  # ← NEW: baseline quality gate
    ],
    "refactor": [
        "symbol_indexer",
        "complexity_scorer",
        "code_clone_detector",
        "tech_debt_quantifier",
        "refactoring_advisor",
        "lint",                  # ← NEW: verify cleanliness after refactor
        "test_and_observe",      # ← NEW: confirm tests still pass
    ],
    "security": [
        "security_scanner",
        "dependency_analyzer",
        "type_checker",
        "linter_enforcer",
        "architecture_validator",
        "lint",                  # ← NEW
    ],
    "docs": [
        "doc_generator",
        "symbol_indexer",
    ],
    "default": [
        "symbol_indexer",
        "linter_enforcer",
        "lint",                  # ← NEW
    ],
}
```

### 7.2 Keyword hints update

Add `"test"` and `"observe"` to appropriate categories:

```python
_GOAL_TYPE_HINTS: Dict[str, list[str]] = {
    "bug_fix": [..., "test", "observe", "diagnose", "traceback"],
    ...
}
```

---

## 8. Phase F — Integration Tests

### 8.1 End-to-end self-healing scenario

Create `tests/integration/test_self_healing_loop.py`:

```python
"""Integration test: orchestrator recovers from a simulated verify failure."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from core.orchestrator import LoopOrchestrator
from core.policy import Policy


def test_orchestrator_retries_on_verify_fail(tmp_path):
    """
    Arrange: verify agent fails twice, succeeds on third attempt.
    Assert: orchestrator does not abort; cycle completes with success.
    """
    # ... setup agents with verify failing twice then succeeding ...
    # ... assert stop_reason != "ABORT" ...
    # ... assert phase_outputs has retry_count == 2 ...


def test_circuit_breaker_halts_runaway_cycles(tmp_path):
    """
    Arrange: verify always fails.
    Assert: after cb_open_threshold failures, stop_reason == "ABORT".
    """
    # ... assert stop_reason == "ABORT" before max_cycles reached ...


def test_health_monitor_degraded_does_not_block_cycle(tmp_path):
    """
    Arrange: brain_db check fails (simulate DB error).
    Assert: cycle still proceeds; health degradation logged but not fatal.
    """
    # ... assert cycle_result["stop_reason"] not in ("ABORT", "ERROR") ...
    # ... assert phase_outputs["health"]["all_ok"] == False ...
```

### 8.2 Lint + TestAndObserve skill integration

Create `tests/integration/test_skill_pipeline.py`:

```python
def test_lint_skill_feeds_diagnostics_into_planner_context(tmp_path):
    """Skill dispatcher returns lint violations; planner receives them."""
    ...

def test_test_and_observe_pytest_failures_parsed_to_diagnostics():
    """Full skill run with real pytest subprocess; diagnostic extracted."""
    ...
```

---

## 9. Implementation Sequence and Effort

| Phase | Files Created/Modified | Est. Effort | Depends On |
|---|---|---|---|
| A — TestAndObserve complete | `agents/skills/test_and_observe.py` | 1.5h | — |
| A — Tests | `tests/test_test_and_observe_skill.py` | 0.5h | A |
| B — LintSkill as SkillBase | `agents/skills/lint.py` | 1h | A (reuses `parse_flake8_output`) |
| B — Registry entry | `agents/skills/registry.py` | 0.1h | B |
| B — Tests | `tests/test_lint_skill.py` | 0.5h | B |
| C — HealthMonitor | `core/health_monitor.py` (new) | 2h | — |
| C — Orchestrator integration | `core/orchestrator.py` | 0.5h | C |
| C — Doctor command | `aura_cli/commands.py` | 0.5h | C |
| C — Tests | `tests/test_health_monitor.py` (recreate) | 1h | C |
| D — CircuitBreaker | `core/circuit_breaker.py` (new) | 1h | — |
| D — Retry utility | `core/retry.py` (new) | 0.5h | — |
| D — Wire into orchestrator | `core/orchestrator.py` | 1h | C, D |
| D — Tests | `tests/test_orchestrator_phases.py` additions | 1h | D |
| E — SKILL_MAP updates | `core/skill_dispatcher.py` | 0.3h | A, B |
| F — Integration tests | `tests/integration/*.py` | 2h | A–E |
| **Total** | | **~14h** | |

---

## 10. File Change Summary

### New files to create
- `core/health_monitor.py`
- `core/circuit_breaker.py`
- `core/retry.py`
- `tests/test_health_monitor.py`
- `tests/test_lint_skill.py`
- `tests/integration/test_self_healing_loop.py`
- `tests/integration/test_skill_pipeline.py`

### Files to modify
- `agents/skills/test_and_observe.py` — add 3 parsers + populate `suggested_next_commands`
- `agents/skills/lint.py` — rewrite as `LintSkill(SkillBase)`, remove `sys.exit()`
- `agents/skills/registry.py` — add `"lint": LintSkill(...)`
- `core/orchestrator.py` — instantiate `HealthMonitor`, `CircuitBreaker`; wire into cycle
- `core/skill_dispatcher.py` — add `lint` and `test_and_observe` to `SKILL_MAP`
- `aura_cli/commands.py` — surface `HealthMonitor.run_all()` in `doctor`
- `tests/test_test_and_observe_skill.py` — add tests for 3 new parsers
- `tests/test_orchestrator_phases.py` — add circuit breaker + retry tests

### Files to leave unchanged
- `core/evolution_loop.py` — already integrates with planner; self-healing feeds it indirectly
- `memory/brain.py` — health monitor probes it but does not modify it
- `agents/planner.py` — receives richer skill context automatically

---

## 11. Rollback Strategy

Each phase is independently revertable:

- **Phase A/B:** Delete new parsers / revert `lint.py` and remove registry entry.
- **Phase C:** Remove `HealthMonitor` instantiation from orchestrator `__init__`; delete `core/health_monitor.py`.
- **Phase D:** Remove `CircuitBreaker` from `__init__`; delete `core/circuit_breaker.py` and `core/retry.py`. `_route_failure` reverts to current stub.
- **Phase E:** Revert `SKILL_MAP` additions in `skill_dispatcher.py`.

All phases are additive — no existing behavior is removed.

---

## 12. Definition of Done

- [ ] All new tests pass: `pytest tests/ -q`
- [ ] `python3 scripts/generate_cli_reference.py --check` passes
- [ ] `aura doctor` exits 0 on healthy system, 1 on degraded
- [ ] Circuit breaker state visible in `aura goal status --json`
- [ ] `lint` and `test_and_observe` appear in `aura --list-skills`
- [ ] No `sys.exit()` calls in any skill file
- [ ] `tests/test_health_monitor.py` covers all 6 check methods
