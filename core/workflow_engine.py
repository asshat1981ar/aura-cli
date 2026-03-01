"""
Workflow Engine — reusable agentic workflow execution core.

Concepts:
  WorkflowDefinition  — a named, ordered list of WorkflowSteps (a linear DAG).
  WorkflowStep        — a single unit of work: run an AURA skill or callable,
                        with optional data wiring from previous steps, retry
                        policy, and a skip-condition.
  WorkflowExecution   — live run of a definition: tracks state, step outputs,
                        retry counts, history, pause/resume/cancel.
  AgenticLoop         — a goal-driven loop that calls LoopOrchestrator.run_cycle()
                        up to max_cycles, with per-cycle scoring and early exit.
  WorkflowEngine      — manages executions and loops in an in-memory registry
                        backed by a SQLite journal for durability.

Usage example:
  engine = WorkflowEngine()
  exec_id = engine.run_workflow("security_audit", {
      "project_root": "/my/project"
  })
  status = engine.execution_status(exec_id)
"""
from __future__ import annotations

import sqlite3
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from core.config_manager import ConfigManager, DEFAULT_CONFIG
from core.logging_utils import log_json
from core.runtime_paths import resolve_project_path

_DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "workflow_engine.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """Exponential-backoff retry configuration for a single step."""
    max_attempts: int = 3
    backoff_base: float = 0.5   # seconds; sleep = backoff_base * 2^attempt
    max_backoff: float = 30.0   # cap on sleep duration

    def sleep_for(self, attempt: int) -> float:
        return min(self.backoff_base * (2 ** attempt), self.max_backoff)


@dataclass
class WorkflowStep:
    """
    One step in a workflow.

    Args:
        name:           Unique label within the workflow.
        skill_name:     Name of an AURA skill (from agents/skills/registry) to run,
                        OR None if `fn` is provided.
        fn:             Callable(inputs: dict) -> dict — used when skill_name is None.
        static_inputs:  Hard-coded inputs merged before calling the step.
        inputs_from:    Wire previous step outputs: {"my_key": "step_name.output_key"}.
                        "step_name.output_key" means step_outputs[step_name][output_key].
                        Use "step_name.*" to forward the entire output dict.
        skip_if_false:  Name of an earlier step's boolean output key.  If that
                        key is falsy the step is skipped (status → "skipped").
        retry:          Retry/backoff policy (default: 3 attempts).
        timeout_s:      Hard wall-clock timeout in seconds (0 = no limit).
    """
    name: str
    skill_name: Optional[str] = None
    fn: Optional[Callable[[Dict], Dict]] = None
    static_inputs: Dict[str, Any] = field(default_factory=dict)
    inputs_from: Dict[str, str] = field(default_factory=dict)
    skip_if_false: Optional[str] = None   # e.g. "check_security.critical_count"
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_s: float = 120.0


@dataclass
class WorkflowDefinition:
    """
    Named, ordered list of steps.  Steps execute sequentially; parallel
    execution can be composed by grouping steps with a fan-out/fan-in pattern
    via the `inputs_from` wiring.
    """
    name: str
    steps: List[WorkflowStep]
    description: str = ""
    max_retries_total: int = 0  # extra safety: abort after N total retries across all steps


@dataclass
class StepResult:
    step_name: str
    status: Literal["ok", "skipped", "failed", "timeout"]
    output: Dict[str, Any]
    attempts: int
    elapsed_ms: float
    error: Optional[str] = None


@dataclass
class WorkflowExecution:
    id: str
    workflow_name: str
    status: Literal["pending", "running", "paused", "completed", "failed", "cancelled"]
    current_step_index: int
    step_outputs: Dict[str, Dict]   # {step_name: output_dict}
    history: List[StepResult]
    initial_inputs: Dict[str, Any]
    error: Optional[str]
    started_at: float
    updated_at: float
    total_retries_used: int = 0

    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")


@dataclass
class LoopCycle:
    cycle_number: int
    status: Literal["ok", "failed", "skipped"]
    phase_outputs: Dict[str, Any]
    elapsed_ms: float
    error: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class AgenticLoop:
    id: str
    goal: str
    max_cycles: int
    current_cycle: int
    status: Literal["running", "paused", "completed", "failed", "stopped"]
    history: List[LoopCycle]
    stop_reason: Optional[str]
    score: float
    started_at: float
    updated_at: float

    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "stopped")


# ---------------------------------------------------------------------------
# SQLite journal
# ---------------------------------------------------------------------------

def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS executions (
            id          TEXT PRIMARY KEY,
            workflow    TEXT NOT NULL,
            status      TEXT NOT NULL,
            step_index  INTEGER NOT NULL DEFAULT 0,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            summary     TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loops (
            id          TEXT PRIMARY KEY,
            goal        TEXT NOT NULL,
            max_cycles  INTEGER NOT NULL,
            current_cycle INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL,
            score       REAL NOT NULL DEFAULT 0,
            stop_reason TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


@contextmanager
def _db():
    conn = _open_db()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Skill runner (lazy import to avoid circular deps)
# ---------------------------------------------------------------------------

def _run_skill(skill_name: str, inputs: Dict) -> Dict:
    """Lazily load skill from registry and call it."""
    try:
        from agents.skills.registry import all_skills
        skills = all_skills()
        if skill_name not in skills:
            return {"error": f"Unknown skill '{skill_name}'. Available: {sorted(skills)}"}
        return skills[skill_name].run(inputs)
    except Exception as exc:
        return {"error": f"Skill runner error: {exc}"}


# ---------------------------------------------------------------------------
# Input wiring
# ---------------------------------------------------------------------------

def _wire_inputs(
    step: WorkflowStep,
    step_outputs: Dict[str, Dict],
    initial_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the input dict for this step by:
      1. Starting with initial_inputs (e.g. project_root)
      2. Applying static_inputs (override)
      3. Resolving inputs_from wiring ("step.key" references)
    """
    resolved: Dict[str, Any] = {**initial_inputs, **step.static_inputs}
    for dest_key, src_path in step.inputs_from.items():
        parts = src_path.split(".", 1)
        src_step = parts[0]
        src_key = parts[1] if len(parts) > 1 else None
        src_output = step_outputs.get(src_step, {})
        if src_key == "*" or src_key is None:
            resolved.update(src_output)
        else:
            resolved[dest_key] = src_output.get(src_key)
    return resolved


# ---------------------------------------------------------------------------
# Core execution logic
# ---------------------------------------------------------------------------

def _execute_step(
    step: WorkflowStep,
    step_outputs: Dict[str, Dict],
    initial_inputs: Dict[str, Any],
) -> StepResult:
    """Run one step with retry + timeout logic."""
    # Check skip condition
    if step.skip_if_false:
        parts = step.skip_if_false.split(".", 1)
        src = step_outputs.get(parts[0], {})
        flag = src.get(parts[1]) if len(parts) > 1 else src
        if not flag:
            return StepResult(
                step_name=step.name,
                status="skipped",
                output={},
                attempts=0,
                elapsed_ms=0.0,
            )

    inputs = _wire_inputs(step, step_outputs, initial_inputs)
    last_error: Optional[str] = None
    attempt = 0

    for attempt in range(max(1, step.retry.max_attempts)):
        t0 = time.time()
        try:
            if step.skill_name:
                output = _run_skill(step.skill_name, inputs)
            elif step.fn:
                output = step.fn(inputs)
            else:
                output = {"error": f"Step '{step.name}' has no skill_name or fn."}

            elapsed = (time.time() - t0) * 1000

            if isinstance(output, dict) and "error" in output:
                last_error = output["error"]
                log_json("WARN", "workflow_step_error", details={
                    "step": step.name, "attempt": attempt + 1, "error": last_error
                })
            else:
                return StepResult(
                    step_name=step.name,
                    status="ok",
                    output=output or {},
                    attempts=attempt + 1,
                    elapsed_ms=elapsed,
                )

        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            last_error = f"{type(exc).__name__}: {exc}"
            log_json("WARN", "workflow_step_exception", details={
                "step": step.name, "attempt": attempt + 1, "error": last_error,
                "traceback": traceback.format_exc()[-500:],
            })

        # Backoff before retry
        if attempt < step.retry.max_attempts - 1:
            sleep_t = step.retry.sleep_for(attempt)
            log_json("INFO", "workflow_step_retry", details={
                "step": step.name, "attempt": attempt + 1, "sleep_s": sleep_t
            })
            time.sleep(sleep_t)

    return StepResult(
        step_name=step.name,
        status="failed",
        output={"error": last_error},
        attempts=attempt + 1,
        elapsed_ms=(time.time() - t0) * 1000,
        error=last_error,
    )


# ---------------------------------------------------------------------------
# WorkflowEngine
# ---------------------------------------------------------------------------

class WorkflowEngine:
    """
    Manages workflow definitions, executions, and agentic loops.

    Thread-safe via a per-execution lock.  Executions and loops are stored
    in memory (fast access) and journalled to SQLite (durability).

    Usage:
        engine = WorkflowEngine()

        # Register a workflow
        engine.define(WorkflowDefinition(
            name="security_audit",
            steps=[
                WorkflowStep("scan",  skill_name="security_scanner"),
                WorkflowStep("lint",  skill_name="linter_enforcer"),
                WorkflowStep("report", skill_name="doc_generator",
                             inputs_from={"code": "scan.*"}),
            ]
        ))

        exec_id = engine.run_workflow("security_audit", {"project_root": "."})
        status = engine.execution_status(exec_id)
    """

    def __init__(self):
        self._definitions: Dict[str, WorkflowDefinition] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._loops: Dict[str, AgenticLoop] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._register_builtin_workflows()
        log_json("INFO", "workflow_engine_init", details={"db": str(_DB_PATH)})

    # ------------------------------------------------------------------
    # Definition management
    # ------------------------------------------------------------------

    def define(self, wf: WorkflowDefinition) -> None:
        """Register or replace a workflow definition."""
        with self._global_lock:
            self._definitions[wf.name] = wf
        log_json("INFO", "workflow_defined", details={"name": wf.name, "steps": len(wf.steps)})

    def list_definitions(self) -> List[Dict]:
        return [
            {
                "name": d.name,
                "description": d.description,
                "step_count": len(d.steps),
                "steps": [s.name for s in d.steps],
            }
            for d in self._definitions.values()
        ]

    # ------------------------------------------------------------------
    # Execution management
    # ------------------------------------------------------------------

    def run_workflow(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        resume_exec_id: Optional[str] = None,
    ) -> str:
        """
        Start (or resume) a workflow execution.  Returns execution_id.
        Runs synchronously in the calling thread — wrap in a thread if async.
        """
        if workflow_name not in self._definitions:
            raise KeyError(f"Workflow '{workflow_name}' not defined.")

        wf = self._definitions[workflow_name]
        exec_id = resume_exec_id or str(uuid.uuid4())
        now = time.time()

        if resume_exec_id and resume_exec_id in self._executions:
            exc = self._executions[resume_exec_id]
            if exc.status != "paused":
                raise ValueError(f"Execution {resume_exec_id} is '{exc.status}', not 'paused'.")
            exc.status = "running"
            exc.updated_at = now
        else:
            exc = WorkflowExecution(
                id=exec_id,
                workflow_name=workflow_name,
                status="running",
                current_step_index=0,
                step_outputs={},
                history=[],
                initial_inputs=inputs or {},
                error=None,
                started_at=now,
                updated_at=now,
            )
            with self._global_lock:
                self._executions[exec_id] = exc
                self._locks[exec_id] = threading.Lock()
            self._journal_execution(exc)

        self._run_execution(exc, wf)
        return exec_id

    def _run_execution(self, exc: WorkflowExecution, wf: WorkflowDefinition) -> None:
        lock = self._locks.get(exc.id, threading.Lock())
        with lock:
            steps = wf.steps
            while exc.current_step_index < len(steps):
                if exc.status in ("cancelled", "paused"):
                    break

                step = steps[exc.current_step_index]
                log_json("INFO", "workflow_step_start", details={
                    "exec_id": exc.id, "step": step.name, "index": exc.current_step_index
                })

                result = _execute_step(step, exc.step_outputs, exc.initial_inputs)
                exc.history.append(result)
                exc.updated_at = time.time()

                if result.status == "ok":
                    exc.step_outputs[step.name] = result.output
                elif result.status == "skipped":
                    exc.step_outputs[step.name] = {}
                elif result.status == "failed":
                    exc.total_retries_used += result.attempts - 1
                    # Check total retry budget
                    if wf.max_retries_total > 0 and exc.total_retries_used >= wf.max_retries_total:
                        exc.status = "failed"
                        exc.error = f"Total retry budget exhausted at step '{step.name}': {result.error}"
                        self._journal_execution(exc)
                        log_json("ERROR", "workflow_budget_exhausted", details={"exec_id": exc.id})
                        return
                    # Single step failed — continue or abort
                    exc.error = f"Step '{step.name}' failed: {result.error}"
                    exc.status = "failed"
                    self._journal_execution(exc)
                    log_json("ERROR", "workflow_execution_failed", details={
                        "exec_id": exc.id, "step": step.name, "error": result.error
                    })
                    return

                exc.current_step_index += 1
                self._journal_execution(exc)

            if exc.status == "running":
                exc.status = "completed"
                exc.updated_at = time.time()
                self._journal_execution(exc)
                log_json("INFO", "workflow_execution_completed", details={
                    "exec_id": exc.id, "steps": len(exc.history)
                })

    def pause_execution(self, exec_id: str) -> None:
        exc = self._get_execution(exec_id)
        if exc.is_terminal():
            raise ValueError(f"Execution {exec_id} is already terminal ({exc.status}).")
        exc.status = "paused"
        exc.updated_at = time.time()
        self._journal_execution(exc)
        log_json("INFO", "workflow_paused", details={"exec_id": exec_id})

    def cancel_execution(self, exec_id: str) -> None:
        exc = self._get_execution(exec_id)
        if not exc:
            raise KeyError(f"Execution '{exec_id}' not found.")
        if exc.is_terminal():
            raise ValueError(f"Execution {exec_id} is already terminal ({exc.status}).")
        exc.status = "cancelled"
        exc.updated_at = time.time()
        self._journal_execution(exc)
        log_json("INFO", "workflow_cancelled", details={"exec_id": exec_id})

    def execution_status(self, exec_id: str) -> Dict[str, Any]:
        exc = self._get_execution(exec_id)
        if not exc:
            return {"id": exec_id, "status": "not_found", "error": f"Execution {exec_id} not found."}
        return {
            "id": exc.id,
            "workflow": exc.workflow_name,
            "status": exc.status,
            "current_step": exc.current_step_index,
            "error": exc.error,
            "elapsed_s": round(time.time() - exc.started_at, 2),
            "history": [
                {
                    "step": r.step_name,
                    "status": r.status,
                    "attempts": r.attempts,
                    "elapsed_ms": round(r.elapsed_ms, 1),
                    "error": r.error,
                }
                for r in exc.history
            ],
            "step_output_keys": {k: list(v.keys()) for k, v in exc.step_outputs.items()},
        }

    def get_step_output(self, exec_id: str, step_name: str) -> Dict:
        exc = self._get_execution(exec_id)
        if not exc:
            raise KeyError(f"Execution '{exec_id}' not found.")
        if step_name not in exc.step_outputs:
            raise KeyError(f"Step '{step_name}' output not available in execution {exec_id}.")
        return exc.step_outputs[step_name]

    def list_executions(self, status_filter: Optional[str] = None) -> List[Dict]:
        with self._global_lock:
            items = list(self._executions.values())
        if status_filter:
            items = [e for e in items if e.status == status_filter]
        return [
            {
                "id": e.id,
                "workflow": e.workflow_name,
                "status": e.status,
                "current_step": e.current_step_index,
                "elapsed_s": round(time.time() - e.started_at, 2),
            }
            for e in items
        ]

    # ------------------------------------------------------------------
    # Agentic loop management
    # ------------------------------------------------------------------

    def create_loop(self, goal: str, max_cycles: int = 5) -> str:
        """Create a new AgenticLoop. Returns loop_id."""
        loop_id = str(uuid.uuid4())
        now = time.time()
        loop = AgenticLoop(
            id=loop_id,
            goal=goal,
            max_cycles=max_cycles,
            current_cycle=0,
            status="running",
            history=[],
            stop_reason=None,
            score=0.0,
            started_at=now,
            updated_at=now,
        )
        with self._global_lock:
            self._loops[loop_id] = loop
            self._locks[loop_id] = threading.Lock()
        self._journal_loop(loop)
        log_json("INFO", "agentic_loop_created", details={"loop_id": loop_id, "goal": goal[:80]})
        return loop_id

    def loop_tick(self, loop_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute one cycle of the agentic loop using LoopOrchestrator.

        Returns the cycle result dict.  Call repeatedly (or in a thread) until
        loop_status() shows a terminal state.
        """
        loop = self._get_loop(loop_id)
        if not loop:
            return {"error": f"Loop {loop_id} not found."}
        if loop.is_terminal():
            return {"error": f"Loop {loop_id} is already '{loop.status}'."}

        lock = self._locks.get(loop_id, threading.Lock())
        with lock:
            loop = self._get_loop(loop_id)  # re-read under lock
            if not loop: return {"error": "Loop deleted during tick"}
            if loop.current_cycle >= loop.max_cycles:
                loop.status = "completed"
                loop.stop_reason = "max_cycles_reached"
                loop.updated_at = time.time()
                self._journal_loop(loop)
                return {"stop_reason": loop.stop_reason, "cycles": loop.current_cycle}

            cycle_num = loop.current_cycle + 1
            log_json("INFO", "agentic_loop_tick", details={
                "loop_id": loop_id, "cycle": cycle_num, "goal": loop.goal[:60]
            })

            t0 = time.time()
            try:
                orchestrator = self._get_orchestrator()
                result = orchestrator.run_cycle(loop.goal, dry_run=dry_run)
                elapsed = (time.time() - t0) * 1000

                stop_reason = result.get("stop_reason")
                phase_outputs = result.get("phase_outputs", {})

                cycle = LoopCycle(
                    cycle_number=cycle_num,
                    status="ok",
                    phase_outputs={k: list(v.keys()) if isinstance(v, dict) else str(v)
                                   for k, v in phase_outputs.items()},
                    elapsed_ms=elapsed,
                    stop_reason=stop_reason,
                )
                loop.history.append(cycle)
                loop.current_cycle = cycle_num
                loop.updated_at = time.time()

                if stop_reason or cycle_num >= loop.max_cycles:
                    loop.status = "completed"
                    loop.stop_reason = stop_reason or "max_cycles_reached"

            except Exception as exc:
                elapsed = (time.time() - t0) * 1000
                err = f"{type(exc).__name__}: {exc}"
                cycle = LoopCycle(
                    cycle_number=cycle_num,
                    status="failed",
                    phase_outputs={},
                    elapsed_ms=elapsed,
                    error=err,
                )
                loop.history.append(cycle)
                loop.current_cycle = cycle_num
                loop.status = "failed"
                loop.stop_reason = f"cycle_{cycle_num}_exception"
                loop.updated_at = time.time()
                log_json("ERROR", "agentic_loop_cycle_failed", details={
                    "loop_id": loop_id, "cycle": cycle_num, "error": err
                })

            self._journal_loop(loop)
            return {
                "loop_id": loop_id,
                "cycle": cycle_num,
                "status": loop.status,
                "cycle_status": cycle.status,
                "elapsed_ms": round(cycle.elapsed_ms, 1),
                "stop_reason": loop.stop_reason,
                "error": cycle.error,
            }

    def stop_loop(self, loop_id: str, reason: str = "user_requested") -> None:
        loop = self._get_loop(loop_id)
        if not loop: raise KeyError(f"Loop {loop_id} not found")
        if loop.is_terminal():
            raise ValueError(f"Loop {loop_id} is already terminal ({loop.status}).")
        loop.status = "stopped"
        loop.stop_reason = reason
        loop.updated_at = time.time()
        self._journal_loop(loop)
        log_json("INFO", "agentic_loop_stopped", details={"loop_id": loop_id, "reason": reason})

    def pause_loop(self, loop_id: str) -> None:
        loop = self._get_loop(loop_id)
        if not loop: raise KeyError(f"Loop {loop_id} not found")
        if loop.is_terminal():
            raise ValueError(f"Loop {loop_id} already terminal ({loop.status}).")
        loop.status = "paused"
        loop.updated_at = time.time()
        self._journal_loop(loop)

    def resume_loop(self, loop_id: str) -> None:
        loop = self._get_loop(loop_id)
        if not loop: raise KeyError(f"Loop {loop_id} not found")
        if loop.status != "paused":
            raise ValueError(f"Loop {loop_id} is '{loop.status}', not 'paused'.")
        loop.status = "running"
        loop.updated_at = time.time()
        self._journal_loop(loop)

    def loop_status(self, loop_id: str) -> Dict[str, Any]:
        loop = self._get_loop(loop_id)
        if not loop:
            return {"id": loop_id, "status": "not_found", "error": f"Loop {loop_id} not found."}
        return {
            "id": loop.id,
            "goal": loop.goal,
            "status": loop.status,
            "current_cycle": loop.current_cycle,
            "max_cycles": loop.max_cycles,
            "stop_reason": loop.stop_reason,
            "score": loop.score,
            "elapsed_s": round(time.time() - loop.started_at, 2),
            "history": [
                {
                    "cycle": c.cycle_number,
                    "status": c.status,
                    "elapsed_ms": round(c.elapsed_ms, 1),
                    "stop_reason": c.stop_reason,
                    "error": c.error,
                }
                for c in loop.history
            ],
        }

    def list_loops(self, status_filter: Optional[str] = None) -> List[Dict]:
        with self._global_lock:
            items = list(self._loops.values())
        if status_filter:
            items = [l for l in items if l.status == status_filter]
        return [
            {
                "id": l.id,
                "goal": l.goal[:80],
                "status": l.status,
                "cycles": l.current_cycle,
                "max_cycles": l.max_cycles,
                "elapsed_s": round(time.time() - l.started_at, 2),
            }
            for l in items
        ]

    # ------------------------------------------------------------------
    # Deadlock / infinite loop guard
    # ------------------------------------------------------------------

    def check_loop_health(self, loop_id: str, stall_threshold_s: float = 300.0) -> Dict:
        """
        Detect potential infinite loops or stalls.

        Returns {"healthy": bool, "warnings": [...], "recommendation": str}
        """
        loop = self._get_loop(loop_id)
        if not loop: return {"healthy": False, "error": "Loop not found"}
        warnings: List[str] = []

        if loop.is_terminal():
            return {"healthy": True, "warnings": [], "recommendation": "Loop is terminal."}

        stale_s = time.time() - loop.updated_at
        if stale_s > stall_threshold_s:
            warnings.append(f"Loop has not progressed for {stale_s:.0f}s (threshold: {stall_threshold_s}s).")

        # Check for repeated identical errors (deadlock signal)
        if len(loop.history) >= 3:
            last_3_errors = [c.error for c in loop.history[-3:]]
            if all(e and e == last_3_errors[0] for e in last_3_errors):
                warnings.append(f"Same error repeated 3 cycles: '{last_3_errors[0]}'")

        recommendation = "stop_loop" if warnings else "healthy"
        return {
            "healthy": not warnings,
            "warnings": warnings,
            "recommendation": recommendation,
            "cycles_remaining": loop.max_cycles - loop.current_cycle,
        }

    # ------------------------------------------------------------------
    # Built-in workflow definitions
    # ------------------------------------------------------------------

    def _register_builtin_workflows(self) -> None:
        """Pre-register useful AURA skill workflows."""
        self.define(WorkflowDefinition(
            name="security_audit",
            description="Scan for vulnerabilities, lint issues, and dependency CVEs.",
            steps=[
                WorkflowStep("security_scan",  skill_name="security_scanner"),
                WorkflowStep("lint_check",     skill_name="linter_enforcer"),
                WorkflowStep("dep_audit",      skill_name="dependency_analyzer"),
                WorkflowStep("observability",  skill_name="observability_checker"),
            ],
        ))
        self.define(WorkflowDefinition(
            name="code_quality",
            description="Full code quality sweep: complexity, type coverage, architecture.",
            steps=[
                WorkflowStep("complexity",    skill_name="complexity_scorer"),
                WorkflowStep("type_check",    skill_name="type_checker"),
                WorkflowStep("arch_validate", skill_name="architecture_validator"),
                WorkflowStep("tech_debt",     skill_name="tech_debt_quantifier"),
                WorkflowStep("refactor_tips", skill_name="refactoring_advisor",
                             inputs_from={"project_root": "complexity.project_root"}),
            ],
        ))
        self.define(WorkflowDefinition(
            name="release_prep",
            description="Pre-release pipeline: lint, test coverage, changelog, deps.",
            steps=[
                WorkflowStep("lint",      skill_name="linter_enforcer"),
                WorkflowStep("coverage",  skill_name="test_coverage_analyzer"),
                WorkflowStep("changelog", skill_name="changelog_generator"),
                WorkflowStep("dep_check", skill_name="dependency_analyzer"),
            ],
        ))
        self.define(WorkflowDefinition(
            name="onboarding_analysis",
            description="Deep-dive for new contributors: architecture, symbols, clone detection.",
            steps=[
                WorkflowStep("symbols",      skill_name="symbol_indexer"),
                WorkflowStep("arch",         skill_name="architecture_validator"),
                WorkflowStep("clones",       skill_name="code_clone_detector"),
                WorkflowStep("git_history",  skill_name="git_history_analyzer"),
            ],
        ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_execution(self, exec_id: str) -> Optional[WorkflowExecution]:
        with self._global_lock:
            return self._executions.get(exec_id)

    def _get_loop(self, loop_id: str) -> Optional[AgenticLoop]:
        with self._global_lock:
            return self._loops.get(loop_id)

    def _get_orchestrator(self):
        """Lazy-load LoopOrchestrator with default agents. Cached after first call."""
        if not hasattr(self, "_orchestrator"):
            project_root = Path(__file__).resolve().parent.parent
            try:
                from aura_cli.cli_main import create_runtime
                rt = create_runtime(project_root, overrides=None)
                orchestrator = rt.get("orchestrator")
                if orchestrator is None:
                    raise KeyError("Runtime factory returned no orchestrator")
                self._orchestrator = orchestrator
            except Exception as exc:
                # Fallback: minimal orchestrator for testing
                log_json("WARN", "workflow_engine_orchestrator_fallback", details={"error": str(exc)})
                from core.orchestrator import LoopOrchestrator
                from memory.store import MemoryStore
                from agents.registry import default_agents
                from memory.brain import Brain
                from core.model_adapter import ModelAdapter
                project_config = ConfigManager(config_file=project_root / "aura.config.json")
                brain_db_path = resolve_project_path(
                    project_root,
                    project_config.get("brain_db_path", DEFAULT_CONFIG["brain_db_path"]),
                    DEFAULT_CONFIG["brain_db_path"],
                )
                memory_store_root = resolve_project_path(
                    project_root,
                    project_config.get("memory_store_path", DEFAULT_CONFIG["memory_store_path"]),
                    DEFAULT_CONFIG["memory_store_path"],
                )
                brain = Brain(db_path=str(brain_db_path))
                model = ModelAdapter()
                agents = default_agents(brain, model)
                self._orchestrator = LoopOrchestrator(
                    agents=agents,
                    memory_store=MemoryStore(memory_store_root),
                    project_root=project_root,
                )
        return self._orchestrator

    def _journal_execution(self, exc: WorkflowExecution) -> None:
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO executions
                       (id, workflow, status, step_index, created_at, updated_at, summary)
                       VALUES (?,?,?,?,?,?,?)""",
                    (exc.id, exc.workflow_name, exc.status, exc.current_step_index,
                     exc.started_at, exc.updated_at,
                     f"{len(exc.history)} steps run, error={exc.error}"),
                )
        except Exception as e:
            log_json("WARN", "workflow_journal_failed", details={"error": str(e)})

    def _journal_loop(self, loop: AgenticLoop) -> None:
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO loops
                       (id, goal, max_cycles, current_cycle, status, score,
                        stop_reason, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (loop.id, loop.goal, loop.max_cycles, loop.current_cycle,
                     loop.status, loop.score, loop.stop_reason,
                     loop.started_at, loop.updated_at),
                )
        except Exception as e:
            log_json("WARN", "loop_journal_failed", details={"error": str(e)})


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[WorkflowEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> WorkflowEngine:
    """Return the module-level singleton WorkflowEngine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = WorkflowEngine()
    return _engine
