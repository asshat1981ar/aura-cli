from __future__ import annotations

import builtins
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json
from core.workflow_models import (
    AgenticLoop,
    LoopCycle,
    RetryPolicy,
    StepResult,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStep,
)
from core.workflow_registry import WorkflowRegistry, register_builtin_workflows
from core.workflow_steps import execute_step, run_skill, wire_inputs
from core.workflow_storage import (
    get_execution_status,
    get_loop_status,
    journal_execution,
    journal_loop,
)

# Allow tests to override the on-disk location; we still operate primarily in-memory.
_DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "workflow_engine.db"

# Expose helpers under underscored names for backward-compatibility in tests.
_wire_inputs = wire_inputs
_execute_step = execute_step
# Make underscored helpers available at module import for legacy tests.
builtins._wire_inputs = _wire_inputs
builtins._execute_step = _execute_step

# Module-level engine for legacy accessors (monkeypatched in tests)
_engine = None


class WorkflowEngine:
    """
    Orchestrates workflow definitions and lightweight agentic loops.
    Provides an in-memory implementation with optional journaling to SQLite.
    """

    def __init__(self, brain=None, model=None, planner=None) -> None:
        register_builtin_workflows()
        self._executions: Dict[str, WorkflowExecution] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._loops: Dict[str, AgenticLoop] = {}

        # Optional dependencies for orchestrator construction.
        self._brain = brain
        self._model = model
        self._planner = planner

    # ------------------------------------------------------------------ #
    # Workflow definitions
    # ------------------------------------------------------------------ #
    def define(self, workflow_def: WorkflowDefinition) -> None:
        WorkflowRegistry.register(workflow_def)

    def list_definitions(self) -> List[Dict[str, Any]]:
        return [
            {"name": name, "steps": [s.name for s in defn.steps], "description": defn.description}
            for name, defn in WorkflowRegistry.list_all().items()
        ]

    def list_workflows(self) -> List[str]:
        return list(WorkflowRegistry.list_all().keys())

    # ------------------------------------------------------------------ #
    # Workflow execution
    # ------------------------------------------------------------------ #
    def _get_lock(self, exec_id: str) -> threading.Lock:
        if exec_id not in self._locks:
            self._locks[exec_id] = threading.Lock()
        return self._locks[exec_id]

    def run_workflow(
        self, workflow_name: str, inputs: Optional[Dict[str, Any]] = None, resume_exec_id: Optional[str] = None
    ) -> str:
        inputs = inputs or {}

        if resume_exec_id:
            if resume_exec_id not in self._executions:
                raise KeyError(f"Execution '{resume_exec_id}' not found.")
            exc = self._executions[resume_exec_id]
            lock = self._get_lock(resume_exec_id)
            with lock:
                exc.status = "running"
                self._run_steps(exc, inputs, start_index=exc.current_step_index)
            return exc.id

        defn = WorkflowRegistry.get(workflow_name)
        if not defn:
            raise ValueError(f"Workflow '{workflow_name}' not found.")

        exec_id = str(uuid.uuid4())
        exc = WorkflowExecution(
            id=exec_id,
            workflow_name=workflow_name,
            status="running",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs=inputs,
            error=None,
            started_at=time.time(),
            updated_at=time.time(),
        )
        self._executions[exec_id] = exc
        self._get_lock(exec_id)  # ensure lock exists
        journal_execution(exc)
        self._run_steps(exc, inputs, start_index=0)
        return exec_id

    def _run_steps(self, exc: WorkflowExecution, initial_inputs: Dict[str, Any], start_index: int) -> None:
        defn = WorkflowRegistry.get(exc.workflow_name)
        if not defn:
            exc.status = "failed"
            exc.error = f"Workflow '{exc.workflow_name}' not found."
            return

        for idx in range(start_index, len(defn.steps)):
            step = defn.steps[idx]
            exc.current_step_index = idx
            exc.updated_at = time.time()
            journal_execution(exc)

            res = execute_step(step, exc.step_outputs, {**initial_inputs, **exc.initial_inputs})
            exc.history.append(res)
            if res.status == "ok":
                exc.step_outputs[step.name] = res.output
            elif res.status == "skipped":
                exc.step_outputs[step.name] = res.output
                continue
            else:
                exc.status = "failed"
                exc.error = res.error or res.output.get("error")
                exc.updated_at = time.time()
                journal_execution(exc)
                return

        exc.status = "completed"
        exc.updated_at = time.time()
        journal_execution(exc)

    def execution_status(self, execution_id: str) -> Dict[str, Any]:
        if execution_id in self._executions:
            exc = self._executions[execution_id]
            return {
                "id": exc.id,
                "workflow": exc.workflow_name,
                "status": exc.status,
                "step_index": exc.current_step_index,
                "history": [
                    {
                        "step": r.step_name,
                        "status": r.status,
                        "output": r.output,
                        "attempts": r.attempts,
                        "elapsed_ms": r.elapsed_ms,
                        "error": r.error,
                    }
                    for r in exc.history
                ],
                "error": exc.error,
                "started_at": exc.started_at,
                "updated_at": exc.updated_at,
            }
        row = get_execution_status(execution_id)
        if row:
            return row
        raise KeyError(f"Execution '{execution_id}' not found.")

    def get_step_output(self, execution_id: str, step_name: str) -> Dict[str, Any]:
        exc = self._executions.get(execution_id)
        if not exc:
            raise KeyError(f"Execution '{execution_id}' not found.")
        if step_name not in exc.step_outputs:
            raise KeyError(f"Step '{step_name}' output not found for execution '{execution_id}'.")
        return exc.step_outputs[step_name]

    def cancel_execution(self, execution_id: str) -> None:
        exc = self._executions.get(execution_id)
        if not exc:
            raise KeyError(f"Execution '{execution_id}' not found.")
        if exc.is_terminal():
            raise ValueError("Execution already terminal")
        exc.status = "cancelled"
        exc.updated_at = time.time()
        journal_execution(exc)

    def pause_execution(self, execution_id: str) -> None:
        exc = self._executions.get(execution_id)
        if not exc:
            raise KeyError(f"Execution '{execution_id}' not found.")
        exc.status = "paused"
        exc.updated_at = time.time()
        journal_execution(exc)

    def list_executions(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        execs = list(self._executions.values())
        if status_filter:
            execs = [e for e in execs if e.status == status_filter]
        return [
            {
                "id": e.id,
                "workflow": e.workflow_name,
                "status": e.status,
                "started_at": e.started_at,
                "updated_at": e.updated_at,
            }
            for e in execs
        ]

    def list_loops(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        loops = list(self._loops.values())
        if status_filter:
            loops = [l for l in loops if l.status == status_filter]
        return [
            {
                "id": l.id,
                "goal": l.goal,
                "status": l.status,
                "current_cycle": l.current_cycle,
                "max_cycles": l.max_cycles,
                "stop_reason": l.stop_reason,
                "score": l.score,
                "updated_at": l.updated_at,
            }
            for l in loops
        ]

    # ------------------------------------------------------------------ #
    # Agentic loops
    # ------------------------------------------------------------------ #
    def _get_loop(self, loop_id: str) -> AgenticLoop:
        loop = self._loops.get(loop_id)
        if loop:
            return loop
        row = get_loop_status(loop_id)
        if row:
            return AgenticLoop(
                id=row["id"],
                goal=row["goal"],
                max_cycles=row["max_cycles"],
                current_cycle=row["current_cycle"],
                status=row["status"],
                history=[],
                stop_reason=row["stop_reason"],
                score=row["score"],
                started_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        raise KeyError(f"Loop '{loop_id}' not found.")

    def _get_orchestrator(self, loop: AgenticLoop):
        """Lazy import to avoid heavy dependencies when unused."""
        try:
            from orchestrator.agentic_loop_orchestrator import AgenticLoopOrchestrator
            return AgenticLoopOrchestrator(self._brain, self._model, self._planner)
        except Exception:
            class _StubOrchestrator:
                def run_cycle(self, loop_obj):
                    return {"stop_reason": None, "phase_outputs": {}}
            return _StubOrchestrator()

    def create_loop(self, goal: str, max_cycles: int = 10, score: float = 0.0) -> str:
        loop_id = str(uuid.uuid4())
        loop = AgenticLoop(
            id=loop_id,
            goal=goal,
            max_cycles=max_cycles,
            current_cycle=0,
            status="running",
            history=[],
            stop_reason=None,
            score=score,
            started_at=time.time(),
            updated_at=time.time(),
        )
        self._loops[loop_id] = loop
        journal_loop(loop)
        return loop_id

    def loop_status(self, loop_id: str) -> Dict[str, Any]:
        loop = self._get_loop(loop_id)
        return {
            "id": loop.id,
            "goal": loop.goal,
            "max_cycles": loop.max_cycles,
            "current_cycle": loop.current_cycle,
            "status": loop.status,
            "stop_reason": loop.stop_reason,
            "score": loop.score,
            "history": [
                {
                    "cycle": c.cycle_number,
                    "cycle_status": c.status,
                    "phase_outputs": c.phase_outputs,
                    "stop_reason": c.stop_reason,
                    "error": c.error,
                }
                for c in loop.history
            ],
        }

    def loop_tick(self, loop_id: str, dry_run: bool = False) -> Dict[str, Any]:
        loop = self._get_loop(loop_id)
        if loop.is_terminal():
            return {
                "id": loop.id,
                "status": loop.status,
                "cycle": loop.current_cycle,
                "stop_reason": loop.stop_reason,
                "cycle_status": "terminal",
                "error": "loop is terminal",
            }

        orch = self._get_orchestrator(loop)
        result = orch.run_cycle(loop)
        stop_reason = result.get("stop_reason")

        loop.current_cycle += 1
        loop.updated_at = time.time()
        if stop_reason is None and loop.current_cycle >= loop.max_cycles:
            stop_reason = "max_cycles_reached"

        cycle_status = "ok" if not stop_reason else "completed"
        loop_cycle = LoopCycle(
            cycle_number=loop.current_cycle,
            status="ok" if not stop_reason else "failed" if stop_reason == "error" else "ok",
            phase_outputs=result.get("phase_outputs", {}),
            elapsed_ms=0.0,
            error=None,
            stop_reason=stop_reason,
        )
        loop.history.append(loop_cycle)

        if stop_reason:
            loop.status = "completed"
            loop.stop_reason = stop_reason
        elif loop.current_cycle >= loop.max_cycles:
            loop.status = "completed"
            loop.stop_reason = "max_cycles_reached"
        else:
            loop.status = "running"

        journal_loop(loop)
        return {
            "id": loop.id,
            "cycle": loop.current_cycle,
            "status": loop.status,
            "cycle_status": loop_cycle.status,
            "stop_reason": loop.stop_reason,
            "phase_outputs": result.get("phase_outputs", {}),
        }

    def stop_loop(self, loop_id: str, reason: str = "stopped") -> None:
        loop = self._get_loop(loop_id)
        if loop.is_terminal():
            raise ValueError("Loop is already terminal")
        loop.status = "stopped"
        loop.stop_reason = reason
        loop.updated_at = time.time()
        journal_loop(loop)

    def pause_loop(self, loop_id: str) -> None:
        loop = self._get_loop(loop_id)
        if loop.is_terminal():
            raise ValueError("Loop is already terminal")
        loop.status = "paused"
        loop.updated_at = time.time()
        journal_loop(loop)

    def resume_loop(self, loop_id: str) -> None:
        loop = self._get_loop(loop_id)
        if loop.status == "paused":
            loop.status = "running"
            loop.updated_at = time.time()
            journal_loop(loop)

    def check_loop_health(self, loop_id: str, stall_threshold_s: float = 300.0) -> Dict[str, Any]:
        loop = self._get_loop(loop_id)
        warnings: List[str] = []
        healthy = True
        age = time.time() - loop.updated_at
        if age > stall_threshold_s:
            healthy = False
            warnings.append(f"Loop has not progressed for {int(age)}s")
        return {"loop_id": loop_id, "healthy": healthy, "warnings": warnings}


def get_engine() -> WorkflowEngine:
    """Return singleton WorkflowEngine (created on first use)."""
    global _engine
    if _engine is None:
        _engine = WorkflowEngine()
    return _engine
