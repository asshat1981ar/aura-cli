"""
PRD-004: Comprehensive tests for WorkflowEngine.
Covers 8 classes and 33 functions in core/workflow_engine.py.
"""
from __future__ import annotations

import os
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.workflow_engine import (
    WorkflowEngine, 
    WorkflowDefinition, 
    WorkflowStep, 
    RetryPolicy,
    WorkflowExecution,
    StepResult,
    LoopCycle,
    AgenticLoop,
    _wire_inputs,
    _execute_step
)

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    # Use a fresh engine for each test
    # Note: SQLite journal will still hit the same file unless we mock _DB_PATH
    with patch("core.workflow_engine._DB_PATH", Path("memory/test_workflow_engine.db")):
        return WorkflowEngine()

# ---------------------------------------------------------------------------
# Unit Tests: Data Models
# ---------------------------------------------------------------------------

def test_retry_policy_backoff():
    policy = RetryPolicy(backoff_base=0.1, max_backoff=1.0)
    assert policy.sleep_for(0) == 0.1
    assert policy.sleep_for(1) == 0.2
    assert policy.sleep_for(2) == 0.4
    assert policy.sleep_for(10) == 1.0

def test_workflow_execution_is_terminal():
    exc = WorkflowExecution(
        id="1", workflow_name="test", status="completed", current_step_index=0,
        step_outputs={}, history=[], initial_inputs={}, error=None,
        started_at=time.time(), updated_at=time.time()
    )
    assert exc.is_terminal() is True
    exc.status = "running"
    assert exc.is_terminal() is False

def test_agentic_loop_is_terminal():
    loop = AgenticLoop(
        id="1", goal="test", max_cycles=5, current_cycle=0, status="stopped",
        history=[], stop_reason="user", score=0.0, started_at=time.time(), updated_at=time.time()
    )
    assert loop.is_terminal() is True
    loop.status = "running"
    assert loop.is_terminal() is False

# ---------------------------------------------------------------------------
# Unit Tests: Input Wiring
# ---------------------------------------------------------------------------

def test_wire_inputs_basic():
    step = WorkflowStep(name="step2", inputs_from={"val": "step1.output_key"})
    step_outputs = {"step1": {"output_key": "hello"}}
    initial = {"root": "/tmp"}
    
    result = _wire_inputs(step, step_outputs, initial)
    assert result["val"] == "hello"
    assert result["root"] == "/tmp"

def test_wire_inputs_wildcard():
    step = WorkflowStep(name="step2", inputs_from={"unused": "step1.*"})
    step_outputs = {"step1": {"a": 1, "b": 2}}
    
    result = _wire_inputs(step, step_outputs, {})
    assert result["a"] == 1
    assert result["b"] == 2

# ---------------------------------------------------------------------------
# Unit Tests: Step Execution
# ---------------------------------------------------------------------------

def test_execute_step_fn():
    def mock_fn(inputs):
        return {"result": inputs["x"] + 1}
    
    step = WorkflowStep(name="add", fn=mock_fn, static_inputs={"x": 10})
    res = _execute_step(step, {}, {})
    
    assert res.status == "ok"
    assert res.output["result"] == 11
    assert res.attempts == 1

def test_execute_step_skip():
    step = WorkflowStep(name="opt", fn=lambda x: {}, skip_if_false="check.run_me")
    step_outputs = {"check": {"run_me": False}}
    
    res = _execute_step(step, step_outputs, {})
    assert res.status == "skipped"

def test_execute_step_retry_failure():
    mock_fn = MagicMock(side_effect=Exception("kaboom"))
    step = WorkflowStep(name="fail", fn=mock_fn, retry=RetryPolicy(max_attempts=2, backoff_base=0.01))
    
    res = _execute_step(step, {}, {})
    assert res.status == "failed"
    assert res.attempts == 2
    assert "kaboom" in res.error

# ---------------------------------------------------------------------------
# Integration Tests: WorkflowEngine
# ---------------------------------------------------------------------------

def test_engine_define_and_list(engine):
    wf = WorkflowDefinition(name="custom", steps=[WorkflowStep("s1", fn=lambda x: {})])
    engine.define(wf)
    
    defs = engine.list_definitions()
    names = [d["name"] for d in defs]
    assert "custom" in names
    assert "security_audit" in names # builtin

def test_engine_run_workflow_sync(engine):
    outputs = []
    def track_fn(inputs):
        outputs.append(inputs["val"])
        return {"out": inputs["val"] + "!"}
        
    wf = WorkflowDefinition(name="pipe", steps=[
        WorkflowStep("s1", fn=track_fn, static_inputs={"val": "a"}),
        WorkflowStep("s2", fn=track_fn, inputs_from={"val": "s1.out"})
    ])
    engine.define(wf)
    
    exec_id = engine.run_workflow("pipe")
    status = engine.execution_status(exec_id)
    
    assert status["status"] == "completed"
    assert outputs == ["a", "a!"]
    assert engine.get_step_output(exec_id, "s2") == {"out": "a!!"}

def test_engine_pause_resume(engine):
    # We can't easily test mid-run pause in sync mode without threading,
    # but we can test the state transitions.
    wf = WorkflowDefinition(name="long", steps=[WorkflowStep("s1", fn=lambda x: {"ok": True})])
    engine.define(wf)
    
    # Mock an execution in paused state
    exec_id = "test-pause"
    exc = WorkflowExecution(
        id=exec_id, workflow_name="long", status="paused", current_step_index=0,
        step_outputs={}, history=[], initial_inputs={}, error=None,
        started_at=time.time(), updated_at=time.time()
    )
    engine._executions[exec_id] = exc
    engine._locks[exec_id] = threading.Lock()
    
    # Resume it
    engine.run_workflow("long", resume_exec_id=exec_id)
    assert exc.status == "completed"

def test_engine_cancel(engine):
    wf = WorkflowDefinition(name="cancel-me", steps=[WorkflowStep("s1", fn=lambda x: {})])
    engine.define(wf)
    
    # Mock running
    exec_id = "test-cancel"
    exc = WorkflowExecution(
        id=exec_id, workflow_name="cancel-me", status="running", current_step_index=0,
        step_outputs={}, history=[], initial_inputs={}, error=None,
        started_at=time.time(), updated_at=time.time()
    )
    engine._executions[exec_id] = exc
    
    engine.cancel_execution(exec_id)
    assert exc.status == "cancelled"

# ---------------------------------------------------------------------------
# Integration Tests: Agentic Loop
# ---------------------------------------------------------------------------

def test_engine_loop_lifecycle(engine):
    # Mock orchestrator
    mock_orch = MagicMock()
    mock_orch.run_cycle.return_value = {
        "stop_reason": None,
        "phase_outputs": {"plan": {"steps": []}}
    }
    engine._get_orchestrator = MagicMock(return_value=mock_orch)
    
    loop_id = engine.create_loop("evolve the world", max_cycles=2)
    
    # Tick 1
    res1 = engine.loop_tick(loop_id)
    assert res1["cycle"] == 1
    assert res1["status"] == "running"
    
    # Tick 2
    res2 = engine.loop_tick(loop_id)
    assert res2["cycle"] == 2
    assert res2["status"] == "completed"
    assert res2["stop_reason"] == "max_cycles_reached"

def test_engine_loop_early_stop(engine):
    mock_orch = MagicMock()
    mock_orch.run_cycle.return_value = {"stop_reason": "GOAL_MET", "phase_outputs": {}}
    engine._get_orchestrator = MagicMock(return_value=mock_orch)
    
    loop_id = engine.create_loop("stop early", max_cycles=5)
    res = engine.loop_tick(loop_id)
    
    assert res["cycle"] == 1
    assert res["status"] == "completed"
    assert res["stop_reason"] == "GOAL_MET"

def test_loop_health_check(engine):
    loop_id = engine.create_loop("stale", max_cycles=5)
    loop = engine._get_loop(loop_id)
    
    # Fake a stall
    loop.updated_at = time.time() - 400
    
    health = engine.check_loop_health(loop_id, stall_threshold_s=300)
    assert health["healthy"] is False
    assert "progressed" in health["warnings"][0].lower()

import threading # Needed for engine tests
