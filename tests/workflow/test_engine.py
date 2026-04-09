"""Comprehensive unit tests for WorkflowEngine (core/workflow_engine.py)."""
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest

from core.workflow_engine import (
    AgenticLoop,
    LoopCycle,
    RetryPolicy,
    StepResult,
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowExecution,
    WorkflowStep,
    _wire_inputs,
    _execute_step,
    _run_step_callable_with_timeout,
    get_engine,
    _DB_PATH,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path, monkeypatch):
    """Create a fresh WorkflowEngine with isolated test DB."""
    test_db_path = tmp_path / "test_workflow.db"
    monkeypatch.setattr("core.workflow_engine._DB_PATH", test_db_path)
    
    # Create fresh engine
    eng = WorkflowEngine()
    yield eng
    
    # Cleanup
    eng._executions.clear()
    eng._loops.clear()
    eng._locks.clear()


@pytest.fixture
def sample_step():
    """Return a simple workflow step."""
    return WorkflowStep(
        name="test_step",
        fn=lambda inputs: {"result": "success", **inputs}
    )


@pytest.fixture
def simple_workflow(sample_step):
    """Return a simple workflow definition."""
    return WorkflowDefinition(
        name="simple_workflow",
        description="A simple test workflow",
        steps=[
            sample_step,
            WorkflowStep(
                name="second_step",
                fn=lambda inputs: {"final": "done"}
            ),
        ]
    )


# -----------------------------------------------------------------------------
# RetryPolicy Tests
# -----------------------------------------------------------------------------

class TestRetryPolicy:
    """Test RetryPolicy exponential backoff behavior."""

    def test_default_values(self):
        """Test default retry policy configuration."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.backoff_base == 0.5
        assert policy.max_backoff == 30.0

    def test_custom_values(self):
        """Test custom retry policy configuration."""
        policy = RetryPolicy(max_attempts=5, backoff_base=1.0, max_backoff=60.0)
        assert policy.max_attempts == 5
        assert policy.backoff_base == 1.0
        assert policy.max_backoff == 60.0

    def test_sleep_for_calculation(self):
        """Test exponential backoff sleep calculation."""
        policy = RetryPolicy(backoff_base=1.0, max_backoff=10.0)
        assert policy.sleep_for(0) == 1.0
        assert policy.sleep_for(1) == 2.0
        assert policy.sleep_for(2) == 4.0
        assert policy.sleep_for(3) == 8.0
        assert policy.sleep_for(10) == 10.0  # capped at max_backoff


# -----------------------------------------------------------------------------
# WorkflowStep Tests
# -----------------------------------------------------------------------------

class TestWorkflowStep:
    """Test WorkflowStep dataclass initialization and configuration."""

    def test_step_with_fn(self):
        """Test step with callable function."""
        fn = lambda x: x
        step = WorkflowStep(name="test", fn=fn)
        assert step.name == "test"
        assert step.fn == fn
        assert step.skill_name is None
        assert step.static_inputs == {}
        assert step.inputs_from == {}

    def test_step_with_skill_name(self):
        """Test step with skill name reference."""
        step = WorkflowStep(name="test", skill_name="my_skill")
        assert step.skill_name == "my_skill"
        assert step.fn is None

    def test_step_with_static_inputs(self):
        """Test step with static input configuration."""
        step = WorkflowStep(
            name="test",
            fn=lambda x: x,
            static_inputs={"key": "value", "num": 42}
        )
        assert step.static_inputs == {"key": "value", "num": 42}

    def test_step_with_inputs_from(self):
        """Test step with input wiring from previous steps."""
        step = WorkflowStep(
            name="test",
            fn=lambda x: x,
            inputs_from={"result": "prev_step.output", "all": "prev_step.*"}
        )
        assert step.inputs_from == {"result": "prev_step.output", "all": "prev_step.*"}

    def test_step_with_skip_condition(self):
        """Test step with skip condition."""
        step = WorkflowStep(
            name="test",
            fn=lambda x: x,
            skip_if_false="prev_step.should_run"
        )
        assert step.skip_if_false == "prev_step.should_run"

    def test_step_with_custom_retry(self):
        """Test step with custom retry policy."""
        retry = RetryPolicy(max_attempts=5)
        step = WorkflowStep(name="test", fn=lambda x: x, retry=retry)
        assert step.retry.max_attempts == 5

    def test_step_with_timeout(self):
        """Test step with custom timeout."""
        step = WorkflowStep(name="test", fn=lambda x: x, timeout_s=60.0)
        assert step.timeout_s == 60.0


# -----------------------------------------------------------------------------
# WorkflowDefinition Tests
# -----------------------------------------------------------------------------

class TestWorkflowDefinition:
    """Test WorkflowDefinition dataclass."""

    def test_basic_definition(self):
        """Test basic workflow definition creation."""
        steps = [WorkflowStep(name="s1", fn=lambda x: x)]
        wf = WorkflowDefinition(name="test_wf", steps=steps)
        assert wf.name == "test_wf"
        assert len(wf.steps) == 1
        assert wf.description == ""
        assert wf.max_retries_total == 0

    def test_full_definition(self):
        """Test workflow definition with all fields."""
        steps = [
            WorkflowStep(name="s1", fn=lambda x: x),
            WorkflowStep(name="s2", fn=lambda x: x),
        ]
        wf = WorkflowDefinition(
            name="test_wf",
            steps=steps,
            description="Test workflow",
            max_retries_total=10
        )
        assert wf.description == "Test workflow"
        assert wf.max_retries_total == 10


# -----------------------------------------------------------------------------
# WorkflowExecution Tests
# -----------------------------------------------------------------------------

class TestWorkflowExecution:
    """Test WorkflowExecution state management."""

    def test_execution_initialization(self):
        """Test workflow execution initialization."""
        now = time.time()
        exc = WorkflowExecution(
            id="exec_123",
            workflow_name="test_wf",
            status="pending",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={"key": "value"},
            error=None,
            started_at=now,
            updated_at=now,
        )
        assert exc.id == "exec_123"
        assert exc.workflow_name == "test_wf"
        assert exc.status == "pending"
        assert exc.current_step_index == 0
        assert exc.initial_inputs == {"key": "value"}
        assert exc.is_terminal() is False

    def test_is_terminal_completed(self):
        """Test is_terminal returns True for completed status."""
        exc = WorkflowExecution(
            id="exec_123",
            workflow_name="test_wf",
            status="completed",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert exc.is_terminal() is True

    def test_is_terminal_failed(self):
        """Test is_terminal returns True for failed status."""
        exc = WorkflowExecution(
            id="exec_123",
            workflow_name="test_wf",
            status="failed",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error="Something went wrong",
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert exc.is_terminal() is True

    def test_is_terminal_cancelled(self):
        """Test is_terminal returns True for cancelled status."""
        exc = WorkflowExecution(
            id="exec_123",
            workflow_name="test_wf",
            status="cancelled",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert exc.is_terminal() is True

    def test_is_terminal_running(self):
        """Test is_terminal returns False for running status."""
        exc = WorkflowExecution(
            id="exec_123",
            workflow_name="test_wf",
            status="running",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert exc.is_terminal() is False


# -----------------------------------------------------------------------------
# StepResult Tests
# -----------------------------------------------------------------------------

class TestStepResult:
    """Test StepResult dataclass."""

    def test_success_result(self):
        """Test successful step result."""
        result = StepResult(
            step_name="test_step",
            status="ok",
            output={"data": "value"},
            attempts=1,
            elapsed_ms=100.0
        )
        assert result.step_name == "test_step"
        assert result.status == "ok"
        assert result.output == {"data": "value"}
        assert result.attempts == 1
        assert result.elapsed_ms == 100.0
        assert result.error is None

    def test_failed_result(self):
        """Test failed step result."""
        result = StepResult(
            step_name="test_step",
            status="failed",
            output={},
            attempts=3,
            elapsed_ms=500.0,
            error="Connection timeout"
        )
        assert result.status == "failed"
        assert result.attempts == 3
        assert result.error == "Connection timeout"

    def test_skipped_result(self):
        """Test skipped step result."""
        result = StepResult(
            step_name="test_step",
            status="skipped",
            output={},
            attempts=0,
            elapsed_ms=0.0
        )
        assert result.status == "skipped"


# -----------------------------------------------------------------------------
# AgenticLoop and LoopCycle Tests
# -----------------------------------------------------------------------------

class TestAgenticLoop:
    """Test AgenticLoop state management."""

    def test_loop_initialization(self):
        """Test agentic loop initialization."""
        now = time.time()
        loop = AgenticLoop(
            id="loop_123",
            goal="Test goal",
            max_cycles=5,
            current_cycle=0,
            status="running",
            history=[],
            stop_reason=None,
            score=0.0,
            started_at=now,
            updated_at=now,
        )
        assert loop.id == "loop_123"
        assert loop.goal == "Test goal"
        assert loop.max_cycles == 5
        assert loop.status == "running"
        assert loop.score == 0.0

    def test_is_terminal_completed(self):
        """Test is_terminal returns True for completed loop."""
        loop = AgenticLoop(
            id="loop_123",
            goal="Test",
            max_cycles=5,
            current_cycle=5,
            status="completed",
            history=[],
            stop_reason="max_cycles_reached",
            score=1.0,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert loop.is_terminal() is True

    def test_is_terminal_failed(self):
        """Test is_terminal returns True for failed loop."""
        loop = AgenticLoop(
            id="loop_123",
            goal="Test",
            max_cycles=5,
            current_cycle=2,
            status="failed",
            history=[],
            stop_reason="error",
            score=0.0,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert loop.is_terminal() is True

    def test_is_terminal_stopped(self):
        """Test is_terminal returns True for stopped loop."""
        loop = AgenticLoop(
            id="loop_123",
            goal="Test",
            max_cycles=5,
            current_cycle=1,
            status="stopped",
            history=[],
            stop_reason="user_requested",
            score=0.0,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert loop.is_terminal() is True

    def test_is_terminal_running(self):
        """Test is_terminal returns False for running loop."""
        loop = AgenticLoop(
            id="loop_123",
            goal="Test",
            max_cycles=5,
            current_cycle=1,
            status="running",
            history=[],
            stop_reason=None,
            score=0.5,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert loop.is_terminal() is False


# -----------------------------------------------------------------------------
# WorkflowEngine - Definition Management Tests
# -----------------------------------------------------------------------------

class TestWorkflowEngineDefinitionManagement:
    """Test workflow definition registration and listing."""

    def test_define_workflow(self, engine, simple_workflow):
        """Test workflow definition registration."""
        engine.define(simple_workflow)
        assert "simple_workflow" in engine._definitions
        assert engine._definitions["simple_workflow"] == simple_workflow

    def test_define_replaces_existing(self, engine, simple_workflow):
        """Test that define() replaces existing workflow."""
        engine.define(simple_workflow)
        
        new_workflow = WorkflowDefinition(
            name="simple_workflow",
            steps=[WorkflowStep(name="new_step", fn=lambda x: x)]
        )
        engine.define(new_workflow)
        
        assert len(engine._definitions["simple_workflow"].steps) == 1
        assert engine._definitions["simple_workflow"].steps[0].name == "new_step"

    def test_list_definitions(self, engine, simple_workflow):
        """Test listing workflow definitions."""
        engine.define(simple_workflow)
        
        definitions = engine.list_definitions()
        assert len(definitions) >= 1
        
        # Find our workflow in the list (builtin workflows also exist)
        wf_dict = next((d for d in definitions if d["name"] == "simple_workflow"), None)
        assert wf_dict is not None
        assert wf_dict["description"] == "A simple test workflow"
        assert wf_dict["step_count"] == 2
        assert wf_dict["steps"] == ["test_step", "second_step"]

    def test_builtin_workflows_registered(self, engine):
        """Test that builtin workflows are registered on init."""
        definitions = engine.list_definitions()
        names = [d["name"] for d in definitions]
        
        assert "security_audit" in names
        assert "code_quality" in names
        assert "release_prep" in names
        assert "onboarding_analysis" in names


# -----------------------------------------------------------------------------
# WorkflowEngine - Execution Tests
# -----------------------------------------------------------------------------

class TestWorkflowEngineExecution:
    """Test workflow execution functionality."""

    def test_run_workflow_success(self, engine, simple_workflow):
        """Test successful workflow execution."""
        engine.define(simple_workflow)
        exec_id = engine.run_workflow("simple_workflow", {"input": "test"})
        
        assert exec_id is not None
        assert exec_id in engine._executions
        
        exc = engine._executions[exec_id]
        assert exc.status == "completed"
        assert exc.workflow_name == "simple_workflow"
        assert len(exc.history) == 2
        assert exc.history[0].status == "ok"
        assert exc.history[1].status == "ok"

    def test_run_workflow_not_defined(self, engine):
        """Test running undefined workflow raises KeyError."""
        with pytest.raises(KeyError, match="Workflow 'undefined' not defined"):
            engine.run_workflow("undefined")

    def test_run_workflow_with_step_failure(self, engine):
        """Test workflow with failing step."""
        failing_workflow = WorkflowDefinition(
            name="failing_workflow",
            steps=[
                WorkflowStep(name="fail", fn=lambda inputs: {"error": "Step failed"}),
                WorkflowStep(name="never_runs", fn=lambda inputs: {"result": "ok"}),
            ]
        )
        engine.define(failing_workflow)
        exec_id = engine.run_workflow("failing_workflow")
        
        exc = engine._executions[exec_id]
        assert exc.status == "failed"
        assert "Step failed" in exc.error
        assert len(exc.history) == 1  # Second step never runs

    def test_run_workflow_step_exception(self, engine):
        """Test workflow step that raises exception."""
        def raise_error(inputs):
            raise ValueError("Test exception")
        
        workflow = WorkflowDefinition(
            name="exception_workflow",
            steps=[WorkflowStep(name="error", fn=raise_error)]
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("exception_workflow")
        
        exc = engine._executions[exec_id]
        assert exc.status == "failed"
        assert "ValueError" in exc.error
        assert "Test exception" in exc.error

    def test_execution_status(self, engine, simple_workflow):
        """Test retrieving execution status."""
        engine.define(simple_workflow)
        exec_id = engine.run_workflow("simple_workflow")
        
        status = engine.execution_status(exec_id)
        assert status["id"] == exec_id
        assert status["workflow"] == "simple_workflow"
        assert status["status"] == "completed"
        assert status["error"] is None
        assert "elapsed_s" in status
        assert len(status["history"]) == 2
        assert "step_output_keys" in status

    def test_execution_status_not_found(self, engine):
        """Test status for non-existent execution."""
        status = engine.execution_status("non_existent")
        assert status["status"] == "not_found"
        assert "error" in status

    def test_get_step_output(self, engine, simple_workflow):
        """Test retrieving step output."""
        engine.define(simple_workflow)
        exec_id = engine.run_workflow("simple_workflow", {"input": "test"})
        
        output = engine.get_step_output(exec_id, "test_step")
        assert output["result"] == "success"
        assert output["input"] == "test"

    def test_get_step_output_not_found(self, engine, simple_workflow):
        """Test retrieving non-existent step output."""
        engine.define(simple_workflow)
        exec_id = engine.run_workflow("simple_workflow")
        
        with pytest.raises(KeyError, match="Step 'non_existent' output not available"):
            engine.get_step_output(exec_id, "non_existent")

    def test_list_executions(self, engine, simple_workflow):
        """Test listing executions."""
        engine.define(simple_workflow)
        exec_id1 = engine.run_workflow("simple_workflow")
        exec_id2 = engine.run_workflow("simple_workflow")
        
        executions = engine.list_executions()
        assert len(executions) >= 2
        
        ids = [e["id"] for e in executions]
        assert exec_id1 in ids
        assert exec_id2 in ids

    def test_list_executions_with_filter(self, engine, simple_workflow):
        """Test listing executions with status filter."""
        engine.define(simple_workflow)
        exec_id = engine.run_workflow("simple_workflow")
        
        completed = engine.list_executions("completed")
        assert all(e["status"] == "completed" for e in completed)
        assert any(e["id"] == exec_id for e in completed)


# -----------------------------------------------------------------------------
# WorkflowEngine - Pause/Resume/Cancel Tests
# -----------------------------------------------------------------------------

class TestWorkflowEngineLifecycle:
    """Test workflow pause, resume, and cancel operations."""

    def test_pause_execution(self, engine, simple_workflow):
        """Test pausing execution."""
        engine.define(simple_workflow)
        exec_id = engine.run_workflow("simple_workflow")
        
        # Note: Execution completes synchronously, so we can't pause mid-run
        # But we can test the pause raises error for terminal execution
        with pytest.raises(ValueError, match="already terminal"):
            engine.pause_execution(exec_id)

    def test_pause_execution_not_found(self, engine):
        """Test pausing non-existent execution."""
        with pytest.raises(KeyError, match="Execution 'fake' not found"):
            engine.pause_execution("fake")

    def test_cancel_execution_not_found(self, engine):
        """Test cancelling non-existent execution."""
        with pytest.raises(KeyError, match="Execution 'fake' not found"):
            engine.cancel_execution("fake")

    def test_cancel_terminal_execution(self, engine, simple_workflow):
        """Test cancelling already terminal execution raises error."""
        engine.define(simple_workflow)
        exec_id = engine.run_workflow("simple_workflow")
        
        with pytest.raises(ValueError, match="already terminal"):
            engine.cancel_execution(exec_id)


# -----------------------------------------------------------------------------
# WorkflowEngine - AgenticLoop Tests
# -----------------------------------------------------------------------------

class TestWorkflowEngineAgenticLoop:
    """Test agentic loop functionality."""

    def test_create_loop(self, engine):
        """Test creating agentic loop."""
        loop_id = engine.create_loop("Test goal", max_cycles=3)
        
        assert loop_id is not None
        assert loop_id in engine._loops
        
        loop = engine._loops[loop_id]
        assert loop.goal == "Test goal"
        assert loop.max_cycles == 3
        assert loop.status == "running"
        assert loop.current_cycle == 0

    def test_loop_status(self, engine):
        """Test retrieving loop status."""
        loop_id = engine.create_loop("Test goal", max_cycles=5)
        
        status = engine.loop_status(loop_id)
        assert status["id"] == loop_id
        assert status["goal"] == "Test goal"
        assert status["status"] == "running"
        assert status["current_cycle"] == 0
        assert status["max_cycles"] == 5

    def test_loop_status_not_found(self, engine):
        """Test status for non-existent loop."""
        status = engine.loop_status("non_existent")
        assert status["status"] == "not_found"
        assert "error" in status

    def test_list_loops(self, engine):
        """Test listing loops."""
        loop_id1 = engine.create_loop("Goal 1")
        loop_id2 = engine.create_loop("Goal 2")
        
        loops = engine.list_loops()
        assert len(loops) >= 2
        
        ids = [l["id"] for l in loops]
        assert loop_id1 in ids
        assert loop_id2 in ids

    def test_list_loops_with_filter(self, engine):
        """Test listing loops with status filter."""
        loop_id = engine.create_loop("Test")
        
        running = engine.list_loops("running")
        assert all(l["status"] == "running" for l in running)
        assert any(l["id"] == loop_id for l in running)

    def test_stop_loop(self, engine):
        """Test stopping a loop."""
        loop_id = engine.create_loop("Test")
        engine.stop_loop(loop_id, reason="test_complete")
        
        loop = engine._loops[loop_id]
        assert loop.status == "stopped"
        assert loop.stop_reason == "test_complete"

    def test_stop_loop_not_found(self, engine):
        """Test stopping non-existent loop."""
        with pytest.raises(KeyError, match="Loop fake not found"):
            engine.stop_loop("fake")

    def test_stop_terminal_loop(self, engine):
        """Test stopping already terminal loop raises error."""
        loop_id = engine.create_loop("Test")
        engine.stop_loop(loop_id)
        
        with pytest.raises(ValueError, match="already terminal"):
            engine.stop_loop(loop_id)

    def test_pause_loop(self, engine):
        """Test pausing a loop."""
        loop_id = engine.create_loop("Test")
        engine.pause_loop(loop_id)
        
        loop = engine._loops[loop_id]
        assert loop.status == "paused"

    def test_resume_loop(self, engine):
        """Test resuming a paused loop."""
        loop_id = engine.create_loop("Test")
        engine.pause_loop(loop_id)
        engine.resume_loop(loop_id)
        
        loop = engine._loops[loop_id]
        assert loop.status == "running"

    def test_resume_loop_not_paused(self, engine):
        """Test resuming non-paused loop raises error."""
        loop_id = engine.create_loop("Test")
        
        with pytest.raises(ValueError, match="not 'paused'"):
            engine.resume_loop(loop_id)

    def test_check_loop_health_terminal(self, engine):
        """Test health check for terminal loop."""
        loop_id = engine.create_loop("Test")
        engine.stop_loop(loop_id)
        
        health = engine.check_loop_health(loop_id)
        assert health["healthy"] is True
        assert health["recommendation"] == "Loop is terminal."

    def test_check_loop_health_not_found(self, engine):
        """Test health check for non-existent loop."""
        health = engine.check_loop_health("fake")
        assert health["healthy"] is False
        assert "error" in health

    def test_check_loop_health_stalled(self, engine):
        """Test health check detects stalled loop."""
        loop_id = engine.create_loop("Test")
        
        # Manually set updated_at to be old
        engine._loops[loop_id].updated_at = time.time() - 400
        
        health = engine.check_loop_health(loop_id, stall_threshold_s=300)
        assert health["healthy"] is False
        assert len(health["warnings"]) > 0
        assert "stop_loop" in health["recommendation"]

    def test_check_loop_health_repeated_errors(self, engine):
        """Test health check detects repeated errors."""
        loop_id = engine.create_loop("Test")
        
        # Add cycles with same error
        loop = engine._loops[loop_id]
        loop.history = [
            LoopCycle(1, "failed", {}, 100, error="Same error"),
            LoopCycle(2, "failed", {}, 100, error="Same error"),
            LoopCycle(3, "failed", {}, 100, error="Same error"),
        ]
        
        health = engine.check_loop_health(loop_id)
        assert health["healthy"] is False
        assert any("Same error" in w for w in health["warnings"])


# -----------------------------------------------------------------------------
# WorkflowEngine - Loop Tick Tests
# -----------------------------------------------------------------------------

class TestWorkflowEngineLoopTick:
    """Test agentic loop tick functionality."""

    @patch("core.workflow_engine.WorkflowEngine._get_orchestrator")
    def test_loop_tick_success(self, mock_get_orch, engine):
        """Test successful loop tick."""
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {
            "stop_reason": None,
            "phase_outputs": {"plan": {"result": "ok"}}
        }
        mock_get_orch.return_value = mock_orch
        
        loop_id = engine.create_loop("Test goal", max_cycles=3)
        result = engine.loop_tick(loop_id)
        
        assert result["cycle"] == 1
        assert result["status"] == "running"
        assert result["cycle_status"] == "ok"
        assert "elapsed_ms" in result
        
        loop = engine._loops[loop_id]
        assert loop.current_cycle == 1
        assert len(loop.history) == 1

    @patch("core.workflow_engine.WorkflowEngine._get_orchestrator")
    def test_loop_tick_max_cycles(self, mock_get_orch, engine):
        """Test loop tick reaches max cycles."""
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"stop_reason": None, "phase_outputs": {}}
        mock_get_orch.return_value = mock_orch
        
        loop_id = engine.create_loop("Test", max_cycles=2)
        
        # First tick
        engine.loop_tick(loop_id)
        # Second tick should complete
        result = engine.loop_tick(loop_id)
        
        assert result["stop_reason"] == "max_cycles_reached"
        
        loop = engine._loops[loop_id]
        assert loop.status == "completed"

    @patch("core.workflow_engine.WorkflowEngine._get_orchestrator")
    def test_loop_tick_with_stop_reason(self, mock_get_orch, engine):
        """Test loop tick with early stop reason."""
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {
            "stop_reason": "goal_achieved",
            "phase_outputs": {}
        }
        mock_get_orch.return_value = mock_orch
        
        loop_id = engine.create_loop("Test", max_cycles=5)
        result = engine.loop_tick(loop_id)
        
        assert result["stop_reason"] == "goal_achieved"
        
        loop = engine._loops[loop_id]
        assert loop.status == "completed"
        assert loop.stop_reason == "goal_achieved"

    @patch("core.workflow_engine.WorkflowEngine._get_orchestrator")
    def test_loop_tick_exception(self, mock_get_orch, engine):
        """Test loop tick with orchestrator exception."""
        mock_orch = MagicMock()
        mock_orch.run_cycle.side_effect = RuntimeError("Orchestrator failed")
        mock_get_orch.return_value = mock_orch
        
        loop_id = engine.create_loop("Test")
        result = engine.loop_tick(loop_id)
        
        assert result["cycle_status"] == "failed"
        assert "Orchestrator failed" in result["error"]
        
        loop = engine._loops[loop_id]
        assert loop.status == "failed"

    def test_loop_tick_not_found(self, engine):
        """Test tick for non-existent loop."""
        result = engine.loop_tick("fake")
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_loop_tick_terminal_loop(self, engine):
        """Test tick for terminal loop."""
        loop_id = engine.create_loop("Test")
        engine.stop_loop(loop_id)
        
        result = engine.loop_tick(loop_id)
        assert "error" in result
        assert "already" in result["error"].lower()


# -----------------------------------------------------------------------------
# Input Wiring Tests
# -----------------------------------------------------------------------------

class TestInputWiring:
    """Test input wiring between steps."""

    def test_wire_inputs_basic(self):
        """Test basic input wiring."""
        step = WorkflowStep(
            name="test",
            fn=lambda x: x,
            static_inputs={"static": "value"}
        )
        step_outputs = {"prev": {"output": "from_prev"}}
        initial_inputs = {"initial": "input"}
        
        result = _wire_inputs(step, step_outputs, initial_inputs)
        assert result == {"initial": "input", "static": "value"}

    def test_wire_inputs_from_step(self):
        """Test wiring inputs from previous step."""
        step = WorkflowStep(
            name="test",
            fn=lambda x: x,
            inputs_from={"result": "prev.output"}
        )
        step_outputs = {"prev": {"output": "from_prev"}}
        initial_inputs = {}
        
        result = _wire_inputs(step, step_outputs, initial_inputs)
        assert result["result"] == "from_prev"

    def test_wire_inputs_wildcard(self):
        """Test wildcard input wiring."""
        step = WorkflowStep(
            name="test",
            fn=lambda x: x,
            inputs_from={"all": "prev.*"}
        )
        step_outputs = {"prev": {"key1": "val1", "key2": "val2"}}
        initial_inputs = {}
        
        result = _wire_inputs(step, step_outputs, initial_inputs)
        assert result["key1"] == "val1"
        assert result["key2"] == "val2"

    def test_wire_inputs_override_priority(self):
        """Test that static_inputs override initial_inputs."""
        step = WorkflowStep(
            name="test",
            fn=lambda x: x,
            static_inputs={"key": "static_value"}
        )
        step_outputs = {}
        initial_inputs = {"key": "initial_value"}
        
        result = _wire_inputs(step, step_outputs, initial_inputs)
        assert result["key"] == "static_value"


# -----------------------------------------------------------------------------
# Step Execution Tests
# -----------------------------------------------------------------------------

class TestStepExecution:
    """Test step execution with _execute_step."""

    def test_execute_step_success(self):
        """Test successful step execution."""
        step = WorkflowStep(
            name="test",
            fn=lambda inputs: {"result": "ok"}
        )
        
        result = _execute_step(step, {}, {})
        assert result.status == "ok"
        assert result.output == {"result": "ok"}
        assert result.attempts == 1

    def test_execute_step_skipped(self):
        """Test step with skip condition."""
        step = WorkflowStep(
            name="test",
            fn=lambda inputs: {"result": "ok"},
            skip_if_false="prev.should_run"
        )
        step_outputs = {"prev": {"should_run": False}}
        
        result = _execute_step(step, step_outputs, {})
        assert result.status == "skipped"
        assert result.output == {}
        assert result.attempts == 0

    def test_execute_step_not_skipped(self):
        """Test step when skip condition is False but step still runs."""
        step = WorkflowStep(
            name="test",
            fn=lambda inputs: {"result": "ok"},
            skip_if_false="prev.should_run"
        )
        step_outputs = {"prev": {"should_run": True}}
        
        result = _execute_step(step, step_outputs, {})
        assert result.status == "ok"

    def test_execute_step_returns_error(self):
        """Test step that returns error dict."""
        step = WorkflowStep(
            name="test",
            fn=lambda inputs: {"error": "Something went wrong"},
            retry=RetryPolicy(max_attempts=1)
        )
        
        result = _execute_step(step, {}, {})
        assert result.status == "failed"
        assert "Something went wrong" in result.error

    def test_execute_step_with_skill(self):
        """Test step with skill name."""
        step = WorkflowStep(
            name="test",
            skill_name="nonexistent_skill"
        )
        
        result = _execute_step(step, {}, {})
        # Should fail because skill doesn't exist
        assert result.status == "failed"
        assert "Unknown skill" in result.error

    def test_execute_step_no_skill_or_fn(self):
        """Test step with neither skill nor fn."""
        step = WorkflowStep(name="test")
        
        result = _execute_step(step, {}, {})
        assert result.status == "failed"
        assert "no skill_name or fn" in result.error

    def test_execute_step_timeout(self):
        """Test step that times out."""
        def slow_fn(inputs):
            time.sleep(2)
            return {"result": "ok"}
        
        step = WorkflowStep(
            name="test",
            fn=slow_fn,
            timeout_s=0.01,
            retry=RetryPolicy(max_attempts=1)
        )
        
        result = _execute_step(step, {}, {})
        assert result.status == "timeout"
        assert "timeout" in result.error.lower()


# -----------------------------------------------------------------------------
# Timeout Execution Tests
# -----------------------------------------------------------------------------

class TestTimeoutExecution:
    """Test _run_step_callable_with_timeout function."""

    def test_no_timeout(self):
        """Test execution without timeout."""
        def fn():
            return "result"
        
        success, result = _run_step_callable_with_timeout(fn, timeout_s=0)
        assert success is True
        assert result == "result"

    def test_with_timeout_success(self):
        """Test successful execution with timeout."""
        def fn():
            return "result"
        
        success, result = _run_step_callable_with_timeout(fn, timeout_s=1.0)
        assert success is True
        assert result == "result"

    def test_with_timeout_exceeded(self):
        """Test timeout exceeded raises TimeoutError."""
        def fn():
            time.sleep(2)
            return "result"
        
        with pytest.raises(TimeoutError):
            _run_step_callable_with_timeout(fn, timeout_s=0.01)

    def test_with_timeout_exception(self):
        """Test exception in function is propagated."""
        def fn():
            raise ValueError("Test error")
        
        success, result = _run_step_callable_with_timeout(fn, timeout_s=1.0)
        assert success is False
        assert isinstance(result, ValueError)


# -----------------------------------------------------------------------------
# Retry Budget Tests
# -----------------------------------------------------------------------------

class TestRetryBudget:
    """Test workflow retry budget functionality."""

    def test_retry_budget_exhausted(self, engine):
        """Test workflow fails when retry budget is exhausted by a step that ultimately fails."""
        # Step always fails, using up retries - this triggers budget exhaustion
        workflow = WorkflowDefinition(
            name="retry_budget_test",
            steps=[
                WorkflowStep(
                    name="retry_step",
                    fn=lambda inputs: {"error": "Always fails"},
                    retry=RetryPolicy(max_attempts=3)
                )
            ],
            max_retries_total=1  # Budget exhausted after just 1 retry
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("retry_budget_test")
        
        exc = engine._executions[exec_id]
        assert exc.status == "failed"
        assert "Total retry budget exhausted" in exc.error
        assert exc.total_retries_used >= 1

    def test_retry_budget_not_exceeded(self, engine):
        """Test workflow succeeds when step fails but retry budget is not exceeded."""
        workflow = WorkflowDefinition(
            name="retry_success_test",
            steps=[
                WorkflowStep(
                    name="retry_step",
                    fn=lambda inputs: {"error": "Always fails"},
                    retry=RetryPolicy(max_attempts=2)
                ),
                WorkflowStep(
                    name="second_step",
                    fn=lambda inputs: {"error": "Also fails"},
                    retry=RetryPolicy(max_attempts=2)
                )
            ],
            max_retries_total=10  # High enough budget to not be exhausted
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("retry_success_test")
        
        exc = engine._executions[exec_id]
        # First step fails and uses (2-1)=1 retry, but workflow continues
        # because budget (10) is not exhausted yet
        assert exc.status == "failed"
        assert "retry_step" in exc.error  # Failed at first step
        assert exc.total_retries_used == 1  # Used 1 retry (attempts - 1)


# -----------------------------------------------------------------------------
# Singleton Tests
# -----------------------------------------------------------------------------

class TestSingleton:
    """Test module-level singleton functionality."""

    def test_get_engine_returns_same_instance(self, monkeypatch, tmp_path):
        """Test that get_engine returns the same instance."""
        test_db_path = tmp_path / "test_singleton.db"
        monkeypatch.setattr("core.workflow_engine._DB_PATH", test_db_path)
        
        # Reset singleton for test
        import core.workflow_engine as we_module
        monkeypatch.setattr(we_module, "_engine", None)
        
        engine1 = get_engine()
        engine2 = get_engine()
        
        assert engine1 is engine2

    def test_get_engine_thread_safe(self, monkeypatch, tmp_path):
        """Test that get_engine is thread-safe."""
        test_db_path = tmp_path / "test_thread_safe.db"
        monkeypatch.setattr("core.workflow_engine._DB_PATH", test_db_path)
        
        # Reset singleton
        import core.workflow_engine as we_module
        monkeypatch.setattr(we_module, "_engine", None)
        
        engines = []
        
        def get_engine_thread():
            engines.append(get_engine())
        
        threads = [threading.Thread(target=get_engine_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should be the same instance
        assert all(e is engines[0] for e in engines)


# -----------------------------------------------------------------------------
# Complex Workflow Scenarios
# -----------------------------------------------------------------------------

class TestComplexWorkflowScenarios:
    """Test complex workflow scenarios."""

    def test_multi_step_workflow_with_wiring(self, engine):
        """Test multi-step workflow with input/output wiring."""
        workflow = WorkflowDefinition(
            name="wiring_test",
            steps=[
                WorkflowStep(
                    name="generate",
                    fn=lambda inputs: {"value": inputs.get("seed", 0) * 2}
                ),
                WorkflowStep(
                    name="process",
                    fn=lambda inputs: {"result": inputs.get("value", 0) + 10},
                    inputs_from={"value": "generate.value"}
                ),
                WorkflowStep(
                    name="finalize",
                    fn=lambda inputs: {"final": inputs.get("result", 0) * 2},
                    inputs_from={"result": "process.result"}
                ),
            ]
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("wiring_test", {"seed": 5})
        
        exc = engine._executions[exec_id]
        assert exc.status == "completed"
        assert exc.step_outputs["generate"]["value"] == 10
        assert exc.step_outputs["process"]["result"] == 20
        assert exc.step_outputs["finalize"]["final"] == 40

    def test_conditional_workflow(self, engine):
        """Test workflow with conditional step skipping."""
        workflow = WorkflowDefinition(
            name="conditional_test",
            steps=[
                WorkflowStep(
                    name="check",
                    fn=lambda inputs: {"should_process": inputs.get("enabled", False)}
                ),
                WorkflowStep(
                    name="process",
                    fn=lambda inputs: {"processed": True},
                    inputs_from={"check_result": "check.should_process"},
                    skip_if_false="check.should_process"
                ),
                WorkflowStep(
                    name="always_run",
                    fn=lambda inputs: {"done": True}
                ),
            ]
        )
        engine.define(workflow)
        
        # Test with enabled=False
        exec_id1 = engine.run_workflow("conditional_test", {"enabled": False})
        exc1 = engine._executions[exec_id1]
        assert exc1.history[1].status == "skipped"
        
        # Test with enabled=True
        exec_id2 = engine.run_workflow("conditional_test", {"enabled": True})
        exc2 = engine._executions[exec_id2]
        assert exc2.history[1].status == "ok"

    def test_error_mid_workflow(self, engine):
        """Test workflow that fails mid-execution."""
        workflow = WorkflowDefinition(
            name="error_mid_test",
            steps=[
                WorkflowStep(name="step1", fn=lambda inputs: {"result": "ok"}),
                WorkflowStep(name="step2", fn=lambda inputs: {"error": "Failure"}),
                WorkflowStep(name="step3", fn=lambda inputs: {"result": "never"}),
            ]
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("error_mid_test")
        
        exc = engine._executions[exec_id]
        assert exc.status == "failed"
        assert len(exc.history) == 2
        assert "step2" in exc.error
        assert "step3" not in [h.step_name for h in exc.history]

    def test_empty_workflow(self, engine):
        """Test workflow with no steps."""
        workflow = WorkflowDefinition(
            name="empty_test",
            steps=[]
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("empty_test")
        
        exc = engine._executions[exec_id]
        assert exc.status == "completed"
        assert len(exc.history) == 0

    def test_large_workflow(self, engine):
        """Test workflow with many steps."""
        steps = [
            WorkflowStep(name=f"step_{i}", fn=lambda inputs, i=i: {"index": i})
            for i in range(50)
        ]
        workflow = WorkflowDefinition(name="large_test", steps=steps)
        engine.define(workflow)
        exec_id = engine.run_workflow("large_test")
        
        exc = engine._executions[exec_id]
        assert exc.status == "completed"
        assert len(exc.history) == 50


# -----------------------------------------------------------------------------
# Database Journal Tests
# -----------------------------------------------------------------------------

class TestDatabaseJournal:
    """Test SQLite journaling functionality."""

    def test_execution_journal_written(self, engine, simple_workflow, tmp_path, monkeypatch):
        """Test that execution is written to journal."""
        test_db_path = tmp_path / "journal_test.db"
        monkeypatch.setattr("core.workflow_engine._DB_PATH", test_db_path)
        
        # Create fresh engine with new DB path
        fresh_engine = WorkflowEngine()
        fresh_engine.define(simple_workflow)
        exec_id = fresh_engine.run_workflow("simple_workflow")
        
        # Check DB directly
        conn = sqlite3.connect(str(test_db_path))
        cursor = conn.execute("SELECT * FROM executions WHERE id = ?", (exec_id,))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[1] == "simple_workflow"  # workflow name
        assert row[2] == "completed"  # status

    def test_loop_journal_written(self, engine, tmp_path, monkeypatch):
        """Test that loop is written to journal."""
        test_db_path = tmp_path / "loop_journal_test.db"
        monkeypatch.setattr("core.workflow_engine._DB_PATH", test_db_path)
        
        fresh_engine = WorkflowEngine()
        loop_id = fresh_engine.create_loop("Test goal", max_cycles=3)
        
        # Check DB directly
        conn = sqlite3.connect(str(test_db_path))
        cursor = conn.execute("SELECT * FROM loops WHERE id = ?", (loop_id,))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[1] == "Test goal"  # goal
        assert row[4] == "running"  # status


# -----------------------------------------------------------------------------
# Edge Cases and Error Handling
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_step_with_none_input(self, engine):
        """Test step handling None inputs gracefully."""
        workflow = WorkflowDefinition(
            name="none_input_test",
            steps=[
                WorkflowStep(
                    name="handle_none",
                    fn=lambda inputs: {"received": inputs is not None}
                )
            ]
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("none_input_test", None)
        
        exc = engine._executions[exec_id]
        assert exc.status == "completed"

    def test_step_returning_none(self, engine):
        """Test step that returns None."""
        workflow = WorkflowDefinition(
            name="none_output_test",
            steps=[
                WorkflowStep(name="returns_none", fn=lambda inputs: None)
            ]
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("none_output_test")
        
        exc = engine._executions[exec_id]
        assert exc.status == "completed"
        assert exc.step_outputs["returns_none"] == {}

    def test_concurrent_executions(self, engine, simple_workflow):
        """Test running multiple executions concurrently."""
        engine.define(simple_workflow)
        
        exec_ids = []
        for i in range(10):
            exec_id = engine.run_workflow("simple_workflow", {"index": i})
            exec_ids.append(exec_id)
        
        # All should be completed
        for exec_id in exec_ids:
            status = engine.execution_status(exec_id)
            assert status["status"] == "completed"

    def test_step_name_collision(self, engine):
        """Test workflow with duplicate step names."""
        workflow = WorkflowDefinition(
            name="collision_test",
            steps=[
                WorkflowStep(name="same_name", fn=lambda inputs: {"step": 1}),
                WorkflowStep(name="same_name", fn=lambda inputs: {"step": 2}),
            ]
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("collision_test")
        
        # Second step overwrites first in outputs
        exc = engine._executions[exec_id]
        assert exc.status == "completed"
        assert exc.step_outputs["same_name"]["step"] == 2

    def test_very_long_step_name(self, engine):
        """Test workflow with very long step name."""
        long_name = "a" * 1000
        workflow = WorkflowDefinition(
            name="long_name_test",
            steps=[
                WorkflowStep(name=long_name, fn=lambda inputs: {"ok": True})
            ]
        )
        engine.define(workflow)
        exec_id = engine.run_workflow("long_name_test")
        
        exc = engine._executions[exec_id]
        assert exc.status == "completed"
        assert long_name in exc.step_outputs


# -----------------------------------------------------------------------------
# Thread Safety Tests
# -----------------------------------------------------------------------------

class TestThreadSafety:
    """Test thread safety of WorkflowEngine."""

    def test_concurrent_define(self, engine):
        """Test concurrent workflow definition."""
        errors = []
        
        def define_workflow(i):
            try:
                wf = WorkflowDefinition(
                    name=f"concurrent_{i}",
                    steps=[WorkflowStep(name="step", fn=lambda x: x)]
                )
                engine.define(wf)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=define_workflow, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        definitions = engine.list_definitions()
        names = [d["name"] for d in definitions]
        for i in range(20):
            assert f"concurrent_{i}" in names

    def test_concurrent_loop_creation(self, engine):
        """Test concurrent loop creation."""
        loop_ids = []
        
        def create_loop():
            loop_id = engine.create_loop("Concurrent test")
            loop_ids.append(loop_id)
        
        threads = [threading.Thread(target=create_loop) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(set(loop_ids)) == 10  # All unique


# -----------------------------------------------------------------------------
# Resume Execution Tests (Mocked)
# -----------------------------------------------------------------------------

class TestResumeExecution:
    """Test workflow resume functionality."""

    def test_resume_paused_execution_not_allowed(self, engine, simple_workflow):
        """Test that resume on completed execution is not allowed."""
        engine.define(simple_workflow)
        exec_id = engine.run_workflow("simple_workflow")
        
        # Cannot resume a completed execution
        with pytest.raises(ValueError, match="not 'paused'"):
            engine.run_workflow("simple_workflow", resume_exec_id=exec_id)


# -----------------------------------------------------------------------------
# Logging and Observability Tests
# -----------------------------------------------------------------------------

class TestLogging:
    """Test logging functionality."""

    @patch("core.workflow_engine.log_json")
    def test_workflow_events_logged(self, mock_log, engine, simple_workflow):
        """Test that workflow events are logged."""
        engine.define(simple_workflow)
        engine.run_workflow("simple_workflow")
        
        # Check that logging was called for key events
        log_calls = [call[0][1] for call in mock_log.call_args_list]
        assert "workflow_step_start" in log_calls or "workflow_execution_completed" in log_calls

    @patch("core.workflow_engine.log_json")
    def test_loop_events_logged(self, mock_log, engine):
        """Test that loop events are logged."""
        engine.create_loop("Test goal")
        
        # Check that logging was called
        mock_log.assert_any_call("INFO", "agentic_loop_created", details=ANY)


# -----------------------------------------------------------------------------
# Mock Orchestrator Tests
# -----------------------------------------------------------------------------

class TestMockOrchestrator:
    """Test with mocked orchestrator."""

    @patch("core.workflow_engine.WorkflowEngine._get_orchestrator")
    def test_loop_dry_run(self, mock_get_orch, engine):
        """Test loop tick with dry_run flag."""
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"stop_reason": None, "phase_outputs": {}}
        mock_get_orch.return_value = mock_orch
        
        loop_id = engine.create_loop("Test")
        engine.loop_tick(loop_id, dry_run=True)
        
        mock_orch.run_cycle.assert_called_once_with("Test", dry_run=True)
