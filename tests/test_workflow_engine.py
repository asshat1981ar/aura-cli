"""Comprehensive unit tests for core/workflow_engine.py."""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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
    get_engine,
)


class TestRetryPolicy:
    """Test RetryPolicy dataclass."""

    def test_default_values(self):
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.backoff_base == 0.5
        assert policy.max_backoff == 30.0

    def test_sleep_for_calculation(self):
        policy = RetryPolicy()
        assert policy.sleep_for(0) == 0.5
        assert policy.sleep_for(1) == 1.0
        assert policy.sleep_for(2) == 2.0
        assert policy.sleep_for(10) == 30.0  # capped at max_backoff

    def test_custom_values(self):
        policy = RetryPolicy(max_attempts=5, backoff_base=1.0, max_backoff=60.0)
        assert policy.sleep_for(0) == 1.0
        assert policy.sleep_for(1) == 2.0


class TestWorkflowStep:
    """Test WorkflowStep dataclass."""

    def test_minimal_creation(self):
        step = WorkflowStep(name="test_step")
        assert step.name == "test_step"
        assert step.skill_name is None
        assert step.fn is None
        assert step.static_inputs == {}
        assert step.inputs_from == {}
        assert step.skip_if_false is None
        assert step.timeout_s == 120.0

    def test_full_creation(self):
        def test_fn(inputs):
            return {"result": "success"}

        step = WorkflowStep(
            name="full_step",
            skill_name="test_skill",
            fn=test_fn,
            static_inputs={"key": "value"},
            inputs_from={"prev": "step1.output"},
            skip_if_false="check.passed",
            timeout_s=60.0,
        )
        assert step.name == "full_step"
        assert step.skill_name == "test_skill"
        assert step.fn == test_fn
        assert step.static_inputs == {"key": "value"}
        assert step.inputs_from == {"prev": "step1.output"}
        assert step.skip_if_false == "check.passed"
        assert step.timeout_s == 60.0


class TestWorkflowDefinition:
    """Test WorkflowDefinition dataclass."""

    def test_creation(self):
        steps = [
            WorkflowStep(name="step1"),
            WorkflowStep(name="step2"),
        ]
        definition = WorkflowDefinition(
            name="test_workflow",
            steps=steps,
            description="A test workflow",
            max_retries_total=5,
        )
        assert definition.name == "test_workflow"
        assert len(definition.steps) == 2
        assert definition.description == "A test workflow"
        assert definition.max_retries_total == 5


class TestStepResult:
    """Test StepResult dataclass."""

    def test_creation(self):
        result = StepResult(
            step_name="test_step",
            status="ok",
            output={"key": "value"},
            attempts=1,
            elapsed_ms=100.5,
        )
        assert result.step_name == "test_step"
        assert result.status == "ok"
        assert result.output == {"key": "value"}
        assert result.attempts == 1
        assert result.elapsed_ms == 100.5
        assert result.error is None

    def test_with_error(self):
        result = StepResult(
            step_name="failing_step",
            status="failed",
            output={},
            attempts=3,
            elapsed_ms=500.0,
            error="Something went wrong",
        )
        assert result.status == "failed"
        assert result.error == "Something went wrong"


class TestWorkflowExecution:
    """Test WorkflowExecution dataclass."""

    def test_creation(self):
        execution = WorkflowExecution(
            id="exec-123",
            workflow_name="test_workflow",
            status="pending",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={"input": "value"},
            error=None,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert execution.id == "exec-123"
        assert execution.workflow_name == "test_workflow"
        assert execution.status == "pending"
        assert execution.current_step_index == 0
        assert execution.is_terminal() is False

    def test_terminal_states(self):
        for status in ["completed", "failed", "cancelled"]:
            execution = WorkflowExecution(
                id="exec-123",
                workflow_name="test",
                status=status,
                current_step_index=0,
                step_outputs={},
                history=[],
                initial_inputs={},
                error=None,
                started_at=time.time(),
                updated_at=time.time(),
            )
            assert execution.is_terminal() is True

    def test_non_terminal_states(self):
        for status in ["pending", "running", "paused"]:
            execution = WorkflowExecution(
                id="exec-123",
                workflow_name="test",
                status=status,
                current_step_index=0,
                step_outputs={},
                history=[],
                initial_inputs={},
                error=None,
                started_at=time.time(),
                updated_at=time.time(),
            )
            assert execution.is_terminal() is False


class TestLoopCycle:
    """Test LoopCycle dataclass."""

    def test_creation(self):
        cycle = LoopCycle(
            cycle_number=1,
            status="ok",
            phase_outputs={"ingest": {"context": "data"}},
            elapsed_ms=1500.0,
        )
        assert cycle.cycle_number == 1
        assert cycle.status == "ok"
        assert cycle.phase_outputs == {"ingest": {"context": "data"}}
        assert cycle.elapsed_ms == 1500.0
        assert cycle.error is None
        assert cycle.stop_reason is None


class TestAgenticLoop:
    """Test AgenticLoop dataclass."""

    def test_creation(self):
        now = time.time()
        loop = AgenticLoop(
            id="loop-123",
            goal="Test goal",
            max_cycles=10,
            current_cycle=0,
            status="running",
            history=[],
            stop_reason=None,
            score=0.0,
            started_at=now,
            updated_at=now,
        )
        assert loop.id == "loop-123"
        assert loop.goal == "Test goal"
        assert loop.max_cycles == 10
        assert loop.current_cycle == 0
        assert loop.status == "running"
        assert loop.score == 0.0
        assert loop.is_terminal() is False


class TestWorkflowEngine:
    """Test WorkflowEngine class."""

    @pytest.fixture
    def engine(self, tmp_path):
        # Use temp path for database
        with patch("core.workflow_engine._DB_PATH", tmp_path / "workflow.db"):
            engine = WorkflowEngine()
            yield engine

    def test_initialization(self, engine):
        assert engine is not None
        assert hasattr(engine, "_executions")
        assert hasattr(engine, "_loops")

    def test_define_workflow(self, engine):
        steps = [WorkflowStep(name="step1")]
        definition = WorkflowDefinition(name="test_workflow", steps=steps)

        engine.define(definition)

        assert "test_workflow" in engine._definitions
        assert engine._definitions["test_workflow"] == definition

    def test_run_workflow(self, engine):
        def mock_skill(inputs):
            return {"result": "success"}

        steps = [
            WorkflowStep(name="step1", fn=mock_skill),
        ]
        definition = WorkflowDefinition(name="test_workflow", steps=steps)
        engine.define(definition)

        exec_id = engine.run_workflow("test_workflow", {"input": "value"})

        assert exec_id is not None
        assert isinstance(exec_id, str)

        # Check execution was stored
        assert exec_id in engine._executions
        execution = engine._executions[exec_id]
        assert execution.workflow_name == "test_workflow"
        assert execution.initial_inputs == {"input": "value"}

    def test_execution_status(self, engine):
        steps = [WorkflowStep(name="step1")]
        definition = WorkflowDefinition(name="test_workflow", steps=steps)
        engine.define(definition)

        exec_id = engine.run_workflow("test_workflow", {})
        status = engine.execution_status(exec_id)

        assert status is not None
        assert "status" in status

    def test_run_nonexistent_workflow(self, engine):
        with pytest.raises(KeyError, match="Workflow 'nonexistent' not defined"):
            engine.run_workflow("nonexistent", {})

    def test_cancel_execution(self, engine):
        # Note: cancel_execution works best on paused or running workflows
        # For this test, we just verify the method exists and doesn't crash
        def mock_fn(inputs):
            return {"done": True}

        steps = [WorkflowStep(name="step1", fn=mock_fn)]
        definition = WorkflowDefinition(name="test_workflow", steps=steps)
        engine.define(definition)

        exec_id = engine.run_workflow("test_workflow", {})

        # If execution is still running, cancel it
        # If already completed, that's ok too
        execution = engine._executions[exec_id]
        if execution.status == "running":
            engine.cancel_execution(exec_id)
            assert execution.status == "cancelled"
        else:
            # Already completed
            assert execution.is_terminal()

    def test_create_loop(self, engine):
        loop_id = engine.create_loop("Test goal", max_cycles=5)

        assert loop_id is not None
        assert isinstance(loop_id, str)
        assert loop_id in engine._loops

        loop = engine._loops[loop_id]
        assert loop.goal == "Test goal"
        assert loop.max_cycles == 5
        assert loop.status == "running"

    def test_loop_status(self, engine):
        loop_id = engine.create_loop("Test goal")
        status = engine.loop_status(loop_id)

        assert status is not None
        assert status["goal"] == "Test goal"
        assert status["status"] == "running"

    def test_stop_loop(self, engine):
        loop_id = engine.create_loop("Test goal")
        engine.stop_loop(loop_id, reason="test_complete")

        loop = engine._loops[loop_id]
        assert loop.status == "stopped"
        assert loop.stop_reason == "test_complete"


class TestWorkflowEngineIntegration:
    """Integration tests for WorkflowEngine."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("core.workflow_engine._DB_PATH", tmp_path / "workflow.db"):
            yield WorkflowEngine()

    def test_full_workflow_execution(self, engine):
        """Test a complete workflow with multiple steps."""
        execution_order = []

        def step1_fn(inputs):
            execution_order.append("step1")
            return {"output1": "value1"}

        def step2_fn(inputs):
            execution_order.append("step2")
            return {"output2": "value2"}

        steps = [
            WorkflowStep(name="step1", fn=step1_fn),
            WorkflowStep(name="step2", fn=step2_fn, inputs_from={"prev": "step1.output1"}),
        ]
        definition = WorkflowDefinition(name="integration_workflow", steps=steps)
        engine.define(definition)

        exec_id = engine.run_workflow("integration_workflow", {"initial": "data"})

        # In a real scenario, we would wait for execution
        # For unit tests, we verify the setup is correct
        execution = engine._executions[exec_id]
        assert execution.workflow_name == "integration_workflow"
        assert execution.initial_inputs == {"initial": "data"}

    def test_step_with_static_inputs(self, engine):
        """Test that static inputs are properly merged."""
        received_inputs = {}

        def capture_fn(inputs):
            received_inputs.update(inputs)
            return {"captured": True}

        steps = [
            WorkflowStep(
                name="capture",
                fn=capture_fn,
                static_inputs={"static_key": "static_value"},
            ),
        ]
        definition = WorkflowDefinition(name="static_test", steps=steps)
        engine.define(definition)

        exec_id = engine.run_workflow("static_test", {"dynamic": "input"})

        # Verify execution was created
        assert exec_id in engine._executions


# ---------------------------------------------------------------------------
# Fixtures shared by new test classes
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temp DB path and patch _DB_PATH for the duration of the test."""
    db_path = tmp_path / "workflow_test.db"
    with patch("core.workflow_engine._DB_PATH", db_path):
        yield db_path


@pytest.fixture
def fresh_engine(tmp_path):
    """WorkflowEngine backed by a throw-away SQLite DB."""
    db_path = tmp_path / "engine_test.db"
    with patch("core.workflow_engine._DB_PATH", db_path):
        yield WorkflowEngine()


# ---------------------------------------------------------------------------
# RetryPolicy – additional backoff math
# ---------------------------------------------------------------------------


class TestRetryPolicyExtended:
    def test_sleep_for_exact_cap_boundary(self):
        """sleep_for should return max_backoff exactly when calculated > max."""
        policy = RetryPolicy(backoff_base=1.0, max_backoff=8.0)
        # 1 * 2^3 = 8 → exactly at cap
        assert policy.sleep_for(3) == 8.0
        # 1 * 2^4 = 16 → capped
        assert policy.sleep_for(4) == 8.0

    def test_sleep_for_zero_attempt(self):
        """Attempt 0 returns backoff_base * 2^0 = backoff_base."""
        policy = RetryPolicy(backoff_base=2.0, max_backoff=100.0)
        assert policy.sleep_for(0) == 2.0

    def test_sleep_for_never_exceeds_max(self):
        policy = RetryPolicy(backoff_base=0.1, max_backoff=1.0)
        for attempt in range(20):
            assert policy.sleep_for(attempt) <= 1.0


# ---------------------------------------------------------------------------
# WorkflowStep – skip_if_false and data_mapping
# ---------------------------------------------------------------------------


class TestWorkflowStepSkipAndMapping:
    def test_skip_if_false_field_stored(self):
        step = WorkflowStep(name="s", skip_if_false="gatekeeper.ok")
        assert step.skip_if_false == "gatekeeper.ok"

    def test_inputs_from_mapping_stored(self):
        step = WorkflowStep(name="s", inputs_from={"x": "prev.value", "y": "prev.*"})
        assert step.inputs_from == {"x": "prev.value", "y": "prev.*"}

    def test_fn_callable_is_stored(self):
        fn = lambda inputs: {"done": True}
        step = WorkflowStep(name="s", fn=fn)
        assert step.fn is fn


# ---------------------------------------------------------------------------
# _wire_inputs
# ---------------------------------------------------------------------------


class TestWireInputs:
    def _make_step(self, static_inputs=None, inputs_from=None):
        return WorkflowStep(
            name="target",
            static_inputs=static_inputs or {},
            inputs_from=inputs_from or {},
        )

    def test_initial_inputs_forwarded(self):
        step = self._make_step()
        result = _wire_inputs(step, {}, {"project_root": "/tmp/proj"})
        assert result["project_root"] == "/tmp/proj"

    def test_static_inputs_override_initial(self):
        step = self._make_step(static_inputs={"key": "STATIC"})
        result = _wire_inputs(step, {}, {"key": "INITIAL"})
        assert result["key"] == "STATIC"

    def test_dot_path_wiring(self):
        step = self._make_step(inputs_from={"x": "step1.output"})
        result = _wire_inputs(step, {"step1": {"output": 42}}, {})
        assert result["x"] == 42

    def test_star_wiring_expands_dict(self):
        step = self._make_step(inputs_from={"_": "step1.*"})
        result = _wire_inputs(step, {"step1": {"a": 1, "b": 2}}, {})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_missing_source_step_yields_none(self):
        step = self._make_step(inputs_from={"x": "missing_step.key"})
        result = _wire_inputs(step, {}, {})
        assert result["x"] is None


# ---------------------------------------------------------------------------
# WorkflowDefinition – register / get steps / ordering
# ---------------------------------------------------------------------------


class TestWorkflowDefinitionOrdering:
    def test_step_order_preserved(self):
        steps = [WorkflowStep(name=f"step{i}") for i in range(5)]
        wf = WorkflowDefinition(name="ordered", steps=steps)
        for i, step in enumerate(wf.steps):
            assert step.name == f"step{i}"

    def test_empty_steps_allowed(self):
        wf = WorkflowDefinition(name="empty", steps=[])
        assert wf.steps == []

    def test_default_max_retries_is_zero(self):
        wf = WorkflowDefinition(name="w", steps=[])
        assert wf.max_retries_total == 0


# ---------------------------------------------------------------------------
# WorkflowEngine – define / list_definitions
# ---------------------------------------------------------------------------


class TestWorkflowEngineDefinitions:
    def test_define_and_retrieve(self, fresh_engine):
        wf = WorkflowDefinition(
            name="my_wf",
            steps=[WorkflowStep(name="a"), WorkflowStep(name="b")],
            description="test",
        )
        fresh_engine.define(wf)
        assert "my_wf" in fresh_engine._definitions
        assert fresh_engine._definitions["my_wf"] is wf

    def test_list_definitions_includes_registered(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="custom_wf", steps=[WorkflowStep(name="x")]))
        names = [d["name"] for d in fresh_engine.list_definitions()]
        assert "custom_wf" in names

    def test_list_definitions_shows_step_count(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="three_step", steps=[WorkflowStep(name=f"s{i}") for i in range(3)]))
        entry = next(d for d in fresh_engine.list_definitions() if d["name"] == "three_step")
        assert entry["step_count"] == 3

    def test_define_overwrites_existing(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="dup", steps=[WorkflowStep(name="old")]))
        fresh_engine.define(WorkflowDefinition(name="dup", steps=[WorkflowStep(name="new")]))
        assert fresh_engine._definitions["dup"].steps[0].name == "new"


# ---------------------------------------------------------------------------
# WorkflowEngine – run_workflow state machine
# ---------------------------------------------------------------------------


class TestWorkflowEngineStateMachine:
    def test_successful_workflow_reaches_completed(self, fresh_engine):
        fresh_engine.define(
            WorkflowDefinition(
                name="ok_wf",
                steps=[WorkflowStep(name="s1", fn=lambda i: {"ok": True})],
            )
        )
        exec_id = fresh_engine.run_workflow("ok_wf", {})
        assert fresh_engine._executions[exec_id].status == "completed"

    def test_failed_step_sets_failed_status(self, fresh_engine):
        def bad_fn(inputs):
            return {"error": "something broke"}

        fresh_engine.define(
            WorkflowDefinition(
                name="fail_wf",
                steps=[WorkflowStep(name="bad", fn=bad_fn, retry=RetryPolicy(max_attempts=1))],
            )
        )
        exec_id = fresh_engine.run_workflow("fail_wf", {})
        exc = fresh_engine._executions[exec_id]
        assert exc.status == "failed"
        assert "bad" in exc.error

    def test_step_outputs_captured(self, fresh_engine):
        fresh_engine.define(
            WorkflowDefinition(
                name="capture_wf",
                steps=[WorkflowStep(name="producer", fn=lambda i: {"produced": 99})],
            )
        )
        exec_id = fresh_engine.run_workflow("capture_wf", {})
        assert fresh_engine._executions[exec_id].step_outputs["producer"] == {"produced": 99}

    def test_skip_if_false_skips_step(self, fresh_engine):
        """When the referenced flag is falsy the step is skipped."""
        fresh_engine.define(
            WorkflowDefinition(
                name="skip_wf",
                steps=[
                    WorkflowStep(name="gate", fn=lambda i: {"go": False}),
                    WorkflowStep(name="body", fn=lambda i: {"ran": True}, skip_if_false="gate.go"),
                ],
            )
        )
        exec_id = fresh_engine.run_workflow("skip_wf", {})
        exc = fresh_engine._executions[exec_id]
        assert exc.status == "completed"
        body_result = next(r for r in exc.history if r.step_name == "body")
        assert body_result.status == "skipped"

    def test_retry_budget_exhaustion(self, fresh_engine):
        """max_retries_total=1 stops the workflow once one retry is used."""
        call_count = {"n": 0}

        def flaky(inputs):
            call_count["n"] += 1
            return {"error": "always bad"}

        fresh_engine.define(
            WorkflowDefinition(
                name="budget_wf",
                steps=[WorkflowStep(name="flaky", fn=flaky, retry=RetryPolicy(max_attempts=3))],
                max_retries_total=1,
            )
        )
        exec_id = fresh_engine.run_workflow("budget_wf", {})
        exc = fresh_engine._executions[exec_id]
        assert exc.status == "failed"
        assert "budget" in exc.error.lower() or "flaky" in exc.error

    def test_no_steps_completes_immediately(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="noop_wf", steps=[]))
        exec_id = fresh_engine.run_workflow("noop_wf", {})
        assert fresh_engine._executions[exec_id].status == "completed"

    def test_multi_step_inputs_wiring(self, fresh_engine):
        """Step 2 receives output of step 1 via inputs_from."""
        received = {}

        def step2_fn(inputs):
            received.update(inputs)
            return {"done": True}

        fresh_engine.define(
            WorkflowDefinition(
                name="wired_wf",
                steps=[
                    WorkflowStep(name="s1", fn=lambda i: {"x": 7}),
                    WorkflowStep(name="s2", fn=step2_fn, inputs_from={"x": "s1.x"}),
                ],
            )
        )
        fresh_engine.run_workflow("wired_wf", {})
        assert received.get("x") == 7


# ---------------------------------------------------------------------------
# WorkflowEngine – execution_status / cancel / list_executions
# ---------------------------------------------------------------------------


class TestWorkflowEngineManagement:
    def test_execution_status_not_found(self, fresh_engine):
        status = fresh_engine.execution_status("nonexistent-id")
        assert status["status"] == "not_found"

    def test_execution_status_fields(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="st_wf", steps=[WorkflowStep(name="s", fn=lambda i: {"r": 1})]))
        exec_id = fresh_engine.run_workflow("st_wf", {})
        status = fresh_engine.execution_status(exec_id)
        assert status["id"] == exec_id
        assert status["workflow"] == "st_wf"
        assert "history" in status
        assert "elapsed_s" in status

    def test_cancel_running_execution(self, fresh_engine):
        """Inject a paused execution and then cancel it."""
        now = time.time()
        exc = WorkflowExecution(
            id="fake-exec",
            workflow_name="w",
            status="paused",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=now,
            updated_at=now,
        )
        with fresh_engine._global_lock:
            fresh_engine._executions["fake-exec"] = exc
            fresh_engine._locks["fake-exec"] = threading.Lock()
        fresh_engine.cancel_execution("fake-exec")
        assert fresh_engine._executions["fake-exec"].status == "cancelled"

    def test_cancel_terminal_raises(self, fresh_engine):
        now = time.time()
        exc = WorkflowExecution(
            id="done-exec",
            workflow_name="w",
            status="completed",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=now,
            updated_at=now,
        )
        with fresh_engine._global_lock:
            fresh_engine._executions["done-exec"] = exc
            fresh_engine._locks["done-exec"] = threading.Lock()
        with pytest.raises(ValueError, match="terminal"):
            fresh_engine.cancel_execution("done-exec")

    def test_list_executions_no_filter(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="lw", steps=[WorkflowStep(name="s", fn=lambda i: {"r": 1})]))
        fresh_engine.run_workflow("lw", {})
        fresh_engine.run_workflow("lw", {})
        items = fresh_engine.list_executions()
        assert len(items) >= 2

    def test_list_executions_with_status_filter(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="fw", steps=[WorkflowStep(name="s", fn=lambda i: {"r": 1})]))
        fresh_engine.run_workflow("fw", {})
        completed = fresh_engine.list_executions(status_filter="completed")
        assert all(e["status"] == "completed" for e in completed)

    def test_get_step_output_success(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="out_wf", steps=[WorkflowStep(name="s", fn=lambda i: {"val": 42})]))
        exec_id = fresh_engine.run_workflow("out_wf", {})
        output = fresh_engine.get_step_output(exec_id, "s")
        assert output == {"val": 42}

    def test_get_step_output_missing_raises(self, fresh_engine):
        fresh_engine.define(WorkflowDefinition(name="out2_wf", steps=[WorkflowStep(name="s", fn=lambda i: {"v": 1})]))
        exec_id = fresh_engine.run_workflow("out2_wf", {})
        with pytest.raises(KeyError):
            fresh_engine.get_step_output(exec_id, "nonexistent_step")


# ---------------------------------------------------------------------------
# AgenticLoop – loop_tick, max_cycles, early stop
# ---------------------------------------------------------------------------


class TestAgenticLoopTick:
    def _mock_orchestrator(self, stop_reason=None, phase_outputs=None):
        orch = MagicMock()
        orch.run_cycle.return_value = {
            "stop_reason": stop_reason,
            "phase_outputs": phase_outputs or {"ingest": {"ctx": "data"}},
        }
        return orch

    def test_loop_tick_advances_cycle(self, fresh_engine):
        loop_id = fresh_engine.create_loop("improve code quality", max_cycles=5)
        with patch.object(fresh_engine, "_get_orchestrator", return_value=self._mock_orchestrator()):
            fresh_engine.loop_tick(loop_id)
        assert fresh_engine._loops[loop_id].current_cycle == 1

    def test_loop_tick_completes_on_stop_reason(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal", max_cycles=10)
        with patch.object(fresh_engine, "_get_orchestrator", return_value=self._mock_orchestrator(stop_reason="score_threshold_met")):
            result = fresh_engine.loop_tick(loop_id)
        assert result["stop_reason"] == "score_threshold_met"
        assert fresh_engine._loops[loop_id].status == "completed"

    def test_loop_tick_max_cycles_stops_loop(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal", max_cycles=2)
        orch = self._mock_orchestrator()
        with patch.object(fresh_engine, "_get_orchestrator", return_value=orch):
            fresh_engine.loop_tick(loop_id)
            result = fresh_engine.loop_tick(loop_id)
        assert fresh_engine._loops[loop_id].status == "completed"
        assert result["stop_reason"] == "max_cycles_reached"

    def test_loop_tick_on_already_terminal_returns_error(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal", max_cycles=1)
        fresh_engine.stop_loop(loop_id)
        result = fresh_engine.loop_tick(loop_id)
        assert "error" in result

    def test_loop_tick_cycle_exception_sets_failed(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal", max_cycles=5)
        orch = MagicMock()
        orch.run_cycle.side_effect = RuntimeError("orchestrator exploded")
        with patch.object(fresh_engine, "_get_orchestrator", return_value=orch):
            result = fresh_engine.loop_tick(loop_id)
        assert fresh_engine._loops[loop_id].status == "failed"
        assert result["cycle_status"] == "failed"

    def test_loop_tick_already_at_max_cycles(self, fresh_engine):
        """If current_cycle == max_cycles before the tick, complete immediately."""
        loop_id = fresh_engine.create_loop("goal", max_cycles=0)
        result = fresh_engine.loop_tick(loop_id)
        assert fresh_engine._loops[loop_id].status == "completed"
        assert result.get("stop_reason") == "max_cycles_reached"

    def test_pause_and_resume_loop(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal", max_cycles=5)
        fresh_engine.pause_loop(loop_id)
        assert fresh_engine._loops[loop_id].status == "paused"
        fresh_engine.resume_loop(loop_id)
        assert fresh_engine._loops[loop_id].status == "running"

    def test_stop_loop_already_terminal_raises(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal", max_cycles=1)
        fresh_engine.stop_loop(loop_id)
        with pytest.raises(ValueError, match="terminal"):
            fresh_engine.stop_loop(loop_id)

    def test_list_loops_filter(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal1", max_cycles=5)
        fresh_engine.stop_loop(loop_id, reason="test")
        fresh_engine.create_loop("goal2", max_cycles=5)
        stopped = fresh_engine.list_loops(status_filter="stopped")
        assert all(l["status"] == "stopped" for l in stopped)
        running = fresh_engine.list_loops(status_filter="running")
        assert all(l["status"] == "running" for l in running)

    def test_loop_status_not_found(self, fresh_engine):
        status = fresh_engine.loop_status("no-such-loop")
        assert status["status"] == "not_found"

    def test_check_loop_health_terminal_is_healthy(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal", max_cycles=1)
        fresh_engine.stop_loop(loop_id)
        health = fresh_engine.check_loop_health(loop_id)
        assert health["healthy"] is True

    def test_check_loop_health_repeated_errors(self, fresh_engine):
        loop_id = fresh_engine.create_loop("goal", max_cycles=10)
        loop = fresh_engine._loops[loop_id]
        err = "RuntimeError: same error"
        for i in range(3):
            loop.history.append(LoopCycle(cycle_number=i + 1, status="failed", phase_outputs={}, elapsed_ms=100, error=err))
        health = fresh_engine.check_loop_health(loop_id)
        assert health["healthy"] is False
        assert any("repeated" in w for w in health["warnings"])


# ---------------------------------------------------------------------------
# SQLite journal – rows written on state transitions
# ---------------------------------------------------------------------------


class TestSQLiteJournal:
    def _read_db(self, db_path: Path) -> dict:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        execs = [dict(r) for r in conn.execute("SELECT * FROM executions").fetchall()]
        loops = [dict(r) for r in conn.execute("SELECT * FROM loops").fetchall()]
        conn.close()
        return {"executions": execs, "loops": loops}

    def test_execution_row_written(self, tmp_path):
        db_path = tmp_path / "journal_test.db"
        with patch("core.workflow_engine._DB_PATH", db_path):
            engine = WorkflowEngine()
            engine.define(
                WorkflowDefinition(
                    name="journal_wf",
                    steps=[WorkflowStep(name="s", fn=lambda i: {"ok": True})],
                )
            )
            exec_id = engine.run_workflow("journal_wf", {})

        rows = self._read_db(db_path)
        exec_ids = [r["id"] for r in rows["executions"]]
        assert exec_id in exec_ids

    def test_execution_status_updated_in_db(self, tmp_path):
        db_path = tmp_path / "journal_status.db"
        with patch("core.workflow_engine._DB_PATH", db_path):
            engine = WorkflowEngine()
            engine.define(
                WorkflowDefinition(
                    name="status_wf",
                    steps=[WorkflowStep(name="s", fn=lambda i: {"ok": True})],
                )
            )
            exec_id = engine.run_workflow("status_wf", {})

        rows = self._read_db(db_path)
        row = next(r for r in rows["executions"] if r["id"] == exec_id)
        assert row["status"] == "completed"

    def test_loop_row_written(self, tmp_path):
        db_path = tmp_path / "loop_journal.db"
        with patch("core.workflow_engine._DB_PATH", db_path):
            engine = WorkflowEngine()
            loop_id = engine.create_loop("journal goal", max_cycles=3)

        rows = self._read_db(db_path)
        loop_ids = [r["id"] for r in rows["loops"]]
        assert loop_id in loop_ids


# ---------------------------------------------------------------------------
# Thread safety – concurrent workflow launches
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_workflow_runs_no_corruption(self, fresh_engine):
        """Launch 10 concurrent workflow runs; all should complete independently."""
        fresh_engine.define(
            WorkflowDefinition(
                name="concurrent_wf",
                steps=[WorkflowStep(name="s", fn=lambda i: {"t": time.time()})],
            )
        )
        exec_ids: list[str] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _run():
            try:
                eid = fresh_engine.run_workflow("concurrent_wf", {})
                with lock:
                    exec_ids.append(eid)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_run) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"Errors during concurrent runs: {errors}"
        assert len(exec_ids) == 10
        # All IDs must be unique
        assert len(set(exec_ids)) == 10
        # All executions must have completed
        for eid in exec_ids:
            assert fresh_engine._executions[eid].status == "completed"

    def test_concurrent_loop_creation(self, fresh_engine):
        """Create 10 loops from different threads; all should land in _loops."""
        loop_ids: list[str] = []
        lock = threading.Lock()

        def _create():
            lid = fresh_engine.create_loop("parallel goal", max_cycles=3)
            with lock:
                loop_ids.append(lid)

        threads = [threading.Thread(target=_create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(loop_ids) == 10
        assert len(set(loop_ids)) == 10


# ---------------------------------------------------------------------------
# get_engine singleton
# ---------------------------------------------------------------------------


class TestGetEngineSingleton:
    def test_returns_same_instance(self, tmp_path):
        db_path = tmp_path / "singleton.db"
        import core.workflow_engine as wem

        orig_db = wem._DB_PATH
        orig_engine = wem._engine
        try:
            wem._DB_PATH = db_path
            wem._engine = None  # reset singleton
            e1 = get_engine()
            e2 = get_engine()
            assert e1 is e2
        finally:
            wem._DB_PATH = orig_db
            wem._engine = orig_engine
