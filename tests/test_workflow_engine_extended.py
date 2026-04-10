"""Extended unit tests for core/workflow_engine.py — Sprint 9 coverage."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

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
    _execute_step,
    _run_skill,
    _run_step_callable_with_timeout,
    _wire_inputs,
    get_engine,
)


def _simple_step(name="s1", fn=None) -> WorkflowStep:
    if fn is None:
        fn = lambda inputs: {"result": "ok"}
    return WorkflowStep(name=name, fn=fn, retry=RetryPolicy(max_attempts=1))


def _simple_def(name="wf", step_fn=None) -> WorkflowDefinition:
    return WorkflowDefinition(name=name, steps=[_simple_step("s1", fn=step_fn)])


class TestRunSkill:
    def test_run_skill_calls_registry(self):
        mock_skill = MagicMock()
        mock_skill.run.return_value = {"output": "data"}
        with patch("agents.skills.registry.all_skills", return_value={"my_skill": mock_skill}):
            result = _run_skill("my_skill", {"key": "val"})
        assert result == {"output": "data"}
        mock_skill.run.assert_called_once_with({"key": "val"})

    def test_run_skill_unknown_returns_error(self):
        with patch("agents.skills.registry.all_skills", return_value={}):
            result = _run_skill("nonexistent_skill", {})
        assert "error" in result
        assert "Unknown skill" in result["error"]

    def test_run_skill_import_error_returns_error(self):
        with patch("agents.skills.registry.all_skills", side_effect=ImportError("no module")):
            result = _run_skill("any_skill", {})
        assert "error" in result

    def test_run_skill_exception_in_run_returns_error(self):
        mock_skill = MagicMock()
        mock_skill.run.side_effect = RuntimeError("skill crashed")
        with patch("agents.skills.registry.all_skills", return_value={"bad_skill": mock_skill}):
            result = _run_skill("bad_skill", {})
        assert "error" in result


class TestRunStepCallableWithTimeout:
    def test_no_timeout_runs_directly(self):
        fn = lambda: {"answer": 42}
        ok, result = _run_step_callable_with_timeout(fn, 0)
        assert ok is True
        assert result == {"answer": 42}

    def test_negative_timeout_runs_directly(self):
        fn = lambda: "direct"
        ok, result = _run_step_callable_with_timeout(fn, -1)
        assert ok is True
        assert result == "direct"

    def test_timeout_within_limit_succeeds(self):
        fn = lambda: {"fast": True}
        ok, result = _run_step_callable_with_timeout(fn, 5.0)
        assert ok is True
        assert result == {"fast": True}

    def test_timeout_exceeded_raises(self):
        def slow():
            time.sleep(10)
            return {}

        with pytest.raises(TimeoutError):
            _run_step_callable_with_timeout(slow, 0.05)


class TestExecuteStepEdgeCases:
    def test_step_with_no_skill_and_no_fn_returns_error(self):
        step = WorkflowStep(name="empty", retry=RetryPolicy(max_attempts=1))
        result = _execute_step(step, {}, {})
        assert result.status == "failed"
        assert "no skill_name or fn" in result.error

    def test_fn_raises_returns_failed(self):
        def boom(inputs):
            raise ValueError("exploded")

        step = WorkflowStep(name="boom", fn=boom, retry=RetryPolicy(max_attempts=1), timeout_s=5.0)
        result = _execute_step(step, {}, {})
        assert result.status == "failed"
        assert "ValueError" in result.error

    def test_skip_if_false_single_part_truthy(self):
        step = WorkflowStep(
            name="conditional",
            fn=lambda i: {"ran": True},
            skip_if_false="prev_step",
            retry=RetryPolicy(max_attempts=1),
        )
        result = _execute_step(step, {"prev_step": {"some_key": True}}, {})
        assert result.status == "ok"

    def test_skip_if_false_single_part_falsy_empty_dict(self):
        step = WorkflowStep(
            name="conditional",
            fn=lambda i: {"ran": True},
            skip_if_false="prev_step",
            retry=RetryPolicy(max_attempts=1),
        )
        result = _execute_step(step, {"prev_step": {}}, {})
        assert result.status == "skipped"

    def test_skip_if_false_dotted_falsy_key(self):
        step = WorkflowStep(
            name="conditioned",
            fn=lambda i: {"done": True},
            skip_if_false="checker.passed",
            retry=RetryPolicy(max_attempts=1),
        )
        result = _execute_step(step, {"checker": {"passed": False}}, {})
        assert result.status == "skipped"

    def test_skip_if_false_missing_source_step(self):
        step = WorkflowStep(
            name="conditioned",
            fn=lambda i: {"done": True},
            skip_if_false="missing_step.key",
            retry=RetryPolicy(max_attempts=1),
        )
        result = _execute_step(step, {}, {})
        assert result.status == "skipped"

    def test_retry_on_error_output(self):
        call_count = {"n": 0}

        def error_fn(inputs):
            call_count["n"] += 1
            return {"error": "transient"}

        step = WorkflowStep(
            name="retry_step",
            fn=error_fn,
            retry=RetryPolicy(max_attempts=3, backoff_base=0.0),
            timeout_s=0,
        )
        result = _execute_step(step, {}, {})
        assert result.status == "failed"
        assert call_count["n"] == 3
        assert result.attempts == 3

    def test_retry_succeeds_on_second_attempt(self):
        attempts = {"n": 0}

        def flaky(inputs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {"error": "first attempt"}
            return {"value": "success"}

        step = WorkflowStep(
            name="flaky",
            fn=flaky,
            retry=RetryPolicy(max_attempts=3, backoff_base=0.0),
            timeout_s=0,
        )
        result = _execute_step(step, {}, {})
        assert result.status == "ok"
        assert result.attempts == 2

    def test_timeout_step_returns_timeout_status(self):
        def slow(inputs):
            time.sleep(10)
            return {"done": True}

        step = WorkflowStep(
            name="slow_step",
            fn=slow,
            retry=RetryPolicy(max_attempts=1),
            timeout_s=0.05,
        )
        result = _execute_step(step, {}, {})
        assert result.status == "timeout"

    def test_step_uses_skill_name(self):
        mock_skill = MagicMock()
        mock_skill.run.return_value = {"skill_output": 1}
        with patch("agents.skills.registry.all_skills", return_value={"my_sk": mock_skill}):
            step = WorkflowStep(
                name="skill_step",
                skill_name="my_sk",
                retry=RetryPolicy(max_attempts=1),
                timeout_s=0,
            )
            result = _execute_step(step, {}, {"project_root": "."})
        assert result.status == "ok"
        assert result.output["skill_output"] == 1

    def test_execute_step_elapsed_ms_positive(self):
        step = WorkflowStep(
            name="timed",
            fn=lambda i: {"x": 1},
            retry=RetryPolicy(max_attempts=1),
            timeout_s=0,
        )
        result = _execute_step(step, {}, {})
        assert result.elapsed_ms >= 0.0


class TestWireInputsExtended:
    def test_wildcard_merges_entire_output(self):
        step = WorkflowStep(
            name="s",
            fn=lambda i: i,
            inputs_from={"_unused": "prev.*"},
        )
        result = _wire_inputs(step, {"prev": {"a": 1, "b": 2}}, {})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_inputs_from_missing_key_gives_none(self):
        step = WorkflowStep(
            name="s",
            fn=lambda i: i,
            inputs_from={"x": "prev_step.no_such_key"},
        )
        result = _wire_inputs(step, {"prev_step": {"other": 99}}, {})
        assert result["x"] is None

    def test_static_inputs_override_initial(self):
        step = WorkflowStep(
            name="s",
            fn=lambda i: i,
            static_inputs={"key": "static_val"},
        )
        result = _wire_inputs(step, {}, {"key": "initial_val"})
        assert result["key"] == "static_val"

    def test_initial_inputs_propagated(self):
        step = WorkflowStep(name="s", fn=lambda i: i)
        result = _wire_inputs(step, {}, {"project_root": "/root"})
        assert result["project_root"] == "/root"


class TestWorkflowEngineStateMachineExtended:
    def setup_method(self):
        self.engine = WorkflowEngine()
        self.engine.define(_simple_def("wf"))

    def test_run_workflow_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="not defined"):
            self.engine.run_workflow("no_such_workflow", {})

    def test_pause_terminal_execution_raises_value_error(self):
        exec_id = self.engine.run_workflow("wf", {})
        with pytest.raises(ValueError, match="already terminal"):
            self.engine.pause_execution(exec_id)

    def test_cancel_terminal_execution_raises_value_error(self):
        exec_id = self.engine.run_workflow("wf", {})
        with pytest.raises(ValueError, match="already terminal"):
            self.engine.cancel_execution(exec_id)

    def test_pause_missing_exec_raises_key_error(self):
        with pytest.raises(KeyError):
            self.engine.pause_execution("nonexistent-id")

    def test_cancel_missing_exec_raises_key_error(self):
        with pytest.raises(KeyError):
            self.engine.cancel_execution("nonexistent-id")

    def test_execution_status_not_found(self):
        status = self.engine.execution_status("nonexistent")
        assert status["status"] == "not_found"

    def test_get_step_output_missing_execution_raises(self):
        with pytest.raises(KeyError):
            self.engine.get_step_output("bad-id", "step1")

    def test_get_step_output_missing_step_raises(self):
        exec_id = self.engine.run_workflow("wf", {})
        with pytest.raises(KeyError):
            self.engine.get_step_output(exec_id, "no_such_step")

    def test_get_step_output_returns_dict(self):
        exec_id = self.engine.run_workflow("wf", {})
        output = self.engine.get_step_output(exec_id, "s1")
        assert isinstance(output, dict)

    def test_list_executions_no_filter(self):
        self.engine.run_workflow("wf", {})
        items = self.engine.list_executions()
        assert len(items) >= 1

    def test_list_executions_with_status_filter(self):
        self.engine.run_workflow("wf", {})
        completed = self.engine.list_executions(status_filter="completed")
        for item in completed:
            assert item["status"] == "completed"

    def test_execution_status_has_history(self):
        exec_id = self.engine.run_workflow("wf", {})
        status = self.engine.execution_status(exec_id)
        assert "history" in status
        assert len(status["history"]) >= 1
        assert status["history"][0]["step"] == "s1"

    def test_execution_status_has_step_output_keys(self):
        exec_id = self.engine.run_workflow("wf", {})
        status = self.engine.execution_status(exec_id)
        assert "step_output_keys" in status

    def test_resume_non_paused_raises(self):
        exec_id = self.engine.run_workflow("wf", {})
        with pytest.raises(ValueError, match="not 'paused'"):
            self.engine.run_workflow("wf", {}, resume_exec_id=exec_id)


class TestWorkflowEngineFailurePaths:
    def test_step_failure_sets_execution_failed(self):
        engine = WorkflowEngine()
        engine.define(
            WorkflowDefinition(
                name="fail_wf",
                steps=[
                    WorkflowStep(
                        name="bad",
                        fn=lambda i: {"error": "always fails"},
                        retry=RetryPolicy(max_attempts=1),
                    )
                ],
            )
        )
        exec_id = engine.run_workflow("fail_wf", {})
        status = engine.execution_status(exec_id)
        assert status["status"] == "failed"

    def test_total_retry_budget_exhausted(self):
        engine = WorkflowEngine()
        engine.define(
            WorkflowDefinition(
                name="budget_wf",
                max_retries_total=1,
                steps=[
                    WorkflowStep(
                        name="fail1",
                        fn=lambda i: {"error": "boom"},
                        retry=RetryPolicy(max_attempts=2, backoff_base=0.0),
                    )
                ],
            )
        )
        exec_id = engine.run_workflow("budget_wf", {})
        status = engine.execution_status(exec_id)
        assert status["status"] == "failed"

    def test_step_timeout_marks_execution_failed(self):
        engine = WorkflowEngine()
        engine.define(
            WorkflowDefinition(
                name="timeout_wf",
                steps=[
                    WorkflowStep(
                        name="slow",
                        fn=lambda i: time.sleep(10) or {},
                        retry=RetryPolicy(max_attempts=1),
                        timeout_s=0.05,
                    )
                ],
            )
        )
        exec_id = engine.run_workflow("timeout_wf", {})
        status = engine.execution_status(exec_id)
        assert status["status"] == "failed"
        assert "timed out" in status["error"]

    def test_second_step_failure_stops_workflow(self):
        engine = WorkflowEngine()
        engine.define(
            WorkflowDefinition(
                name="two_step",
                steps=[
                    WorkflowStep("ok_step", fn=lambda i: {"x": 1}, retry=RetryPolicy(max_attempts=1)),
                    WorkflowStep("bad_step", fn=lambda i: {"error": "boom"}, retry=RetryPolicy(max_attempts=1)),
                ],
            )
        )
        exec_id = engine.run_workflow("two_step", {})
        status = engine.execution_status(exec_id)
        assert status["status"] == "failed"
        assert "bad_step" in status["error"]


class TestWorkflowEngineMultiStep:
    def test_multi_step_workflow_all_ok(self):
        engine = WorkflowEngine()
        engine.define(
            WorkflowDefinition(
                name="multi",
                steps=[
                    WorkflowStep("step_a", fn=lambda i: {"val": 10}, retry=RetryPolicy(max_attempts=1)),
                    WorkflowStep(
                        "step_b",
                        fn=lambda i: {"doubled": i.get("val", 0) * 2},
                        inputs_from={"val": "step_a.val"},
                        retry=RetryPolicy(max_attempts=1),
                    ),
                ],
            )
        )
        exec_id = engine.run_workflow("multi", {})
        status = engine.execution_status(exec_id)
        assert status["status"] == "completed"
        b_out = engine.get_step_output(exec_id, "step_b")
        assert b_out["doubled"] == 20

    def test_skip_step_propagates(self):
        engine = WorkflowEngine()
        engine.define(
            WorkflowDefinition(
                name="skip_wf",
                steps=[
                    WorkflowStep("check", fn=lambda i: {"ok": False}, retry=RetryPolicy(max_attempts=1)),
                    WorkflowStep(
                        "guarded",
                        fn=lambda i: {"ran": True},
                        skip_if_false="check.ok",
                        retry=RetryPolicy(max_attempts=1),
                    ),
                ],
            )
        )
        exec_id = engine.run_workflow("skip_wf", {})
        status = engine.execution_status(exec_id)
        assert status["status"] == "completed"
        assert status["history"][1]["status"] == "skipped"


class TestAgenticLoopOperations:
    def setup_method(self):
        self.engine = WorkflowEngine()

    def test_create_loop_returns_id(self):
        loop_id = self.engine.create_loop("test goal", max_cycles=3)
        assert loop_id
        status = self.engine.loop_status(loop_id)
        assert status["status"] == "running"

    def test_loop_status_not_found(self):
        result = self.engine.loop_status("bad-id")
        assert result["status"] == "not_found"

    def test_stop_loop_not_found_raises(self):
        with pytest.raises(KeyError):
            self.engine.stop_loop("nonexistent")

    def test_stop_loop_terminal_raises(self):
        loop_id = self.engine.create_loop("goal")
        loop = self.engine._get_loop(loop_id)
        loop.status = "completed"
        with pytest.raises(ValueError, match="already terminal"):
            self.engine.stop_loop(loop_id)

    def test_stop_loop_succeeds(self):
        loop_id = self.engine.create_loop("goal")
        self.engine.stop_loop(loop_id, reason="test_stop")
        status = self.engine.loop_status(loop_id)
        assert status["status"] == "stopped"
        assert status["stop_reason"] == "test_stop"

    def test_pause_loop_not_found_raises(self):
        with pytest.raises(KeyError):
            self.engine.pause_loop("nonexistent")

    def test_pause_loop_terminal_raises(self):
        loop_id = self.engine.create_loop("goal")
        loop = self.engine._get_loop(loop_id)
        loop.status = "stopped"
        with pytest.raises(ValueError, match="already terminal"):
            self.engine.pause_loop(loop_id)

    def test_pause_loop_succeeds(self):
        loop_id = self.engine.create_loop("goal")
        self.engine.pause_loop(loop_id)
        status = self.engine.loop_status(loop_id)
        assert status["status"] == "paused"

    def test_resume_loop_not_found_raises(self):
        with pytest.raises(KeyError):
            self.engine.resume_loop("nonexistent")

    def test_resume_loop_not_paused_raises(self):
        loop_id = self.engine.create_loop("goal")
        with pytest.raises(ValueError, match="not 'paused'"):
            self.engine.resume_loop(loop_id)

    def test_resume_loop_succeeds(self):
        loop_id = self.engine.create_loop("goal")
        self.engine.pause_loop(loop_id)
        self.engine.resume_loop(loop_id)
        status = self.engine.loop_status(loop_id)
        assert status["status"] == "running"

    def test_list_loops_no_filter(self):
        self.engine.create_loop("goal A")
        self.engine.create_loop("goal B")
        items = self.engine.list_loops()
        assert len(items) >= 2

    def test_list_loops_status_filter(self):
        loop_id = self.engine.create_loop("filtered goal")
        self.engine.stop_loop(loop_id)
        stopped = self.engine.list_loops(status_filter="stopped")
        assert all(l["status"] == "stopped" for l in stopped)


class TestLoopTick:
    def setup_method(self):
        self.engine = WorkflowEngine()

    def test_loop_tick_not_found(self):
        result = self.engine.loop_tick("nonexistent")
        assert "error" in result

    def test_loop_tick_already_terminal(self):
        loop_id = self.engine.create_loop("goal")
        loop = self.engine._get_loop(loop_id)
        loop.status = "completed"
        result = self.engine.loop_tick(loop_id)
        assert "error" in result

    def test_loop_tick_max_cycles_reached_at_zero(self):
        loop_id = self.engine.create_loop("goal", max_cycles=0)
        result = self.engine.loop_tick(loop_id)
        assert result.get("stop_reason") == "max_cycles_reached"

    def test_loop_tick_orchestrator_exception(self):
        loop_id = self.engine.create_loop("goal", max_cycles=5)
        mock_orch = MagicMock()
        mock_orch.run_cycle.side_effect = RuntimeError("orch boom")
        self.engine._orchestrator = mock_orch
        result = self.engine.loop_tick(loop_id)
        assert result["cycle_status"] == "failed"
        status = self.engine.loop_status(loop_id)
        assert status["status"] == "failed"

    def test_loop_tick_with_stop_reason(self):
        loop_id = self.engine.create_loop("goal", max_cycles=5)
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {
            "stop_reason": "goal_achieved",
            "phase_outputs": {"plan": {"key": "val"}},
        }
        self.engine._orchestrator = mock_orch
        result = self.engine.loop_tick(loop_id)
        assert result["stop_reason"] == "goal_achieved"
        assert self.engine.loop_status(loop_id)["status"] == "completed"

    def test_loop_tick_no_stop_reason_increments(self):
        loop_id = self.engine.create_loop("goal", max_cycles=3)
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"phase_outputs": {}}
        self.engine._orchestrator = mock_orch
        result = self.engine.loop_tick(loop_id)
        assert result["cycle"] == 1
        assert result["cycle_status"] == "ok"

    def test_loop_tick_last_cycle_completes(self):
        loop_id = self.engine.create_loop("goal", max_cycles=1)
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"phase_outputs": {}}
        self.engine._orchestrator = mock_orch
        result = self.engine.loop_tick(loop_id)
        assert result["status"] == "completed"
        assert result["stop_reason"] == "max_cycles_reached"

    def test_loop_tick_history_updated(self):
        loop_id = self.engine.create_loop("goal", max_cycles=3)
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"phase_outputs": {}}
        self.engine._orchestrator = mock_orch
        self.engine.loop_tick(loop_id)
        status = self.engine.loop_status(loop_id)
        assert len(status["history"]) == 1

    def test_loop_tick_elapsed_ms_in_result(self):
        loop_id = self.engine.create_loop("goal", max_cycles=3)
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"phase_outputs": {}}
        self.engine._orchestrator = mock_orch
        result = self.engine.loop_tick(loop_id)
        assert "elapsed_ms" in result
        assert result["elapsed_ms"] >= 0


class TestCheckLoopHealth:
    def setup_method(self):
        self.engine = WorkflowEngine()

    def test_loop_not_found(self):
        result = self.engine.check_loop_health("missing")
        assert result["healthy"] is False
        assert "error" in result

    def test_terminal_loop_is_healthy(self):
        loop_id = self.engine.create_loop("goal")
        self.engine.stop_loop(loop_id)
        result = self.engine.check_loop_health(loop_id)
        assert result["healthy"] is True

    def test_active_recent_loop_is_healthy(self):
        loop_id = self.engine.create_loop("goal")
        result = self.engine.check_loop_health(loop_id, stall_threshold_s=9999)
        assert result["healthy"] is True

    def test_stale_loop_detected(self):
        loop_id = self.engine.create_loop("goal")
        loop = self.engine._get_loop(loop_id)
        loop.updated_at = time.time() - 400
        result = self.engine.check_loop_health(loop_id, stall_threshold_s=300)
        assert result["healthy"] is False

    def test_repeated_error_detected(self):
        loop_id = self.engine.create_loop("goal")
        loop = self.engine._get_loop(loop_id)
        err = "same error repeated"
        loop.history = [
            LoopCycle(1, "failed", {}, 100, error=err),
            LoopCycle(2, "failed", {}, 100, error=err),
            LoopCycle(3, "failed", {}, 100, error=err),
        ]
        result = self.engine.check_loop_health(loop_id, stall_threshold_s=9999)
        assert result["healthy"] is False
        assert len(result["warnings"]) > 0

    def test_different_errors_no_deadlock(self):
        loop_id = self.engine.create_loop("goal")
        loop = self.engine._get_loop(loop_id)
        loop.history = [
            LoopCycle(1, "failed", {}, 100, error="err1"),
            LoopCycle(2, "failed", {}, 100, error="err2"),
            LoopCycle(3, "failed", {}, 100, error="err3"),
        ]
        result = self.engine.check_loop_health(loop_id, stall_threshold_s=9999)
        assert result["healthy"] is True

    def test_fewer_than_3_cycles_no_repeated_error(self):
        loop_id = self.engine.create_loop("goal")
        loop = self.engine._get_loop(loop_id)
        loop.history = [
            LoopCycle(1, "failed", {}, 100, error="e"),
            LoopCycle(2, "failed", {}, 100, error="e"),
        ]
        result = self.engine.check_loop_health(loop_id, stall_threshold_s=9999)
        assert result["healthy"] is True

    def test_cycles_remaining_computed(self):
        loop_id = self.engine.create_loop("goal", max_cycles=5)
        loop = self.engine._get_loop(loop_id)
        loop.current_cycle = 2
        result = self.engine.check_loop_health(loop_id, stall_threshold_s=9999)
        assert result["cycles_remaining"] == 3


class TestLoopStatusDetail:
    def test_loop_status_fields(self):
        engine = WorkflowEngine()
        loop_id = engine.create_loop("my goal", max_cycles=7)
        status = engine.loop_status(loop_id)
        assert status["id"] == loop_id
        assert status["goal"] == "my goal"
        assert status["max_cycles"] == 7
        assert status["current_cycle"] == 0
        assert "elapsed_s" in status
        assert isinstance(status["history"], list)


class TestBuiltinWorkflowDefinitions:
    def setup_method(self):
        self.engine = WorkflowEngine()

    def test_security_audit_registered(self):
        defs = {d["name"]: d for d in self.engine.list_definitions()}
        assert "security_audit" in defs
        assert defs["security_audit"]["step_count"] == 4

    def test_code_quality_registered(self):
        defs = {d["name"]: d for d in self.engine.list_definitions()}
        assert "code_quality" in defs
        assert defs["code_quality"]["step_count"] == 5

    def test_release_prep_registered(self):
        defs = {d["name"]: d for d in self.engine.list_definitions()}
        assert "release_prep" in defs
        assert defs["release_prep"]["step_count"] == 4

    def test_onboarding_analysis_registered(self):
        defs = {d["name"]: d for d in self.engine.list_definitions()}
        assert "onboarding_analysis" in defs
        assert defs["onboarding_analysis"]["step_count"] == 4

    def test_define_overwrites_existing(self):
        self.engine.define(
            WorkflowDefinition(
                name="security_audit",
                description="overwritten",
                steps=[WorkflowStep("only_step", fn=lambda i: {})],
            )
        )
        defs = {d["name"]: d for d in self.engine.list_definitions()}
        assert defs["security_audit"]["step_count"] == 1

    def test_list_definitions_has_steps_key(self):
        for d in self.engine.list_definitions():
            assert "steps" in d
            assert isinstance(d["steps"], list)


class TestGetEngineSingletonExtended:
    def test_get_engine_returns_engine_instance(self):
        engine = get_engine()
        assert isinstance(engine, WorkflowEngine)

    def test_get_engine_is_singleton(self):
        e1 = get_engine()
        e2 = get_engine()
        assert e1 is e2

    def test_get_engine_has_definitions(self):
        engine = get_engine()
        names = [d["name"] for d in engine.list_definitions()]
        assert "security_audit" in names


class TestIsTerminal:
    def test_execution_is_terminal_completed(self):
        exc = WorkflowExecution(
            id="x",
            workflow_name="w",
            status="completed",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=0.0,
            updated_at=0.0,
        )
        assert exc.is_terminal() is True

    def test_execution_is_terminal_failed(self):
        exc = WorkflowExecution(
            id="x",
            workflow_name="w",
            status="failed",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=0.0,
            updated_at=0.0,
        )
        assert exc.is_terminal() is True

    def test_execution_is_terminal_cancelled(self):
        exc = WorkflowExecution(
            id="x",
            workflow_name="w",
            status="cancelled",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=0.0,
            updated_at=0.0,
        )
        assert exc.is_terminal() is True

    def test_execution_not_terminal_running(self):
        exc = WorkflowExecution(
            id="x",
            workflow_name="w",
            status="running",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=0.0,
            updated_at=0.0,
        )
        assert exc.is_terminal() is False

    def test_execution_not_terminal_paused(self):
        exc = WorkflowExecution(
            id="x",
            workflow_name="w",
            status="paused",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=0.0,
            updated_at=0.0,
        )
        assert exc.is_terminal() is False

    def test_loop_is_terminal_completed(self):
        loop = AgenticLoop(
            id="l",
            goal="g",
            max_cycles=5,
            current_cycle=5,
            status="completed",
            history=[],
            stop_reason="done",
            score=0.0,
            started_at=0.0,
            updated_at=0.0,
        )
        assert loop.is_terminal() is True

    def test_loop_is_terminal_failed(self):
        loop = AgenticLoop(
            id="l",
            goal="g",
            max_cycles=5,
            current_cycle=5,
            status="failed",
            history=[],
            stop_reason="err",
            score=0.0,
            started_at=0.0,
            updated_at=0.0,
        )
        assert loop.is_terminal() is True

    def test_loop_is_terminal_stopped(self):
        loop = AgenticLoop(
            id="l",
            goal="g",
            max_cycles=5,
            current_cycle=2,
            status="stopped",
            history=[],
            stop_reason="user",
            score=0.0,
            started_at=0.0,
            updated_at=0.0,
        )
        assert loop.is_terminal() is True

    def test_loop_not_terminal_running(self):
        loop = AgenticLoop(
            id="l",
            goal="g",
            max_cycles=5,
            current_cycle=1,
            status="running",
            history=[],
            stop_reason=None,
            score=0.0,
            started_at=0.0,
            updated_at=0.0,
        )
        assert loop.is_terminal() is False

    def test_loop_not_terminal_paused(self):
        loop = AgenticLoop(
            id="l",
            goal="g",
            max_cycles=5,
            current_cycle=1,
            status="paused",
            history=[],
            stop_reason=None,
            score=0.0,
            started_at=0.0,
            updated_at=0.0,
        )
        assert loop.is_terminal() is False


class TestJournalResilience:
    def test_journal_execution_handles_db_error(self):
        engine = WorkflowEngine()
        exc = WorkflowExecution(
            id="x",
            workflow_name="wf",
            status="running",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=0.0,
            updated_at=0.0,
        )
        with patch("core.workflow_engine._open_db", side_effect=Exception("db down")):
            try:
                engine._journal_execution(exc)
            except Exception:
                pytest.fail("_journal_execution should swallow DB errors")

    def test_journal_loop_handles_db_error(self):
        engine = WorkflowEngine()
        loop = AgenticLoop(
            id="l",
            goal="g",
            max_cycles=3,
            current_cycle=0,
            status="running",
            history=[],
            stop_reason=None,
            score=0.0,
            started_at=0.0,
            updated_at=0.0,
        )
        with patch("core.workflow_engine._open_db", side_effect=Exception("db down")):
            try:
                engine._journal_loop(loop)
            except Exception:
                pytest.fail("_journal_loop should swallow DB errors")


class TestGetOrchestrator:
    def test_cached_after_first_call(self):
        engine = WorkflowEngine()
        mock_orch = MagicMock()
        engine._orchestrator = mock_orch
        assert engine._get_orchestrator() is mock_orch
        assert engine._get_orchestrator() is mock_orch


class TestRetryPolicyArithmetic:
    def test_sleep_for_capped_at_max_backoff(self):
        policy = RetryPolicy(backoff_base=1.0, max_backoff=10.0)
        assert policy.sleep_for(100) == 10.0

    def test_sleep_for_zero_attempt(self):
        policy = RetryPolicy(backoff_base=2.0)
        assert policy.sleep_for(0) == 2.0

    def test_sleep_for_first_attempt(self):
        policy = RetryPolicy(backoff_base=1.0, max_backoff=100.0)
        assert policy.sleep_for(1) == 2.0

    def test_sleep_for_second_attempt(self):
        policy = RetryPolicy(backoff_base=0.5, max_backoff=100.0)
        assert policy.sleep_for(2) == 2.0

    def test_sleep_for_custom_max_backoff(self):
        policy = RetryPolicy(backoff_base=5.0, max_backoff=8.0)
        assert policy.sleep_for(3) == 8.0  # 5*8=40 > 8 → capped


class TestDataclasses:
    def test_step_result_with_error(self):
        sr = StepResult(
            step_name="s",
            status="failed",
            output={"error": "boom"},
            attempts=2,
            elapsed_ms=123.4,
            error="boom",
        )
        assert sr.step_name == "s"
        assert sr.status == "failed"
        assert sr.error == "boom"

    def test_loop_cycle_fields(self):
        lc = LoopCycle(
            cycle_number=3,
            status="ok",
            phase_outputs={"plan": ["step1"]},
            elapsed_ms=55.5,
        )
        assert lc.cycle_number == 3
        assert lc.stop_reason is None
        assert lc.error is None

    def test_workflow_execution_total_retries_default(self):
        exc = WorkflowExecution(
            id="e1",
            workflow_name="wf",
            status="running",
            current_step_index=0,
            step_outputs={},
            history=[],
            initial_inputs={},
            error=None,
            started_at=time.time(),
            updated_at=time.time(),
        )
        assert exc.total_retries_used == 0

    def test_workflow_step_default_retry_policy(self):
        step = WorkflowStep(name="s")
        assert step.retry.max_attempts == 3
        assert step.retry.backoff_base == 0.5

    def test_workflow_definition_default_max_retries(self):
        wf = WorkflowDefinition(name="w", steps=[])
        assert wf.max_retries_total == 0
