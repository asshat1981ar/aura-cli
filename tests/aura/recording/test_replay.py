"""Tests for aura/recording/replay.py — VariableInterpolator, ReplayEngine."""

import pytest

from aura.recording.replay import VariableInterpolator, ReplayEngine
from aura.recording.models import Recording, RecordingStep, StepStatus


# ---------------------------------------------------------------------------
# VariableInterpolator
# ---------------------------------------------------------------------------

class TestVariableInterpolatorInterpolate:
    def test_no_variables(self):
        result = VariableInterpolator.interpolate("hello world", {})
        assert result == "hello world"

    def test_single_variable(self):
        result = VariableInterpolator.interpolate("hello ${name}", {"name": "world"})
        assert result == "hello world"

    def test_multiple_variables(self):
        result = VariableInterpolator.interpolate("${a} + ${b}", {"a": "x", "b": "y"})
        assert result == "x + y"

    def test_missing_variable_left_as_is(self):
        result = VariableInterpolator.interpolate("${missing}", {})
        assert result == "${missing}"

    def test_variable_repeated(self):
        result = VariableInterpolator.interpolate("${v} and ${v}", {"v": "Z"})
        assert result == "Z and Z"

    def test_partial_variables(self):
        result = VariableInterpolator.interpolate("${found} ${notfound}", {"found": "yes"})
        assert result == "yes ${notfound}"

    def test_empty_string(self):
        result = VariableInterpolator.interpolate("", {"k": "v"})
        assert result == ""

    def test_no_placeholders(self):
        result = VariableInterpolator.interpolate("plain text", {"k": "v"})
        assert result == "plain text"


class TestVariableInterpolatorInterpolateDict:
    def test_string_values_interpolated(self):
        result = VariableInterpolator.interpolate_dict(
            {"key": "${val}"}, {"val": "hello"}
        )
        assert result["key"] == "hello"

    def test_non_string_values_unchanged(self):
        result = VariableInterpolator.interpolate_dict(
            {"num": 42, "lst": [1, 2]}, {"x": "y"}
        )
        assert result["num"] == 42
        assert result["lst"] == [1, 2]

    def test_empty_dict(self):
        assert VariableInterpolator.interpolate_dict({}, {"k": "v"}) == {}

    def test_mixed_values(self):
        result = VariableInterpolator.interpolate_dict(
            {"a": "${x}", "b": 99, "c": "${y}"},
            {"x": "A", "y": "C"},
        )
        assert result == {"a": "A", "b": 99, "c": "C"}


class TestVariableInterpolatorInterpolateList:
    def test_string_items_interpolated(self):
        result = VariableInterpolator.interpolate_list(["${a}", "${b}"], {"a": "1", "b": "2"})
        assert result == ["1", "2"]

    def test_non_string_items_unchanged(self):
        result = VariableInterpolator.interpolate_list([1, None, True], {})
        assert result == [1, None, True]

    def test_empty_list(self):
        assert VariableInterpolator.interpolate_list([], {"k": "v"}) == []

    def test_mixed_list(self):
        result = VariableInterpolator.interpolate_list(["${x}", 5, "${y}"], {"x": "A", "y": "B"})
        assert result == ["A", 5, "B"]


# ---------------------------------------------------------------------------
# ReplayEngine — handler registration
# ---------------------------------------------------------------------------

class TestReplayEngineRegisterHandler:
    def test_register_handler_stored(self):
        engine = ReplayEngine()
        handler = lambda: None
        engine.register_handler("my_cmd", handler)
        assert engine.handlers["my_cmd"] is handler

    def test_register_multiple_handlers(self):
        engine = ReplayEngine()
        engine.register_handler("a", lambda: "a")
        engine.register_handler("b", lambda: "b")
        assert "a" in engine.handlers
        assert "b" in engine.handlers

    def test_init_with_handlers(self):
        h = {"cmd": lambda: None}
        engine = ReplayEngine(handlers=h)
        assert "cmd" in engine.handlers

    def test_init_no_handlers_defaults_empty(self):
        engine = ReplayEngine()
        assert engine.handlers == {}


# ---------------------------------------------------------------------------
# ReplayEngine — _evaluate_condition
# ---------------------------------------------------------------------------

class TestEvaluateCondition:
    def test_equals_true(self):
        engine = ReplayEngine()
        assert engine._evaluate_condition("prod == prod") is True

    def test_equals_false(self):
        engine = ReplayEngine()
        assert engine._evaluate_condition("dev == prod") is False

    def test_not_equals_true(self):
        engine = ReplayEngine()
        assert engine._evaluate_condition("dev != prod") is True

    def test_not_equals_false(self):
        engine = ReplayEngine()
        assert engine._evaluate_condition("prod != prod") is False

    def test_nonempty_string_truthy(self):
        engine = ReplayEngine()
        assert engine._evaluate_condition("some_value") is True

    def test_empty_string_falsy(self):
        engine = ReplayEngine()
        assert engine._evaluate_condition("") is False

    def test_quoted_right_side(self):
        engine = ReplayEngine()
        assert engine._evaluate_condition('prod == "prod"') is True


# ---------------------------------------------------------------------------
# ReplayEngine — replay
# ---------------------------------------------------------------------------

def _make_recording(name="test", steps=None, variables=None) -> Recording:
    r = Recording(name=name)
    r.variables = variables or {}
    for step in (steps or []):
        r.add_step(step)
    return r


class TestReplayEngineReplay:
    async def test_empty_recording_succeeds(self):
        engine = ReplayEngine()
        result = await engine.replay(_make_recording())
        assert result.success is True
        assert result.step_results == []

    async def test_no_handler_step_fails(self):
        engine = ReplayEngine()
        step = RecordingStep(command="unknown", retry_count=1)
        recording = _make_recording(steps=[step])
        result = await engine.replay(recording)
        assert result.success is False
        assert "No handler" in result.step_results[0]["error"]

    async def test_successful_step(self):
        engine = ReplayEngine()
        engine.register_handler("greet", lambda name: f"hello {name}")
        step = RecordingStep(command="greet", args=["world"], retry_count=1)
        recording = _make_recording(steps=[step])
        result = await engine.replay(recording)
        assert result.success is True
        assert result.step_results[0]["output"] == "hello world"

    async def test_stop_on_error_default(self):
        engine = ReplayEngine()
        # Two steps, first has no handler → stops
        s1 = RecordingStep(command="bad", retry_count=1)
        s2 = RecordingStep(command="good", retry_count=1)
        engine.register_handler("good", lambda: "ok")
        recording = _make_recording(steps=[s1, s2])
        result = await engine.replay(recording)
        assert len(result.step_results) == 1

    async def test_continue_on_error(self):
        engine = ReplayEngine()
        s1 = RecordingStep(command="bad", retry_count=1)
        s2 = RecordingStep(command="good", retry_count=1)
        engine.register_handler("good", lambda: "ok")
        recording = _make_recording(steps=[s1, s2])
        result = await engine.replay(recording, stop_on_error=False)
        assert len(result.step_results) == 2

    async def test_variables_interpolated_in_args(self):
        engine = ReplayEngine()
        received = []
        engine.register_handler("capture", lambda x: received.append(x) or x)
        step = RecordingStep(command="capture", args=["${env}"], retry_count=1)
        recording = _make_recording(steps=[step], variables={"env": "staging"})
        await engine.replay(recording)
        assert received == ["staging"]

    async def test_provided_variables_override_recording(self):
        engine = ReplayEngine()
        received = []
        engine.register_handler("capture", lambda x: received.append(x) or x)
        step = RecordingStep(command="capture", args=["${env}"], retry_count=1)
        recording = _make_recording(steps=[step], variables={"env": "default"})
        await engine.replay(recording, variables={"env": "override"})
        assert received == ["override"]

    async def test_skipped_step_when_condition_not_met(self):
        engine = ReplayEngine()
        engine.register_handler("cmd", lambda: "ran")
        step = RecordingStep(command="cmd", condition="false == true", retry_count=1)
        recording = _make_recording(steps=[step])
        result = await engine.replay(recording)
        assert result.success is True
        assert result.step_results[0].get("skipped") is True

    async def test_async_handler_supported(self):
        import asyncio
        engine = ReplayEngine()

        async def async_handler(x):
            await asyncio.sleep(0)
            return f"async_{x}"

        engine.register_handler("async_cmd", async_handler)
        step = RecordingStep(command="async_cmd", args=["test"], retry_count=1)
        recording = _make_recording(steps=[step])
        result = await engine.replay(recording)
        assert result.success is True
        assert result.step_results[0]["output"] == "async_test"

    async def test_success_count_and_failed_count(self):
        engine = ReplayEngine()
        engine.register_handler("ok", lambda: "good")
        s_ok = RecordingStep(command="ok", retry_count=1)
        s_bad = RecordingStep(command="missing", retry_count=1)
        recording = _make_recording(steps=[s_ok, s_bad])
        result = await engine.replay(recording, stop_on_error=False)
        assert result.success_count == 1
        assert result.failed_count == 1

    async def test_replay_result_has_recording_name(self):
        engine = ReplayEngine()
        recording = _make_recording(name="special_name")
        result = await engine.replay(recording)
        assert result.recording_name == "special_name"
