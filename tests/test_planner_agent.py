"""Unit tests for agents.planner.PlannerAgent — targets ≥90% coverage.

Covers:
- __init__ with and without SCHEMAS_AVAILABLE
- run() routing logic
- _respond() role dispatch
- plan() → _plan_legacy() path including all parse branches
- plan() → _plan_structured() path including fallbacks and edge cases
- _parse_legacy_response() all branches
- _update_plan() all branches
- get_cache_stats() both availability states
- Edge cases: empty goal, model exception, backfill_context variants
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from agents.planner import PlannerAgent
from tests.fixtures.mock_llm import MockModelAdapter


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_brain() -> MagicMock:
    brain = MagicMock()
    brain.get_memories.return_value = []
    brain.remember.return_value = None
    return brain


def _make_model(response: str = "{}") -> MagicMock:
    """Return a MagicMock model whose .respond() returns *response*.

    Uses spec=["respond"] so ``inspect.getattr_static`` won't find
    ``respond_for_role``, exercising the fallback branch of _respond().
    """
    model = MagicMock(spec=["respond"])
    model.respond.return_value = response
    return model


# Valid structured planner output (PlannerOutput-compatible dict).
VALID_PLANNER_DICT: dict = {
    "analysis": "The goal requires implementing a new feature.",
    "gap_assessment": "Tests are missing for module X.",
    "approach": "TDD approach with incremental commits.",
    "risk_assessment": "Low risk due to isolated scope.",
    "plan": [
        {
            "step_number": 1,
            "description": "Write unit tests for feature X",
            "target_file": "tests/test_feature_x.py",
            "verification": "pytest tests/test_feature_x.py",
        }
    ],
    "confidence": 0.85,
    "total_steps": 1,
    "estimated_complexity": "low",
}

VALID_LEGACY_LIST: list = ["Step 1: Analyse the codebase", "Step 2: Implement changes"]


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestPlannerAgentInit:
    def test_stores_brain_and_model(self):
        brain = _make_brain()
        model = _make_model()
        agent = PlannerAgent(brain, model)
        assert agent.brain is brain
        assert agent.model is model

    def test_capabilities_class_attribute(self):
        assert "planning" in PlannerAgent.capabilities
        assert "decomposition" in PlannerAgent.capabilities

    def test_use_structured_true_when_schemas_available(self):
        with patch("agents.planner.SCHEMAS_AVAILABLE", True):
            agent = PlannerAgent(_make_brain(), _make_model())
        assert agent.use_structured is True

    def test_use_structured_false_when_schemas_unavailable(self):
        with patch("agents.planner.SCHEMAS_AVAILABLE", False):
            agent = PlannerAgent(_make_brain(), _make_model())
        assert agent.use_structured is False


# ---------------------------------------------------------------------------
# _respond()
# ---------------------------------------------------------------------------

class TestPlannerRespond:
    """Tests for the internal _respond() role-dispatch helper."""

    def test_falls_back_to_respond_when_model_has_no_respond_for_role(self):
        """Model without respond_for_role → model.respond() is called."""
        brain = _make_brain()
        model = _make_model("plain response")
        agent = PlannerAgent(brain, model)
        result = agent._respond("some prompt")
        assert result == "plain response"
        model.respond.assert_called_once_with("some prompt")

    def test_uses_respond_for_role_when_available(self):
        """MockModelAdapter defines respond_for_role → that method is used."""
        brain = _make_brain()
        model = MockModelAdapter({"keyword": "role_response"})
        agent = PlannerAgent(brain, model)
        result = agent._respond("a prompt with keyword in it")
        assert result == "role_response"

    def test_falls_back_to_respond_when_respond_for_role_not_callable(self):
        """respond_for_role attribute exists but is not callable → model.respond()."""

        class ModelWithNonCallableRFR:
            respond_for_role = "this_is_a_string_not_a_function"

            def respond(self, prompt):
                return "respond_result"

        agent = PlannerAgent(_make_brain(), ModelWithNonCallableRFR())
        result = agent._respond("hello")
        assert result == "respond_result"


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestPlannerRun:
    """Tests for the public run() method that wraps plan()."""

    def _make_agent(self) -> PlannerAgent:
        agent = PlannerAgent(_make_brain(), _make_model())
        agent.use_structured = False
        return agent

    def test_run_wraps_list_result_in_steps_key(self):
        agent = self._make_agent()
        with patch.object(agent, "plan", return_value=VALID_LEGACY_LIST):
            result = agent.run({"goal": "do something"})
        assert result == {"steps": VALID_LEGACY_LIST}

    def test_run_passes_through_dict_with_plan_key(self):
        agent = self._make_agent()
        plan_dict = {"plan": "structured_plan_value", "steps": ["a"]}
        with patch.object(agent, "plan", return_value=plan_dict):
            result = agent.run({"goal": "x"})
        assert result is plan_dict

    def test_run_wraps_dict_without_plan_key_in_steps(self):
        """Structured dicts that lack 'plan' key are wrapped as {"steps": <dict>}."""
        agent = self._make_agent()
        structured = {"steps": ["a"], "confidence": 0.9}  # no "plan" key
        with patch.object(agent, "plan", return_value=structured):
            result = agent.run({"goal": "x"})
        assert result == {"steps": structured}

    def test_run_defaults_empty_strings_for_missing_input_keys(self):
        agent = self._make_agent()
        captured: dict = {}

        def _capture(goal, mem, sim, weak, backfill_context=None):
            captured.update({"goal": goal, "mem": mem, "sim": sim, "weak": weak})
            return []

        with patch.object(agent, "plan", side_effect=_capture):
            agent.run({})
        assert captured["goal"] == ""
        assert captured["mem"] == ""
        assert captured["sim"] == ""
        assert captured["weak"] == ""

    def test_run_passes_all_fields_to_plan(self):
        agent = self._make_agent()
        captured: dict = {}

        def _capture(goal, mem, sim, weak, backfill_context=None):
            captured.update({
                "goal": goal, "sim": sim,
                "weak": weak, "bc": backfill_context,
            })
            return []

        with patch.object(agent, "plan", side_effect=_capture):
            agent.run({
                "goal": "test goal",
                "memory_snapshot": "mem_text",
                "similar_past_problems": "sim_text",
                "known_weaknesses": "weak_text",
                "backfill_context": [{"file": "f.py", "coverage_pct": 20}],
            })
        assert captured["goal"] == "test goal"
        assert captured["bc"] == [{"file": "f.py", "coverage_pct": 20}]


# ---------------------------------------------------------------------------
# plan() → _plan_legacy() path
# ---------------------------------------------------------------------------

class TestPlanLegacy:
    """Tests for the legacy (non-structured) planning path."""

    def _agent(self, model_response: str) -> PlannerAgent:
        agent = PlannerAgent(_make_brain(), _make_model(model_response))
        agent.use_structured = False
        return agent

    def test_happy_path_returns_list_of_strings(self):
        response = json.dumps(VALID_LEGACY_LIST)
        agent = self._agent(response)
        with patch("agents.planner.log_json"):
            result = agent.plan("my goal", "", "", "")
        assert result == VALID_LEGACY_LIST
        agent.brain.remember.assert_called()

    def test_non_json_response_returns_error_list(self):
        agent = self._agent("this is not json at all")
        with patch("agents.planner.log_json"):
            result = agent.plan("my goal", "", "", "")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].startswith("ERROR:")

    def test_response_is_dict_not_list_returns_error(self):
        agent = self._agent('{"key": "value"}')
        with patch("agents.planner.log_json"):
            result = agent.plan("my goal", "", "", "")
        assert isinstance(result, list)
        assert result[0].startswith("ERROR:")

    def test_response_is_list_of_non_strings_returns_error(self):
        agent = self._agent(json.dumps([1, 2, 3]))
        with patch("agents.planner.log_json"):
            result = agent.plan("my goal", "", "", "")
        assert isinstance(result, list)
        assert result[0].startswith("ERROR:")

    def test_backfill_context_with_coverage_pct_key(self):
        response = json.dumps(VALID_LEGACY_LIST)
        agent = self._agent(response)
        backfill = [{"file": "src/main.py", "coverage_pct": 15.0}]
        with patch("agents.planner.log_json"):
            result = agent.plan("goal", "", "", "", backfill_context=backfill)
        assert result == VALID_LEGACY_LIST

    def test_backfill_context_with_coverage_key_fallback(self):
        """backfill items may use 'coverage' instead of 'coverage_pct'."""
        response = json.dumps(VALID_LEGACY_LIST)
        agent = self._agent(response)
        backfill = [{"file": "src/main.py", "coverage": 5.0}]
        with patch("agents.planner.log_json"):
            result = agent.plan("goal", "", "", "", backfill_context=backfill)
        assert result == VALID_LEGACY_LIST

    def test_empty_goal_still_returns_list(self):
        response = json.dumps(VALID_LEGACY_LIST)
        agent = self._agent(response)
        with patch("agents.planner.log_json"):
            result = agent.plan("", "", "", "")
        assert isinstance(result, list)

    def test_model_exception_propagates(self):
        """If _respond raises, the exception propagates out of plan()."""
        agent = PlannerAgent(_make_brain(), _make_model())
        agent.use_structured = False
        agent.model.respond.side_effect = RuntimeError("model down")
        with pytest.raises(RuntimeError, match="model down"):
            agent.plan("goal", "", "", "")


# ---------------------------------------------------------------------------
# plan() → _plan_structured() path
# ---------------------------------------------------------------------------

class TestPlanStructured:
    """Tests for the structured planning path using PlannerOutput schema."""

    def _agent(self) -> PlannerAgent:
        agent = PlannerAgent(_make_brain(), _make_model("fake_response"))
        agent.use_structured = True
        return agent

    def test_happy_path_returns_dict_with_required_keys(self):
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="rendered_prompt") as mock_render,
            patch("agents.planner._aura_safe_loads", return_value=dict(VALID_PLANNER_DICT)),
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("build feature X", "mem", "sim", "weak")

        assert isinstance(result, dict)
        assert "steps" in result
        assert "confidence" in result
        assert "complexity" in result
        assert "reasoning" in result
        assert "structured_output" in result
        assert isinstance(result["steps"], list)
        mock_render.assert_called_once()

    def test_happy_path_step_with_target_file_includes_brackets(self):
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads", return_value=dict(VALID_PLANNER_DICT)),
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("goal", "", "", "")
        # VALID_PLANNER_DICT has target_file set → should appear in brackets
        assert "tests/test_feature_x.py" in result["steps"][0]

    def test_happy_path_step_without_target_file_no_brackets(self):
        plan_dict = dict(VALID_PLANNER_DICT)
        plan_dict["plan"] = [
            {
                "step_number": 1,
                "description": "Do something abstract",
                "target_file": None,
                "verification": "check output",
            }
        ]
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads", return_value=plan_dict),
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("goal", "", "", "")
        assert "[" not in result["steps"][0]

    def test_calls_brain_remember(self):
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads", return_value=dict(VALID_PLANNER_DICT)),
            patch("agents.planner.log_json"),
        ):
            agent.plan("goal", "", "", "")
        agent.brain.remember.assert_called()

    def test_fallback_on_json_decode_error(self):
        """JSONDecodeError triggers fallback to _parse_legacy_response()."""
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads",
                  side_effect=json.JSONDecodeError("bad json", "", 0)),
            patch.object(agent, "_parse_legacy_response",
                         return_value=["fallback step"]) as mock_legacy,
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("goal", "", "", "")
        mock_legacy.assert_called_once()
        assert result == ["fallback step"]

    def test_fallback_on_validation_error_bad_dict(self):
        """Dict that fails PlannerOutput validation triggers fallback."""
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads", return_value={"bad": "data"}),
            patch.object(agent, "_parse_legacy_response",
                         return_value=["fallback"]) as mock_legacy,
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("goal", "", "", "")
        mock_legacy.assert_called_once()
        assert result == ["fallback"]

    def test_fallback_on_type_error(self):
        """TypeError during PlannerOutput(**parsed) triggers legacy fallback."""
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads", side_effect=TypeError("bad type")),
            patch.object(agent, "_parse_legacy_response",
                         return_value=["type_fallback"]) as mock_legacy,
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("goal", "", "", "")
        mock_legacy.assert_called_once()
        assert result == ["type_fallback"]

    def test_fallback_on_key_error(self):
        """KeyError triggers legacy fallback."""
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads", side_effect=KeyError("missing")),
            patch.object(agent, "_parse_legacy_response",
                         return_value=["key_fallback"]) as mock_legacy,
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("goal", "", "", "")
        mock_legacy.assert_called_once()
        assert result == ["key_fallback"]

    def test_unexpected_exception_returns_error_list(self):
        """Unexpected exceptions (not JSON/Validation) return error list."""
        agent = self._agent()
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads",
                  side_effect=RuntimeError("unexpected boom")),
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("goal", "", "", "")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "ERROR:" in result[0]
        assert "unexpected boom" in result[0]

    def test_with_backfill_context_structured_path(self):
        agent = self._agent()
        backfill = [{"file": "a.py", "coverage_pct": 30.0}]
        with (
            patch("agents.planner.render_prompt", return_value="p") as mock_render,
            patch("agents.planner._aura_safe_loads", return_value=dict(VALID_PLANNER_DICT)),
            patch("agents.planner.log_json"),
        ):
            result = agent.plan("goal", "", "", "", backfill_context=backfill)
        # render_prompt is still called once on the structured path
        mock_render.assert_called_once()
        assert "steps" in result

    def test_render_prompt_receives_correct_params(self):
        agent = self._agent()
        call_args_captured: dict = {}

        def _fake_render(template_name, role, params):
            call_args_captured.update({"template_name": template_name, "role": role,
                                       "params": params})
            return "rendered"

        with (
            patch("agents.planner.render_prompt", side_effect=_fake_render),
            patch("agents.planner._aura_safe_loads", return_value=dict(VALID_PLANNER_DICT)),
            patch("agents.planner.log_json"),
        ):
            agent.plan("my goal", "mem_text", "sim_text", "weak_text")

        assert call_args_captured["template_name"] == "planner"
        assert call_args_captured["role"] == "planner"
        assert call_args_captured["params"]["goal"] == "my goal"


# ---------------------------------------------------------------------------
# _update_plan()
# ---------------------------------------------------------------------------

class TestUpdatePlan:
    """Tests for the plan revision method."""

    def _agent(self, response: str) -> PlannerAgent:
        agent = PlannerAgent(_make_brain(), _make_model(response))
        return agent

    def test_happy_path_returns_revised_list(self):
        revised = ["Step 1: revised", "Step 2: added"]
        agent = self._agent(json.dumps(revised))
        with patch("agents.planner.log_json"):
            result = agent._update_plan(VALID_LEGACY_LIST, "needs improvement")
        assert result == revised
        agent.brain.remember.assert_called()

    def test_accepts_dict_input_with_steps_key(self):
        revised = ["Step 1: v2"]
        agent = self._agent(json.dumps(revised))
        plan_dict = {"steps": VALID_LEGACY_LIST, "confidence": 0.9}
        with patch("agents.planner.log_json"):
            result = agent._update_plan(plan_dict, "feedback")
        assert result == revised

    def test_invalid_response_returns_original_plan(self):
        agent = self._agent('{"not": "a list"}')
        with patch("agents.planner.log_json"):
            result = agent._update_plan(VALID_LEGACY_LIST, "feedback")
        assert result == VALID_LEGACY_LIST

    def test_exception_returns_original_plan(self):
        agent = self._agent("bad json {{{{")
        with patch("agents.planner.log_json"):
            result = agent._update_plan(VALID_LEGACY_LIST, "feedback")
        assert result == VALID_LEGACY_LIST

    def test_list_of_non_strings_returns_original_plan(self):
        agent = self._agent(json.dumps([1, 2]))
        with patch("agents.planner.log_json"):
            result = agent._update_plan(VALID_LEGACY_LIST, "feedback")
        assert result == VALID_LEGACY_LIST


# ---------------------------------------------------------------------------
# get_cache_stats()
# ---------------------------------------------------------------------------

class TestGetCacheStats:
    def test_returns_stats_dict_when_schemas_available(self):
        brain = _make_brain()
        model = _make_model()
        agent = PlannerAgent(brain, model)
        agent.use_structured = True  # ensure structured path
        fake_stats = {"hits": 3, "misses": 1, "size": 2}
        with patch("agents.planner.get_cached_prompt_stats", return_value=fake_stats):
            result = agent.get_cache_stats()
        assert result == fake_stats

    def test_returns_error_dict_when_schemas_unavailable(self):
        brain = _make_brain()
        model = _make_model()
        with patch("agents.planner.SCHEMAS_AVAILABLE", False):
            agent = PlannerAgent(brain, model)
        # Need to also patch the module-level flag since get_cache_stats checks it
        with patch("agents.planner.SCHEMAS_AVAILABLE", False):
            result = agent.get_cache_stats()
        assert "error" in result


# ---------------------------------------------------------------------------
# Integration-style: run() → plan() end-to-end (legacy path)
# ---------------------------------------------------------------------------

class TestRunPlanIntegration:
    """Light integration tests exercising run() all the way through plan()."""

    def test_full_legacy_flow(self):
        response = json.dumps(["Step 1: analyse", "Step 2: implement"])
        agent = PlannerAgent(_make_brain(), _make_model(response))
        agent.use_structured = False
        with patch("agents.planner.log_json"):
            result = agent.run({"goal": "build feature", "memory_snapshot": "nothing"})
        assert "steps" in result
        assert result["steps"] == ["Step 1: analyse", "Step 2: implement"]

    def test_full_structured_flow(self):
        agent = PlannerAgent(_make_brain(), _make_model("fake_json"))
        agent.use_structured = True
        with (
            patch("agents.planner.render_prompt", return_value="p"),
            patch("agents.planner._aura_safe_loads", return_value=dict(VALID_PLANNER_DICT)),
            patch("agents.planner.log_json"),
        ):
            result = agent.run({"goal": "structured goal"})
        # structured path returns dict without "plan" key → wrapped in {"steps": ...}
        assert "steps" in result
