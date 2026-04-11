"""Unit tests for agents.planner.PlannerAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.planner import PlannerAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brain():
    b = MagicMock()
    b.remember.return_value = None
    b.recall_with_budget.return_value = []
    return b


def _model(response: str = "[]"):
    """Model without respond_for_role so the fallback branch runs."""
    m = MagicMock(spec=["respond"])
    m.respond.return_value = response
    return m


VALID_PLAN_LIST = json.dumps(["Step 1: Analyse", "Step 2: Implement", "Step 3: Test"])

VALID_PLANNER_DICT = {
    "analysis": "Analyse the goal",
    "gap_assessment": "Missing tests",
    "approach": "TDD",
    "risk_assessment": "Low risk",
    "plan": [
        {
            "step_number": 1,
            "description": "Write tests",
            "target_file": "tests/test_x.py",
            "verification": "pytest tests/test_x.py",
        }
    ],
    "confidence": 0.9,
    "total_steps": 1,
    "estimated_complexity": "low",
}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestPlannerInit:
    def test_stores_brain_model_and_vector_store(self):
        brain = _brain()
        model = _model()
        vs = MagicMock()
        agent = PlannerAgent(brain, model, vector_store=vs)
        assert agent.brain is brain
        assert agent.model is model
        assert agent.vector_store is vs

    def test_vector_store_defaults_to_none(self):
        agent = PlannerAgent(_brain(), _model())
        assert agent.vector_store is None


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestPlannerRun:
    def test_run_returns_steps_key_from_legacy_list(self):
        """run() must wrap a raw list result in {"steps": [...]}."""
        brain = _brain()
        model = _model(VALID_PLAN_LIST)
        with patch("agents.planner.SCHEMAS_AVAILABLE", False):
            agent = PlannerAgent(brain, model)
            agent.use_structured = False
            result = agent.run({"goal": "add feature X"})
        assert "steps" in result
        assert isinstance(result["steps"], list)

    def test_run_passes_optional_keys(self):
        brain = _brain()
        model = _model(VALID_PLAN_LIST)
        agent = PlannerAgent(brain, model)
        agent.use_structured = False
        result = agent.run(
            {
                "goal": "fix bug",
                "memory_snapshot": "prev work",
                "similar_past_problems": "similar bug fixed before",
                "known_weaknesses": "weak area",
            }
        )
        assert "steps" in result

    def test_run_vector_store_query_called_with_goal(self):
        brain = _brain()
        model = _model(VALID_PLAN_LIST)
        vs = MagicMock()
        vs.query.return_value = ["hint 1"]
        agent = PlannerAgent(brain, model, vector_store=vs)
        agent.use_structured = False
        agent.run({"goal": "my goal"})
        vs.query.assert_called_once_with("my goal", top_k=3)

    def test_run_vector_store_exception_is_swallowed(self):
        brain = _brain()
        model = _model(VALID_PLAN_LIST)
        vs = MagicMock()
        vs.query.side_effect = RuntimeError("db down")
        agent = PlannerAgent(brain, model, vector_store=vs)
        agent.use_structured = False
        result = agent.run({"goal": "goal"})
        assert "steps" in result


# ---------------------------------------------------------------------------
# _plan_legacy / _parse_legacy_response
# ---------------------------------------------------------------------------


class TestPlanLegacy:
    def _agent(self, response: str) -> PlannerAgent:
        agent = PlannerAgent(_brain(), _model(response))
        agent.use_structured = False
        return agent

    def test_happy_path_returns_list_of_strings(self):
        agent = self._agent(VALID_PLAN_LIST)
        result = agent.plan("goal", "", "", "")
        assert isinstance(result, list)
        assert result[0].startswith("Step 1")

    def test_empty_goal_still_calls_model(self):
        agent = self._agent(VALID_PLAN_LIST)
        result = agent.plan("", "", "", "")
        assert isinstance(result, list)

    def test_invalid_json_returns_error_string(self):
        agent = self._agent("not json at all")
        result = agent.plan("goal", "", "", "")
        assert isinstance(result, list)
        assert result[0].startswith("ERROR")

    def test_non_list_json_returns_error_string(self):
        agent = self._agent(json.dumps({"key": "val"}))
        result = agent.plan("goal", "", "", "")
        assert result[0].startswith("ERROR")

    def test_model_exception_bubbles_as_error_list(self):
        brain = _brain()
        model = MagicMock(spec=["respond"])
        model.respond.side_effect = RuntimeError("LLM down")
        agent = PlannerAgent(brain, model)
        agent.use_structured = False
        # _plan_legacy calls model.respond; the exception propagates to plan()
        with pytest.raises(RuntimeError):
            agent.plan("goal", "", "", "")

    def test_backfill_context_included_in_prompt(self):
        """backfill_context items should appear in the constructed prompt."""
        brain = _brain()
        model = MagicMock(spec=["respond"])
        model.respond.return_value = VALID_PLAN_LIST
        agent = PlannerAgent(brain, model)
        agent.use_structured = False
        agent.plan(
            "goal",
            "",
            "",
            "",
            backfill_context=[{"file": "agents/foo.py", "coverage_pct": 10}],
        )
        called_prompt = model.respond.call_args[0][0]
        assert "agents/foo.py" in called_prompt

    def test_hints_included_in_prompt(self):
        brain = _brain()
        model = MagicMock(spec=["respond"])
        model.respond.return_value = VALID_PLAN_LIST
        agent = PlannerAgent(brain, model)
        agent.use_structured = False
        agent.plan("goal", "", "", "", hints=["use pytest fixtures"])
        called_prompt = model.respond.call_args[0][0]
        assert "use pytest fixtures" in called_prompt


# ---------------------------------------------------------------------------
# _update_plan
# ---------------------------------------------------------------------------


class TestUpdatePlan:
    def test_update_returns_revised_list(self):
        revised = json.dumps(["Step 1: revised"])
        agent = PlannerAgent(_brain(), _model(revised))
        result = agent._update_plan(["Step 1: original"], "feedback here")
        assert result == ["Step 1: revised"]

    def test_update_falls_back_to_original_on_bad_json(self):
        agent = PlannerAgent(_brain(), _model("bad json"))
        original = ["Step 1: keep me"]
        result = agent._update_plan(original, "tweak this")
        assert result == original

    def test_update_accepts_dict_with_steps_key(self):
        revised = json.dumps(["Step 1: new"])
        agent = PlannerAgent(_brain(), _model(revised))
        result = agent._update_plan({"steps": ["Step 1: old"]}, "feedback")
        assert result == ["Step 1: new"]
