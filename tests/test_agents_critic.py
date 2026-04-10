"""Unit tests for agents.critic.CriticAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.critic import CriticAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brain():
    b = MagicMock()
    b.remember.return_value = None
    b.recall_with_budget.return_value = []
    return b


def _model(response: str = "{}"):
    m = MagicMock(spec=["respond"])
    m.respond.return_value = response
    return m


VALID_CRITIC_DICT = {
    "initial_assessment": "Looks reasonable",
    "completeness_check": "All steps present",
    "feasibility_analysis": "Feasible within scope",
    "risk_identification": "Minor risks only",
    "overall_assessment": "approve",
    "confidence": 0.85,
    "issues": [],
    "positive_aspects": ["Clear structure"],
    "summary": "Good plan overall",
}

VALID_MUTATION_DICT = {
    "impact_analysis": "Minimal impact",
    "safety_assessment": "Safe to apply",
    "effectiveness_evaluation": "Likely effective",
    "decision": "APPROVED",
    "confidence_score": 0.9,
    "impact_assessment": "Positive change",
    "reasoning": "Logic is sound",
    "recommendations": None,
}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestCriticInit:
    def test_stores_brain_and_model(self):
        brain = _brain()
        model = _model()
        agent = CriticAgent(brain, model)
        assert agent.brain is brain
        assert agent.model is model


# ---------------------------------------------------------------------------
# critique_plan — legacy path
# ---------------------------------------------------------------------------


class TestCritiquePlanLegacy:
    def _agent(self, response: str = "some critique") -> CriticAgent:
        agent = CriticAgent(_brain(), _model(response))
        agent.use_structured = False
        return agent

    def test_returns_string_feedback(self):
        agent = self._agent("plan looks fine")
        result = agent.critique_plan("my task", ["Step 1", "Step 2"])
        assert isinstance(result, str)
        assert "plan looks fine" in result

    def test_empty_plan_list_still_calls_model(self):
        agent = self._agent("ok")
        result = agent.critique_plan("task", [])
        assert isinstance(result, str)

    def test_model_exception_propagates(self):
        brain = _brain()
        model = MagicMock(spec=["respond"])
        model.respond.side_effect = RuntimeError("LLM error")
        agent = CriticAgent(brain, model)
        agent.use_structured = False
        with pytest.raises(RuntimeError):
            agent.critique_plan("task", ["Step 1"])

    def test_brain_remember_called(self):
        brain = _brain()
        agent = CriticAgent(brain, _model("feedback"))
        agent.use_structured = False
        agent.critique_plan("fix bug", ["Step 1"])
        brain.remember.assert_called_once()


# ---------------------------------------------------------------------------
# critique_plan — structured path
# ---------------------------------------------------------------------------


class TestCritiquePlanStructured:
    def _agent_structured(self, response: str) -> CriticAgent:
        agent = CriticAgent(_brain(), _model(response))
        agent.use_structured = True
        return agent

    def test_happy_path_returns_dict_with_assessment(self):
        with patch("agents.critic.SCHEMAS_AVAILABLE", True), \
             patch("agents.critic.render_prompt", return_value="rendered"):
            agent = self._agent_structured(json.dumps(VALID_CRITIC_DICT))
            result = agent.critique_plan("task", ["Step 1"])
        assert isinstance(result, dict)
        assert result["assessment"] == "approve"

    def test_requires_changes_false_on_approve(self):
        with patch("agents.critic.SCHEMAS_AVAILABLE", True), \
             patch("agents.critic.render_prompt", return_value="rendered"):
            agent = self._agent_structured(json.dumps(VALID_CRITIC_DICT))
            result = agent.critique_plan("task", ["Step 1"])
        assert result["requires_changes"] is False

    def test_requires_changes_true_on_critical_issue(self):
        critic_with_critical = {
            **VALID_CRITIC_DICT,
            "overall_assessment": "request_changes",
            "issues": [
                {
                    "severity": "critical",
                    "category": "correctness",
                    "description": "broken logic",
                    "recommendation": "fix it",
                }
            ],
        }
        with patch("agents.critic.SCHEMAS_AVAILABLE", True), \
             patch("agents.critic.render_prompt", return_value="rendered"):
            agent = self._agent_structured(json.dumps(critic_with_critical))
            result = agent.critique_plan("task", ["Step 1"])
        assert result["requires_changes"] is True

    def test_invalid_json_falls_back_to_legacy(self):
        with patch("agents.critic.SCHEMAS_AVAILABLE", True), \
             patch("agents.critic.render_prompt", return_value="rendered"):
            agent = self._agent_structured("not json")
            result = agent.critique_plan("task", ["Step 1"])
        # fallback gives feedback_text (string from legacy)
        assert "feedback_text" in result


# ---------------------------------------------------------------------------
# critique_code — legacy path
# ---------------------------------------------------------------------------


class TestCritiqueCodeLegacy:
    def test_returns_string_from_model(self):
        agent = CriticAgent(_brain(), _model("code looks ok"))
        agent.use_structured = False
        result = agent.critique_code("fix bug", "def foo(): pass")
        assert isinstance(result, str)
        assert "code looks ok" in result

    def test_with_requirements_string(self):
        agent = CriticAgent(_brain(), _model("ok"))
        agent.use_structured = False
        result = agent.critique_code("task", "x = 1", requirements="must be fast")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# validate_mutation — legacy path
# ---------------------------------------------------------------------------


class TestValidateMutationLegacy:
    def test_approved_decision(self):
        legacy_response = json.dumps({
            "decision": "APPROVED",
            "confidence_score": 0.8,
            "impact_assessment": "positive",
            "reasoning": "valid change",
        })
        agent = CriticAgent(_brain(), _model(legacy_response))
        agent.use_structured = False
        result = agent.validate_mutation("swap function X for Y")
        assert result["approved"] is True
        assert result["decision"] == "APPROVED"

    def test_rejected_decision(self):
        legacy_response = json.dumps({
            "decision": "REJECTED",
            "confidence_score": 0.3,
            "impact_assessment": "risky",
            "reasoning": "breaks invariants",
        })
        agent = CriticAgent(_brain(), _model(legacy_response))
        agent.use_structured = False
        result = agent.validate_mutation("delete module X")
        assert result["approved"] is False

    def test_bad_json_returns_rejected(self):
        agent = CriticAgent(_brain(), _model("not json"))
        agent.use_structured = False
        result = agent.validate_mutation("some mutation")
        assert result["approved"] is False
        assert result["decision"] == "REJECTED"


# ---------------------------------------------------------------------------
# validate_mutation — structured path
# ---------------------------------------------------------------------------


class TestValidateMutationStructured:
    def test_happy_path_structured(self):
        with patch("agents.critic.SCHEMAS_AVAILABLE", True), \
             patch("agents.critic.render_prompt", return_value="rendered"):
            agent = CriticAgent(_brain(), _model(json.dumps(VALID_MUTATION_DICT)))
            agent.use_structured = True
            result = agent.validate_mutation("change X to Y")
        assert result["approved"] is True
        assert result["decision"] == "APPROVED"

    def test_structured_parse_failure_falls_back(self):
        with patch("agents.critic.SCHEMAS_AVAILABLE", True), \
             patch("agents.critic.render_prompt", return_value="rendered"):
            agent = CriticAgent(_brain(), _model("garbage"))
            agent.use_structured = True
            result = agent.validate_mutation("change X")
        # Legacy fallback should return a dict with approved key
        assert "approved" in result
