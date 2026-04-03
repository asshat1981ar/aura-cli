"""Unit tests for agents/investigation_agent.py."""
from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from agents.investigation_agent import InvestigationAgent


# ---------------------------------------------------------------------------
# Mock-based unit tests
# ---------------------------------------------------------------------------

class TestInvestigationAgentUnit(unittest.TestCase):
    """Isolated unit tests using mocks for all core helpers."""

    def _mock_investigation(self, summary="inv summary"):
        return {"summary": summary, "details": []}

    def _mock_remediation(self, summary="rem summary"):
        return {"summary": summary, "actions": []}

    @patch("agents.investigation_agent.build_remediation_plan")
    @patch("agents.investigation_agent.investigate_verification_failure")
    def test_run_returns_investigated_status(self, mock_verify, mock_remediation):
        mock_verify.return_value = self._mock_investigation()
        mock_remediation.return_value = self._mock_remediation()

        result = InvestigationAgent().run({"goal": "fix tests"})

        self.assertEqual(result["status"], "investigated")
        self.assertEqual(result["goal"], "fix tests")
        self.assertIn("verification_investigation", result)
        self.assertIn("remediation_plan", result)
        self.assertIsNone(result["test_drop_investigation"])

    @patch("agents.investigation_agent.build_remediation_plan")
    @patch("agents.investigation_agent.investigate_verification_failure")
    def test_none_input_treated_as_empty_dict(self, mock_verify, mock_remediation):
        mock_verify.return_value = self._mock_investigation()
        mock_remediation.return_value = self._mock_remediation()

        result = InvestigationAgent().run(None)

        self.assertEqual(result["status"], "investigated")
        self.assertIsNone(result["goal"])

    @patch("agents.investigation_agent.build_remediation_plan")
    @patch("agents.investigation_agent.investigate_verification_failure")
    def test_non_dict_input_treated_as_empty_dict(self, mock_verify, mock_remediation):
        mock_verify.return_value = self._mock_investigation()
        mock_remediation.return_value = self._mock_remediation()

        result = InvestigationAgent().run("not a dict")

        self.assertEqual(result["status"], "investigated")

    @patch("agents.investigation_agent.investigate_test_count_drop")
    @patch("agents.investigation_agent.build_remediation_plan")
    @patch("agents.investigation_agent.investigate_verification_failure")
    def test_test_drop_triggered_when_both_counts_present(
        self, mock_verify, mock_remediation, mock_drop
    ):
        mock_verify.return_value = self._mock_investigation("v-summary")
        mock_remediation.return_value = self._mock_remediation("r-summary")
        mock_drop.return_value = {"summary": "drop-summary"}

        result = InvestigationAgent().run(
            {"previous_test_count": 100, "current_test_count": 80}
        )

        mock_drop.assert_called_once()
        self.assertIsNotNone(result["test_drop_investigation"])
        self.assertIn("drop-summary", result["summary"])

    @patch("agents.investigation_agent.investigate_test_count_drop")
    @patch("agents.investigation_agent.build_remediation_plan")
    @patch("agents.investigation_agent.investigate_verification_failure")
    def test_test_drop_not_triggered_with_only_one_count(
        self, mock_verify, mock_remediation, mock_drop
    ):
        mock_verify.return_value = self._mock_investigation()
        mock_remediation.return_value = self._mock_remediation()

        InvestigationAgent().run({"previous_test_count": 50})

        mock_drop.assert_not_called()

    @patch("agents.investigation_agent.build_remediation_plan")
    @patch("agents.investigation_agent.investigate_verification_failure")
    def test_summary_joins_investigation_and_remediation_parts(
        self, mock_verify, mock_remediation
    ):
        mock_verify.return_value = self._mock_investigation("alpha")
        mock_remediation.return_value = self._mock_remediation("beta")

        result = InvestigationAgent().run({})

        self.assertIn("alpha", result["summary"])
        self.assertIn("beta", result["summary"])

    def test_agent_name_attribute(self):
        self.assertEqual(InvestigationAgent.name, "investigation")


# ---------------------------------------------------------------------------
# Integration-style tests (call real core helpers)
# ---------------------------------------------------------------------------


def test_investigation_agent_combines_failure_analysis_and_remediation():
    agent = InvestigationAgent()

    result = agent.run(
        {
            "goal": "Fix parser",
            "verification": {
                "failures": ["SyntaxError: invalid syntax"],
                "logs": "SyntaxError: invalid syntax",
            },
            "context": {"goal": "Fix parser", "phase": "verify", "route": "plan"},
            "route": "plan",
            "root_cause_analysis": {
                "patterns": ["syntax_error"],
                "recommended_actions": ["Inspect the generated file."],
            },
            "history": [
                {"phase_outputs": {"verification": {"failures": ["SyntaxError: invalid syntax"]}}}
            ],
        }
    )

    assert result["status"] == "investigated"
    assert result["verification_investigation"]["repeated_failure_detected"] is True
    assert result["remediation_plan"]["route"] == "replan"
    assert "repeating" in result["summary"]


def test_investigation_agent_includes_test_drop_analysis_when_counts_present():
    agent = InvestigationAgent()

    result = agent.run(
        {
            "goal": "Stabilize suite",
            "verification": {"status": "fail", "failures": ["collection error"], "logs": ""},
            "context": {"goal": "Stabilize suite", "phase": "verify", "route": "plan"},
            "route": "plan",
            "previous_test_count": 25,
            "current_test_count": 0,
        }
    )

    assert result["test_drop_investigation"] is not None
    assert result["test_drop_investigation"]["severity"] == "critical"
    assert "25 to 0" in result["test_drop_investigation"]["summary"]
