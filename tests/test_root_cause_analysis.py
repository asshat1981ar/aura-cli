"""Unit tests for agents/root_cause_analysis.py."""
from __future__ import annotations

import pytest

from agents.root_cause_analysis import RootCauseAnalysisAgent


class TestRootCauseAnalysisAgentInit:
    def test_name(self):
        agent = RootCauseAnalysisAgent()
        assert agent.name == "root_cause_analysis"


class TestPatternMatching:
    def setup_method(self):
        self.agent = RootCauseAnalysisAgent()

    def test_import_error_pattern(self):
        result = self.agent.run({"failures": ["ModuleNotFoundError: No module named 'foo'"]})
        assert "import_error" in result["patterns"]
        assert result["status"] == "analyzed"

    def test_syntax_error_pattern(self):
        result = self.agent.run({"failures": ["SyntaxError: invalid syntax at line 42"]})
        assert "syntax_error" in result["patterns"]

    def test_timeout_pattern(self):
        result = self.agent.run({"failures": ["Task timed out after 60 seconds"]})
        assert "timeout" in result["patterns"]

    def test_assertion_failure_pattern(self):
        result = self.agent.run({"failures": ["AssertionError: expected True"]})
        assert "assertion_failure" in result["patterns"]

    def test_permission_pattern(self):
        result = self.agent.run({"failures": ["PermissionError: Permission denied: '/etc/secret'"]})
        assert "permission" in result["patterns"]

    def test_network_pattern(self):
        result = self.agent.run({"failures": ["Connection refused to 127.0.0.1:8001"]})
        assert "network" in result["patterns"]

    def test_fallback_pattern_on_no_match(self):
        result = self.agent.run({"failures": ["Something totally generic happened"]})
        assert "unknown_failure" in result["patterns"]
        assert result["confidence"] == 0.45

    def test_logs_also_matched(self):
        result = self.agent.run({"failures": [], "logs": "ImportError: cannot import name X"})
        assert "import_error" in result["patterns"]


class TestConfidenceAndSummary:
    def setup_method(self):
        self.agent = RootCauseAnalysisAgent()

    def test_confidence_is_float_between_0_and_1(self):
        result = self.agent.run({"failures": ["SyntaxError: bad syntax"]})
        assert 0.0 < result["confidence"] <= 1.0

    def test_summary_contains_goal_when_provided(self):
        result = self.agent.run({"failures": ["SyntaxError: x"], "goal": "refactor module"})
        assert "refactor module" in result["summary"]

    def test_summary_contains_phase_from_context(self):
        result = self.agent.run({
            "failures": ["SyntaxError: x"],
            "context": {"phase": "apply"},
        })
        assert "apply" in result["summary"]

    def test_repeated_failure_detected(self):
        failure_text = "syntaxerror: bad code"
        history = [
            {"failure": failure_text},
            {"failure": failure_text},
        ]
        result = self.agent.run({"failures": [failure_text], "history": history})
        assert result["repeated_failure_detected"] is True
        assert any("repeat" in a.lower() for a in result["recommended_actions"])

    def test_no_repeated_failure_without_history(self):
        result = self.agent.run({"failures": ["SyntaxError: x"]})
        assert result["repeated_failure_detected"] is False


class TestEvidence:
    def setup_method(self):
        self.agent = RootCauseAnalysisAgent()

    def test_evidence_contains_failures(self):
        result = self.agent.run({"failures": ["ModuleNotFoundError: foo"]})
        assert "ModuleNotFoundError: foo" in result["evidence"]["failures"]

    def test_evidence_history_count(self):
        result = self.agent.run({"failures": ["err"], "history": [{"x": 1}, {"x": 2}]})
        assert result["evidence"]["history_count"] == 2

    def test_evidence_logs_truncated(self):
        long_log = "x" * 5000
        result = self.agent.run({"failures": [], "logs": long_log})
        assert len(result["evidence"]["logs"]) <= 2000


class TestDeduplication:
    def test_recommended_actions_are_unique(self):
        agent = RootCauseAnalysisAgent()
        # syntax_error and assertion_failure both produce actions; ensure no dupes
        result = agent.run({"failures": ["SyntaxError: x", "AssertionError: y"]})
        actions = result["recommended_actions"]
        assert len(actions) == len(set(actions))
