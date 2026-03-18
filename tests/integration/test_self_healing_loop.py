"""Integration test: self-healing feedback loop.

Tests the full cycle: failure detection → route_failure → re-plan/re-act → verify pass.
Uses mock agents that return controlled outputs to simulate the loop behavior.
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.orchestrator import LoopOrchestrator
from core.policy import Policy


def _make_agent(name, return_value=None):
    agent = MagicMock()
    agent.name = name
    agent.run.return_value = return_value or {"status": "success"}
    return agent


class TestSelfHealingLoop(unittest.TestCase):
    """Test that the orchestrator can recover from failures via re-planning."""

    def setUp(self):
        self.agents = {
            "ingest": _make_agent("ingest", {"goal": "test", "context": "ctx"}),
            "plan": _make_agent("plan", {"steps": ["step1"]}),
            "critique": _make_agent("critique", {"feedback": "ok"}),
            "synthesize": _make_agent("synthesize", {"tasks": [{"tests": []}]}),
            "act": _make_agent("act", {"changes": []}),
            "verify": _make_agent("verify"),
            "reflect": _make_agent("reflect", {"summary": "done"}),
        }
        self.orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=MagicMock(**{
                "get.return_value": None,
                "recall_all.return_value": [],
                "recall_recent.return_value": [],
                "recall_with_budget.return_value": [],
                "set.return_value": None,
                "remember.return_value": None,
            }),
            project_root=Path("."),
            policy=Policy.from_config({}),
        )

    def test_successful_cycle_has_no_stop_reason(self):
        """A clean cycle completes without a stop reason."""
        self.agents["verify"].run.return_value = {"status": "pass", "failures": [], "logs": ""}
        result = self.orchestrator.run_cycle("test goal", dry_run=True)
        assert result.get("stop_reason") is None

    def test_circuit_breaker_activates_after_consecutive_failures(self):
        """After threshold failures, the circuit breaker blocks new cycles."""
        self.agents["verify"].run.return_value = {"status": "fail", "failures": ["err"], "logs": ""}
        for _ in range(5):
            self.orchestrator.run_cycle("failing goal", dry_run=True)

        result = self.orchestrator.run_cycle("blocked goal", dry_run=True)
        assert result["stop_reason"] == "CIRCUIT_BREAKER_OPEN"

    def test_improvement_loop_receives_cycle_entry(self):
        """Attached improvement loops are called with the cycle entry."""
        mock_loop = MagicMock()
        self.orchestrator.attach_improvement_loops(mock_loop)
        self.agents["verify"].run.return_value = {"status": "pass", "failures": [], "logs": ""}

        self.orchestrator.run_cycle("test goal", dry_run=True)

        mock_loop.on_cycle_complete.assert_called_once()
        entry = mock_loop.on_cycle_complete.call_args[0][0]
        assert entry["goal"] == "test goal"

    def test_improvement_loop_error_does_not_crash_cycle(self):
        """An error in an improvement loop is logged but doesn't crash."""
        failing_loop = MagicMock()
        failing_loop.on_cycle_complete.side_effect = RuntimeError("boom")
        self.orchestrator.attach_improvement_loops(failing_loop)
        self.agents["verify"].run.return_value = {"status": "pass", "failures": [], "logs": ""}

        # Should not raise
        result = self.orchestrator.run_cycle("test goal", dry_run=True)
        assert result.get("stop_reason") is None


if __name__ == "__main__":
    unittest.main()
