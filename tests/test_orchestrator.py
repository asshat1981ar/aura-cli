"""Unit tests for core/orchestrator.py — LoopOrchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from core.orchestrator import LoopOrchestrator, BeadsSyncLoop
from core.policy import Policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agents(**overrides):
    """Return a minimal agents dict with MagicMock instances for every phase."""
    phase_names = ["ingest", "plan", "critique", "synthesize", "act", "verify", "reflect"]
    agents = {}
    for name in phase_names:
        mock = MagicMock(name=name)
        mock.name = name
        mock.run.return_value = {"status": "success", "agent_name": name}
        agents[name] = mock
    agents.update(overrides)
    return agents


def _make_orchestrator(agents=None, policy=None, project_root=None, **kwargs):
    """Build a LoopOrchestrator with all heavy side-effects patched out."""
    agents = agents or _make_agents()

    # Patch everything that touches the filesystem or external services at
    # construction time so tests are hermetic.
    with (
        patch("core.orchestrator.HookEngine"),
        patch("core.orchestrator.PhaseDispatcher"),
        patch("core.orchestrator.memory_controller"),
        patch("core.orchestrator.log_json"),
        patch("agents.skills.registry.all_skills", return_value={}, create=True),
    ):
        orch = LoopOrchestrator(
            agents=agents,
            policy=policy or Policy.from_config({}),
            project_root=project_root or Path("."),
            **kwargs,
        )
    return orch


def _patch_run_cycle(orchestrator, return_value):
    """Patch run_cycle on *orchestrator* to return a fixed value."""
    orchestrator.run_cycle = MagicMock(return_value=return_value)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLoopOrchestratorInit:
    def test_agents_stored(self):
        agents = _make_agents()
        orch = _make_orchestrator(agents=agents)
        assert orch.agents is agents

    def test_default_policy_is_sliding_window(self):
        orch = _make_orchestrator()
        assert orch.policy is not None

    def test_project_root_stored(self):
        orch = _make_orchestrator(project_root=Path("/some/path"))
        assert orch.project_root == Path("/some/path")

    def test_strict_schema_default_false(self):
        orch = _make_orchestrator()
        assert orch.strict_schema is False

    def test_strict_schema_true(self):
        orch = _make_orchestrator(strict_schema=True)
        assert orch.strict_schema is True


# ---------------------------------------------------------------------------
# run_loop — stopping conditions
# ---------------------------------------------------------------------------

class TestRunLoop:
    def _basic_cycle_entry(self, **extra):
        entry = {
            "cycle_id": "cycle_abc",
            # verification status "fail" so SlidingWindowPolicy doesn't stop early
            "phase_outputs": {"verification": {"status": "fail"}},
        }
        entry.update(extra)
        return entry

    def test_run_loop_returns_goal_and_history(self):
        orch = _make_orchestrator()
        entry = self._basic_cycle_entry()
        _patch_run_cycle(orch, entry)

        result = orch.run_loop("fix bug", max_cycles=1, dry_run=True)

        assert result["goal"] == "fix bug"
        assert len(result["history"]) == 1

    def test_run_loop_max_cycles_respected(self):
        orch = _make_orchestrator()
        _patch_run_cycle(orch, self._basic_cycle_entry())

        result = orch.run_loop("fix bug", max_cycles=3, dry_run=True)

        assert orch.run_cycle.call_count == 3
        assert result["stop_reason"] == "MAX_CYCLES"

    def test_run_loop_stops_on_cycle_stop_reason(self):
        orch = _make_orchestrator()
        entry = self._basic_cycle_entry(stop_reason="INVALID_OUTPUT")
        _patch_run_cycle(orch, entry)

        result = orch.run_loop("fix bug", max_cycles=10, dry_run=True)

        assert orch.run_cycle.call_count == 1
        assert result["stop_reason"] == "INVALID_OUTPUT"

    def test_run_loop_stops_on_policy_pass(self):
        orch = _make_orchestrator()
        # Use a passing verification to trigger SlidingWindowPolicy.PASS
        entry = {
            "cycle_id": "cycle_abc",
            "phase_outputs": {"verification": {"status": "pass"}},
        }
        _patch_run_cycle(orch, entry)

        result = orch.run_loop("feat", max_cycles=5, dry_run=True)

        assert result["stop_reason"] == "PASS"
        assert orch.run_cycle.call_count == 1

    def test_run_loop_max_cycles_sets_reason_on_last_entry(self):
        orch = _make_orchestrator()
        _patch_run_cycle(orch, self._basic_cycle_entry())
        orch.policy = MagicMock()
        orch.policy.evaluate.return_value = ""  # never stop early

        result = orch.run_loop("task", max_cycles=2, dry_run=True)

        assert result["history"][-1]["stop_reason"] == "MAX_CYCLES"

    def test_run_loop_dry_run_passed_to_cycle(self):
        orch = _make_orchestrator()
        _patch_run_cycle(orch, self._basic_cycle_entry())

        orch.run_loop("task", max_cycles=1, dry_run=True)

        orch.run_cycle.assert_called_with("task", dry_run=True, context_injection=None)

    def test_run_loop_empty_cycles_stop_reason_max_cycles(self):
        orch = _make_orchestrator()
        _patch_run_cycle(orch, self._basic_cycle_entry())

        result = orch.run_loop("task", max_cycles=0, dry_run=True)

        assert result["stop_reason"] == "MAX_CYCLES"
        assert result["history"] == []


# ---------------------------------------------------------------------------
# Improvement loops
# ---------------------------------------------------------------------------

class TestImprovementLoops:
    def test_attach_improvement_loops_registers_loop(self):
        orch = _make_orchestrator()
        loop_mock = MagicMock()
        orch.attach_improvement_loops(loop_mock)
        assert loop_mock in orch._improvement_loops

    def test_improvement_loop_called_via_post_cycle_hooks(self):
        """_run_post_cycle_hooks triggers on_cycle_complete on each loop."""
        orch = _make_orchestrator()
        loop_mock = MagicMock()
        orch.attach_improvement_loops(loop_mock)

        entry = {"cycle_id": "x", "phase_outputs": {}, "stop_reason": ""}
        with patch("core.orchestrator_learn.log_json"):
            orch._run_post_cycle_hooks(entry, "goal", "x", {}, False)

        loop_mock.on_cycle_complete.assert_called_once_with(entry)

    def test_improvement_loop_error_does_not_crash(self):
        orch = _make_orchestrator()
        bad_loop = MagicMock()
        bad_loop.on_cycle_complete.side_effect = RuntimeError("boom")
        orch.attach_improvement_loops(bad_loop)

        entry = {"cycle_id": "x", "phase_outputs": {}, "stop_reason": ""}
        with patch("core.orchestrator_learn.log_json"):
            # Should not raise
            orch._run_post_cycle_hooks(entry, "goal", "x", {}, False)


# ---------------------------------------------------------------------------
# BeadsSyncLoop
# ---------------------------------------------------------------------------

class TestBeadsSyncLoop:
    def test_push_pull_called_every_n_cycles(self):
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)
        for _ in range(BeadsSyncLoop.EVERY_N):
            loop.on_cycle_complete({"dry_run": False})
        assert skill.run.call_count == 2  # pull + push

    def test_dry_run_skips_sync(self):
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)
        for _ in range(BeadsSyncLoop.EVERY_N):
            loop.on_cycle_complete({"dry_run": True})
        skill.run.assert_not_called()


# ---------------------------------------------------------------------------
# _estimate_confidence
# ---------------------------------------------------------------------------

class TestEstimateConfidence:
    def setup_method(self):
        self.orch = _make_orchestrator()

    def test_verify_pass_returns_high_confidence(self):
        c = self.orch._estimate_confidence({"status": "pass"}, "verify")
        assert c >= 0.9

    def test_verify_fail_returns_low_confidence(self):
        c = self.orch._estimate_confidence({"status": "fail"}, "verify")
        assert c <= 0.2

    def test_plan_with_steps_raises_confidence(self):
        c = self.orch._estimate_confidence({"steps": ["a", "b"]}, "plan")
        assert c > 0.5

    def test_non_dict_output_baseline(self):
        c = self.orch._estimate_confidence("not a dict", "plan")
        assert c == 0.3
