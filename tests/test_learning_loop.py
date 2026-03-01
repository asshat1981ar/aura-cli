"""35+ tests for PRD-003: Autonomous Learning Loop."""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.cycle_outcome import CycleOutcome
from core.quality_snapshot import run_quality_snapshot
from core.adaptive_pipeline import AdaptivePipeline
from core.autonomous_discovery import AutonomousDiscovery


# ── TestCycleOutcome ─────────────────────────────────────────────────────────

class TestCycleOutcome:
    def test_default_fields(self):
        co = CycleOutcome()
        assert co.goal == ""
        assert co.goal_type == ""
        assert co.success is False
        assert co.failure_phase is None
        assert co.failure_reason is None
        assert co.phases_completed == []

    def test_mark_complete_success(self):
        co = CycleOutcome()
        co.mark_complete(True)
        assert co.success is True
        assert co.completed_at > 0

    def test_mark_complete_failure_with_reason(self):
        co = CycleOutcome()
        co.mark_complete(False, failure_phase="verify", failure_reason="lint fail")
        assert co.success is False
        assert co.failure_phase == "verify"
        assert co.failure_reason == "lint fail"

    def test_tests_delta_computed(self):
        co = CycleOutcome(tests_before=5, tests_after=8)
        co.mark_complete(True)
        assert co.tests_delta == 3

    def test_to_json_from_json_roundtrip(self):
        co = CycleOutcome(goal="fix x", goal_type="bug_fix", changes_applied=2)
        co.mark_complete(True)
        co2 = CycleOutcome.from_json(co.to_json())
        assert co2.cycle_id == co.cycle_id
        assert co2.goal == "fix x"
        assert co2.changes_applied == 2
        assert co2.success is True

    def test_cycle_id_unique(self):
        ids = {CycleOutcome().cycle_id for _ in range(10)}
        assert len(ids) == 10

    def test_duration_s(self):
        co = CycleOutcome()
        co.started_at = 50.0
        co.completed_at = 52.5
        assert co.duration_s() == pytest.approx(2.5)

    def test_to_json_is_string(self):
        assert isinstance(CycleOutcome().to_json(), str)

    def test_from_json_valid(self):
        data = CycleOutcome(goal="hello")
        restored = CycleOutcome.from_json(data.to_json())
        assert restored.goal == "hello"

    def test_brain_entries_added_default_zero(self):
        co = CycleOutcome()
        assert co.brain_entries_added == 0


# ── TestQualitySnapshot ──────────────────────────────────────────────────────

class TestQualitySnapshot:
    def test_returns_dict(self, tmp_path):
        result = run_quality_snapshot(tmp_path)
        assert isinstance(result, dict)

    def test_has_test_count_key(self, tmp_path):
        result = run_quality_snapshot(tmp_path)
        assert "test_count" in result or "error" in result

    def test_has_syntax_errors_key(self, tmp_path):
        result = run_quality_snapshot(tmp_path)
        if "error" not in result:
            assert "syntax_errors" in result

    def test_has_import_errors_key(self, tmp_path):
        result = run_quality_snapshot(tmp_path)
        if "error" not in result:
            assert "import_errors" in result

    def test_has_elapsed_ms_key(self, tmp_path):
        result = run_quality_snapshot(tmp_path)
        if "error" not in result:
            assert "elapsed_ms" in result

    def test_never_raises_on_bad_path(self):
        result = run_quality_snapshot(Path("/nonexistent/path"))
        assert isinstance(result, dict)

    def test_never_raises_on_none_changed_files(self, tmp_path):
        result = run_quality_snapshot(tmp_path, changed_files=None)
        assert isinstance(result, dict)

    def test_syntax_error_detected(self, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def foo(\n  broken syntax !!!")
        result = run_quality_snapshot(tmp_path, changed_files=["bad.py"])
        if "error" not in result:
            assert len(result["syntax_errors"]) > 0

    def test_good_file_no_syntax_errors(self, tmp_path):
        good_file = tmp_path / "good.py"
        good_file.write_text("def foo():\n    return 1\n")
        result = run_quality_snapshot(tmp_path, changed_files=["good.py"])
        if "error" not in result:
            assert result["syntax_errors"] == []

    def test_test_count_is_int(self, tmp_path):
        result = run_quality_snapshot(tmp_path)
        if "error" not in result:
            assert isinstance(result["test_count"], int)


# ── TestAdaptivePipelineOutcome ──────────────────────────────────────────────

class TestAdaptivePipelineOutcome:
    def _make_brain(self):
        brain = MagicMock()
        brain._stored = []
        def remember(entry):
            brain._stored.append(entry)
        def recall_recent(limit=100):
            return list(brain._stored[-limit:])
        brain.remember.side_effect = remember
        brain.recall_recent.side_effect = recall_recent
        return brain

    def test_record_outcome_no_brain_no_error(self):
        ap = AdaptivePipeline()
        ap.record_outcome("bug_fix", "deep", True)  # should not raise

    def test_record_outcome_stores_to_brain(self):
        brain = self._make_brain()
        ap = AdaptivePipeline(brain=brain)
        ap.record_outcome("bug_fix", "deep", True)
        assert brain.remember.called

    def test_win_rate_no_brain_returns_zero(self):
        ap = AdaptivePipeline()
        assert ap.win_rate("bug_fix", "deep") == 0.0

    def test_win_rate_after_win(self):
        brain = self._make_brain()
        ap = AdaptivePipeline(brain=brain)
        ap.record_outcome("refactor", "normal", True)
        rate = ap.win_rate("refactor", "normal")
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_win_rate_100_percent(self):
        brain = self._make_brain()
        ap = AdaptivePipeline(brain=brain)
        ap.record_outcome("feature", "deep", True)
        assert ap.win_rate("feature", "deep") == pytest.approx(1.0)

    def test_win_rate_50_percent(self):
        brain = self._make_brain()
        ap = AdaptivePipeline(brain=brain)
        ap.record_outcome("feature", "minimal", True)
        ap.record_outcome("feature", "minimal", False)
        rate = ap.win_rate("feature", "minimal")
        assert rate == pytest.approx(0.5)

    def test_brain_param_stored(self):
        brain = MagicMock()
        ap = AdaptivePipeline(brain=brain)
        assert ap._brain is brain

    def test_brain_default_none(self):
        ap = AdaptivePipeline()
        assert ap._brain is None


# ── TestHybridClosedLoopDeprecation ──────────────────────────────────────────

class TestHybridClosedLoopDeprecation:
    def _make_deps(self):
        model = MagicMock()
        brain = MagicMock()
        brain.recall_recent.return_value = []
        git = MagicMock()
        return model, brain, git

    def test_instantiation_warns(self):
        from core.hybrid_loop import HybridClosedLoop
        model, brain, git = self._make_deps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HybridClosedLoop(model, brain, git)
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_warning_message_contains_looporch(self):
        from core.hybrid_loop import HybridClosedLoop
        model, brain, git = self._make_deps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HybridClosedLoop(model, brain, git)
        msgs = [str(warning.message) for warning in w]
        assert any("LoopOrchestrator" in m for m in msgs)

    def test_warning_is_deprecation_warning(self):
        from core.hybrid_loop import HybridClosedLoop
        model, brain, git = self._make_deps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HybridClosedLoop(model, brain, git)
        categories = [warning.category for warning in w]
        assert DeprecationWarning in categories


# ── TestAutonomousDiscoveryEnqueue ───────────────────────────────────────────

class TestAutonomousDiscoveryEnqueue:
    def _make_queue(self):
        q = MagicMock()
        q.added = []
        def batch_add(goals):
            q.added.extend(goals)
        q.batch_add.side_effect = batch_add
        return q

    def test_enqueue_findings_returns_zero_no_queue(self, tmp_path):
        memory = MagicMock()
        memory.query.return_value = []
        ad = AutonomousDiscovery(MagicMock(), memory, project_root=str(tmp_path))
        findings = {"suggestions": [{"suggested_goal": "fix lint"}]}
        count = ad._enqueue_findings(findings)
        assert count == 0

    def test_enqueue_findings_calls_batch_add(self, tmp_path):
        q = self._make_queue()
        memory = MagicMock()
        memory.query.return_value = []
        ad = AutonomousDiscovery(
            MagicMock(), memory,
            project_root=str(tmp_path),
            goal_queue_extra=q,
        )
        findings = {"suggestions": [
            {"suggested_goal": "goal one"},
            {"suggested_goal": "goal two"},
        ]}
        count = ad._enqueue_findings(findings)
        assert count == 2
        assert q.batch_add.called

    def test_enqueue_findings_caps_at_3(self, tmp_path):
        q = self._make_queue()
        memory = MagicMock()
        memory.query.return_value = []
        ad = AutonomousDiscovery(
            MagicMock(), memory,
            project_root=str(tmp_path),
            goal_queue_extra=q,
        )
        findings = {"suggestions": [
            {"suggested_goal": f"goal {i}"} for i in range(6)
        ]}
        count = ad._enqueue_findings(findings)
        assert count == 3

    def test_enqueue_findings_empty_suggestions(self, tmp_path):
        q = self._make_queue()
        memory = MagicMock()
        memory.query.return_value = []
        ad = AutonomousDiscovery(
            MagicMock(), memory,
            project_root=str(tmp_path),
            goal_queue_extra=q,
        )
        count = ad._enqueue_findings({"suggestions": []})
        assert count == 0


# ── TestCycleOutcomeBrainIntegration ─────────────────────────────────────────

class TestCycleOutcomeBrainIntegration:
    def _make_brain(self):
        brain = MagicMock()
        brain._stored = []
        brain.remember.side_effect = lambda e: brain._stored.append(e)
        brain.recall_recent.side_effect = lambda limit=100: list(brain._stored[-limit:])
        return brain

    def test_to_json_then_remember(self):
        brain = self._make_brain()
        co = CycleOutcome(goal="integration test", goal_type="feature")
        co.mark_complete(True)
        brain.remember(co.to_json())
        assert len(brain._stored) == 1

    def test_recall_and_restore(self):
        brain = self._make_brain()
        co = CycleOutcome(goal="learn loop", goal_type="refactor", changes_applied=5)
        co.mark_complete(True)
        brain.remember(co.to_json())
        entries = brain.recall_recent(limit=10)
        assert len(entries) == 1
        restored = CycleOutcome.from_json(entries[0])
        assert restored.goal == "learn loop"
        assert restored.changes_applied == 5
        assert restored.success is True

    def test_multiple_outcomes_stored(self):
        brain = self._make_brain()
        for i in range(5):
            co = CycleOutcome(goal=f"goal {i}")
            co.mark_complete(i % 2 == 0)
            brain.remember(co.to_json())
        entries = brain.recall_recent(limit=10)
        assert len(entries) == 5
        outcomes = [CycleOutcome.from_json(e) for e in entries]
        goals = [o.goal for o in outcomes]
        assert "goal 0" in goals
        assert "goal 4" in goals
