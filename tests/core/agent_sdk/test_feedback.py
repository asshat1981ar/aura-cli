"""Tests for core/agent_sdk/feedback.py — SkillWeightUpdater, FeedbackCollector."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from core.agent_sdk.feedback import SkillWeightUpdater, FeedbackCollector


# ---------------------------------------------------------------------------
# SkillWeightUpdater
# ---------------------------------------------------------------------------

class TestSkillWeightUpdaterInit:
    def test_empty_weights_when_file_missing(self, tmp_path):
        updater = SkillWeightUpdater(weights_path=tmp_path / "weights.json")
        assert updater.get_weights() == {}

    def test_loads_existing_weights(self, tmp_path):
        p = tmp_path / "weights.json"
        p.write_text(json.dumps({"linter": 0.8, "type_checker": 0.6}))
        updater = SkillWeightUpdater(weights_path=p)
        assert updater.get_weights() == {"linter": 0.8, "type_checker": 0.6}

    def test_corrupted_json_starts_empty(self, tmp_path):
        p = tmp_path / "weights.json"
        p.write_text("{BROKEN")
        updater = SkillWeightUpdater(weights_path=p)
        assert updater.get_weights() == {}


class TestSkillWeightUpdaterUpdate:
    def test_success_increases_weight(self, tmp_path):
        updater = SkillWeightUpdater(
            weights_path=tmp_path / "w.json",
            success_delta=0.1,
        )
        updater.update(["linter"], success=True)
        weights = updater.get_weights()
        assert weights["linter"] == pytest.approx(0.6)  # 0.5 + 0.1

    def test_failure_decreases_weight(self, tmp_path):
        updater = SkillWeightUpdater(
            weights_path=tmp_path / "w.json",
            failure_delta=-0.05,
        )
        updater.update(["linter"], success=False)
        weights = updater.get_weights()
        assert weights["linter"] == pytest.approx(0.45)  # 0.5 - 0.05

    def test_weight_capped_at_cap(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"skill": 0.95}))
        updater = SkillWeightUpdater(weights_path=p, success_delta=0.1, cap=1.0)
        updater.update(["skill"], success=True)
        assert updater.get_weights()["skill"] == pytest.approx(1.0)

    def test_weight_floored_at_floor(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"skill": 0.12}))
        updater = SkillWeightUpdater(weights_path=p, failure_delta=-0.1, floor=0.1)
        updater.update(["skill"], success=False)
        assert updater.get_weights()["skill"] == pytest.approx(0.1)

    def test_multiple_skills_updated(self, tmp_path):
        updater = SkillWeightUpdater(weights_path=tmp_path / "w.json", success_delta=0.1)
        updater.update(["a", "b", "c"], success=True)
        weights = updater.get_weights()
        assert "a" in weights and "b" in weights and "c" in weights

    def test_update_persists_to_file(self, tmp_path):
        p = tmp_path / "w.json"
        updater = SkillWeightUpdater(weights_path=p)
        updater.update(["skill_x"], success=True)
        data = json.loads(p.read_text())
        assert "skill_x" in data

    def test_parent_dir_created_if_missing(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "w.json"
        updater = SkillWeightUpdater(weights_path=p)
        updater.update(["s"], success=True)
        assert p.exists()


# ---------------------------------------------------------------------------
# FeedbackCollector
# ---------------------------------------------------------------------------

def _make_collector(tmp_path, with_brain=False, with_store=False):
    router = MagicMock()
    router.record_outcome = MagicMock()
    updater = SkillWeightUpdater(weights_path=tmp_path / "w.json")
    brain = MagicMock() if with_brain else None
    store = MagicMock() if with_store else None
    fc = FeedbackCollector(
        model_router=router,
        skill_updater=updater,
        brain=brain,
        session_store=store,
    )
    return fc, router, updater, brain, store


class TestFeedbackCollectorOnGoalComplete:
    def test_calls_model_router_record_outcome(self, tmp_path):
        fc, router, *_ = _make_collector(tmp_path)
        fc.on_goal_complete(1, "fix bug", "bug_fix", "claude-sonnet", [], True, {}, 0.01)
        router.record_outcome.assert_called_once_with("bug_fix", "claude-sonnet", True)

    def test_updates_skill_weights(self, tmp_path):
        fc, _, updater, *_ = _make_collector(tmp_path)
        fc.on_goal_complete(1, "g", "t", "m", ["linter"], True, {}, 0.0)
        assert "linter" in updater.get_weights()

    def test_stores_in_brain_when_provided(self, tmp_path):
        fc, _, _, brain, _ = _make_collector(tmp_path, with_brain=True)
        fc.on_goal_complete(1, "my goal", "t", "m", [], True, {"ok": True}, 0.05)
        brain.remember.assert_called_once()
        call_arg = brain.remember.call_args[0][0]
        assert call_arg["goal"] == "my goal"
        assert call_arg["success"] is True

    def test_no_brain_no_crash(self, tmp_path):
        fc, *_ = _make_collector(tmp_path, with_brain=False)
        result = fc.on_goal_complete(1, "g", "t", "m", [], False, {}, 0.0)
        assert result["brain_stored"] is False

    def test_returns_correct_summary(self, tmp_path):
        fc, *_ = _make_collector(tmp_path)
        result = fc.on_goal_complete(1, "g", "t", "m", ["s1", "s2"], True, {}, 0.0)
        assert result["model_updated"] is True
        assert result["skills_updated"] == ["s1", "s2"]

    def test_brain_exception_does_not_propagate(self, tmp_path):
        fc, _, _, brain, _ = _make_collector(tmp_path, with_brain=True)
        brain.remember.side_effect = RuntimeError("brain down")
        # Should not raise
        result = fc.on_goal_complete(1, "g", "t", "m", [], True, {}, 0.0)
        assert result["brain_stored"] is True  # brain was provided


class TestFeedbackCollectorGetFailurePatterns:
    def test_no_store_returns_empty(self, tmp_path):
        fc, *_ = _make_collector(tmp_path, with_store=False)
        result = fc.get_failure_patterns("bug_fix")
        assert result == []

    def test_not_enough_failures_returns_empty(self, tmp_path):
        fc, _, _, _, store = _make_collector(tmp_path, with_store=True)
        store.list_sessions.return_value = [
            {"goal_type": "bug_fix", "error_summary": "err1"},
            {"goal_type": "bug_fix", "error_summary": "err2"},
        ]
        result = fc.get_failure_patterns("bug_fix", limit=3)
        assert result == []

    def test_enough_failures_returns_summaries(self, tmp_path):
        fc, _, _, _, store = _make_collector(tmp_path, with_store=True)
        store.list_sessions.return_value = [
            {"goal_type": "bug_fix", "error_summary": f"err{i}"} for i in range(5)
        ]
        result = fc.get_failure_patterns("bug_fix", limit=3)
        assert len(result) == 3

    def test_store_exception_returns_empty(self, tmp_path):
        fc, _, _, _, store = _make_collector(tmp_path, with_store=True)
        store.list_sessions.side_effect = RuntimeError("db error")
        result = fc.get_failure_patterns("any")
        assert result == []
