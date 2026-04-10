# tests/test_agent_sdk_feedback.py
"""Tests for feedback collector and skill weight updater."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TestSkillWeightUpdater(unittest.TestCase):
    """Test skill weight adjustment."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.weights_path = Path(self.tmpdir) / "skill_weights.json"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_update_increases_on_success(self):
        from core.agent_sdk.feedback import SkillWeightUpdater

        self.weights_path.write_text(json.dumps({"linter": 0.5}))
        updater = SkillWeightUpdater(weights_path=self.weights_path)
        updater.update(["linter"], success=True)
        weights = updater.get_weights()
        self.assertAlmostEqual(weights["linter"], 0.6)

    def test_update_decreases_on_failure(self):
        from core.agent_sdk.feedback import SkillWeightUpdater

        self.weights_path.write_text(json.dumps({"linter": 0.5}))
        updater = SkillWeightUpdater(weights_path=self.weights_path)
        updater.update(["linter"], success=False)
        weights = updater.get_weights()
        self.assertAlmostEqual(weights["linter"], 0.45)

    def test_weight_capped_at_max(self):
        from core.agent_sdk.feedback import SkillWeightUpdater

        self.weights_path.write_text(json.dumps({"linter": 0.95}))
        updater = SkillWeightUpdater(weights_path=self.weights_path, cap=1.0, success_delta=0.1)
        updater.update(["linter"], success=True)
        weights = updater.get_weights()
        self.assertAlmostEqual(weights["linter"], 1.0)

    def test_weight_floored_at_min(self):
        from core.agent_sdk.feedback import SkillWeightUpdater

        self.weights_path.write_text(json.dumps({"linter": 0.12}))
        updater = SkillWeightUpdater(weights_path=self.weights_path, floor=0.1, failure_delta=-0.05)
        updater.update(["linter"], success=False)
        weights = updater.get_weights()
        self.assertAlmostEqual(weights["linter"], 0.1)

    def test_new_skill_starts_at_default(self):
        from core.agent_sdk.feedback import SkillWeightUpdater

        updater = SkillWeightUpdater(weights_path=self.weights_path)
        updater.update(["brand_new_skill"], success=True)
        weights = updater.get_weights()
        self.assertIn("brand_new_skill", weights)

    def test_handles_missing_file(self):
        from core.agent_sdk.feedback import SkillWeightUpdater

        updater = SkillWeightUpdater(weights_path=self.weights_path)
        weights = updater.get_weights()
        self.assertIsInstance(weights, dict)


class TestFeedbackCollector(unittest.TestCase):
    """Test feedback dispatch to all three systems."""

    def test_on_goal_complete_dispatches_all(self):
        from core.agent_sdk.feedback import FeedbackCollector

        mock_router = MagicMock()
        mock_updater = MagicMock()
        mock_updater.get_weights.return_value = {}
        mock_brain = MagicMock()
        mock_store = MagicMock()
        collector = FeedbackCollector(
            model_router=mock_router,
            skill_updater=mock_updater,
            brain=mock_brain,
            session_store=mock_store,
        )
        collector.on_goal_complete(
            session_pk=1,
            goal="Fix bug",
            goal_type="bug_fix",
            model="claude-sonnet-4-6",
            skills_used=["linter"],
            success=True,
            verification_result={"passed": True},
            cost=0.5,
        )
        mock_router.record_outcome.assert_called_once_with("bug_fix", "claude-sonnet-4-6", True)
        mock_updater.update.assert_called_once_with(["linter"], True)
        mock_brain.remember.assert_called_once()

    def test_get_failure_patterns_empty_when_few_failures(self):
        from core.agent_sdk.feedback import FeedbackCollector

        mock_store = MagicMock()
        mock_store.list_sessions.return_value = [{"status": "failed", "error_summary": "test error"}]
        collector = FeedbackCollector(
            model_router=MagicMock(),
            skill_updater=MagicMock(),
            brain=None,
            session_store=mock_store,
        )
        patterns = collector.get_failure_patterns("bug_fix")
        # Less than 3 failures, should return empty
        self.assertEqual(patterns, [])

    def test_get_failure_patterns_returns_errors(self):
        from core.agent_sdk.feedback import FeedbackCollector

        mock_store = MagicMock()
        mock_store.list_sessions.return_value = [
            {"status": "failed", "goal_type": "bug_fix", "error_summary": "ImportError: no module X"},
            {"status": "failed", "goal_type": "bug_fix", "error_summary": "ImportError: no module Y"},
            {"status": "failed", "goal_type": "bug_fix", "error_summary": "SyntaxError: bad token"},
        ]
        collector = FeedbackCollector(
            model_router=MagicMock(),
            skill_updater=MagicMock(),
            brain=None,
            session_store=mock_store,
        )
        patterns = collector.get_failure_patterns("bug_fix")
        self.assertEqual(len(patterns), 3)


if __name__ == "__main__":
    unittest.main()
