# tests/test_agent_sdk_model_router.py
"""Tests for adaptive model router."""
import json
import os
import tempfile
import unittest
from pathlib import Path


class TestAdaptiveModelRouter(unittest.TestCase):
    """Test model tier selection and learning."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.stats_path = Path(self.tmpdir) / "model_stats.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_selection_is_standard(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")

    def test_selects_cheapest_viable_tier(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        # Pre-seed stats: fast tier has good success rate
        stats = {
            "bug_fix": {
                "fast": {"attempts": 10, "successes": 9, "consecutive_failures": 0,
                         "consecutive_successes": 5, "ema_score": 0.9},
            }
        }
        self.stats_path.write_text(json.dumps(stats))
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-haiku-4-5")

    def test_skips_tier_with_low_ema(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        stats = {
            "bug_fix": {
                "fast": {"attempts": 10, "successes": 3, "consecutive_failures": 1,
                         "consecutive_successes": 0, "ema_score": 0.4},
                "standard": {"attempts": 20, "successes": 18, "consecutive_failures": 0,
                             "consecutive_successes": 6, "ema_score": 0.88},
            }
        }
        self.stats_path.write_text(json.dumps(stats))
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")

    def test_record_outcome_updates_ema(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        router = AdaptiveModelRouter(stats_path=self.stats_path, ema_alpha=0.2)
        router.record_outcome("bug_fix", "claude-sonnet-4-6", success=True)
        router.record_outcome("bug_fix", "claude-sonnet-4-6", success=True)
        stats = router.get_stats()
        tier_stats = stats["bug_fix"]["standard"]
        self.assertEqual(tier_stats["attempts"], 2)
        self.assertEqual(tier_stats["successes"], 2)
        self.assertGreater(tier_stats["ema_score"], 0.5)

    def test_record_outcome_persists_to_file(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        router.record_outcome("feature", "claude-sonnet-4-6", success=True)
        # Re-load from file
        router2 = AdaptiveModelRouter(stats_path=self.stats_path)
        stats = router2.get_stats()
        self.assertIn("feature", stats)

    def test_escalate_returns_next_tier(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        next_model = router.escalate("bug_fix", "claude-haiku-4-5")
        self.assertEqual(next_model, "claude-sonnet-4-6")
        next_model = router.escalate("bug_fix", "claude-sonnet-4-6")
        self.assertEqual(next_model, "claude-opus-4-6")
        # Already at top — stays there
        next_model = router.escalate("bug_fix", "claude-opus-4-6")
        self.assertEqual(next_model, "claude-opus-4-6")

    def test_consecutive_failures_trigger_skip(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        stats = {
            "bug_fix": {
                "fast": {"attempts": 5, "successes": 4, "consecutive_failures": 2,
                         "consecutive_successes": 0, "ema_score": 0.75},
            }
        }
        self.stats_path.write_text(json.dumps(stats))
        router = AdaptiveModelRouter(stats_path=self.stats_path, escalation_threshold=2)
        model = router.select_model("bug_fix")
        # Should skip fast despite good EMA because consecutive_failures >= threshold
        self.assertEqual(model, "claude-sonnet-4-6")

    def test_handles_missing_stats_file(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        missing_path = Path(self.tmpdir) / "nonexistent.json"
        router = AdaptiveModelRouter(stats_path=missing_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")

    def test_handles_corrupt_stats_file(self):
        from core.agent_sdk.model_router import AdaptiveModelRouter
        self.stats_path.write_text("not valid json {{{")
        router = AdaptiveModelRouter(stats_path=self.stats_path)
        model = router.select_model("bug_fix")
        self.assertEqual(model, "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
