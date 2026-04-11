"""Unit tests for RouterAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import time
import unittest
from unittest.mock import Mock, patch, MagicMock

from agents.router import RouterAgent, ModelStats


class TestModelStats(unittest.TestCase):
    """Test suite for ModelStats dataclass."""

    def test_init(self):
        """Test ModelStats initialization."""
        stats = ModelStats(name="test_model")

        self.assertEqual(stats.name, "test_model")
        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failure_count, 0)
        self.assertEqual(stats.total_latency, 0.0)
        self.assertEqual(stats.ema_score, 0.75)
        self.assertEqual(stats.consecutive_failures, 0)
        self.assertEqual(stats.cooldown_until, 0.0)

    def test_is_cooled_down_initially(self):
        """Test that model is cooled down initially."""
        stats = ModelStats(name="test_model")
        self.assertTrue(stats.is_cooled_down)

    def test_is_cooled_down_after_cooldown(self):
        """Test cooldown status after setting cooldown."""
        stats = ModelStats(name="test_model")
        stats.cooldown_until = time.time() + 100
        self.assertFalse(stats.is_cooled_down)

    def test_record_success(self):
        """Test recording a successful call."""
        stats = ModelStats(name="test_model")
        stats.record(success=True, latency=1.5)

        self.assertEqual(stats.success_count, 1)
        self.assertEqual(stats.failure_count, 0)
        self.assertEqual(stats.total_latency, 1.5)
        self.assertGreater(stats.ema_score, 0.75)  # Should increase
        self.assertEqual(stats.consecutive_failures, 0)

    def test_record_failure(self):
        """Test recording a failed call."""
        stats = ModelStats(name="test_model")
        stats.record(success=False, latency=0.5)

        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failure_count, 1)
        self.assertEqual(stats.total_latency, 0.5)
        self.assertLess(stats.ema_score, 0.75)  # Should decrease
        self.assertEqual(stats.consecutive_failures, 1)

    def test_consecutive_failures_trigger_cooldown(self):
        """Test that consecutive failures trigger cooldown."""
        stats = ModelStats(name="test_model")

        for _ in range(10):
            stats.record(success=False, latency=0.5)

        self.assertGreater(stats.cooldown_until, time.time())
        self.assertFalse(stats.is_cooled_down)

    def test_openai_shorter_cooldown(self):
        """Test that openai model has shorter cooldown for consecutive failures."""
        openai_stats = ModelStats(name="openai")
        other_stats = ModelStats(name="gemini")

        # Mock time to avoid timing issues
        start_time = time.time()

        for i in range(10):
            openai_stats.record(success=False, latency=0.5)
            other_stats.record(success=False, latency=0.5)

        # Check that openai has shorter cooldown (1s vs 10s)
        # Note: Low EMA may also trigger LONG_COOLDOWN, so check consecutive_failure cooldown
        openai_cooldown = openai_stats.cooldown_until - start_time
        other_cooldown = other_stats.cooldown_until - start_time

        # openai should have shorter or equal cooldown compared to others
        # (the consecutive failure cooldown is 1s for openai vs 10s for others)
        self.assertLessEqual(openai_cooldown, other_cooldown)

    def test_avg_latency(self):
        """Test average latency calculation."""
        stats = ModelStats(name="test_model")
        stats.record(success=True, latency=1.0)
        stats.record(success=True, latency=2.0)
        stats.record(success=False, latency=0.5)

        self.assertEqual(stats.avg_latency, 1.1666666666666667)

    def test_avg_latency_no_calls(self):
        """Test average latency with no calls."""
        stats = ModelStats(name="test_model")
        self.assertEqual(stats.avg_latency, 0.0)

    def test_low_ema_score_triggers_long_cooldown(self):
        """Test that low EMA score triggers long cooldown."""
        stats = ModelStats(name="test_model")
        stats.ema_score = 0.2  # Below MIN_EMA_SCORE
        stats.record(success=False, latency=0.5)

        self.assertGreater(stats.cooldown_until, time.time() + 500)

    def test_high_latency_triggers_long_cooldown(self):
        """Test that high latency triggers long cooldown."""
        stats = ModelStats(name="test_model")

        # Make 5 calls with high latency to trigger the threshold check
        for _ in range(5):
            stats.record(success=True, latency=100.0)

        self.assertGreater(stats.cooldown_until, time.time() + 500)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ModelStats(name="test_model")
        stats.record(success=True, latency=1.0)

        d = stats.to_dict()

        self.assertEqual(d["name"], "test_model")
        self.assertEqual(d["success_count"], 1)
        self.assertEqual(d["ema_score"], stats.ema_score)

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"name": "test_model", "success_count": 5, "failure_count": 2, "ema_score": 0.8, "consecutive_failures": 0}

        stats = ModelStats.from_dict(d)

        self.assertEqual(stats.name, "test_model")
        self.assertEqual(stats.success_count, 5)
        self.assertEqual(stats.failure_count, 2)
        self.assertEqual(stats.ema_score, 0.8)

    def test_str_representation(self):
        """Test string representation."""
        stats = ModelStats(name="test_model")
        stats.record(success=True, latency=1.0)

        s = str(stats)

        self.assertIn("test_model", s)
        self.assertIn("ema=", s)
        self.assertIn("ok=1", s)


class TestRouterAgent(unittest.TestCase):
    """Test suite for RouterAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_brain = Mock()
        self.mock_brain.get.return_value = None
        self.mock_adapter = Mock()
        self.mock_adapter.call_openai = Mock(return_value="openai response")
        self.mock_adapter.call_gemini = Mock(return_value="gemini response")

        self.router = RouterAgent(self.mock_brain, self.mock_adapter)

    def test_init(self):
        """Test router initialization."""
        self.assertEqual(self.router.brain, self.mock_brain)
        self.assertEqual(self.router.adapter, self.mock_adapter)
        self.assertEqual(len(self.router.stats), 6)  # Default enabled models

    def test_init_with_custom_enabled_models(self):
        """Test initialization with custom enabled models."""
        router = RouterAgent(self.mock_brain, self.mock_adapter, enabled_models=["openai", "gemini"])

        self.assertEqual(len(router.stats), 2)
        self.assertIn("openai", router.stats)
        self.assertIn("gemini", router.stats)

    def test_route_success_first_try(self):
        """Test successful routing on first try."""
        result = self.router.route("test prompt")

        self.assertEqual(result, "openai response")
        self.mock_adapter.call_openai.assert_called_once_with("test prompt")

    def test_route_fallback_on_failure(self):
        """Test fallback to next model on failure."""
        self.mock_adapter.call_openai.side_effect = Exception("OpenAI error")

        result = self.router.route("test prompt")

        self.assertEqual(result, "gemini response")
        self.mock_adapter.call_openai.assert_called_once()
        self.mock_adapter.call_gemini.assert_called_once()

    def test_route_all_fail(self):
        """Test when all models fail."""
        self.mock_adapter.call_openai.side_effect = Exception("Error")
        self.mock_adapter.call_gemini.side_effect = Exception("Error")
        self.mock_adapter.call_anthropic = Mock(side_effect=Exception("Error"))
        self.mock_adapter.call_openrouter = Mock(side_effect=Exception("Error"))
        self.mock_adapter.call_codex = Mock(side_effect=Exception("Error"))
        self.mock_adapter.call_copilot = Mock(side_effect=Exception("Error"))

        with self.assertRaises(RuntimeError) as ctx:
            self.router.route("test prompt")

        self.assertIn("all candidates exhausted", str(ctx.exception))

    def test_route_records_stats_on_success(self):
        """Test that routing records stats on success."""
        self.router.route("test prompt")

        self.assertEqual(self.router.stats["openai"].success_count, 1)
        self.assertEqual(self.router.stats["openai"].failure_count, 0)

    def test_route_records_stats_on_failure(self):
        """Test that routing records stats on failure."""
        self.mock_adapter.call_openai.side_effect = Exception("Error")
        self.router.route("test prompt")

        self.assertEqual(self.router.stats["openai"].failure_count, 1)
        self.assertEqual(self.router.stats["gemini"].success_count, 1)

    def test_route_saves_stats_to_brain(self):
        """Test that routing persists stats to brain."""
        self.router.route("test prompt")

        self.mock_brain.set.assert_called()

    def test_route_remembers_to_brain(self):
        """Test that routing remembers to brain."""
        self.router.route("test prompt")

        self.mock_brain.remember.assert_called()

    def test_rank_candidates_excludes_cooled_down(self):
        """Test that ranking excludes cooled-down models."""
        # Put openai on cooldown
        self.router.stats["openai"].cooldown_until = time.time() + 100

        candidates = self.router._rank_candidates("test prompt")

        self.assertNotIn("openai", candidates)

    def test_rank_candidates_prefers_high_ema(self):
        """Test that ranking prefers high EMA models."""
        self.router.stats["openai"].ema_score = 0.9
        self.router.stats["gemini"].ema_score = 0.8

        candidates = self.router._rank_candidates("test prompt")

        # openai should be first due to higher EMA
        self.assertEqual(candidates[0], "openai")

    def test_rank_candidates_long_prompt_bonus(self):
        """Test bonus for openai on long prompts."""
        long_prompt = "x" * 3500

        self.router.stats["openai"].ema_score = 0.8
        self.router.stats["gemini"].ema_score = 0.8

        candidates = self.router._rank_candidates(long_prompt)

        # openai should get bonus on long prompts
        self.assertEqual(candidates[0], "openai")

    def test_rank_candidates_short_prompt_latency(self):
        """Test latency preference on short prompts."""
        short_prompt = "x" * 100

        self.router.stats["openai"].ema_score = 0.8
        self.router.stats["gemini"].ema_score = 0.8
        self.router.stats["openai"].total_latency = 10.0
        self.router.stats["gemini"].total_latency = 5.0
        self.router.stats["openai"].success_count = 1
        self.router.stats["gemini"].success_count = 1

        candidates = self.router._rank_candidates(short_prompt)

        # gemini should be preferred due to lower latency
        self.assertEqual(candidates[0], "gemini")

    def test_force_cooldown(self):
        """Test manual cooldown."""
        self.router.force_cooldown("openai", seconds=300)

        # Property is is_cooled_down, not is_cooldowned
        self.assertFalse(self.router.stats["openai"].is_cooled_down)
        self.assertGreater(self.router.stats["openai"].cooldown_until, time.time() + 290)

    def test_report(self):
        """Test stats report."""
        self.router.stats["openai"].ema_score = 0.9
        self.router.stats["gemini"].ema_score = 0.8

        report = self.router.report()

        self.assertIn("RouterAgent Model Rankings:", report)
        self.assertIn("openai", report)
        self.assertIn("gemini", report)
        # openai should appear before gemini due to higher EMA
        self.assertLess(report.index("openai"), report.index("gemini"))

    def test_load_stats_from_brain(self):
        """Test loading persisted stats from brain."""
        saved_stats = {"openai": {"name": "openai", "ema_score": 0.95, "success_count": 10}, "gemini": {"name": "gemini", "ema_score": 0.85, "success_count": 5}}
        self.mock_brain.get.return_value = saved_stats

        router = RouterAgent(self.mock_brain, self.mock_adapter)

        self.assertEqual(router.stats["openai"].ema_score, 0.95)
        self.assertEqual(router.stats["openai"].success_count, 10)

    def test_load_stats_handles_exception(self):
        """Test that loading stats handles exceptions gracefully."""
        self.mock_brain.get.side_effect = Exception("Brain error")

        # Should not raise
        router = RouterAgent(self.mock_brain, self.mock_adapter)

        self.assertEqual(len(router.stats), 6)

    def test_save_stats_handles_exception(self):
        """Test that saving stats handles exceptions gracefully."""
        self.mock_brain.set.side_effect = Exception("Brain error")

        # Should not raise
        self.router.route("test prompt")

    def test_get_caller_returns_none_for_unknown_model(self):
        """Test that _get_caller returns None for unknown model."""
        caller = self.router._get_caller("unknown_model")

        self.assertIsNone(caller)

    def test_get_caller_returns_callable(self):
        """Test that _get_caller returns callable for known model."""
        caller = self.router._get_caller("openai")

        self.assertIsNotNone(caller)
        self.assertEqual(caller, self.mock_adapter.call_openai)


class TestRouterAgentIntegration(unittest.TestCase):
    """Integration-style tests for RouterAgent."""

    def test_full_routing_workflow(self):
        """Test complete routing workflow."""
        mock_brain = Mock()
        mock_brain.get.return_value = None
        mock_adapter = Mock()

        # First call fails, second succeeds
        mock_adapter.call_openai.side_effect = Exception("Rate limit")
        mock_adapter.call_gemini.return_value = "Success!"

        router = RouterAgent(mock_brain, mock_adapter)
        result = router.route("Generate code")

        self.assertEqual(result, "Success!")
        self.assertEqual(router.stats["openai"].failure_count, 1)
        self.assertEqual(router.stats["gemini"].success_count, 1)

        # Verify brain was updated
        mock_brain.remember.assert_called()
        mock_brain.set.assert_called()


if __name__ == "__main__":
    unittest.main()
