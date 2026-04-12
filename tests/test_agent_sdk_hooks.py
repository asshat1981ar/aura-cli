# tests/test_agent_sdk_hooks.py
"""Tests for Agent SDK hooks."""

import unittest
from unittest.mock import MagicMock
import time


class TestHookCreation(unittest.TestCase):
    """Test hook factory functions."""

    def test_create_hooks_returns_dict(self):
        from core.agent_sdk.hooks import create_hooks

        hooks = create_hooks()
        self.assertIsInstance(hooks, dict)

    def test_hooks_has_pre_tool_use(self):
        from core.agent_sdk.hooks import create_hooks

        hooks = create_hooks()
        self.assertIn("PreToolUse", hooks)

    def test_hooks_has_post_tool_use(self):
        from core.agent_sdk.hooks import create_hooks

        hooks = create_hooks()
        self.assertIn("PostToolUse", hooks)

    def test_hooks_has_stop(self):
        from core.agent_sdk.hooks import create_hooks

        hooks = create_hooks()
        self.assertIn("Stop", hooks)


class TestMetricsCollector(unittest.TestCase):
    """Test the metrics collector used by hooks."""

    def test_record_tool_call(self):
        from core.agent_sdk.hooks import MetricsCollector

        mc = MetricsCollector()
        mc.record_tool_call("create_plan", 1.5, success=True)
        stats = mc.get_stats()
        self.assertEqual(stats["tool_calls"]["create_plan"]["count"], 1)
        self.assertTrue(stats["tool_calls"]["create_plan"]["success_rate"] > 0)

    def test_record_multiple_calls(self):
        from core.agent_sdk.hooks import MetricsCollector

        mc = MetricsCollector()
        mc.record_tool_call("verify_changes", 2.0, success=True)
        mc.record_tool_call("verify_changes", 1.0, success=False)
        stats = mc.get_stats()
        self.assertEqual(stats["tool_calls"]["verify_changes"]["count"], 2)
        self.assertAlmostEqual(stats["tool_calls"]["verify_changes"]["success_rate"], 0.5)

    def test_get_summary(self):
        from core.agent_sdk.hooks import MetricsCollector

        mc = MetricsCollector()
        mc.record_tool_call("analyze_goal", 0.5, success=True)
        summary = mc.get_summary()
        self.assertIn("total_calls", summary)
        self.assertEqual(summary["total_calls"], 1)


if __name__ == "__main__":
    unittest.main()
