# tests/test_agent_sdk_cli_integration.py
"""Tests for Agent SDK CLI integration."""
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path


class TestCLIIntegration(unittest.TestCase):
    """Test CLI command wiring for agent-run."""

    def test_build_controller_from_args(self):
        from core.agent_sdk.cli_integration import build_controller_from_args

        args = MagicMock()
        args.model = None
        args.max_turns = None
        args.max_budget = None
        args.permission_mode = None
        args.project_root = "/tmp/test"

        controller = build_controller_from_args(args)
        self.assertIsNotNone(controller)

    def test_build_controller_with_overrides(self):
        from core.agent_sdk.cli_integration import build_controller_from_args

        args = MagicMock()
        args.model = "claude-opus-4-6"
        args.max_turns = 50
        args.max_budget = 10.0
        args.permission_mode = "bypassPermissions"
        args.project_root = "/tmp/test"

        controller = build_controller_from_args(args)
        self.assertEqual(controller.config.model, "claude-opus-4-6")
        self.assertEqual(controller.config.max_turns, 50)
        self.assertEqual(controller.config.max_budget_usd, 10.0)

    def test_format_result(self):
        from core.agent_sdk.cli_integration import format_result

        result = {
            "result": "Successfully fixed the bug.",
            "session_id": "abc-123",
            "metrics": {"total_calls": 5, "success_rate": 1.0},
        }
        output = format_result(result)
        self.assertIn("Successfully fixed", output)
        self.assertIn("abc-123", output)


if __name__ == "__main__":
    unittest.main()
