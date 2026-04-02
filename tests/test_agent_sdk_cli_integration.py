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


class TestCLICommands(unittest.TestCase):
    """Tests for handle_agent_status and handle_agent_cost functions."""

    def test_handle_agent_status_returns_list(self):
        from core.agent_sdk.cli_integration import handle_agent_status

        mock_store = MagicMock()
        mock_store.list_sessions.return_value = [
            {"session_id": "abc-1", "status": "completed", "goal": "Fix bug"},
            {"session_id": "abc-2", "status": "running", "goal": "Add feature"},
        ]
        result = handle_agent_status(mock_store, limit=10)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        mock_store.list_sessions.assert_called_once_with(limit=10)

    def test_handle_agent_cost_returns_summary(self):
        from core.agent_sdk.cli_integration import handle_agent_cost

        mock_store = MagicMock()
        mock_store.get_cost_summary.return_value = {
            "total_usd": 1.23,
            "session_count": 5,
            "days": 7,
        }
        result = handle_agent_cost(mock_store, days=7)
        self.assertIsInstance(result, dict)
        self.assertIn("total_usd", result)
        self.assertEqual(result["total_usd"], 1.23)
        mock_store.get_cost_summary.assert_called_once_with(days=7)

    def test_handle_agent_status_default_limit(self):
        from core.agent_sdk.cli_integration import handle_agent_status

        mock_store = MagicMock()
        mock_store.list_sessions.return_value = []
        handle_agent_status(mock_store)
        mock_store.list_sessions.assert_called_once_with(limit=20)


if __name__ == "__main__":
    unittest.main()
