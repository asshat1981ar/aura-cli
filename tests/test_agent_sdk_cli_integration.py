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


class TestCLIIntegrationV2(unittest.TestCase):
    """Test enhanced CLI with subsystem initialization."""

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_controller_initializes_subsystems(self):
        from core.agent_sdk.cli_integration import build_controller_from_args

        args = MagicMock()
        args.model = None
        args.max_turns = None
        args.max_budget = None
        args.permission_mode = None
        args.project_root = self.tmpdir
        # Patch config to use temp paths so no side effects in CWD
        with patch("core.agent_sdk.cli_integration.AgentSDKConfig") as MockConfig:
            from core.agent_sdk.config import AgentSDKConfig

            cfg = AgentSDKConfig(
                model_stats_path=Path(self.tmpdir) / "stats.json",
                session_db_path=Path(self.tmpdir) / "sessions.db",
                skill_weights_path=Path(self.tmpdir) / "weights.json",
            )
            MockConfig.return_value = cfg
            MockConfig.from_aura_config.return_value = cfg
            controller = build_controller_from_args(args)
        self.assertIsNotNone(controller.model_router)
        self.assertIsNotNone(controller.session_store)

    def test_format_result_shows_cost(self):
        from core.agent_sdk.cli_integration import format_result

        result = {
            "result": "Done.",
            "session_id": "abc-123",
            "total_cost_usd": 0.37,
            "metrics": {"total_calls": 5, "success_rate": 1.0},
        }
        output = format_result(result)
        self.assertIn("$0.37", output)


class TestAgentScanCLI(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        (Path(self.tmpdir) / "test.py").write_text("def hello():\n    pass\n")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_handle_agent_scan(self):
        from core.agent_sdk.cli_integration import handle_agent_scan

        result = handle_agent_scan(
            project_root=Path(self.tmpdir),
            db_path=Path(self.tmpdir) / "index.db",
            exclude_patterns=["__pycache__"],
            no_llm=True,
        )
        self.assertIn("files_scanned", result)
        self.assertGreater(result["files_scanned"], 0)

    def test_handle_agent_scan_stats(self):
        from core.agent_sdk.cli_integration import handle_agent_scan, format_scan_stats

        handle_agent_scan(
            project_root=Path(self.tmpdir),
            db_path=Path(self.tmpdir) / "index.db",
            no_llm=True,
        )
        stats = format_scan_stats(Path(self.tmpdir) / "index.db")
        self.assertIn("files", stats.lower())


if __name__ == "__main__":
    unittest.main()
