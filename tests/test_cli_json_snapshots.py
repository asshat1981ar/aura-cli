import json
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from contextlib import redirect_stdout

from aura_cli.cli_options import parse_cli_args
import aura_cli.cli_main as cli_main
from tests.cli_snapshot_utils import normalized_json_text, read_snapshot_text, snapshot_dir_for


class TestCLIJSONSnapshots(unittest.TestCase):
    """
    Ensures canonical CLI commands output stable JSON structures.
    """
    _SNAPSHOT_DIR = snapshot_dir_for(__file__)

    def setUp(self):
        # Create snapshot dir if it doesn't exist (for first run)
        self._SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def _dispatch_and_capture_json(self, argv, *, runtime_factory=None):
        parsed = parse_cli_args(argv)
        rf = runtime_factory or MagicMock()
        out = io.StringIO()
        
        with redirect_stdout(out):
            # We patch check_project_writability to avoid permission checks in tests
            with patch("aura_cli.cli_main.check_project_writability", return_value=True):
                code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=rf)
        
        self.assertEqual(code, 0, f"Command failed with code {code}")
        output = out.getvalue()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            self.fail(f"Command output is not valid JSON:\n{output}")

    def _assert_snapshot(self, data: dict, snapshot_name: str):
        normalized = normalized_json_text(json.dumps(data))
        snapshot_path = self._SNAPSHOT_DIR / snapshot_name
        
        # If snapshot doesn't exist, we might want to create it (during dev)
        # But for CI, we assert.
        if not snapshot_path.exists():
            snapshot_path.write_text(normalized, encoding="utf-8")
            # Fail so we know we generated a new one
            self.fail(f"Snapshot {snapshot_name} did not exist. Created it.")
            
        self.assertEqual(normalized, read_snapshot_text(self._SNAPSHOT_DIR, snapshot_name))

    def test_config_json(self):
        """Test 'aura config' output structure."""
        # Mock config to have stable values
        with patch("core.config_manager.config.show_config", return_value={
            "env": "test", 
            "defaults": {"model": "gpt-4"}, 
            "overrides": {}
        }):
            data = self._dispatch_and_capture_json(["config"])
            self._assert_snapshot(data, "cli_config.json")

    def test_mcp_tools_json(self):
        """Test 'aura mcp tools' output structure."""
        mock_tools = {
            "tools": [{"name": "test_tool", "description": "A test tool"}]
        }
        with patch("aura_cli.cli_main._mcp_request", return_value=(200, mock_tools)):
            data = self._dispatch_and_capture_json(["mcp", "tools"])
            self._assert_snapshot(data, "cli_mcp_tools.json")

    def test_mcp_call_json(self):
        """Test 'aura mcp call' output structure."""
        mock_result = {"result": "success"}
        with patch("aura_cli.cli_main._mcp_request", return_value=(200, mock_result)):
            data = self._dispatch_and_capture_json(["mcp", "call", "test_tool"])
            self._assert_snapshot(data, "cli_mcp_call.json")

    def test_queue_list_json(self):
        """Test 'aura queue list --json' output structure."""
        mock_queue = MagicMock()
        mock_queue.queue = ["Goal 1", "Goal 2"]
        
        runtime = {"goal_queue": mock_queue}
        runtime_factory = MagicMock(return_value=runtime)
        
        data = self._dispatch_and_capture_json(["queue", "list", "--json"], runtime_factory=runtime_factory)
        self._assert_snapshot(data, "cli_queue_list.json")

    def test_scaffold_json(self):
        """Test 'aura scaffold --json' output structure."""
        runtime = {
            "model_adapter": MagicMock(),
            "brain": MagicMock()
        }
        runtime_factory = MagicMock(return_value=runtime)
        
        with patch("agents.scaffolder.ScaffolderAgent") as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.scaffold_project.return_value = "Scaffold complete"
            
            data = self._dispatch_and_capture_json(
                ["scaffold", "demo", "--json"], 
                runtime_factory=runtime_factory
            )
            self._assert_snapshot(data, "cli_scaffold.json")

    def test_workflow_run_json(self):
        """Test 'aura workflow run --json' output structure."""
        mock_orch = MagicMock()
        mock_orch.run_loop.return_value = {
            "stop_reason": "completed",
            "history": [{"cycle": 1}, {"cycle": 2}]
        }
        
        runtime = {"orchestrator": mock_orch}
        runtime_factory = MagicMock(return_value=runtime)
        
        data = self._dispatch_and_capture_json(
            ["workflow", "run", "Test Workflow", "--json"], 
            runtime_factory=runtime_factory
        )
        self._assert_snapshot(data, "cli_workflow_run.json")

    def test_evolve_json(self):
        """Test 'aura evolve --json' output structure."""
        runtime = {
            "brain": MagicMock(),
            "model_adapter": MagicMock(),
            "planner": MagicMock(),
            "act": MagicMock(),
            "critique": MagicMock()
        }
        runtime_factory = MagicMock(return_value=runtime)
        
        with patch("core.evolution_loop.EvolutionLoop") as mock_loop_cls:
            mock_loop = mock_loop_cls.return_value
            mock_loop.run.return_value = {
                "outcome": "success",
                "cycles": 5,
                "improvement": "optimized"
            }
            # Also patch GitTools and MutatorAgent as they are instantiated inside handler
            with patch("core.git_tools.GitTools"), \
                 patch("agents.mutator.MutatorAgent"), \
                 patch("core.vector_store.VectorStore"):
                 
                data = self._dispatch_and_capture_json(
                    ["evolve", "--json"], 
                    runtime_factory=runtime_factory
                )
                self._assert_snapshot(data, "cli_evolve.json")

    def test_goal_once_json(self):
        """Test 'aura goal once --json' output structure."""
        mock_orch = MagicMock()
        mock_orch.run_loop.return_value = {
            "stop_reason": "completed",
            "history": [{"cycle": 1}]
        }
        
        runtime = {"orchestrator": mock_orch}
        runtime_factory = MagicMock(return_value=runtime)
        
        data = self._dispatch_and_capture_json(
            ["goal", "once", "Test Goal", "--json"], 
            runtime_factory=runtime_factory
        )
        self._assert_snapshot(data, "cli_goal_once.json")

    def test_goal_add_json(self):
        """Test 'aura goal add --json' output structure."""
        mock_queue = MagicMock()
        mock_queue.queue = ["New Goal"]
        
        runtime = {"goal_queue": mock_queue}
        runtime_factory = MagicMock(return_value=runtime)
        
        data = self._dispatch_and_capture_json(
            ["goal", "add", "New Goal", "--json"], 
            runtime_factory=runtime_factory
        )
        self._assert_snapshot(data, "cli_goal_add.json") 

    def test_goal_status_json(self):
        """Test 'aura goal status --json' output structure."""
        mock_queue = MagicMock()
        mock_queue.queue = []
        mock_archive = MagicMock()
        mock_archive.completed = []

        capability_report = {
            "applied_bootstrap_actions": [],
            "configured": {
                "auto_add_capabilities": True,
                "auto_provision_mcp": False,
                "auto_queue_missing_capabilities": True,
                "auto_start_mcp_servers": False,
            },
            "failed_bootstrap_actions": [],
            "last_goal": None,
            "last_updated": None,
            "matched_capabilities": [],
            "matched_capability_ids": [],
            "missing_skills": [],
            "pending_bootstrap_actions": [],
            "pending_self_development_goals": [],
            "queue_strategy": None,
            "queued_goals": [],
            "recommended_skills": [],
            "running_bootstrap_actions": [],
            "skipped_goals": [],
        }

        runtime = {
            "goal_queue": mock_queue,
            "goal_archive": mock_archive,
            "orchestrator": SimpleNamespace(),
            "debugger": None,
            "planner": None,
            "loop": SimpleNamespace(),
            "model_adapter": None,
            "brain": None,
            "memory_persistence_path": None,
        }

        with patch("aura_cli.commands.build_capability_status_report", return_value=capability_report):
            data = self._dispatch_and_capture_json(
                ["goal", "status", "--json"],
                runtime_factory=MagicMock(return_value=runtime),
            )
        self._assert_snapshot(data, "cli_goal_status.json")

    def test_metrics_show_json(self):
        """Test 'aura metrics show --json' output structure."""
        brain = MagicMock()
        brain.recall_recent.return_value = [
            "outcome:12345678 -> {\"cycle_id\":\"12345678\",\"success\":true,\"completed_at\":10,\"started_at\":5,\"goal\":\"Test Goal\"}"
        ]
        runtime = {"brain": brain}

        data = self._dispatch_and_capture_json(
            ["metrics", "--json"],
            runtime_factory=MagicMock(return_value=runtime),
        )
        self._assert_snapshot(data, "cli_metrics_show.json")


    def test_contract_report_check(self):
        """Test 'aura contract-report --check' output structure."""
        # This runs actual contract logic, which is fine as it's deterministic-ish
        # We might need to mask some fields if they change (like versions)
        # For now, let's see if we can run it.
        # It relies on aura_cli.options.help_schema()
        
        data = self._dispatch_and_capture_json(["contract-report", "--check"])
        
        # Mask out 'generated_at' or similar if present (not present in help_schema root)
        # help_schema has 'version', 'generated_by', 'deterministic' which are stable.
        
        self._assert_snapshot(data, "cli_contract_report.json")
