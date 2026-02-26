import io
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from aura_cli.cli_options import parse_cli_args
import aura_cli.cli_main as cli_main


class TestCLIMainDispatch(unittest.TestCase):
    def _dispatch(self, argv, *, runtime_factory=None):
        parsed = parse_cli_args(argv)
        rf = runtime_factory or MagicMock()
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=rf)
        return code, out.getvalue(), err.getvalue(), rf

    def test_dispatch_help_does_not_create_runtime(self):
        parsed = parse_cli_args(["help", "goal", "add"])
        runtime_factory = MagicMock()

        with patch("aura_cli.cli_main.render_help", return_value="HELP TEXT"):
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().strip(), "HELP TEXT")
        runtime_factory.assert_not_called()

    def test_dispatch_doctor_does_not_create_runtime(self):
        parsed = parse_cli_args(["doctor"])
        runtime_factory = MagicMock()

        with patch("aura_cli.cli_main._handle_doctor") as mock_doctor:
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        mock_doctor.assert_called_once()
        runtime_factory.assert_not_called()

    def test_dispatch_goal_status_uses_runtime_and_status_handler(self):
        parsed = parse_cli_args(["goal", "status", "--json"])
        fake_runtime = {
            "goal_queue": MagicMock(),
            "goal_archive": MagicMock(),
            "orchestrator": MagicMock(),
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "loop": MagicMock(),
            "model_adapter": MagicMock(),
            "brain": MagicMock(),
        }
        runtime_factory = MagicMock(return_value=fake_runtime)

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main._handle_status") as mock_status, \
             patch("aura_cli.cli_main.log_json"):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        runtime_factory.assert_called_once()
        mock_status.assert_called_once_with(
            fake_runtime["goal_queue"],
            fake_runtime["goal_archive"],
            fake_runtime["orchestrator"],
            as_json=True,
        )

    def test_main_returns_json_parse_error(self):
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cli_main.main(project_root_override=Path(d), argv=["goa", "--json"])

        self.assertEqual(code, 2)
        self.assertIn("\"code\": \"cli_parse_error\"", out.getvalue())
        self.assertEqual(err.getvalue(), "")

    def test_legacy_and_canonical_goal_status_use_same_handler(self):
        fake_runtime = {
            "goal_queue": MagicMock(),
            "goal_archive": MagicMock(),
            "orchestrator": MagicMock(),
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "loop": MagicMock(),
            "model_adapter": MagicMock(),
            "brain": MagicMock(),
        }
        runtime_factory = MagicMock(return_value=fake_runtime)

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main._handle_status") as mock_status, \
             patch("aura_cli.cli_main.log_json"):
            code1, *_ = self._dispatch(["goal", "status", "--json"], runtime_factory=runtime_factory)
            code2, *_ = self._dispatch(["--status", "--json"], runtime_factory=runtime_factory)

        self.assertEqual(code1, 0)
        self.assertEqual(code2, 0)
        self.assertEqual(mock_status.call_args_list, [
            call(fake_runtime["goal_queue"], fake_runtime["goal_archive"], fake_runtime["orchestrator"], as_json=True),
            call(fake_runtime["goal_queue"], fake_runtime["goal_archive"], fake_runtime["orchestrator"], as_json=True),
        ])

    def test_legacy_and_canonical_mcp_tools_use_same_handler_without_runtime(self):
        runtime_factory = MagicMock()

        with patch("aura_cli.cli_main.cmd_mcp_tools") as mock_tools:
            code1, *_ = self._dispatch(["mcp", "tools"], runtime_factory=runtime_factory)
            code2, *_ = self._dispatch(["--mcp-tools"], runtime_factory=runtime_factory)

        self.assertEqual(code1, 0)
        self.assertEqual(code2, 0)
        self.assertEqual(mock_tools.call_count, 2)
        runtime_factory.assert_not_called()

    def test_legacy_and_canonical_mcp_call_use_same_handler_args(self):
        runtime_factory = MagicMock()

        with patch("aura_cli.cli_main.cmd_mcp_call") as mock_call:
            code1, *_ = self._dispatch(["mcp", "call", "limits", "--args", "{\"x\":1}"], runtime_factory=runtime_factory)
            code2, *_ = self._dispatch(["--mcp-call", "limits", "--mcp-args", "{\"x\":1}"], runtime_factory=runtime_factory)

        self.assertEqual(code1, 0)
        self.assertEqual(code2, 0)
        self.assertEqual(mock_call.call_args_list, [call("limits", "{\"x\":1}"), call("limits", "{\"x\":1}")])
        runtime_factory.assert_not_called()

    def test_legacy_and_canonical_workflow_run_call_same_orchestrator(self):
        orchestrator = MagicMock()
        orchestrator.run_loop.return_value = {"stop_reason": "done", "history": [1, 2]}
        fake_runtime = {
            "goal_queue": MagicMock(),
            "goal_archive": MagicMock(),
            "orchestrator": orchestrator,
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "loop": MagicMock(),
            "model_adapter": MagicMock(),
            "brain": MagicMock(),
        }
        runtime_factory = MagicMock(return_value=fake_runtime)

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main.log_json"), \
             patch.object(cli_main.config, "set_runtime_override"):
            code1, *_ = self._dispatch(["workflow", "run", "Summarize repo", "--max-cycles", "3", "--dry-run"], runtime_factory=runtime_factory)
            code2, *_ = self._dispatch(["--workflow-goal", "Summarize repo", "--workflow-max-cycles", "3", "--dry-run"], runtime_factory=runtime_factory)

        self.assertEqual(code1, 0)
        self.assertEqual(code2, 0)
        self.assertEqual(orchestrator.run_loop.call_args_list, [
            call("Summarize repo", max_cycles=3, dry_run=True),
            call("Summarize repo", max_cycles=3, dry_run=True),
        ])

    def test_legacy_and_canonical_goal_add_run_use_same_queue_and_runner(self):
        goal_queue = MagicMock()
        fake_runtime = {
            "goal_queue": goal_queue,
            "goal_archive": MagicMock(),
            "orchestrator": MagicMock(),
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "loop": MagicMock(),
            "model_adapter": MagicMock(),
            "brain": MagicMock(),
        }
        runtime_factory = MagicMock(return_value=fake_runtime)

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main.log_json"), \
             patch("aura_cli.cli_main.run_goals_loop") as mock_run_goals:
            code1, *_ = self._dispatch(["goal", "add", "Fix tests", "--run"], runtime_factory=runtime_factory)
            code2, *_ = self._dispatch(["--add-goal", "Fix tests", "--run-goals"], runtime_factory=runtime_factory)

        self.assertEqual(code1, 0)
        self.assertEqual(code2, 0)
        self.assertEqual(goal_queue.add.call_args_list, [call("Fix tests"), call("Fix tests")])
        self.assertEqual(mock_run_goals.call_count, 2)

        first_args = mock_run_goals.call_args_list[0].args[0]
        second_args = mock_run_goals.call_args_list[1].args[0]
        self.assertEqual(first_args.add_goal, second_args.add_goal)
        self.assertEqual(first_args.run_goals, second_args.run_goals)
        self.assertEqual(first_args.decompose, second_args.decompose)

    def test_logs_command_does_not_require_runtime(self):
        runtime_factory = MagicMock()

        mock_streamer = MagicMock()
        mock_streamer_cls = MagicMock(return_value=mock_streamer)
        with patch("aura_cli.tui.log_streamer.LogStreamer", mock_streamer_cls):
            code, *_ = self._dispatch(["logs", "--tail", "10", "--level", "warn"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        runtime_factory.assert_not_called()
        mock_streamer_cls.assert_called_once_with(level_filter="warn")
        mock_streamer.stream_stdin.assert_called_once_with(tail=10)

    def test_watch_command_uses_runtime_and_studio(self):
        fake_runtime = {
            "goal_queue": MagicMock(),
            "goal_archive": MagicMock(),
            "orchestrator": MagicMock(),
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "loop": MagicMock(),
            "model_adapter": MagicMock(),
            "brain": MagicMock(),
        }
        runtime_factory = MagicMock(return_value=fake_runtime)

        mock_studio = MagicMock()
        mock_studio_cls = MagicMock(return_value=mock_studio)
        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main.log_json"), \
             patch("aura_cli.tui.app.AuraStudio", mock_studio_cls):
            code, *_ = self._dispatch(["watch"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        runtime_factory.assert_called_once()
        mock_studio_cls.assert_called_once_with(runtime=fake_runtime)
        mock_studio.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
