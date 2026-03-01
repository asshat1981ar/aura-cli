import json
import io
import tempfile
import unittest
import builtins
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from aura_cli.cli_options import parse_cli_args
import aura_cli.cli_main as cli_main


class TestCLIMainDispatch(unittest.TestCase):
    _SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

    def _dispatch(self, argv, *, runtime_factory=None):
        parsed = parse_cli_args(argv)
        rf = runtime_factory or MagicMock()
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=rf)
        return code, out.getvalue(), err.getvalue(), rf

    def _snapshot_text(self, name: str) -> str:
        return (self._SNAPSHOT_DIR / name).read_text(encoding="utf-8")

    def _normalized_json_text(self, raw_json: str) -> str:
        return json.dumps(json.loads(raw_json), indent=2, sort_keys=True) + "\n"

    def _assert_json_snapshot(self, raw_json: str, snapshot_name: str) -> None:
        self.assertEqual(self._normalized_json_text(raw_json), self._snapshot_text(snapshot_name))

    def _without_cli_warnings(self, payload: dict) -> dict:
        base = dict(payload)
        base.pop("cli_warnings", None)
        return base

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

    def test_create_runtime_contract_starts_without_legacy_loop(self):
        fake_goal_queue = MagicMock()
        fake_goal_archive = MagicMock()
        fake_brain = MagicMock()
        fake_brain.db = object()
        fake_model_adapter = MagicMock()
        fake_vector_store = MagicMock()
        fake_router = MagicMock()
        fake_debugger = MagicMock()
        fake_planner = MagicMock()
        fake_memory_store = MagicMock()
        fake_policy = MagicMock()
        fake_orchestrator = MagicMock()
        fake_git_tools = MagicMock()

        blocked_optional_imports = {
            "memory.cache_adapter_factory",
            "memory.momento_brain",
            "memory.momento_memory_store",
            "core.context_manager",
            "core.project_syncer",
            "core.reflection_loop",
            "core.health_monitor",
            "core.weakness_remediator",
            "core.skill_weight_adapter",
            "core.convergence_escape",
            "core.memory_compaction",
            "core.context_graph",
            "core.adaptive_pipeline",
            "core.propagation_engine",
            "core.autonomous_discovery",
        }
        real_import = builtins.__import__

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in blocked_optional_imports:
                raise ImportError(f"blocked optional import for test: {name}")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as d, \
             patch("aura_cli.cli_main.GoalQueue", return_value=fake_goal_queue), \
             patch("aura_cli.cli_main.GoalArchive", return_value=fake_goal_archive), \
             patch("aura_cli.cli_main.Brain", return_value=fake_brain), \
             patch("aura_cli.cli_main.ModelAdapter", return_value=fake_model_adapter), \
             patch("aura_cli.cli_main.VectorStore", return_value=fake_vector_store), \
             patch("aura_cli.cli_main.RouterAgent", return_value=fake_router), \
             patch("aura_cli.cli_main.DebuggerAgent", return_value=fake_debugger), \
             patch("aura_cli.cli_main.PlannerAgent", return_value=fake_planner), \
             patch("aura_cli.cli_main.MemoryStore", return_value=fake_memory_store), \
             patch("aura_cli.cli_main.default_agents", return_value={}), \
             patch("aura_cli.cli_main.Policy.from_config", return_value=fake_policy), \
             patch("aura_cli.cli_main.LoopOrchestrator", return_value=fake_orchestrator), \
             patch("aura_cli.cli_main.GitTools", return_value=fake_git_tools), \
             patch("aura_cli.cli_main.HybridClosedLoop") as mock_hybrid_loop, \
             patch("aura_cli.cli_main.log_json"), \
             patch("builtins.__import__", side_effect=_guarded_import):
            runtime = cli_main.create_runtime(Path(d), overrides=None)

        self.assertIs(runtime["goal_queue"], fake_goal_queue)
        self.assertIs(runtime["goal_archive"], fake_goal_archive)
        self.assertIs(runtime["model_adapter"], fake_model_adapter)
        self.assertIs(runtime["brain"], fake_brain)
        self.assertIs(runtime["orchestrator"], fake_orchestrator)
        self.assertIs(runtime["git_tools"], fake_git_tools)
        self.assertIs(runtime["loop"], None)
        mock_hybrid_loop.assert_not_called()

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

    def test_legacy_warning_stderr_text_is_unchanged(self):
        runtime_factory = MagicMock()

        with patch("aura_cli.cli_main.cmd_mcp_tools") as mock_tools:
            code, out, err, _ = self._dispatch(["--mcp-tools"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura mcp tools` instead.\n")
        mock_tools.assert_called_once()
        runtime_factory.assert_not_called()

    def test_legacy_mcp_tools_json_output_includes_cli_warnings(self):
        runtime_factory = MagicMock()

        def _emit_json():
            print('{"status": 200, "data": {"tools": []}}')

        with patch("aura_cli.cli_main.cmd_mcp_tools", side_effect=_emit_json):
            code, out, err, _ = self._dispatch(["--mcp-tools"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura mcp tools` instead.\n")
        payload = json.loads(out)
        self.assertEqual(payload["status"], 200)
        self.assertIn("cli_warnings", payload)
        self.assertEqual(payload["cli_warnings"][0]["code"], "legacy_cli_flags_deprecated")
        self.assertEqual(payload["cli_warnings"][0]["action"], "mcp_tools")
        runtime_factory.assert_not_called()

    def test_legacy_mcp_tools_json_output_matches_snapshot(self):
        runtime_factory = MagicMock()

        def _emit_json():
            print('{"status": 200, "data": {"tools": []}}')

        with patch("aura_cli.cli_main.cmd_mcp_tools", side_effect=_emit_json):
            code, out, err, _ = self._dispatch(["--mcp-tools"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura mcp tools` instead.\n")
        self._assert_json_snapshot(out, "cli_legacy_mcp_tools_dispatch.json")
        runtime_factory.assert_not_called()

    def test_canonical_mcp_tools_json_output_matches_snapshot_and_legacy_base_shape(self):
        runtime_factory = MagicMock()

        def _emit_json():
            print('{"status": 200, "data": {"tools": []}}')

        with patch("aura_cli.cli_main.cmd_mcp_tools", side_effect=_emit_json):
            code_c, out_c, err_c, _ = self._dispatch(["mcp", "tools"], runtime_factory=runtime_factory)
            code_l, out_l, err_l, _ = self._dispatch(["--mcp-tools"], runtime_factory=runtime_factory)

        self.assertEqual(code_c, 0)
        self.assertEqual(code_l, 0)
        self.assertEqual(err_c, "")
        self.assertEqual(err_l, "Warning: Legacy flags are deprecated; use `aura mcp tools` instead.\n")
        canonical_payload = json.loads(out_c)
        legacy_payload = json.loads(out_l)
        self.assertEqual(canonical_payload, self._without_cli_warnings(legacy_payload))
        self._assert_json_snapshot(out_c, "cli_canonical_mcp_tools_dispatch.json")
        runtime_factory.assert_not_called()

    def test_legacy_diag_json_output_matches_snapshot(self):
        runtime_factory = MagicMock()

        def _emit_diag_json():
            print(
                '{"health":{"status":200,"data":{"status":"ok"}},'
                '"metrics":{"status":200,"data":{"skill_metrics":{}}}}'
            )

        with patch("aura_cli.cli_main.cmd_diag", side_effect=_emit_diag_json):
            code, out, err, _ = self._dispatch(["--diag"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura diag` instead.\n")
        self._assert_json_snapshot(out, "cli_legacy_diag_dispatch.json")
        runtime_factory.assert_not_called()

    def test_canonical_diag_json_output_matches_snapshot_and_legacy_base_shape(self):
        runtime_factory = MagicMock()

        def _emit_diag_json():
            print(
                '{"health":{"status":200,"data":{"status":"ok"}},'
                '"metrics":{"status":200,"data":{"skill_metrics":{}}}}'
            )

        with patch("aura_cli.cli_main.cmd_diag", side_effect=_emit_diag_json):
            code_c, out_c, err_c, _ = self._dispatch(["diag"], runtime_factory=runtime_factory)
            code_l, out_l, err_l, _ = self._dispatch(["--diag"], runtime_factory=runtime_factory)

        self.assertEqual(code_c, 0)
        self.assertEqual(code_l, 0)
        self.assertEqual(err_c, "")
        self.assertEqual(err_l, "Warning: Legacy flags are deprecated; use `aura diag` instead.\n")
        canonical_payload = json.loads(out_c)
        legacy_payload = json.loads(out_l)
        self.assertEqual(canonical_payload, self._without_cli_warnings(legacy_payload))
        self._assert_json_snapshot(out_c, "cli_canonical_diag_dispatch.json")
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

    def test_legacy_mcp_call_json_output_matches_snapshot(self):
        runtime_factory = MagicMock()

        def _emit_mcp_call_json(tool, args_json):
            print(
                json.dumps(
                    {
                        "status": 200,
                        "data": {
                            "tool": tool,
                            "args_json": args_json,
                            "result": {"ok": True},
                        },
                    }
                )
            )

        with patch("aura_cli.cli_main.cmd_mcp_call", side_effect=_emit_mcp_call_json):
            code, out, err, _ = self._dispatch(
                ["--mcp-call", "limits", "--mcp-args", "{\"x\":1}"],
                runtime_factory=runtime_factory,
            )

        self.assertEqual(code, 0)
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura mcp call` instead.\n")
        self._assert_json_snapshot(out, "cli_legacy_mcp_call_dispatch.json")
        runtime_factory.assert_not_called()

    def test_canonical_mcp_call_json_output_matches_snapshot_and_legacy_base_shape(self):
        runtime_factory = MagicMock()

        def _emit_mcp_call_json(tool, args_json):
            print(
                json.dumps(
                    {
                        "status": 200,
                        "data": {
                            "tool": tool,
                            "args_json": args_json,
                            "result": {"ok": True},
                        },
                    }
                )
            )

        with patch("aura_cli.cli_main.cmd_mcp_call", side_effect=_emit_mcp_call_json):
            code_c, out_c, err_c, _ = self._dispatch(
                ["mcp", "call", "limits", "--args", "{\"x\":1}"],
                runtime_factory=runtime_factory,
            )
            code_l, out_l, err_l, _ = self._dispatch(
                ["--mcp-call", "limits", "--mcp-args", "{\"x\":1}"],
                runtime_factory=runtime_factory,
            )

        self.assertEqual(code_c, 0)
        self.assertEqual(code_l, 0)
        self.assertEqual(err_c, "")
        self.assertEqual(err_l, "Warning: Legacy flags are deprecated; use `aura mcp call` instead.\n")
        canonical_payload = json.loads(out_c)
        legacy_payload = json.loads(out_l)
        self.assertEqual(canonical_payload, self._without_cli_warnings(legacy_payload))
        self._assert_json_snapshot(out_c, "cli_canonical_mcp_call_dispatch.json")
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

    def test_legacy_workflow_run_json_output_matches_snapshot(self):
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
            code, out, err, _ = self._dispatch(
                ["--workflow-goal", "Summarize repo", "--workflow-max-cycles", "3", "--dry-run"],
                runtime_factory=runtime_factory,
            )

        self.assertEqual(code, 0)
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura workflow run` instead.\n")
        self._assert_json_snapshot(out, "cli_legacy_workflow_run_dispatch.json")

    def test_canonical_workflow_run_json_output_matches_snapshot_and_legacy_base_shape(self):
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
            code_c, out_c, err_c, _ = self._dispatch(
                ["workflow", "run", "Summarize repo", "--max-cycles", "3", "--dry-run"],
                runtime_factory=runtime_factory,
            )
            code_l, out_l, err_l, _ = self._dispatch(
                ["--workflow-goal", "Summarize repo", "--workflow-max-cycles", "3", "--dry-run"],
                runtime_factory=runtime_factory,
            )

        self.assertEqual(code_c, 0)
        self.assertEqual(code_l, 0)
        self.assertEqual(err_c, "")
        self.assertEqual(err_l, "Warning: Legacy flags are deprecated; use `aura workflow run` instead.\n")
        canonical_payload = json.loads(out_c)
        legacy_payload = json.loads(out_l)
        self.assertEqual(canonical_payload, self._without_cli_warnings(legacy_payload))
        self._assert_json_snapshot(out_c, "cli_canonical_workflow_run_dispatch.json")

    def test_legacy_evolve_json_output_matches_snapshot(self):
        fake_runtime = {
            "planner": MagicMock(),
            "model_adapter": MagicMock(),
            "brain": MagicMock(),
        }
        runtime_factory = MagicMock(return_value=fake_runtime)

        fake_evo = MagicMock()
        fake_evo.run.return_value = {
            "status": "ok",
            "mutations": 2,
            "artifacts": ["core/vector_store.py", "aura_cli/cli_main.py"],
            "meta": {"score": 0.9},
        }

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main.log_json"), \
             patch("aura_cli.cli_main.default_agents", return_value={"act": MagicMock(), "critique": MagicMock()}), \
             patch("aura_cli.cli_main.GitTools", return_value=MagicMock()), \
             patch("agents.mutator.MutatorAgent", return_value=MagicMock()), \
             patch("core.vector_store.VectorStore", return_value=MagicMock()), \
             patch("core.evolution_loop.EvolutionLoop", return_value=fake_evo):
            code, out, err, _ = self._dispatch(["--evolve"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura evolve` instead.\n")
        self._assert_json_snapshot(out, "cli_legacy_evolve_dispatch.json")
        fake_evo.run.assert_called_once_with("evolve and improve the AURA system")

    def test_canonical_evolve_json_output_matches_snapshot_and_legacy_base_shape(self):
        fake_runtime = {
            "planner": MagicMock(),
            "model_adapter": MagicMock(),
            "brain": MagicMock(),
        }
        runtime_factory = MagicMock(return_value=fake_runtime)

        fake_evo = MagicMock()
        fake_evo.run.return_value = {
            "status": "ok",
            "mutations": 2,
            "artifacts": ["core/vector_store.py", "aura_cli/cli_main.py"],
            "meta": {"score": 0.9},
        }

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main.log_json"), \
             patch("aura_cli.cli_main.default_agents", return_value={"act": MagicMock(), "critique": MagicMock()}), \
             patch("aura_cli.cli_main.GitTools", return_value=MagicMock()), \
             patch("agents.mutator.MutatorAgent", return_value=MagicMock()), \
             patch("core.vector_store.VectorStore", return_value=MagicMock()), \
             patch("core.evolution_loop.EvolutionLoop", return_value=fake_evo):
            code_c, out_c, err_c, _ = self._dispatch(["evolve"], runtime_factory=runtime_factory)
            code_l, out_l, err_l, _ = self._dispatch(["--evolve"], runtime_factory=runtime_factory)

        self.assertEqual(code_c, 0)
        self.assertEqual(code_l, 0)
        self.assertEqual(err_c, "")
        self.assertEqual(err_l, "Warning: Legacy flags are deprecated; use `aura evolve` instead.\n")
        canonical_payload = json.loads(out_c)
        legacy_payload = json.loads(out_l)
        self.assertEqual(canonical_payload, self._without_cli_warnings(legacy_payload))
        self._assert_json_snapshot(out_c, "cli_canonical_evolve_dispatch.json")
        self.assertEqual(fake_evo.run.call_args_list, [
            call("evolve and improve the AURA system"),
            call("evolve and improve the AURA system"),
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

    def test_goal_run_dispatch_lazily_initializes_legacy_loop_once(self):
        fake_runtime = {
            "goal_queue": MagicMock(),
            "goal_archive": MagicMock(),
            "orchestrator": MagicMock(),
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "loop": None,
            "model_adapter": MagicMock(),
            "brain": MagicMock(),
            "git_tools": MagicMock(),
        }
        runtime_factory = MagicMock(return_value=fake_runtime)
        fake_loop = MagicMock()

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main.log_json"), \
             patch("aura_cli.cli_main.HybridClosedLoop", return_value=fake_loop) as mock_hybrid, \
             patch("aura_cli.cli_main.run_goals_loop") as mock_run_goals:
            code1, *_ = self._dispatch(["goal", "run"], runtime_factory=runtime_factory)
            code2, *_ = self._dispatch(["goal", "run"], runtime_factory=runtime_factory)

        self.assertEqual(code1, 0)
        self.assertEqual(code2, 0)
        mock_hybrid.assert_called_once_with(
            fake_runtime["model_adapter"],
            fake_runtime["brain"],
            fake_runtime["git_tools"],
        )
        self.assertIs(fake_runtime["loop"], fake_loop)
        self.assertEqual(mock_run_goals.call_count, 2)
        self.assertIs(mock_run_goals.call_args_list[0].args[2], fake_loop)
        self.assertIs(mock_run_goals.call_args_list[1].args[2], fake_loop)

    def test_legacy_goal_status_json_output_includes_cli_warnings(self):
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

        def _emit_status_json(*_args, **_kwargs):
            print('{"queue_length": 0, "queue": [], "completed_count": 0, "completed": []}')

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main._handle_status", side_effect=_emit_status_json), \
             patch("aura_cli.cli_main.log_json"):
            code, out, err, _ = self._dispatch(["--status", "--json"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura goal status` instead.\n")
        payload = json.loads(out)
        self.assertEqual(payload["queue_length"], 0)
        self.assertIn("cli_warnings", payload)
        self.assertEqual(payload["cli_warnings"][0]["action"], "goal_status")
        self.assertEqual(payload["cli_warnings"][0]["replacement_command"], "aura goal status")

    def test_legacy_goal_status_json_output_matches_snapshot(self):
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

        def _emit_status_json(*_args, **_kwargs):
            print('{"queue_length": 0, "queue": [], "completed_count": 0, "completed": []}')

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main._handle_status", side_effect=_emit_status_json), \
             patch("aura_cli.cli_main.log_json"):
            code, out, err, _ = self._dispatch(["--status", "--json"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(err, "Warning: Legacy flags are deprecated; use `aura goal status` instead.\n")
        self._assert_json_snapshot(out, "cli_legacy_goal_status_dispatch.json")

    def test_canonical_goal_status_json_output_matches_snapshot_and_legacy_base_shape(self):
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

        def _emit_status_json(*_args, **_kwargs):
            print('{"queue_length": 0, "queue": [], "completed_count": 0, "completed": []}')

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main._handle_status", side_effect=_emit_status_json), \
             patch("aura_cli.cli_main.log_json"):
            code_c, out_c, err_c, _ = self._dispatch(["goal", "status", "--json"], runtime_factory=runtime_factory)
            code_l, out_l, err_l, _ = self._dispatch(["--status", "--json"], runtime_factory=runtime_factory)

        self.assertEqual(code_c, 0)
        self.assertEqual(code_l, 0)
        self.assertEqual(err_c, "")
        self.assertEqual(err_l, "Warning: Legacy flags are deprecated; use `aura goal status` instead.\n")
        canonical_payload = json.loads(out_c)
        legacy_payload = json.loads(out_l)
        self.assertEqual(canonical_payload, self._without_cli_warnings(legacy_payload))
        self._assert_json_snapshot(out_c, "cli_canonical_goal_status_dispatch.json")

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
        mock_studio.run.assert_called_once_with(autonomous=False)
        fake_runtime["orchestrator"].attach_ui_callback.assert_called_once_with(mock_studio)

    def test_watch_and_studio_commands_have_runtime_and_ui_parity(self):
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
            watch_code, watch_out, watch_err, _ = self._dispatch(["watch"], runtime_factory=runtime_factory)
            studio_code, studio_out, studio_err, _ = self._dispatch(["studio"], runtime_factory=runtime_factory)

        self.assertEqual(watch_code, 0)
        self.assertEqual(studio_code, 0)
        self.assertEqual(watch_out, "")
        self.assertEqual(studio_out, "")
        self.assertEqual(watch_err, "")
        self.assertEqual(studio_err, "")
        self.assertEqual(runtime_factory.call_count, 2)
        self.assertEqual(mock_studio_cls.call_args_list, [call(runtime=fake_runtime), call(runtime=fake_runtime)])
        self.assertEqual(mock_studio.run.call_args_list, [call(autonomous=False), call(autonomous=False)])
        self.assertEqual(
            fake_runtime["orchestrator"].attach_ui_callback.call_args_list,
            [call(mock_studio), call(mock_studio)],
        )

    def test_watch_and_studio_autonomous_flag_is_forwarded_to_ui_run(self):
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
            watch_code, *_ = self._dispatch(["watch", "--autonomous"], runtime_factory=runtime_factory)
            studio_code, *_ = self._dispatch(["studio", "--autonomous"], runtime_factory=runtime_factory)

        self.assertEqual(watch_code, 0)
        self.assertEqual(studio_code, 0)
        self.assertEqual(mock_studio.run.call_args_list, [call(autonomous=True), call(autonomous=True)])


if __name__ == "__main__":
    unittest.main()
