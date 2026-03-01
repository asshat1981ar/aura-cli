import json
import io
import os
import tempfile
import unittest
import builtins
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from aura_cli.cli_options import parse_cli_args
import aura_cli.cli_main as cli_main
import aura_cli.options as cli_options_meta
from tests.cli_entrypoint_test_utils import run_main_subprocess
from tests.cli_snapshot_utils import normalized_json_text, read_snapshot_text


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

    def _assert_json_snapshot(self, raw_json: str, snapshot_name: str) -> None:
        self.assertEqual(normalized_json_text(raw_json), read_snapshot_text(self._SNAPSHOT_DIR, snapshot_name))

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

    def test_runtime_initialization_follows_action_contract_for_all_actions(self):
        runtime_factory = MagicMock(return_value={})
        runtime_seen: dict[str, bool] = {}

        def _make_handler(action_name: str):
            def _handler(ctx):
                runtime_seen[action_name] = ctx.runtime is not None
                return 0

            return _handler

        stub_registry = {
            action: cli_main.DispatchRule(
                action=action,
                requires_runtime=rule.requires_runtime,
                handler=_make_handler(action),
            )
            for action, rule in cli_main.COMMAND_DISPATCH_REGISTRY.items()
        }

        with patch.dict(cli_main.COMMAND_DISPATCH_REGISTRY, stub_registry, clear=True), \
             patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main.log_json"):
            for action, spec in sorted(cli_options_meta.CLI_ACTION_SPECS_BY_ACTION.items()):
                with self.subTest(action=action):
                    argv = list(cli_options_meta.action_smoke_argv(action))
                    parsed = parse_cli_args(argv)
                    runtime_factory.reset_mock()
                    code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=runtime_factory)

                    self.assertEqual(code, 0)
                    self.assertEqual(runtime_seen.get(action), spec.requires_runtime)
                    if spec.requires_runtime:
                        runtime_factory.assert_called_once()
                    else:
                        runtime_factory.assert_not_called()

    def test_dispatch_doctor_does_not_create_runtime(self):
        parsed = parse_cli_args(["doctor"])
        runtime_factory = MagicMock()

        with patch("aura_cli.cli_main._handle_doctor") as mock_doctor:
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        mock_doctor.assert_called_once_with(Path("."))
        runtime_factory.assert_not_called()

    def test_dispatch_config_does_not_create_runtime(self):
        parsed = parse_cli_args(["config"])
        runtime_factory = MagicMock()

        out = io.StringIO()
        with patch("aura_cli.cli_main.config.show_config", return_value={"model_name": "gpt-5"}), redirect_stdout(out):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), {"model_name": "gpt-5"})
        runtime_factory.assert_not_called()

    def test_dispatch_contract_report_does_not_create_runtime(self):
        parsed = parse_cli_args(["contract-report", "--check"])
        runtime_factory = MagicMock()

        report = {"ok": True, "failure_keys": [], "missing_in_help_schema": [], "smoke_dispatch_mismatches": []}
        with patch("aura_cli.contract_report.build_cli_contract_report", return_value=report) as mock_build, \
             patch("aura_cli.contract_report.render_cli_contract_report", return_value='{"ok":true}\n') as mock_render, \
             patch("aura_cli.contract_report.cli_contract_report_exit_code", return_value=0) as mock_exit:
            code, out, err, _ = self._dispatch(["contract-report", "--check"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(out, '{"ok":true}\n')
        self.assertEqual(err, "")
        runtime_factory.assert_not_called()
        mock_build.assert_called_once_with(include_dispatch=True, dispatch_registry=cli_main.COMMAND_DISPATCH_REGISTRY)
        mock_render.assert_called_once_with(report, compact=False)
        mock_exit.assert_called_once_with(report, check=True)

    def test_dispatch_contract_report_no_dispatch_compact_uses_flags(self):
        report = {"ok": True, "failure_keys": []}

        with patch("aura_cli.contract_report.build_cli_contract_report", return_value=report) as mock_build, \
             patch("aura_cli.contract_report.render_cli_contract_report", return_value='{"ok":true}\n') as mock_render, \
             patch("aura_cli.contract_report.cli_contract_report_exit_code", return_value=0):
            code, out, err, runtime_factory = self._dispatch(["contract-report", "--no-dispatch", "--compact"])

        self.assertEqual(code, 0)
        self.assertEqual(out, '{"ok":true}\n')
        self.assertEqual(err, "")
        runtime_factory.assert_not_called()
        mock_build.assert_called_once_with(include_dispatch=False, dispatch_registry=cli_main.COMMAND_DISPATCH_REGISTRY)
        mock_render.assert_called_once_with(report, compact=True)

    def test_canonical_contract_report_json_output_matches_snapshot(self):
        runtime_factory = MagicMock()

        code, out, err, _ = self._dispatch(["contract-report", "--check"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self._assert_json_snapshot(out, "cli_contract_report.json")
        runtime_factory.assert_not_called()

    def test_dispatch_contract_report_check_returns_nonzero_for_dirty_report(self):
        runtime_factory = MagicMock()
        dirty_report = {
            "ok": False,
            "failure_keys": ["missing_in_help_schema"],
            "missing_in_help_schema": [["contract-report"]],
        }

        with patch("aura_cli.contract_report.build_cli_contract_report", return_value=dirty_report):
            code, out, err, _ = self._dispatch(["contract-report", "--check"], runtime_factory=runtime_factory)

        self.assertEqual(code, 1)
        self._assert_json_snapshot(out, "cli_contract_report_dirty.json")
        self.assertEqual(
            err,
            "CLI contract failures: missing_in_help_schema\n",
        )
        runtime_factory.assert_not_called()

    def test_canonical_config_json_output_matches_snapshot(self):
        runtime_factory = MagicMock()

        with patch(
            "aura_cli.cli_main.config.show_config",
            return_value={"policy_max_cycles": 5, "model_name": "gpt-5"},
        ):
            code, out, err, _ = self._dispatch(["config"], runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self._assert_json_snapshot(out, "cli_canonical_config_dispatch.json")
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
        self.assertEqual(runtime_factory.call_args.kwargs["overrides"], {"runtime_mode": "queue"})
        mock_status.assert_called_once_with(
            fake_runtime["goal_queue"],
            fake_runtime["goal_archive"],
            fake_runtime["orchestrator"],
            as_json=True,
            project_root=Path("."),
            memory_persistence_path=None,
        )

    def test_create_runtime_queue_mode_skips_heavy_components(self):
        fake_goal_queue = MagicMock()
        fake_goal_archive = MagicMock()

        with tempfile.TemporaryDirectory() as d, \
             patch("aura_cli.cli_main.GoalQueue", return_value=fake_goal_queue), \
             patch("aura_cli.cli_main.GoalArchive", return_value=fake_goal_archive), \
             patch("aura_cli.cli_main.ModelAdapter") as mock_model_adapter, \
             patch("aura_cli.cli_main.VectorStore") as mock_vector_store, \
             patch("aura_cli.cli_main.LoopOrchestrator") as mock_orchestrator, \
             patch("aura_cli.cli_main.GitTools") as mock_git_tools, \
             patch("aura_cli.cli_main.log_json"):
            runtime = cli_main.create_runtime(Path(d), overrides={"runtime_mode": "queue"})

        self.assertIs(runtime["goal_queue"], fake_goal_queue)
        self.assertIs(runtime["goal_archive"], fake_goal_archive)
        self.assertIsNone(runtime["model_adapter"])
        self.assertIsNone(runtime["memory_store"])
        self.assertIsNone(runtime["brain"])
        self.assertIsNone(runtime["vector_store"])
        self.assertFalse(hasattr(runtime["orchestrator"], "run_loop"))
        mock_model_adapter.assert_not_called()
        mock_vector_store.assert_not_called()
        mock_orchestrator.assert_not_called()
        mock_git_tools.assert_not_called()

    def test_create_runtime_queue_mode_resolves_storage_paths_against_project_root(self):
        fake_goal_queue = MagicMock()
        fake_goal_archive = MagicMock()

        def _config_get(key, default=None):
            mapping = {
                "goal_queue_path": "state/custom_queue.json",
                "goal_archive_path": "state/custom_archive.json",
                "api_key": None,
            }
            return mapping.get(key, default)

        with tempfile.TemporaryDirectory() as d, \
             patch("aura_cli.cli_main.GoalQueue", return_value=fake_goal_queue) as mock_goal_queue, \
             patch("aura_cli.cli_main.GoalArchive", return_value=fake_goal_archive) as mock_goal_archive, \
             patch.object(cli_main.config, "get", side_effect=_config_get), \
             patch("aura_cli.cli_main.log_json"):
            cli_main.create_runtime(Path(d), overrides={"runtime_mode": "queue"})

        self.assertEqual(mock_goal_queue.call_args.args[0], str(Path(d) / "state" / "custom_queue.json"))
        self.assertEqual(mock_goal_archive.call_args.args[0], str(Path(d) / "state" / "custom_archive.json"))

    def test_dispatch_goal_add_uses_queue_runtime_mode(self):
        parsed = parse_cli_args(["goal", "add", "Fix tests"])
        goal_queue = MagicMock()
        fake_runtime = {
            "goal_queue": goal_queue,
            "goal_archive": MagicMock(),
            "orchestrator": MagicMock(),
            "debugger": None,
            "planner": None,
            "loop": None,
            "model_adapter": None,
            "brain": None,
        }
        runtime_factory = MagicMock(return_value=fake_runtime)

        with patch("aura_cli.cli_main._check_project_writability", return_value=True), \
             patch("aura_cli.cli_main.log_json"):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=runtime_factory)

        self.assertEqual(code, 0)
        self.assertEqual(runtime_factory.call_args.kwargs["overrides"], {"runtime_mode": "queue"})
        goal_queue.add.assert_called_once_with("Fix tests")

    def test_goal_status_json_subprocess_stdout_is_clean_json_and_skips_heavy_runtime_logs(self):
        proc = run_main_subprocess("goal", "status", "--json")

        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertIn("queue_length", payload)
        self.assertIn("queue", payload)
        self.assertIn("completed_count", payload)
        self.assertIn("completed", payload)
        self.assertNotIn("vector_store_initialized", proc.stderr)
        self.assertNotIn("background_sync_started", proc.stderr)

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
            patch("core.hybrid_loop.HybridClosedLoop") as mock_hybrid_loop, \
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

    def test_create_runtime_full_mode_resolves_brain_and_memory_paths_against_project_root(self):
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

        def _config_get(key, default=None):
            mapping = {
                "goal_queue_path": "state/custom_queue.json",
                "goal_archive_path": "state/custom_archive.json",
                "brain_db_path": "state/custom_brain.db",
                "memory_store_path": "state/store",
                "api_key": None,
                "strict_schema": False,
            }
            return mapping.get(key, default)

        with tempfile.TemporaryDirectory() as d, \
             patch("aura_cli.cli_main.GoalQueue", return_value=fake_goal_queue), \
             patch("aura_cli.cli_main.GoalArchive", return_value=fake_goal_archive), \
             patch("aura_cli.cli_main.Brain", return_value=fake_brain) as mock_brain_cls, \
             patch("aura_cli.cli_main.ModelAdapter", return_value=fake_model_adapter), \
             patch("aura_cli.cli_main.VectorStore", return_value=fake_vector_store), \
             patch("aura_cli.cli_main.RouterAgent", return_value=fake_router), \
             patch("aura_cli.cli_main.DebuggerAgent", return_value=fake_debugger), \
             patch("aura_cli.cli_main.PlannerAgent", return_value=fake_planner), \
             patch("aura_cli.cli_main.MemoryStore", return_value=fake_memory_store) as mock_memory_store_cls, \
             patch("aura_cli.cli_main.default_agents", return_value={}), \
             patch("aura_cli.cli_main.Policy.from_config", return_value=fake_policy), \
             patch("aura_cli.cli_main.LoopOrchestrator", return_value=fake_orchestrator), \
             patch("aura_cli.cli_main.GitTools", return_value=fake_git_tools), \
             patch("core.hybrid_loop.HybridClosedLoop"), \
             patch.object(cli_main.config, "get", side_effect=_config_get), \
             patch("aura_cli.cli_main.log_json"), \
             patch("builtins.__import__", side_effect=_guarded_import):
            cli_main.create_runtime(Path(d), overrides=None)

        mock_brain_cls.assert_called_once_with(db_path=str(Path(d) / "state" / "custom_brain.db"))
        mock_memory_store_cls.assert_called_once_with(Path(d) / "state" / "store")

    def test_create_runtime_full_mode_honors_env_var_storage_paths_end_to_end(self):
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

        original_runtime_overrides = dict(cli_main.config.runtime_overrides)
        try:
            with patch.dict(
                os.environ,
                {
                    "AURA_BRAIN_DB_PATH": "env_state/runtime_brain.db",
                    "AURA_MEMORY_STORE_PATH": "env_state/runtime_store",
                    "AURA_LOCAL_MODEL_COMMAND": "echo local-model",
                },
                clear=False,
            ):
                cli_main.config.refresh()
                with tempfile.TemporaryDirectory() as d, \
                     patch("aura_cli.cli_main.GoalQueue", return_value=fake_goal_queue), \
                     patch("aura_cli.cli_main.GoalArchive", return_value=fake_goal_archive), \
                     patch("aura_cli.cli_main.Brain", return_value=fake_brain) as mock_brain_cls, \
                     patch("aura_cli.cli_main.ModelAdapter", return_value=fake_model_adapter), \
                     patch("aura_cli.cli_main.VectorStore", return_value=fake_vector_store), \
                     patch("aura_cli.cli_main.RouterAgent", return_value=fake_router), \
                     patch("aura_cli.cli_main.DebuggerAgent", return_value=fake_debugger), \
                     patch("aura_cli.cli_main.PlannerAgent", return_value=fake_planner), \
                     patch("aura_cli.cli_main.MemoryStore", return_value=fake_memory_store) as mock_memory_store_cls, \
                     patch("aura_cli.cli_main.default_agents", return_value={}), \
                     patch("aura_cli.cli_main.Policy.from_config", return_value=fake_policy), \
                     patch("aura_cli.cli_main.LoopOrchestrator", return_value=fake_orchestrator), \
                     patch("aura_cli.cli_main.GitTools", return_value=fake_git_tools), \
                     patch("core.hybrid_loop.HybridClosedLoop"), \
                     patch("aura_cli.cli_main.log_json"), \
                     patch("builtins.__import__", side_effect=_guarded_import):
                    cli_main.create_runtime(Path(d), overrides=None)
        finally:
            cli_main.config.runtime_overrides = original_runtime_overrides
            cli_main.config.refresh()

        mock_brain_cls.assert_called_once_with(db_path=str(Path(d) / "env_state" / "runtime_brain.db"))
        mock_memory_store_cls.assert_called_once_with(Path(d) / "env_state" / "runtime_store")


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
            call(
                fake_runtime["goal_queue"],
                fake_runtime["goal_archive"],
                fake_runtime["orchestrator"],
                as_json=True,
                project_root=Path("."),
                memory_persistence_path=None,
            ),
            call(
                fake_runtime["goal_queue"],
                fake_runtime["goal_archive"],
                fake_runtime["orchestrator"],
                as_json=True,
                project_root=Path("."),
                memory_persistence_path=None,
            ),
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
        self.assertEqual(payload["cli_warnings"][0]["code"], cli_options_meta.CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED)
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
             patch("core.hybrid_loop.HybridClosedLoop", return_value=fake_loop) as mock_hybrid, \
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
