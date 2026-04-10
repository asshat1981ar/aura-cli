"""Deep coverage tests for aura_cli/dispatch.py — uncovered lines pass."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    defaults = dict(
        action=None,
        dry_run=False,
        decompose=False,
        model=None,
        anthropic_api_key=None,
        beads=False,
        no_beads=False,
        beads_required=False,
        beads_optional=False,
        json=False,
        help_topics=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_ctx(args=None, *, project_root=None, runtime=None, parsed=None):
    from aura_cli.dispatch import DispatchContext

    ctx = DispatchContext(
        parsed=parsed or SimpleNamespace(warning_records=[], warnings=[]),
        project_root=project_root or Path("/fake/proj"),
        runtime_factory=MagicMock(),
        args=args or _make_args(),
    )
    if runtime is not None:
        ctx.runtime = runtime
    return ctx


# ---------------------------------------------------------------------------
# _run_async_safely
# ---------------------------------------------------------------------------


class TestRunAsyncSafely:
    def test_uses_anyio_when_available(self):
        from aura_cli.dispatch import _run_async_safely

        async def _coro():
            return 42

        coro = _coro()
        mock_anyio = MagicMock()
        mock_anyio.run.return_value = 42
        with patch("aura_cli.dispatch.anyio_available", True), patch(
            "aura_cli.dispatch.anyio", mock_anyio, create=True
        ):
            result = _run_async_safely(coro)
        # Close coroutine in case mock didn't consume it
        try:
            coro.close()
        except Exception:
            pass
        assert result == 42

    def test_falls_back_to_asyncio_when_no_anyio(self):
        from aura_cli.dispatch import _run_async_safely

        async def _coro():
            return "hello"

        with patch("aura_cli.dispatch.anyio_available", False):
            result = _run_async_safely(_coro())
        assert result == "hello"

    def test_reraises_non_already_running_runtime_error(self):
        from aura_cli.dispatch import _run_async_safely

        async def _coro():
            return None

        coro = _coro()
        with patch("aura_cli.dispatch.anyio_available", False), patch(
            "asyncio.run", side_effect=RuntimeError("something else")
        ):
            with pytest.raises(RuntimeError, match="something else"):
                _run_async_safely(coro)
        try:
            coro.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# _sync_cli_compat
# ---------------------------------------------------------------------------


class TestSyncCliCompat:
    def test_imports_and_sets_attributes(self):
        from aura_cli.dispatch import _sync_cli_compat
        import aura_cli.dispatch as dispatch_mod

        mock_cli_main = MagicMock()
        mock_cli_main.log_json = object()
        mock_cli_main._check_project_writability = object()
        mock_cli_main.render_help = object()
        mock_cli_main.attach_cli_warnings = object()
        mock_cli_main.unknown_command_help_topic_payload = object()
        # provide all the names so getattr doesn't fail
        for name in (
            "cmd_diag",
            "cmd_mcp_call",
            "cmd_mcp_tools",
            "config",
            "DEFAULT_CONFIG",
            "_handle_status",
            "_handle_doctor",
            "_handle_help",
            "_handle_readiness",
            "_handle_add",
            "_handle_run",
            "_handle_exit",
            "_handle_clear",
            "run_goals_loop",
            "default_agents",
            "GitTools",
            "VectorStore",
            "Brain",
            "ScaffolderAgent",
        ):
            setattr(mock_cli_main, name, object())

        with patch("importlib.import_module", return_value=mock_cli_main):
            _sync_cli_compat()

        assert dispatch_mod.render_help is mock_cli_main.render_help


# ---------------------------------------------------------------------------
# _prepare_runtime_context
# ---------------------------------------------------------------------------


class TestPrepareRuntimeContext:
    def _make_full_ctx(self, **arg_kwargs):
        ctx = _make_ctx(args=_make_args(**arg_kwargs))
        return ctx

    def test_skips_when_runtime_already_set(self):
        from aura_cli.dispatch import _prepare_runtime_context

        ctx = _make_ctx()
        ctx.runtime = {"already": "set"}
        result = _prepare_runtime_context(ctx)
        assert result is None
        ctx.runtime_factory.assert_not_called()

    def test_calls_runtime_factory_with_overrides(self):
        from aura_cli.dispatch import _prepare_runtime_context

        ctx = self._make_full_ctx(model="claude-3", dry_run=True)
        mock_runtime = MagicMock()
        ctx.runtime_factory.return_value = mock_runtime

        with patch("aura_cli.dispatch._sync_cli_compat"), patch(
            "aura_cli.dispatch.log_json"
        ), patch("aura_cli.dispatch._check_project_writability", return_value=True):
            result = _prepare_runtime_context(ctx)

        assert result is None
        ctx.runtime_factory.assert_called_once()
        _, kwargs = ctx.runtime_factory.call_args
        overrides = kwargs["overrides"]
        assert overrides["model_name"] == "claude-3"
        assert overrides["dry_run"] is True

    def test_returns_one_when_not_writable(self):
        from aura_cli.dispatch import _prepare_runtime_context

        ctx = self._make_full_ctx()
        ctx.runtime_factory.return_value = MagicMock()

        with patch("aura_cli.dispatch._sync_cli_compat"), patch(
            "aura_cli.dispatch.log_json"
        ), patch("aura_cli.dispatch._check_project_writability", return_value=False):
            result = _prepare_runtime_context(ctx)

        assert result == 1

    def test_beads_flag_adds_beads_overrides(self):
        from aura_cli.dispatch import _prepare_runtime_context

        ctx = self._make_full_ctx(beads=True)
        ctx.runtime_factory.return_value = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = {}

        with patch("aura_cli.dispatch._sync_cli_compat"), patch(
            "aura_cli.dispatch.log_json"
        ), patch(
            "aura_cli.dispatch._check_project_writability", return_value=True
        ), patch(
            "aura_cli.dispatch.config", mock_cfg
        ), patch(
            "aura_cli.dispatch.DEFAULT_CONFIG", {"beads": {}}
        ):
            _prepare_runtime_context(ctx)

        _, kwargs = ctx.runtime_factory.call_args
        overrides = kwargs["overrides"]
        assert "beads" in overrides

    def test_decompose_flag_added_to_overrides(self):
        from aura_cli.dispatch import _prepare_runtime_context

        ctx = self._make_full_ctx(decompose=True)
        ctx.runtime_factory.return_value = MagicMock()

        with patch("aura_cli.dispatch._sync_cli_compat"), patch(
            "aura_cli.dispatch.log_json"
        ), patch("aura_cli.dispatch._check_project_writability", return_value=True):
            _prepare_runtime_context(ctx)

        _, kwargs = ctx.runtime_factory.call_args
        assert kwargs["overrides"]["decompose"] is True


# ---------------------------------------------------------------------------
# _print_json_payload
# ---------------------------------------------------------------------------


class TestPrintJsonPayload:
    def test_prints_json_with_warnings(self, capsys):
        from aura_cli.dispatch import _print_json_payload

        with patch(
            "aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p
        ):
            _print_json_payload({"key": "val"}, parsed=None, indent=2)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["key"] == "val"


# ---------------------------------------------------------------------------
# _run_json_printing_callable_with_warnings
# ---------------------------------------------------------------------------


class TestRunJsonPrintingCallable:
    def test_no_warning_records_calls_func_directly(self, capsys):
        from aura_cli.dispatch import _run_json_printing_callable_with_warnings

        ctx = _make_ctx(parsed=SimpleNamespace(warning_records=None, warnings=[]))
        func = MagicMock(return_value=7)
        result = _run_json_printing_callable_with_warnings(ctx, func, "arg1")
        func.assert_called_once_with("arg1")
        assert result == 7

    def test_json_output_is_decorated_with_warnings(self, capsys):
        from aura_cli.dispatch import _run_json_printing_callable_with_warnings

        ctx = _make_ctx(
            parsed=SimpleNamespace(
                warning_records=[SimpleNamespace(message="warn!")], warnings=[]
            )
        )

        def _func():
            print(json.dumps({"ok": True}))
            return 0

        with patch(
            "aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p
        ):
            result = _run_json_printing_callable_with_warnings(ctx, _func)

        assert result == 0
        out = capsys.readouterr().out
        assert json.loads(out)["ok"] is True

    def test_non_json_output_passed_through(self, capsys):
        from aura_cli.dispatch import _run_json_printing_callable_with_warnings

        ctx = _make_ctx(
            parsed=SimpleNamespace(warning_records=["w"], warnings=[])
        )

        def _func():
            print("plain text output")

        _run_json_printing_callable_with_warnings(ctx, _func)
        out = capsys.readouterr().out
        assert "plain text output" in out


# ---------------------------------------------------------------------------
# _handle_bootstrap_dispatch
# ---------------------------------------------------------------------------


class TestBootstrapDispatch:
    def test_calls_interactive_bootstrap(self):
        from aura_cli.dispatch import _handle_bootstrap_dispatch

        mock_cfg = MagicMock()
        ctx = _make_ctx()
        with patch("aura_cli.dispatch.config", mock_cfg):
            code = _handle_bootstrap_dispatch(ctx)

        mock_cfg.interactive_bootstrap.assert_called_once()
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_contract_report_dispatch
# ---------------------------------------------------------------------------


class TestContractReportDispatch:
    def test_returns_exit_code_from_report(self):
        from aura_cli.dispatch import _handle_contract_report_dispatch

        ctx = _make_ctx(args=_make_args())
        ctx.args.no_dispatch = False
        ctx.args.compact = False
        ctx.args.check = False

        mock_report = MagicMock()
        with patch(
            "aura_cli.dispatch._handle_contract_report_dispatch.__module__",
            "aura_cli.dispatch",
        ), patch(
            "aura_cli.contract_report.build_cli_contract_report",
            return_value=mock_report,
        ), patch(
            "aura_cli.contract_report.render_cli_contract_report", return_value="report\n"
        ), patch(
            "aura_cli.contract_report.cli_contract_report_exit_code", return_value=0
        ), patch(
            "aura_cli.contract_report.cli_contract_report_failure_message",
            return_value="",
        ):
            code = _handle_contract_report_dispatch(ctx)

        assert code == 0


# ---------------------------------------------------------------------------
# _handle_mcp_tools_dispatch / _handle_mcp_call_dispatch / _handle_diag_dispatch
# ---------------------------------------------------------------------------


class TestSimpleMcpDispatches:
    def test_mcp_tools_calls_cmd_mcp_tools(self, capsys):
        from aura_cli.dispatch import _handle_mcp_tools_dispatch

        ctx = _make_ctx(parsed=SimpleNamespace(warning_records=None, warnings=[]))
        with patch("aura_cli.dispatch.cmd_mcp_tools", return_value=0) as mock_t:
            _handle_mcp_tools_dispatch(ctx)
        mock_t.assert_called_once()

    def test_mcp_call_dispatch_passes_args(self, capsys):
        from aura_cli.dispatch import _handle_mcp_call_dispatch

        args = _make_args()
        args.mcp_call = "tool_name"
        args.mcp_args = ["a", "b"]
        ctx = _make_ctx(args=args, parsed=SimpleNamespace(warning_records=None, warnings=[]))
        with patch("aura_cli.dispatch.cmd_mcp_call", return_value=0) as mock_c:
            _handle_mcp_call_dispatch(ctx)
        mock_c.assert_called_once_with("tool_name", ["a", "b"])

    def test_diag_dispatch_calls_cmd_diag(self, capsys):
        from aura_cli.dispatch import _handle_diag_dispatch

        ctx = _make_ctx(parsed=SimpleNamespace(warning_records=None, warnings=[]))
        with patch("aura_cli.dispatch.cmd_diag", return_value=0) as mock_d:
            _handle_diag_dispatch(ctx)
        mock_d.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_logs_dispatch
# ---------------------------------------------------------------------------


class TestLogsDispatch:
    def test_stream_file_when_file_arg_present(self):
        from aura_cli.dispatch import _handle_logs_dispatch

        args = _make_args()
        args.level = "debug"
        args.file = "/some/path.log"
        args.tail = 20
        args.follow = False
        ctx = _make_ctx(args=args)

        mock_streamer = MagicMock()
        with patch("aura_cli.tui.log_streamer.LogStreamer", return_value=mock_streamer):
            code = _handle_logs_dispatch(ctx)

        mock_streamer.stream_file.assert_called_once()
        assert code == 0

    def test_stream_stdin_when_no_file_arg(self):
        from aura_cli.dispatch import _handle_logs_dispatch

        args = _make_args()
        args.level = "info"
        args.file = None
        args.tail = None
        ctx = _make_ctx(args=args)

        mock_streamer = MagicMock()
        with patch("aura_cli.tui.log_streamer.LogStreamer", return_value=mock_streamer):
            code = _handle_logs_dispatch(ctx)

        mock_streamer.stream_stdin.assert_called_once()
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_history_dispatch
# ---------------------------------------------------------------------------


class TestHistoryDispatch:
    def test_calls_handle_history_with_runtime(self):
        from aura_cli.dispatch import _handle_history_dispatch

        mock_archive = MagicMock()
        args = _make_args()
        args.limit = 5
        args.json = False
        ctx = _make_ctx(args=args)
        ctx.runtime = {"goal_archive": mock_archive}

        # _handle_history is imported locally inside the handler from aura_cli.commands
        with patch("aura_cli.commands._handle_history") as mock_h:
            code = _handle_history_dispatch(ctx)

        mock_h.assert_called_once_with(mock_archive, limit=5, as_json=False)
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_queue_clear_dispatch (json mode with warnings cleared)
# ---------------------------------------------------------------------------


class TestQueueClearDispatch:
    def test_json_mode_outputs_cleared_count(self, capsys):
        from aura_cli.dispatch import _handle_queue_clear_dispatch

        mock_queue = MagicMock()
        mock_queue.queue = ["a", "b"]
        ctx = _make_ctx(
            args=_make_args(json=True),
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )
        ctx.runtime = {"goal_queue": mock_queue}
        with patch("aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p):
            code = _handle_queue_clear_dispatch(ctx)
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["cleared_count"] == 2


# ---------------------------------------------------------------------------
# _handle_goal_once_dispatch
# ---------------------------------------------------------------------------


class TestGoalOnceDispatch:
    def _make_goal_ctx(self, **arg_kwargs):
        args = _make_args(**arg_kwargs)
        args.goal = "do something"
        args.max_cycles = 3
        args.dry_run = arg_kwargs.get("dry_run", False)
        args.explain = False
        args.json = False
        ctx = _make_ctx(args=args)
        return ctx

    def _mock_orchestrator(self, result=None):
        orch = MagicMock()
        orch.run_loop.return_value = result or {"stop_reason": "done", "history": []}
        return orch

    def test_success_returns_exit_success(self, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch

        ctx = self._make_goal_ctx()
        ctx.runtime = {"orchestrator": self._mock_orchestrator()}

        with patch("aura_cli.dispatch.config") as mock_cfg:
            mock_cfg.get.return_value = 5
            with patch("aura_cli.dispatch._print_json_payload"):
                code = _handle_goal_once_dispatch(ctx)

        assert code == 0  # EXIT_SUCCESS

    def test_keyboard_interrupt_returns_cancelled(self, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch

        ctx = self._make_goal_ctx()
        mock_orch = MagicMock()
        mock_orch.run_loop.side_effect = KeyboardInterrupt()
        ctx.runtime = {"orchestrator": mock_orch}

        with patch("aura_cli.dispatch.config") as mock_cfg:
            mock_cfg.get.return_value = 5
            code = _handle_goal_once_dispatch(ctx)

        assert code != 0
        assert "Cancelled" in capsys.readouterr().err

    def test_generic_exception_returns_failure(self, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch

        ctx = self._make_goal_ctx()
        mock_orch = MagicMock()
        mock_orch.run_loop.side_effect = RuntimeError("boom")
        ctx.runtime = {"orchestrator": mock_orch}

        with patch("aura_cli.dispatch.config") as mock_cfg, patch(
            "aura_cli.dispatch.log_json"
        ):
            mock_cfg.get.return_value = 5
            code = _handle_goal_once_dispatch(ctx)

        assert code != 0
        assert "boom" in capsys.readouterr().err

    def test_json_mode_prints_json_payload(self, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch

        args = _make_args(json=True)
        args.goal = "do something"
        args.max_cycles = 1
        args.dry_run = False
        args.explain = False
        ctx = _make_ctx(args=args, parsed=SimpleNamespace(warning_records=[], warnings=[]))
        ctx.runtime = {"orchestrator": self._mock_orchestrator()}

        with patch("aura_cli.dispatch.config") as mock_cfg, patch(
            "aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p
        ):
            mock_cfg.get.return_value = 5
            with patch(
                "core.operator_runtime.build_beads_runtime_metadata", return_value={}
            ):
                code = _handle_goal_once_dispatch(ctx)

        assert code == 0

    def test_dry_run_prints_mode(self, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch

        ctx = self._make_goal_ctx(dry_run=True)
        ctx.runtime = {"orchestrator": self._mock_orchestrator()}

        with patch("aura_cli.dispatch.config") as mock_cfg:
            mock_cfg.get.return_value = 5
            code = _handle_goal_once_dispatch(ctx)

        out = capsys.readouterr().out
        assert "Dry-run" in out or "dry" in out.lower()
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_goal_run_dispatch
# ---------------------------------------------------------------------------


class TestGoalRunDispatch:
    def _make_goal_run_ctx(self, **arg_kwargs):
        args = _make_args(**arg_kwargs)
        args.resume = False
        args.decompose = False
        ctx = _make_ctx(args=args)
        ctx.runtime = {
            "goal_queue": MagicMock(),
            "orchestrator": MagicMock(),
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "goal_archive": MagicMock(),
        }
        return ctx

    def test_success_returns_zero(self):
        from aura_cli.dispatch import _handle_goal_run_dispatch

        ctx = self._make_goal_run_ctx()

        with patch("aura_cli.dispatch.run_goals_loop") as mock_loop, patch(
            "core.in_flight_tracker.InFlightTracker"
        ) as mock_tracker_cls:
            mock_tracker_cls.return_value.exists.return_value = False
            code = _handle_goal_run_dispatch(ctx)

        mock_loop.assert_called_once()
        assert code == 0

    def test_keyboard_interrupt_returns_cancelled(self, capsys):
        from aura_cli.dispatch import _handle_goal_run_dispatch

        ctx = self._make_goal_run_ctx()

        with patch(
            "aura_cli.dispatch.run_goals_loop", side_effect=KeyboardInterrupt()
        ), patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker_cls.return_value.exists.return_value = False
            code = _handle_goal_run_dispatch(ctx)

        assert code != 0

    def test_resume_flag_prepends_goal(self, capsys):
        from aura_cli.dispatch import _handle_goal_run_dispatch

        args = _make_args()
        args.resume = True
        args.decompose = False
        ctx = _make_ctx(args=args)
        ctx.runtime = {
            "goal_queue": MagicMock(),
            "orchestrator": MagicMock(),
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "goal_archive": MagicMock(),
        }

        with patch("aura_cli.dispatch.run_goals_loop"), patch(
            "core.in_flight_tracker.InFlightTracker"
        ) as mock_tracker_cls:
            mock_inst = mock_tracker_cls.return_value
            mock_inst.exists.return_value = True
            mock_inst.read.return_value = {"goal": "resume-me"}
            code = _handle_goal_run_dispatch(ctx)

        ctx.runtime["goal_queue"].prepend_batch.assert_called_once_with(["resume-me"])
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_interactive_dispatch
# ---------------------------------------------------------------------------


class TestInteractiveDispatch:
    def test_calls_cli_interaction_loop(self):
        from aura_cli.dispatch import _handle_interactive_dispatch

        ctx = _make_ctx()
        ctx.runtime = {"some": "runtime"}

        with patch("aura_cli.cli_main.cli_interaction_loop") as mock_loop:
            code = _handle_interactive_dispatch(ctx)

        mock_loop.assert_called_once_with(ctx.args, ctx.runtime)
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_goal_add_dispatch / _handle_goal_add_run_dispatch
# ---------------------------------------------------------------------------


class TestGoalAddDispatch:
    def test_goal_add_calls_maybe_add_goal(self):
        from aura_cli.dispatch import _handle_goal_add_dispatch

        args = _make_args()
        args.add_goal = "new goal"
        ctx = _make_ctx(args=args)
        mock_queue = MagicMock()
        mock_queue.queue = []
        ctx.runtime = {"goal_queue": mock_queue}

        with patch("aura_cli.dispatch.log_json"):
            code = _handle_goal_add_dispatch(ctx)

        mock_queue.add.assert_called_once_with("new goal")
        assert code == 0

    def test_goal_add_no_add_goal_attr_is_noop(self):
        from aura_cli.dispatch import _handle_goal_add_dispatch

        ctx = _make_ctx()
        ctx.runtime = {"goal_queue": MagicMock()}
        code = _handle_goal_add_dispatch(ctx)
        ctx.runtime["goal_queue"].add.assert_not_called()
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_goal_resume_dispatch
# ---------------------------------------------------------------------------


class TestGoalResumeDispatch:
    def test_no_interrupted_goal_prints_message(self, capsys):
        from aura_cli.dispatch import _handle_goal_resume_dispatch

        ctx = _make_ctx()
        ctx.runtime = {"goal_queue": MagicMock()}

        with patch("core.in_flight_tracker.InFlightTracker") as mock_cls:
            mock_cls.return_value.read.return_value = None
            code = _handle_goal_resume_dispatch(ctx)

        assert code == 0
        assert "No interrupted" in capsys.readouterr().out

    def test_found_goal_is_requeued(self, capsys):
        from aura_cli.dispatch import _handle_goal_resume_dispatch

        args = _make_args()
        args.run = False
        ctx = _make_ctx(args=args)
        ctx.runtime = {"goal_queue": MagicMock()}

        with patch("core.in_flight_tracker.InFlightTracker") as mock_cls:
            mock_inst = mock_cls.return_value
            mock_inst.read.return_value = {
                "goal": "interrupted-goal",
                "started_at": "2025",
                "cycle_limit": 1,
                "phase": "act",
            }
            code = _handle_goal_resume_dispatch(ctx)

        ctx.runtime["goal_queue"].prepend_batch.assert_called_once_with(
            ["interrupted-goal"]
        )
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_sadd_run_dispatch — spec not found
# ---------------------------------------------------------------------------


class TestSaddRunDispatch:
    def test_missing_spec_file_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_sadd_run_dispatch

        args = _make_args()
        args.spec = "/nonexistent/spec.yaml"
        args.dry_run = True
        args.json = False
        ctx = _make_ctx(args=args)

        code = _handle_sadd_run_dispatch(ctx)

        assert code == 1
        assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _handle_sadd_status_dispatch
# ---------------------------------------------------------------------------


class TestSaddStatusDispatch:
    def test_no_sessions_prints_message(self, capsys):
        from aura_cli.dispatch import _handle_sadd_status_dispatch

        args = _make_args()
        args.session_id = None
        args.json = False
        ctx = _make_ctx(args=args)

        mock_store = MagicMock()
        mock_store.list_sessions.return_value = []
        with patch("core.sadd.session_store.SessionStore", return_value=mock_store):
            code = _handle_sadd_status_dispatch(ctx)

        assert code == 0
        assert "No SADD" in capsys.readouterr().out

    def test_json_mode_lists_sessions(self, capsys):
        from aura_cli.dispatch import _handle_sadd_status_dispatch

        args = _make_args(json=True)
        args.session_id = None
        ctx = _make_ctx(args=args)

        mock_store = MagicMock()
        mock_store.list_sessions.return_value = [{"id": "abc", "title": "t"}]
        with patch("core.sadd.session_store.SessionStore", return_value=mock_store):
            code = _handle_sadd_status_dispatch(ctx)

        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)

    def test_unknown_session_id_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_sadd_status_dispatch

        args = _make_args()
        args.session_id = "bad-id"
        args.json = False
        ctx = _make_ctx(args=args)

        mock_store = MagicMock()
        mock_store.get_session.return_value = None
        with patch("core.sadd.session_store.SessionStore", return_value=mock_store):
            code = _handle_sadd_status_dispatch(ctx)

        assert code == 1


# ---------------------------------------------------------------------------
# _handle_sadd_resume_dispatch — missing session_id
# ---------------------------------------------------------------------------


class TestSaddResumeDispatch:
    def test_no_session_id_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_sadd_resume_dispatch

        args = _make_args()
        args.session_id = None
        ctx = _make_ctx(args=args)

        code = _handle_sadd_resume_dispatch(ctx)

        assert code == 1
        assert "--session-id" in capsys.readouterr().err

    def test_not_found_session_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_sadd_resume_dispatch

        args = _make_args()
        args.session_id = "missing-id"
        args.run = False
        ctx = _make_ctx(args=args)

        mock_store = MagicMock()
        mock_store.load_session_for_resume.return_value = None
        with patch("core.sadd.session_store.SessionStore", return_value=mock_store):
            code = _handle_sadd_resume_dispatch(ctx)

        assert code == 1


# ---------------------------------------------------------------------------
# _handle_credentials_status_dispatch
# ---------------------------------------------------------------------------


class TestCredentialsStatusDispatch:
    def _store_info(self):
        return {
            "app_name": "aura",
            "keyring_available": True,
            "fallback_available": True,
            "fallback_path": "/home/.aura_creds",
            "fallback_exists": False,
            "stored_keys_count": 0,
        }

    def test_text_mode_prints_status(self, capsys):
        from aura_cli.dispatch import _handle_credentials_status_dispatch

        ctx = _make_ctx()
        mock_cfg = MagicMock()
        mock_cfg.get_credential_store_info.return_value = self._store_info()
        mock_cfg.secure_retrieve_credential.return_value = None

        with patch("aura_cli.dispatch.config", mock_cfg):
            code = _handle_credentials_status_dispatch(ctx)

        assert code == 0
        out = capsys.readouterr().out
        assert "AURA Credential" in out

    def test_json_mode_returns_json(self, capsys):
        from aura_cli.dispatch import _handle_credentials_status_dispatch

        ctx = _make_ctx(args=_make_args(json=True), parsed=SimpleNamespace(warning_records=[], warnings=[]))
        mock_cfg = MagicMock()
        mock_cfg.get_credential_store_info.return_value = self._store_info()

        with patch("aura_cli.dispatch.config", mock_cfg), patch(
            "aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p
        ):
            code = _handle_credentials_status_dispatch(ctx)

        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["app_name"] == "aura"


# ---------------------------------------------------------------------------
# _handle_mcp_restart_dispatch
# ---------------------------------------------------------------------------


class TestMcpRestartDispatch:
    def test_no_server_name_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_mcp_restart_dispatch

        args = _make_args()
        args.mcp_server = None
        ctx = _make_ctx(args=args)

        code = _handle_mcp_restart_dispatch(ctx)

        assert code == 1
        assert "required" in capsys.readouterr().err

    def test_unknown_server_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_mcp_restart_dispatch

        args = _make_args()
        args.mcp_server = "ghost_server"
        ctx = _make_ctx(args=args)

        with patch(
            "core.mcp_registry.get_registered_service", side_effect=KeyError("ghost")
        ):
            code = _handle_mcp_restart_dispatch(ctx)

        assert code == 1

    def test_healthy_server_returns_zero(self, capsys):
        from aura_cli.dispatch import _handle_mcp_restart_dispatch

        args = _make_args()
        args.mcp_server = "dev_tools"
        ctx = _make_ctx(args=args)

        with patch(
            "core.mcp_registry.get_registered_service",
            return_value={"url": "http://localhost:9000"},
        ), patch(
            "core.mcp_health.check_mcp_health", new=lambda *a, **kw: None
        ), patch(
            "aura_cli.dispatch._run_async_safely",
            return_value={"status": "healthy"},
        ):
            code = _handle_mcp_restart_dispatch(ctx)

        assert code == 0

    def test_unhealthy_server_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_mcp_restart_dispatch

        args = _make_args()
        args.mcp_server = "dev_tools"
        ctx = _make_ctx(args=args)

        with patch(
            "core.mcp_registry.get_registered_service",
            return_value={"url": "http://localhost:9000"},
        ), patch(
            "core.mcp_health.check_mcp_health", new=lambda *a, **kw: None
        ), patch(
            "aura_cli.dispatch._run_async_safely",
            return_value={"status": "offline", "error": "connection refused"},
        ):
            code = _handle_mcp_restart_dispatch(ctx)

        assert code == 1


# ---------------------------------------------------------------------------
# _handle_cancel_dispatch
# ---------------------------------------------------------------------------


class TestCancelDispatch:
    def test_no_run_id_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_cancel_dispatch

        ctx = _make_ctx()
        ctx.args.run_id = None
        code = _handle_cancel_dispatch(ctx)
        assert code == 1

    def test_run_id_not_found_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_cancel_dispatch

        ctx = _make_ctx()
        ctx.args.run_id = "unknown-id"

        with patch("core.running_runs.list_runs", return_value=[]):
            code = _handle_cancel_dispatch(ctx)

        assert code == 1

    def test_successful_cancellation_returns_zero(self, capsys):
        from aura_cli.dispatch import _handle_cancel_dispatch

        ctx = _make_ctx()
        ctx.args.run_id = "run-abc"

        with patch(
            "core.running_runs.list_runs",
            return_value=[{"run_id": "run-abc"}],
        ), patch("core.running_runs.cancel_run", return_value=True), patch(
            "aura_cli.dispatch.log_json"
        ):
            code = _handle_cancel_dispatch(ctx)

        assert code == 0

    def test_cancel_returns_false_means_race_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_cancel_dispatch

        ctx = _make_ctx()
        ctx.args.run_id = "run-xyz"

        with patch(
            "core.running_runs.list_runs",
            return_value=[{"run_id": "run-xyz"}],
        ), patch("core.running_runs.cancel_run", return_value=False):
            code = _handle_cancel_dispatch(ctx)

        assert code == 1


# ---------------------------------------------------------------------------
# dispatch_command — top-level routing
# ---------------------------------------------------------------------------


class TestDispatchCommand:
    def _make_parsed(self, action, **ns_kwargs):
        ns = SimpleNamespace(
            action=action,
            warning_records=[],
            warnings=[],
            **ns_kwargs,
        )
        ns.namespace = _make_args()
        return ns

    def test_unknown_action_returns_one(self, capsys):
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("totally_unknown_action_xyz")

        with patch("aura_cli.dispatch._sync_cli_compat"):
            code = dispatch_command(
                parsed, project_root=Path("/proj"), runtime_factory=MagicMock()
            )

        assert code == 1
        assert "No dispatch rule" in capsys.readouterr().err

    def test_known_no_runtime_action_calls_handler(self, capsys):
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("doctor")

        # COMMAND_DISPATCH_REGISTRY holds the original function reference;
        # just verify the routing works by letting doctor run with mocked internals.
        with patch("aura_cli.dispatch._sync_cli_compat"), patch(
            "aura_cli.dispatch._handle_doctor"
        ) as mock_doc:
            code = dispatch_command(
                parsed, project_root=Path("/proj"), runtime_factory=MagicMock()
            )

        mock_doc.assert_called_once_with(Path("/proj"))
        assert code == 0

    def test_warnings_are_printed(self, capsys):
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("doctor")
        parsed.warnings = ["deprecated flag"]
        parsed.warning_records = []

        with patch("aura_cli.dispatch._sync_cli_compat"), patch(
            "aura_cli.dispatch._handle_doctor_dispatch", return_value=0
        ):
            dispatch_command(
                parsed, project_root=Path("/proj"), runtime_factory=MagicMock()
            )

        assert "deprecated flag" in capsys.readouterr().err

    def test_warning_records_are_printed(self, capsys):
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("doctor")
        parsed.warning_records = [SimpleNamespace(message="structured warning")]

        with patch("aura_cli.dispatch._sync_cli_compat"), patch(
            "aura_cli.dispatch._handle_doctor_dispatch", return_value=0
        ):
            dispatch_command(
                parsed, project_root=Path("/proj"), runtime_factory=MagicMock()
            )

        assert "structured warning" in capsys.readouterr().err

    def test_requires_runtime_action_preps_runtime(self, capsys):
        """Verify that actions requiring runtime call _prepare_runtime_context."""
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("goal_once")
        parsed.namespace = _make_args()
        parsed.namespace.goal = "test goal"
        parsed.namespace.max_cycles = 1
        parsed.namespace.dry_run = False
        parsed.namespace.explain = False
        parsed.namespace.json = False

        mock_rt = {
            "orchestrator": MagicMock(
                run_loop=MagicMock(
                    return_value={"stop_reason": "done", "history": []}
                )
            )
        }

        def _fake_factory(root, *, overrides=None):
            return mock_rt

        with patch("aura_cli.dispatch._sync_cli_compat"), patch(
            "aura_cli.dispatch.log_json"
        ), patch(
            "aura_cli.dispatch._check_project_writability", return_value=True
        ), patch(
            "aura_cli.dispatch.config"
        ) as mock_cfg, patch(
            "aura_cli.dispatch.DEFAULT_CONFIG", {"beads": {}}
        ):
            # config.get must return None/dict so _resolve_beads_runtime_override can dict() it
            mock_cfg.get.return_value = None
            code = dispatch_command(
                parsed,
                project_root=Path("/proj"),
                runtime_factory=_fake_factory,
            )

        assert code == 0


# ---------------------------------------------------------------------------
# _handle_agent_list_dispatch
# ---------------------------------------------------------------------------


class TestAgentListDispatch:
    def test_returns_zero_and_lists_agents(self, capsys):
        from aura_cli.dispatch import _handle_agent_list_dispatch

        ctx = _make_ctx()

        with patch(
            "agents.registry._AGENT_MODULE_MAP",
            {"coder": ("agents.coder", "CoderAgent")},
        ), patch(
            "agents.registry.FALLBACK_CAPABILITIES", {"coder": ["coding"]}
        ), patch(
            "aura_cli.dispatch.log_json"
        ):
            code = _handle_agent_list_dispatch(ctx)

        assert code == 0


# ---------------------------------------------------------------------------
# _handle_innovate_start_dispatch (and other thin innovate wrappers)
# ---------------------------------------------------------------------------


class TestInnovateDispatches:
    def test_innovate_start_calls_command(self):
        from aura_cli.dispatch import _handle_innovate_start_dispatch

        ctx = _make_ctx()
        ctx.runtime = {}

        with patch("aura_cli.commands._handle_innovate_start") as mock_h:
            code = _handle_innovate_start_dispatch(ctx)

        mock_h.assert_called_once()
        assert code == 0

    def test_innovate_techniques_calls_command(self):
        from aura_cli.dispatch import _handle_innovate_techniques_dispatch

        ctx = _make_ctx()
        with patch("aura_cli.commands._handle_innovate_techniques") as mock_h:
            code = _handle_innovate_techniques_dispatch(ctx)
        mock_h.assert_called_once()
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_credentials_migrate / store / delete dispatch
# ---------------------------------------------------------------------------


class TestCredentialsMiscDispatches:
    def test_migrate_calls_handle_migrate(self):
        from aura_cli.dispatch import _handle_credentials_migrate_dispatch

        ctx = _make_ctx()
        mock_cfg = MagicMock()
        with patch("aura_cli.dispatch.config", mock_cfg), patch(
            "aura_cli.dispatch._handle_migrate_credentials"
        ) as mock_m:
            code = _handle_credentials_migrate_dispatch(ctx)
        mock_m.assert_called_once()
        assert code == 0

    def test_store_calls_handle_secure_store(self):
        from aura_cli.dispatch import _handle_credentials_store_dispatch

        ctx = _make_ctx()
        mock_cfg = MagicMock()
        with patch("aura_cli.dispatch.config", mock_cfg), patch(
            "aura_cli.dispatch._handle_secure_store"
        ) as mock_s:
            code = _handle_credentials_store_dispatch(ctx)
        mock_s.assert_called_once()
        assert code == 0

    def test_delete_calls_handle_secure_delete(self):
        from aura_cli.dispatch import _handle_credentials_delete_dispatch

        ctx = _make_ctx()
        mock_cfg = MagicMock()
        with patch("aura_cli.dispatch.config", mock_cfg), patch(
            "aura_cli.dispatch._handle_secure_delete"
        ) as mock_d:
            code = _handle_credentials_delete_dispatch(ctx)
        mock_d.assert_called_once()
        assert code == 0


# ---------------------------------------------------------------------------
# _handle_beads_schemas_dispatch
# ---------------------------------------------------------------------------


class TestBeadsSchemasDispatch:
    def test_json_mode_returns_schema_payload(self, capsys):
        from aura_cli.dispatch import _handle_beads_schemas_dispatch

        ctx = _make_ctx(
            args=_make_args(json=True),
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )
        ctx.project_root = Path("/fake/proj")

        with patch(
            "core.beads_contract.BEADS_SCHEMA_VERSION", "1.0"
        ), patch(
            "core.beads_contract.BeadsInput.__annotations__", {"goal": str}
        ), patch(
            "core.beads_contract.BeadsDecision.__annotations__", {"action": str}
        ), patch(
            "core.beads_contract.BeadsResult.__annotations__", {"success": bool}
        ), patch(
            "aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p
        ):
            code = _handle_beads_schemas_dispatch(ctx)

        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert "schema_version" in data
        assert "schemas" in data
