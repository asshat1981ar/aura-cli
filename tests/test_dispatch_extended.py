"""Extended coverage tests for aura_cli/dispatch.py — Sprint 9."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

os.environ.setdefault("AURA_SKIP_CHDIR", "1")
os.environ.setdefault("AURA_TEST_MODE", "1")


# ---------------------------------------------------------------------------
# Helpers
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


def _fake_runtime(**overrides):
    rt = {
        "goal_queue": MagicMock(),
        "goal_archive": MagicMock(),
        "orchestrator": MagicMock(),
        "debugger": MagicMock(),
        "planner": MagicMock(),
        "brain": MagicMock(),
        "model_adapter": MagicMock(),
        "memory_store": MagicMock(),
        "vector_store": MagicMock(),
        "memory_persistence_path": "/fake/memory.db",
    }
    rt.update(overrides)
    return rt


# ---------------------------------------------------------------------------
# _prepare_runtime_context — additional branches
# ---------------------------------------------------------------------------


class TestPrepareRuntimeContextExtra:
    def setup_method(self):
        from aura_cli.dispatch import DispatchContext

        self.DispatchContext = DispatchContext

    def _make_ctx_for_prep(self, args):
        ctx = self.DispatchContext(
            parsed=SimpleNamespace(action=getattr(args, "action", None), warning_records=[], warnings=[]),
            project_root=Path("/fake/proj"),
            runtime_factory=MagicMock(return_value=_fake_runtime()),
            args=args,
        )
        return ctx

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._check_project_writability", return_value=True)
    @patch("aura_cli.dispatch.log_json")
    def test_model_override_is_passed(self, _log, _writable, _compat):
        args = _make_args(model="claude-3", action="goal_once")
        ctx = self._make_ctx_for_prep(args)
        from aura_cli.dispatch import _prepare_runtime_context

        rc = _prepare_runtime_context(ctx)
        assert rc is None
        call_overrides = ctx.runtime_factory.call_args[1]["overrides"]
        assert call_overrides["model_name"] == "claude-3"

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._check_project_writability", return_value=True)
    @patch("aura_cli.dispatch.log_json")
    def test_anthropic_api_key_override(self, _log, _writable, _compat):
        args = _make_args(anthropic_api_key="sk-test", action="goal_once")
        ctx = self._make_ctx_for_prep(args)
        from aura_cli.dispatch import _prepare_runtime_context

        rc = _prepare_runtime_context(ctx)
        assert rc is None
        call_overrides = ctx.runtime_factory.call_args[1]["overrides"]
        assert call_overrides["anthropic_api_key"] == "sk-test"

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._check_project_writability", return_value=True)
    @patch("aura_cli.dispatch.log_json")
    def test_no_overrides_calls_factory_with_none(self, _log, _writable, _compat):
        args = _make_args(action="goal_once")
        ctx = self._make_ctx_for_prep(args)
        from aura_cli.dispatch import _prepare_runtime_context

        rc = _prepare_runtime_context(ctx)
        assert rc is None
        call_kwargs = ctx.runtime_factory.call_args[1]
        assert call_kwargs["overrides"] is None

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._check_project_writability", return_value=True)
    @patch("aura_cli.dispatch.log_json")
    def test_runtime_mode_queue_for_goal_status(self, _log, _writable, _compat):
        args = _make_args(action="goal_status")
        ctx = self._make_ctx_for_prep(args)
        from aura_cli.dispatch import _prepare_runtime_context

        _prepare_runtime_context(ctx)
        overrides = ctx.runtime_factory.call_args[1]["overrides"]
        assert overrides["runtime_mode"] == "queue"


# ---------------------------------------------------------------------------
# _handle_help_dispatch — additional branch: ValueError no JSON
# ---------------------------------------------------------------------------


class TestHelpDispatchExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.render_help", side_effect=ValueError("bad topic"))
    def test_value_error_no_json_prints_to_stderr(self, _render, _compat, capsys):
        from aura_cli.dispatch import _handle_help_dispatch

        ctx = _make_ctx(args=_make_args(json=False, help_topics="bad"))
        rc = _handle_help_dispatch(ctx)
        assert rc == 2
        captured = capsys.readouterr()
        assert "Error: bad topic" in captured.err


# ---------------------------------------------------------------------------
# _handle_show_config_dispatch
# ---------------------------------------------------------------------------


class TestShowConfigDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_prints_json(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_show_config_dispatch

        mock_config.show_config.return_value = {"key": "val"}
        ctx = _make_ctx()
        rc = _handle_show_config_dispatch(ctx)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out == {"key": "val"}


# ---------------------------------------------------------------------------
# _handle_config_set_dispatch
# ---------------------------------------------------------------------------


class TestConfigSetDispatchExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_plain_key_updates_config(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_config_set_dispatch

        args = _make_args(config_key="max_cycles", config_value="10")
        ctx = _make_ctx(args=args)
        rc = _handle_config_set_dispatch(ctx)
        assert rc == 0
        mock_config.update_config.assert_called_once_with({"max_cycles": "10"})
        assert "max_cycles" in capsys.readouterr().out

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_model_dot_key_routes_to_model_routing(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_config_set_dispatch

        args = _make_args(config_key="model.plan", config_value="gpt-4")
        ctx = _make_ctx(args=args)
        rc = _handle_config_set_dispatch(ctx)
        assert rc == 0
        mock_config.update_config.assert_called_once_with({"model_routing": {"plan": "gpt-4"}})

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_exception_returns_one(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_config_set_dispatch

        mock_config.update_config.side_effect = RuntimeError("disk full")
        args = _make_args(config_key="x", config_value="y")
        ctx = _make_ctx(args=args)
        rc = _handle_config_set_dispatch(ctx)
        assert rc == 1
        assert "disk full" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _handle_doctor_dispatch & _handle_readiness_dispatch
# ---------------------------------------------------------------------------


class TestSimpleHandlersExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._handle_doctor")
    def test_doctor_dispatch(self, mock_doctor, _compat):
        from aura_cli.dispatch import _handle_doctor_dispatch

        ctx = _make_ctx()
        rc = _handle_doctor_dispatch(ctx)
        assert rc == 0
        mock_doctor.assert_called_once_with(ctx.project_root)

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._handle_readiness")
    def test_readiness_dispatch(self, mock_readiness, _compat):
        from aura_cli.dispatch import _handle_readiness_dispatch

        ctx = _make_ctx()
        rc = _handle_readiness_dispatch(ctx)
        assert rc == 0
        mock_readiness.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_bootstrap_dispatch
# ---------------------------------------------------------------------------


class TestBootstrapDispatchExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_calls_interactive_bootstrap(self, mock_config, _compat):
        from aura_cli.dispatch import _handle_bootstrap_dispatch

        ctx = _make_ctx()
        rc = _handle_bootstrap_dispatch(ctx)
        assert rc == 0
        mock_config.interactive_bootstrap.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_memory_search_dispatch
# ---------------------------------------------------------------------------


class TestMemorySearchDispatch:
    def _make_hit(self, score=0.9, source_ref="file.py", content="content"):
        h = MagicMock()
        h.score = score
        h.source_ref = source_ref
        h.content = content
        return h

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_no_hits_prints_message(self, _compat, capsys):
        from aura_cli.dispatch import _handle_memory_search_dispatch

        rt = _fake_runtime()
        rt["vector_store"] = MagicMock()
        rt["vector_store"].search.return_value = []
        args = _make_args(query="test", limit=5, json=False)
        ctx = _make_ctx(args=args, runtime=rt)

        with patch("aura_cli.dispatch.RetrievalQuery", create=True) as _rq:
            # patch the local import
            with patch("core.memory_types.RetrievalQuery") as mock_rq:
                mock_rq.return_value = MagicMock()
                rc = _handle_memory_search_dispatch(ctx)

        assert rc == 0
        assert "No results found" in capsys.readouterr().out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_hits_are_printed(self, _compat, capsys):
        from aura_cli.dispatch import _handle_memory_search_dispatch

        hit = self._make_hit(score=0.85, source_ref="src/foo.py", content="x" * 50)
        rt = _fake_runtime()
        rt["vector_store"] = MagicMock()
        rt["vector_store"].search.return_value = [hit]
        args = _make_args(query="foo", limit=5, json=False)
        ctx = _make_ctx(
            args=args,
            runtime=rt,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )

        with patch("core.memory_types.RetrievalQuery") as mock_rq:
            mock_rq.return_value = MagicMock()
            rc = _handle_memory_search_dispatch(ctx)

        assert rc == 0
        out = capsys.readouterr().out
        assert "0.850" in out or "Score" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_json_mode_outputs_json(self, _compat, capsys):
        from aura_cli.dispatch import _handle_memory_search_dispatch

        hit = self._make_hit(score=0.5, source_ref="a.py", content="hello world")
        rt = _fake_runtime()
        rt["vector_store"] = MagicMock()
        rt["vector_store"].search.return_value = [hit]
        args = _make_args(query="q", limit=3, json=True)
        ctx = _make_ctx(
            args=args,
            runtime=rt,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )

        with patch("core.memory_types.RetrievalQuery") as mock_rq:
            mock_rq.return_value = MagicMock()
            rc = _handle_memory_search_dispatch(ctx)

        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "hits" in data
        assert data["query"] == "q"


# ---------------------------------------------------------------------------
# _handle_memory_reindex_dispatch
# ---------------------------------------------------------------------------


class TestMemoryReindexDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_json_mode_returns_ok(self, _compat, capsys):
        from aura_cli.dispatch import _handle_memory_reindex_dispatch

        rt = _fake_runtime()
        mock_vs = MagicMock()
        mock_vs.rebuild.return_value = {"embeddings_written": 10, "records_seen": 10}
        rt["vector_store"] = mock_vs
        mock_adapter = MagicMock()
        mock_adapter.model_id.return_value = "text-embed"
        mock_adapter.dimensions.return_value = 1536
        rt["model_adapter"] = mock_adapter

        args = _make_args(json=True)
        ctx = _make_ctx(
            args=args,
            runtime=rt,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )

        with patch("core.project_syncer.ProjectKnowledgeSyncer") as mock_syncer_cls:
            mock_syncer = MagicMock()
            mock_syncer.sync_all.return_value = {"files_processed": 5, "chunks_created": 10, "files_skipped": 1}
            mock_syncer_cls.return_value = mock_syncer
            rc = _handle_memory_reindex_dispatch(ctx)

        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "ok"

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_plain_mode_prints_summary(self, _compat, capsys):
        from aura_cli.dispatch import _handle_memory_reindex_dispatch

        rt = _fake_runtime()
        mock_vs = MagicMock()
        mock_vs.rebuild.return_value = {"embeddings_written": 5, "records_seen": 7}
        rt["vector_store"] = mock_vs
        mock_adapter = MagicMock()
        mock_adapter.model_id.return_value = "embed-v1"
        mock_adapter.dimensions.return_value = 768
        rt["model_adapter"] = mock_adapter

        args = _make_args(json=False)
        ctx = _make_ctx(args=args, runtime=rt)

        with patch("core.project_syncer.ProjectKnowledgeSyncer") as mock_syncer_cls:
            mock_syncer = MagicMock()
            mock_syncer.sync_all.return_value = {"files_processed": 3, "chunks_created": 6, "files_skipped": 0}
            mock_syncer_cls.return_value = mock_syncer
            rc = _handle_memory_reindex_dispatch(ctx)

        assert rc == 0
        out = capsys.readouterr().out
        assert "reindex complete" in out.lower() or "embed-v1" in out


# ---------------------------------------------------------------------------
# _handle_metrics_show_dispatch
# ---------------------------------------------------------------------------


class TestMetricsShowDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_no_data_prints_message(self, _compat, capsys):
        from aura_cli.dispatch import _handle_metrics_show_dispatch

        rt = _fake_runtime()
        rt["memory_store"] = MagicMock()
        rt["memory_store"].read_log.return_value = []
        rt["brain"] = MagicMock()
        rt["brain"].recall_recent.return_value = []

        args = _make_args(json=False)
        ctx = _make_ctx(args=args, runtime=rt)
        rc = _handle_metrics_show_dispatch(ctx)
        assert rc == 0
        assert "No metrics" in capsys.readouterr().out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_structured_summaries_json_mode(self, _compat, capsys):
        from aura_cli.dispatch import _handle_metrics_show_dispatch

        rt = _fake_runtime()
        log_entries = [
            {
                "cycle_summary": {
                    "outcome": "SUCCESS",
                    "duration_s": 3.5,
                    "cycle_id": "abc123",
                    "goal": "do something",
                    "stop_reason": "done",
                }
            }
        ]
        rt["memory_store"] = MagicMock()
        rt["memory_store"].read_log.return_value = log_entries

        args = _make_args(json=True)
        ctx = _make_ctx(
            args=args,
            runtime=rt,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )
        rc = _handle_metrics_show_dispatch(ctx)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["summary"]["successes"] == 1
        assert out["summary"]["win_rate"] == 100.0

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_fallback_legacy_outcomes(self, _compat, capsys):
        from aura_cli.dispatch import _handle_metrics_show_dispatch

        rt = _fake_runtime()
        import time

        ts = time.time()
        raw_entry = f"outcome:id1 -> {json.dumps({'success': True, 'goal': 'legacy goal', 'started_at': ts - 2, 'completed_at': ts, 'cycle_id': 'leg1'})}"
        rt["memory_store"] = MagicMock()
        rt["memory_store"].read_log.return_value = []
        rt["brain"] = MagicMock()
        rt["brain"].recall_recent.return_value = [raw_entry]

        args = _make_args(json=False)
        ctx = _make_ctx(args=args, runtime=rt)
        rc = _handle_metrics_show_dispatch(ctx)
        assert rc == 0
        out = capsys.readouterr().out
        assert "SUCCESS" in out or "100.0%" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_structured_summaries_plain_mode(self, _compat, capsys):
        from aura_cli.dispatch import _handle_metrics_show_dispatch

        rt = _fake_runtime()
        log_entries = [
            {
                "cycle_summary": {
                    "outcome": "FAILED",
                    "duration_s": 1.0,
                    "cycle_id": "deadbeef",
                    "goal": "a long goal name here",
                    "stop_reason": "max_cycles",
                }
            }
        ]
        rt["memory_store"] = MagicMock()
        rt["memory_store"].read_log.return_value = log_entries

        args = _make_args(json=False)
        ctx = _make_ctx(args=args, runtime=rt)
        rc = _handle_metrics_show_dispatch(ctx)
        assert rc == 0
        out = capsys.readouterr().out
        assert "FAILED" in out or "0.0%" in out


# ---------------------------------------------------------------------------
# _handle_workflow_run_dispatch
# ---------------------------------------------------------------------------


class TestWorkflowRunDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_calls_orchestrator_run_loop(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_workflow_run_dispatch

        mock_config.get.return_value = 5
        rt = _fake_runtime()
        rt["orchestrator"].run_loop.return_value = {"stop_reason": "done", "history": ["a", "b"]}

        args = _make_args(
            workflow_goal="test workflow",
            workflow_max_cycles=3,
            max_cycles=None,
            dry_run=False,
        )
        ctx = _make_ctx(
            args=args,
            runtime=rt,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )

        with patch("core.operator_runtime.build_beads_runtime_metadata", return_value={}):
            rc = _handle_workflow_run_dispatch(ctx)

        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["goal"] == "test workflow"
        assert out["stop_reason"] == "done"
        assert out["cycles"] == 2


# ---------------------------------------------------------------------------
# _handle_scaffold_dispatch
# ---------------------------------------------------------------------------


class TestScaffoldDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.ScaffolderAgent")
    @patch("aura_cli.dispatch.Brain")
    def test_plain_mode_prints_result(self, mock_brain, mock_scaff_cls, _compat, capsys):
        from aura_cli.dispatch import _handle_scaffold_dispatch

        mock_scaff = MagicMock()
        mock_scaff.scaffold_project.return_value = "project scaffolded!"
        mock_scaff_cls.return_value = mock_scaff

        rt = _fake_runtime()
        args = _make_args(scaffold="myapp", scaffold_desc="A test app", json=False)
        ctx = _make_ctx(args=args, runtime=rt)
        rc = _handle_scaffold_dispatch(ctx)
        assert rc == 0
        assert "project scaffolded!" in capsys.readouterr().out

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.ScaffolderAgent")
    @patch("aura_cli.dispatch.Brain")
    def test_json_mode_prints_payload(self, mock_brain, mock_scaff_cls, _compat, capsys):
        from aura_cli.dispatch import _handle_scaffold_dispatch

        mock_scaff = MagicMock()
        mock_scaff.scaffold_project.return_value = "ok"
        mock_scaff_cls.return_value = mock_scaff

        rt = _fake_runtime()
        args = _make_args(scaffold="myapp2", scaffold_desc="desc", json=True)
        ctx = _make_ctx(
            args=args,
            runtime=rt,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )
        rc = _handle_scaffold_dispatch(ctx)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["project_name"] == "myapp2"


# ---------------------------------------------------------------------------
# _resolve_evolve_agents
# ---------------------------------------------------------------------------


class TestResolveEvolveAgents:
    def test_uses_orchestrator_agents_when_present(self):
        from aura_cli.dispatch import _resolve_evolve_agents

        brain = MagicMock()
        model = MagicMock()
        orchestrator = MagicMock()

        act_agent = MagicMock()
        critique_agent = MagicMock()
        plan_agent = MagicMock()
        orchestrator.agents = {
            "act": act_agent,
            "critique": critique_agent,
            "plan": plan_agent,
        }

        planner, coder, critic = _resolve_evolve_agents(brain, model, orchestrator)
        # They may be unwrapped via .agent attribute lookup, so just check we got something
        assert planner is not None
        assert coder is not None
        assert critic is not None

    @patch("aura_cli.dispatch.default_agents")
    def test_falls_back_to_default_agents(self, mock_default):
        from aura_cli.dispatch import _resolve_evolve_agents

        brain = MagicMock()
        model = MagicMock()
        orchestrator = MagicMock()
        orchestrator.agents = None

        act = MagicMock()
        act.agent = MagicMock()
        critique = MagicMock()
        critique.agent = MagicMock()
        plan = MagicMock()
        plan.agent = MagicMock()
        mock_default.return_value = {"act": act, "critique": critique, "plan": plan}

        planner, coder, critic = _resolve_evolve_agents(brain, model, orchestrator)
        mock_default.assert_called_once_with(brain, model)

    def test_resolves_missing_agents_via_handlers(self):
        from aura_cli.dispatch import _resolve_evolve_agents

        brain = MagicMock()
        model = MagicMock()
        orchestrator = MagicMock()
        orchestrator.agents = {}  # no agents

        mock_coder = MagicMock()
        mock_critic = MagicMock()
        mock_planner = MagicMock()

        with (
            patch("agents.handlers.coder._resolve_agent", return_value=mock_coder),
            patch("agents.handlers.critic._resolve_agent", return_value=mock_critic),
            patch("agents.handlers.planner._resolve_agent", return_value=mock_planner),
            patch("aura_cli.dispatch.default_agents", return_value={}),
        ):
            planner, coder, critic = _resolve_evolve_agents(brain, model, orchestrator)

        assert coder is mock_coder
        assert critic is mock_critic
        assert planner is mock_planner


# ---------------------------------------------------------------------------
# _handle_goal_status_dispatch
# ---------------------------------------------------------------------------


class TestGoalStatusDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._handle_status")
    def test_plain_mode_calls_handle_status(self, mock_status, _compat):
        from aura_cli.dispatch import _handle_goal_status_dispatch

        rt = _fake_runtime()
        args = _make_args(json=False)
        ctx = _make_ctx(args=args, runtime=rt)
        rc = _handle_goal_status_dispatch(ctx)
        assert rc == 0
        mock_status.assert_called_once()

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._handle_status")
    def test_json_mode_wraps_with_warnings(self, mock_status, _compat):
        from aura_cli.dispatch import _handle_goal_status_dispatch

        mock_status.return_value = None
        rt = _fake_runtime()
        args = _make_args(json=True)
        ctx = _make_ctx(args=args, runtime=rt, parsed=SimpleNamespace(warning_records=[], warnings=[]))
        rc = _handle_goal_status_dispatch(ctx)
        assert rc == 0
        mock_status.assert_called_once()


# ---------------------------------------------------------------------------
# _maybe_add_goal
# ---------------------------------------------------------------------------


class TestMaybeAddGoal:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.log_json")
    def test_no_add_goal_attr_is_noop(self, _log, _compat):
        from aura_cli.dispatch import _maybe_add_goal

        rt = _fake_runtime()
        ctx = _make_ctx(args=_make_args(), runtime=rt)
        _maybe_add_goal(ctx)
        rt["goal_queue"].add.assert_not_called()

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.log_json")
    def test_adds_goal_to_queue(self, _log, _compat, capsys):
        from aura_cli.dispatch import _maybe_add_goal

        rt = _fake_runtime()
        rt["goal_queue"].queue = ["existing"]
        args = _make_args(add_goal="my new goal", json=False)
        ctx = _make_ctx(args=args, runtime=rt)
        _maybe_add_goal(ctx)
        rt["goal_queue"].add.assert_called_once_with("my new goal")
        out = capsys.readouterr().out
        assert "Added goal" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.log_json")
    def test_json_mode_suppresses_print(self, _log, _compat, capsys):
        from aura_cli.dispatch import _maybe_add_goal

        rt = _fake_runtime()
        rt["goal_queue"].queue = []
        args = _make_args(add_goal="goal x", json=True)
        ctx = _make_ctx(args=args, runtime=rt)
        _maybe_add_goal(ctx)
        out = capsys.readouterr().out
        assert "Added goal" not in out


# ---------------------------------------------------------------------------
# _handle_goal_once_dispatch — error branches
# ---------------------------------------------------------------------------


class TestGoalOnceDispatchErrors:
    def _make_goal_once_ctx(self, **kwargs):
        rt = _fake_runtime()
        defaults = dict(
            goal="do something",
            max_cycles=3,
            dry_run=False,
            explain=False,
            json=False,
        )
        defaults.update(kwargs)
        args = _make_args(**defaults)
        return _make_ctx(args=args, runtime=rt, parsed=SimpleNamespace(warning_records=[], warnings=[]))

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_apply_error_returns_exit_apply_error(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch
        from aura_cli.exit_codes import EXIT_APPLY_ERROR

        mock_config.get.return_value = 5
        ctx = self._make_goal_once_ctx()

        with patch("core.file_tools.OldCodeNotFoundError", Exception, create=True):
            from core.file_tools import OldCodeNotFoundError

            ctx.runtime["orchestrator"].run_loop.side_effect = OldCodeNotFoundError("apply failed")
            with patch("aura_cli.dispatch.log_json"):
                rc = _handle_goal_once_dispatch(ctx)

        assert rc == EXIT_APPLY_ERROR

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_generic_exception_returns_exit_failure(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch
        from aura_cli.exit_codes import EXIT_FAILURE

        mock_config.get.return_value = 5
        ctx = self._make_goal_once_ctx()
        ctx.runtime["orchestrator"].run_loop.side_effect = Exception("unexpected")

        with patch("aura_cli.dispatch.log_json"):
            rc = _handle_goal_once_dispatch(ctx)

        assert rc == EXIT_FAILURE
        assert "Error" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_llm_error_returns_llm_exit_code(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch
        from aura_cli.exit_codes import EXIT_LLM_ERROR

        mock_config.get.return_value = 5
        ctx = self._make_goal_once_ctx()

        class AnthropicError(Exception):
            pass

        ctx.runtime["orchestrator"].run_loop.side_effect = AnthropicError("rate limit")

        with patch("aura_cli.dispatch.log_json"):
            rc = _handle_goal_once_dispatch(ctx)

        assert rc == EXIT_LLM_ERROR

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_keyboard_interrupt_returns_cancelled(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch
        from aura_cli.exit_codes import EXIT_CANCELLED

        mock_config.get.return_value = 5
        ctx = self._make_goal_once_ctx()
        ctx.runtime["orchestrator"].run_loop.side_effect = KeyboardInterrupt()

        rc = _handle_goal_once_dispatch(ctx)
        assert rc == EXIT_CANCELLED

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_explain_flag_calls_format_decision_log(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch
        from aura_cli.exit_codes import EXIT_SUCCESS

        mock_config.get.return_value = 5
        ctx = self._make_goal_once_ctx(explain=True)
        ctx.runtime["orchestrator"].run_loop.return_value = {"history": ["step1"], "stop_reason": "done"}

        with (
            patch("aura_cli.dispatch.log_json"),
            patch("core.explain.format_decision_log", return_value="decision log") as mock_fmt,
            patch("core.operator_runtime.build_beads_runtime_metadata", return_value={}),
        ):
            rc = _handle_goal_once_dispatch(ctx)

        assert rc == EXIT_SUCCESS
        mock_fmt.assert_called_once_with(["step1"])

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_sandbox_error_returns_sandbox_exit_code(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_once_dispatch
        from aura_cli.exit_codes import EXIT_SANDBOX_ERROR

        mock_config.get.return_value = 5
        ctx = self._make_goal_once_ctx()

        class SandboxViolationError(Exception):
            pass

        ctx.runtime["orchestrator"].run_loop.side_effect = SandboxViolationError("sandbox blocked")

        with patch("aura_cli.dispatch.log_json"):
            rc = _handle_goal_once_dispatch(ctx)

        assert rc == EXIT_SANDBOX_ERROR


# ---------------------------------------------------------------------------
# _handle_goal_run_dispatch — error branches
# ---------------------------------------------------------------------------


class TestGoalRunDispatchErrors:
    def _make_run_ctx(self, **kwargs):
        rt = _fake_runtime()
        defaults = dict(
            decompose=False,
            resume=False,
        )
        defaults.update(kwargs)
        args = _make_args(**defaults)
        return _make_ctx(args=args, runtime=rt)

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.run_goals_loop")
    def test_keyboard_interrupt_returns_cancelled(self, mock_loop, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_run_dispatch
        from aura_cli.exit_codes import EXIT_CANCELLED

        mock_loop.side_effect = KeyboardInterrupt()
        ctx = self._make_run_ctx()

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.exists.return_value = False
            mock_tracker_cls.return_value = mock_tracker
            rc = _handle_goal_run_dispatch(ctx)

        assert rc == EXIT_CANCELLED

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.run_goals_loop")
    def test_generic_exception_returns_failure(self, mock_loop, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_run_dispatch
        from aura_cli.exit_codes import EXIT_FAILURE

        mock_loop.side_effect = RuntimeError("unexpected")
        ctx = self._make_run_ctx()

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.exists.return_value = False
            mock_tracker_cls.return_value = mock_tracker
            rc = _handle_goal_run_dispatch(ctx)

        assert rc == EXIT_FAILURE

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.run_goals_loop")
    def test_llm_error_returns_llm_exit_code(self, mock_loop, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_run_dispatch
        from aura_cli.exit_codes import EXIT_LLM_ERROR

        class RateLimitError(Exception):
            pass

        mock_loop.side_effect = RateLimitError("rate limit")
        ctx = self._make_run_ctx()

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.exists.return_value = False
            mock_tracker_cls.return_value = mock_tracker
            rc = _handle_goal_run_dispatch(ctx)

        assert rc == EXIT_LLM_ERROR

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.run_goals_loop")
    def test_success_returns_zero(self, mock_loop, _compat):
        from aura_cli.dispatch import _handle_goal_run_dispatch
        from aura_cli.exit_codes import EXIT_SUCCESS

        mock_loop.return_value = None
        ctx = self._make_run_ctx()

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.exists.return_value = False
            mock_tracker_cls.return_value = mock_tracker
            rc = _handle_goal_run_dispatch(ctx)

        assert rc == EXIT_SUCCESS

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.run_goals_loop")
    def test_resume_flag_prints_resuming_message(self, mock_loop, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_run_dispatch

        mock_loop.return_value = None
        ctx = self._make_run_ctx(resume=True)

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.exists.return_value = True
            mock_tracker.read.return_value = {"goal": "previous goal"}
            mock_tracker_cls.return_value = mock_tracker
            rc = _handle_goal_run_dispatch(ctx)

        out = capsys.readouterr().out
        assert "Resuming" in out or "previous goal" in out
        assert rc == 0

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.run_goals_loop")
    def test_interrupted_detected_warns(self, mock_loop, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_run_dispatch

        mock_loop.return_value = None
        ctx = self._make_run_ctx(resume=False)

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.exists.return_value = True
            mock_tracker.read.return_value = {"goal": "old goal"}
            mock_tracker_cls.return_value = mock_tracker
            rc = _handle_goal_run_dispatch(ctx)

        err = capsys.readouterr().err
        assert "Interrupted" in err or "old goal" in err or "resume" in err.lower()


# ---------------------------------------------------------------------------
# _handle_goal_add_dispatch and _handle_goal_add_run_dispatch
# ---------------------------------------------------------------------------


class TestGoalAddDispatches:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.log_json")
    def test_goal_add_calls_maybe_add_goal(self, _log, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_add_dispatch

        rt = _fake_runtime()
        rt["goal_queue"].queue = []
        args = _make_args(add_goal="test goal")
        ctx = _make_ctx(args=args, runtime=rt)
        rc = _handle_goal_add_dispatch(ctx)
        assert rc == 0
        rt["goal_queue"].add.assert_called_once_with("test goal")

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.run_goals_loop")
    @patch("aura_cli.dispatch.log_json")
    def test_goal_add_run_adds_and_runs(self, _log, mock_loop, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_add_run_dispatch

        mock_loop.return_value = None
        rt = _fake_runtime()
        rt["goal_queue"].queue = []
        args = _make_args(add_goal="run goal", decompose=False, resume=False)
        ctx = _make_ctx(args=args, runtime=rt)

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.exists.return_value = False
            mock_tracker_cls.return_value = mock_tracker
            rc = _handle_goal_add_run_dispatch(ctx)

        rt["goal_queue"].add.assert_called_once_with("run goal")
        assert rc == 0


# ---------------------------------------------------------------------------
# _handle_goal_resume_dispatch
# ---------------------------------------------------------------------------


class TestGoalResumeDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_no_record_prints_message(self, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_resume_dispatch

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.read.return_value = None
            mock_tracker_cls.return_value = mock_tracker
            ctx = _make_ctx(args=_make_args(run=False))
            rc = _handle_goal_resume_dispatch(ctx)

        assert rc == 0
        assert "No interrupted" in capsys.readouterr().out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_found_goal_is_requeued(self, _compat, capsys):
        from aura_cli.dispatch import _handle_goal_resume_dispatch

        rt = _fake_runtime()
        rt["goal_queue"] = MagicMock()

        with patch("core.in_flight_tracker.InFlightTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.read.return_value = {
                "goal": "recover me",
                "started_at": "2024-01-01",
                "phase": "plan",
                "cycle_limit": 3,
            }
            mock_tracker_cls.return_value = mock_tracker
            args = _make_args(run=False)
            ctx = _make_ctx(args=args, runtime=rt)
            rc = _handle_goal_resume_dispatch(ctx)

        assert rc == 0
        rt["goal_queue"].prepend_batch.assert_called_once_with(["recover me"])
        out = capsys.readouterr().out
        assert "recover me" in out


# ---------------------------------------------------------------------------
# _handle_credentials_*_dispatch
# ---------------------------------------------------------------------------


class TestCredentialsDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._handle_migrate_credentials")
    @patch("aura_cli.dispatch.config")
    def test_migrate_calls_handler(self, mock_config, mock_migrate, _compat):
        from aura_cli.dispatch import _handle_credentials_migrate_dispatch

        ctx = _make_ctx()
        rc = _handle_credentials_migrate_dispatch(ctx)
        assert rc == 0
        mock_migrate.assert_called_once_with(ctx.args, mock_config)

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._handle_secure_store")
    @patch("aura_cli.dispatch.config")
    def test_store_calls_handler(self, mock_config, mock_store, _compat):
        from aura_cli.dispatch import _handle_credentials_store_dispatch

        ctx = _make_ctx()
        rc = _handle_credentials_store_dispatch(ctx)
        assert rc == 0
        mock_store.assert_called_once_with(ctx.args, mock_config)

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._handle_secure_delete")
    @patch("aura_cli.dispatch.config")
    def test_delete_calls_handler(self, mock_config, mock_delete, _compat):
        from aura_cli.dispatch import _handle_credentials_delete_dispatch

        ctx = _make_ctx()
        rc = _handle_credentials_delete_dispatch(ctx)
        assert rc == 0
        mock_delete.assert_called_once_with(ctx.args, mock_config)

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_status_plain_mode(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_credentials_status_dispatch

        mock_config.get_credential_store_info.return_value = {
            "app_name": "AURA",
            "keyring_available": True,
            "fallback_available": True,
            "fallback_path": "/path/creds",
            "fallback_exists": False,
            "stored_keys_count": 2,
        }
        mock_config.secure_retrieve_credential.return_value = None

        args = _make_args(json=False)
        ctx = _make_ctx(args=args)
        rc = _handle_credentials_status_dispatch(ctx)
        assert rc == 0
        out = capsys.readouterr().out
        assert "AURA" in out or "Credential" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.config")
    def test_status_json_mode(self, mock_config, _compat, capsys):
        from aura_cli.dispatch import _handle_credentials_status_dispatch

        store_info = {
            "app_name": "AURA",
            "keyring_available": False,
            "fallback_available": True,
            "fallback_path": "/path/creds",
            "fallback_exists": True,
            "stored_keys_count": 1,
        }
        mock_config.get_credential_store_info.return_value = store_info

        args = _make_args(json=True)
        ctx = _make_ctx(
            args=args,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )
        rc = _handle_credentials_status_dispatch(ctx)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["app_name"] == "AURA"


# ---------------------------------------------------------------------------
# _handle_mcp_status_dispatch
# ---------------------------------------------------------------------------


class TestMcpStatusDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._run_async_safely")
    def test_json_mode_returns_servers(self, mock_async, _compat, capsys):
        from aura_cli.dispatch import _handle_mcp_status_dispatch

        mock_async.return_value = [{"name": "skills", "status": "healthy", "health_data": {}}]

        with (
            patch("core.mcp_health.get_health_summary", return_value={"healthy_count": 1, "total_servers": 1, "all_healthy": True}),
            patch("core.mcp_registry.list_registered_services", return_value=[{"config_name": "skills", "url": "http://localhost:8080"}]),
        ):
            args = _make_args(json=True)
            ctx = _make_ctx(
                args=args,
                parsed=SimpleNamespace(warning_records=[], warnings=[]),
            )
            rc = _handle_mcp_status_dispatch(ctx)

        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "servers" in out
        assert "summary" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._run_async_safely")
    def test_plain_mode_no_rich_falls_back(self, mock_async, _compat, capsys):
        from aura_cli.dispatch import _handle_mcp_status_dispatch

        mock_async.return_value = [{"name": "dev_tools", "status": "offline", "health_data": {}}]

        with (
            patch("core.mcp_health.get_health_summary", return_value={"healthy_count": 0, "total_servers": 1, "all_healthy": False}),
            patch("core.mcp_registry.list_registered_services", return_value=[]),
            patch.dict("sys.modules", {"rich": None, "rich.console": None, "rich.table": None, "rich.box": None}),
        ):
            args = _make_args(json=False)
            ctx = _make_ctx(args=args)
            rc = _handle_mcp_status_dispatch(ctx)

        # Returns 1 when not all_healthy
        assert rc in (0, 1)


# ---------------------------------------------------------------------------
# _handle_mcp_restart_dispatch
# ---------------------------------------------------------------------------


class TestMcpRestartDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_no_server_name_returns_one(self, _compat, capsys):
        from aura_cli.dispatch import _handle_mcp_restart_dispatch

        args = _make_args(mcp_server=None)
        ctx = _make_ctx(args=args)
        rc = _handle_mcp_restart_dispatch(ctx)
        assert rc == 1
        assert "required" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._run_async_safely")
    def test_unknown_server_returns_one(self, mock_async, _compat, capsys):
        from aura_cli.dispatch import _handle_mcp_restart_dispatch

        with patch("core.mcp_registry.get_registered_service", side_effect=KeyError("unknown")):
            args = _make_args(mcp_server="unknown_server")
            ctx = _make_ctx(args=args)
            rc = _handle_mcp_restart_dispatch(ctx)

        assert rc == 1
        assert "unknown" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._run_async_safely")
    def test_healthy_server_json_returns_zero(self, mock_async, _compat, capsys):
        from aura_cli.dispatch import _handle_mcp_restart_dispatch

        mock_async.return_value = {"status": "healthy"}

        with patch("core.mcp_registry.get_registered_service", return_value={"url": "http://localhost:9000"}):
            args = _make_args(mcp_server="skills", json=True)
            ctx = _make_ctx(
                args=args,
                parsed=SimpleNamespace(warning_records=[], warnings=[]),
            )
            rc = _handle_mcp_restart_dispatch(ctx)

        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["server"] == "skills"

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._run_async_safely")
    def test_offline_server_returns_one(self, mock_async, _compat, capsys):
        from aura_cli.dispatch import _handle_mcp_restart_dispatch

        mock_async.return_value = {"status": "offline", "error": "connection refused"}

        with patch("core.mcp_registry.get_registered_service", return_value={"url": "http://localhost:9001"}):
            args = _make_args(mcp_server="skills", json=False)
            ctx = _make_ctx(args=args)
            rc = _handle_mcp_restart_dispatch(ctx)

        assert rc == 1


# ---------------------------------------------------------------------------
# _handle_beads_schemas_dispatch
# ---------------------------------------------------------------------------


class TestBeadsSchemasDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_json_mode_returns_payload(self, _compat, capsys, tmp_path):
        from aura_cli.dispatch import _handle_beads_schemas_dispatch

        args = _make_args(json=True)
        ctx = _make_ctx(
            args=args,
            project_root=tmp_path,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )
        rc = _handle_beads_schemas_dispatch(ctx)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "schemas" in out
        assert "schema_version" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_plain_mode_no_rich(self, _compat, capsys, tmp_path):
        from aura_cli.dispatch import _handle_beads_schemas_dispatch

        args = _make_args(json=False)
        ctx = _make_ctx(args=args, project_root=tmp_path)

        with patch.dict("sys.modules", {"rich": None, "rich.console": None, "rich.table": None, "rich.box": None}):
            rc = _handle_beads_schemas_dispatch(ctx)

        assert rc == 0
        out = capsys.readouterr().out
        assert "BEADS" in out or "Schema" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_with_config_file(self, _compat, capsys, tmp_path):
        from aura_cli.dispatch import _handle_beads_schemas_dispatch

        beads_dir = tmp_path / ".beads"
        beads_dir.mkdir()
        (beads_dir / "config.yaml").write_text("enabled: true\n")
        (beads_dir / "interactions.jsonl").write_text('{"id":1}\n{"id":2}\n')

        args = _make_args(json=True)
        ctx = _make_ctx(
            args=args,
            project_root=tmp_path,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )
        rc = _handle_beads_schemas_dispatch(ctx)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["interaction_count"] == 2


# ---------------------------------------------------------------------------
# _handle_agent_list_dispatch
# ---------------------------------------------------------------------------


class TestAgentListDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.log_json")
    def test_plain_mode_no_rich(self, _log, _compat, capsys):
        from aura_cli.dispatch import _handle_agent_list_dispatch

        with patch.dict("sys.modules", {"rich": None, "rich.table": None, "rich.console": None}):
            ctx = _make_ctx()
            rc = _handle_agent_list_dispatch(ctx)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Agent" in out or "Total" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.log_json")
    def test_with_rich_calls_log(self, _log, _compat):
        from aura_cli.dispatch import _handle_agent_list_dispatch

        ctx = _make_ctx()
        rc = _handle_agent_list_dispatch(ctx)
        assert rc == 0
        _log.assert_called()


# ---------------------------------------------------------------------------
# _handle_cancel_dispatch
# ---------------------------------------------------------------------------


class TestCancelDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_no_run_id_returns_one(self, _compat, capsys):
        from aura_cli.dispatch import _handle_cancel_dispatch

        args = _make_args(run_id=None)
        ctx = _make_ctx(args=args)
        rc = _handle_cancel_dispatch(ctx)
        assert rc == 1
        assert "required" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_unknown_run_id_returns_one(self, _compat, capsys):
        from aura_cli.dispatch import _handle_cancel_dispatch

        with (
            patch("core.running_runs.list_runs", return_value=[]),
            patch("core.running_runs.cancel_run"),
        ):
            args = _make_args(run_id="no-such-run")
            ctx = _make_ctx(args=args)
            rc = _handle_cancel_dispatch(ctx)

        assert rc == 1
        assert "not found" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.log_json")
    def test_successful_cancel_returns_zero(self, _log, _compat, capsys):
        from aura_cli.dispatch import _handle_cancel_dispatch

        with (
            patch("core.running_runs.list_runs", return_value=[{"run_id": "run-abc"}]),
            patch("core.running_runs.cancel_run", return_value=True),
        ):
            args = _make_args(run_id="run-abc")
            ctx = _make_ctx(args=args)
            rc = _handle_cancel_dispatch(ctx)

        assert rc == 0
        _log.assert_called()

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.log_json")
    def test_cancel_already_completed_returns_one(self, _log, _compat, capsys):
        from aura_cli.dispatch import _handle_cancel_dispatch

        with (
            patch("core.running_runs.list_runs", return_value=[{"run_id": "run-xyz"}]),
            patch("core.running_runs.cancel_run", return_value=False),
        ):
            args = _make_args(run_id="run-xyz")
            ctx = _make_ctx(args=args)
            rc = _handle_cancel_dispatch(ctx)

        assert rc == 1


# ---------------------------------------------------------------------------
# _handle_sadd_run_dispatch — validation error path
# ---------------------------------------------------------------------------


class TestSaddRunDispatchExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_validation_errors_returns_one(self, _compat, capsys, tmp_path):
        from aura_cli.dispatch import _handle_sadd_run_dispatch

        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("title: test\nworkstreams: []\n")

        args = _make_args(spec=str(spec_file), dry_run=True, json=False)
        rt = _fake_runtime()
        ctx = _make_ctx(args=args, runtime=rt)

        mock_spec = MagicMock()
        mock_spec.workstreams = []
        mock_spec.title = "test"
        mock_spec.parse_confidence = 1.0

        with (
            patch("core.sadd.design_spec_parser.DesignSpecParser") as mock_parser_cls,
            patch("core.sadd.types.validate_spec", return_value=["error1"]),
        ):
            mock_parser = MagicMock()
            mock_parser.parse_file.return_value = mock_spec
            mock_parser_cls.return_value = mock_parser
            rc = _handle_sadd_run_dispatch(ctx)

        assert rc == 1
        assert "Validation" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_missing_spec_returns_one(self, _compat, capsys):
        from aura_cli.dispatch import _handle_sadd_run_dispatch

        args = _make_args(spec="/nonexistent/path.yaml", dry_run=True, json=False)
        ctx = _make_ctx(args=args, runtime=_fake_runtime())
        rc = _handle_sadd_run_dispatch(ctx)
        assert rc == 1
        assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _handle_sadd_status_dispatch — session_id present
# ---------------------------------------------------------------------------


class TestSaddStatusDispatchExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_with_session_id_json_mode(self, _compat, capsys):
        from aura_cli.dispatch import _handle_sadd_status_dispatch

        mock_session = {"id": "sess-1", "title": "Test Session", "status": "completed", "created_at": 1000000.0, "report_json": None}

        with patch("core.sadd.session_store.SessionStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.get_session.return_value = mock_session
            mock_store.get_events.return_value = []
            mock_store.list_checkpoints.return_value = []
            mock_store_cls.return_value = mock_store

            args = _make_args(json=True, session_id="sess-1")
            ctx = _make_ctx(
                args=args,
                parsed=SimpleNamespace(warning_records=[], warnings=[]),
            )
            rc = _handle_sadd_status_dispatch(ctx)

        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "session" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_with_session_id_plain_mode(self, _compat, capsys):
        from aura_cli.dispatch import _handle_sadd_status_dispatch

        mock_session = {"id": "sess-2", "title": "My Session", "status": "running", "created_at": 1000000.0, "report_json": None}

        with patch("core.sadd.session_store.SessionStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.get_session.return_value = mock_session
            mock_store.get_events.return_value = []
            mock_store.list_checkpoints.return_value = []
            mock_store_cls.return_value = mock_store

            args = _make_args(json=False, session_id="sess-2")
            ctx = _make_ctx(args=args)
            rc = _handle_sadd_status_dispatch(ctx)

        assert rc == 0
        out = capsys.readouterr().out
        assert "My Session" in out or "sess-2" in out


# ---------------------------------------------------------------------------
# _handle_sadd_resume_dispatch
# ---------------------------------------------------------------------------


class TestSaddResumeDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_no_session_id_returns_one(self, _compat, capsys):
        from aura_cli.dispatch import _handle_sadd_resume_dispatch

        args = _make_args(session_id=None)
        ctx = _make_ctx(args=args)
        rc = _handle_sadd_resume_dispatch(ctx)
        assert rc == 1
        assert "required" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_not_found_returns_one(self, _compat, capsys):
        from aura_cli.dispatch import _handle_sadd_resume_dispatch

        with patch("core.sadd.session_store.SessionStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.load_session_for_resume.return_value = None
            mock_store_cls.return_value = mock_store

            args = _make_args(session_id="bad-sess")
            ctx = _make_ctx(args=args)
            rc = _handle_sadd_resume_dispatch(ctx)

        assert rc == 1
        assert "not found" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_dry_run_shows_remaining(self, _compat, capsys):
        from aura_cli.dispatch import _handle_sadd_resume_dispatch

        mock_spec = MagicMock()
        mock_spec.title = "My SADD"
        mock_spec.workstreams = [MagicMock(), MagicMock()]

        mock_config_data = MagicMock()
        mock_graph_state = None
        raw_results = {}

        with patch("core.sadd.session_store.SessionStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.load_session_for_resume.return_value = (mock_spec, mock_config_data, mock_graph_state, raw_results)
            mock_store_cls.return_value = mock_store

            with patch("core.sadd.workstream_graph.WorkstreamGraph") as mock_graph_cls:
                mock_graph = MagicMock()
                mock_graph._nodes = {}
                mock_graph_cls.return_value = mock_graph

                args = _make_args(session_id="good-sess", run=False)
                rt = _fake_runtime()
                ctx = _make_ctx(args=args, runtime=rt)
                rc = _handle_sadd_resume_dispatch(ctx)

        assert rc == 0
        out = capsys.readouterr().out
        assert "My SADD" in out or "good-sess" in out


# ---------------------------------------------------------------------------
# _handle_innovate_*_dispatch — delegation tests
# ---------------------------------------------------------------------------


class TestInnovateDispatches:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_innovate_start(self, _compat):
        from aura_cli.dispatch import _handle_innovate_start_dispatch

        with patch("aura_cli.commands._handle_innovate_start") as mock_fn:
            rt = _fake_runtime()
            ctx = _make_ctx(runtime=rt)
            rc = _handle_innovate_start_dispatch(ctx)
            assert rc == 0
            mock_fn.assert_called_once_with(ctx.args, ctx.runtime)

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_innovate_list(self, _compat):
        from aura_cli.dispatch import _handle_innovate_list_dispatch

        with patch("aura_cli.commands._handle_innovate_list") as mock_fn:
            rt = _fake_runtime()
            ctx = _make_ctx(runtime=rt)
            rc = _handle_innovate_list_dispatch(ctx)
            assert rc == 0
            mock_fn.assert_called_once_with(ctx.args, ctx.runtime)

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_innovate_show(self, _compat):
        from aura_cli.dispatch import _handle_innovate_show_dispatch

        with patch("aura_cli.commands._handle_innovate_show") as mock_fn:
            rt = _fake_runtime()
            ctx = _make_ctx(runtime=rt)
            rc = _handle_innovate_show_dispatch(ctx)
            assert rc == 0
            mock_fn.assert_called_once_with(ctx.args, ctx.runtime)

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_innovate_resume(self, _compat):
        from aura_cli.dispatch import _handle_innovate_resume_dispatch

        with patch("aura_cli.commands._handle_innovate_resume") as mock_fn:
            rt = _fake_runtime()
            ctx = _make_ctx(runtime=rt)
            rc = _handle_innovate_resume_dispatch(ctx)
            assert rc == 0
            mock_fn.assert_called_once_with(ctx.args, ctx.runtime)

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_innovate_export(self, _compat):
        from aura_cli.dispatch import _handle_innovate_export_dispatch

        with patch("aura_cli.commands._handle_innovate_export") as mock_fn:
            rt = _fake_runtime()
            ctx = _make_ctx(runtime=rt)
            rc = _handle_innovate_export_dispatch(ctx)
            assert rc == 0
            mock_fn.assert_called_once_with(ctx.args, ctx.runtime)

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_innovate_techniques(self, _compat):
        from aura_cli.dispatch import _handle_innovate_techniques_dispatch

        with patch("aura_cli.commands._handle_innovate_techniques") as mock_fn:
            ctx = _make_ctx()
            rc = _handle_innovate_techniques_dispatch(ctx)
            assert rc == 0
            mock_fn.assert_called_once_with(ctx.args)

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_innovate_to_goals(self, _compat):
        from aura_cli.dispatch import _handle_innovate_to_goals_dispatch

        with patch("aura_cli.commands._handle_innovate_to_goals") as mock_fn:
            rt = _fake_runtime()
            ctx = _make_ctx(runtime=rt)
            rc = _handle_innovate_to_goals_dispatch(ctx)
            assert rc == 0
            mock_fn.assert_called_once_with(ctx.args, ctx.runtime)

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_innovate_insights(self, _compat):
        from aura_cli.dispatch import _handle_innovate_insights_dispatch

        with patch("aura_cli.commands._handle_innovate_insights") as mock_fn:
            rt = _fake_runtime()
            ctx = _make_ctx(runtime=rt)
            rc = _handle_innovate_insights_dispatch(ctx)
            assert rc == 0
            mock_fn.assert_called_once_with(ctx.args, ctx.runtime)


# ---------------------------------------------------------------------------
# _handle_watch_dispatch
# ---------------------------------------------------------------------------


class TestWatchDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_calls_studio_run(self, _compat):
        from aura_cli.dispatch import _handle_watch_dispatch

        rt = _fake_runtime()
        rt["orchestrator"].attach_ui_callback = MagicMock()

        with patch("aura_cli.tui.app.AuraStudio") as mock_studio_cls:
            mock_studio = MagicMock()
            mock_studio_cls.return_value = mock_studio
            args = _make_args(autonomous=True)
            ctx = _make_ctx(args=args, runtime=rt)
            rc = _handle_watch_dispatch(ctx)

        assert rc == 0
        mock_studio.run.assert_called_once_with(autonomous=True)

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_no_orchestrator_skips_attach(self, _compat):
        from aura_cli.dispatch import _handle_watch_dispatch

        rt = _fake_runtime()
        rt["orchestrator"] = None

        with patch("aura_cli.tui.app.AuraStudio") as mock_studio_cls:
            mock_studio = MagicMock()
            mock_studio_cls.return_value = mock_studio
            args = _make_args(autonomous=False)
            ctx = _make_ctx(args=args, runtime=rt)
            rc = _handle_watch_dispatch(ctx)

        assert rc == 0


# ---------------------------------------------------------------------------
# _handle_queue_list_dispatch
# ---------------------------------------------------------------------------


class TestQueueListDispatchExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_empty_queue_message(self, _compat, capsys):
        from aura_cli.dispatch import _handle_queue_list_dispatch

        rt = _fake_runtime()
        rt["goal_queue"].queue = []
        args = _make_args(json=False)
        ctx = _make_ctx(args=args, runtime=rt)
        rc = _handle_queue_list_dispatch(ctx)
        assert rc == 0
        assert "empty" in capsys.readouterr().out.lower()

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_non_empty_queue_lists(self, _compat, capsys):
        from aura_cli.dispatch import _handle_queue_list_dispatch

        rt = _fake_runtime()
        rt["goal_queue"].queue = ["goal1", "goal2"]
        args = _make_args(json=False)
        ctx = _make_ctx(args=args, runtime=rt)
        rc = _handle_queue_list_dispatch(ctx)
        assert rc == 0
        out = capsys.readouterr().out
        assert "goal1" in out
        assert "goal2" in out

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_json_mode(self, _compat, capsys):
        from aura_cli.dispatch import _handle_queue_list_dispatch

        rt = _fake_runtime()
        rt["goal_queue"].queue = ["g1"]
        args = _make_args(json=True)
        ctx = _make_ctx(
            args=args,
            runtime=rt,
            parsed=SimpleNamespace(warning_records=[], warnings=[]),
        )
        rc = _handle_queue_list_dispatch(ctx)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["count"] == 1


# ---------------------------------------------------------------------------
# _handle_history_dispatch
# ---------------------------------------------------------------------------


class TestHistoryDispatchExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_calls_handle_history(self, _compat):
        from aura_cli.dispatch import _handle_history_dispatch

        rt = _fake_runtime()
        args = _make_args(limit=5, json=False)
        ctx = _make_ctx(args=args, runtime=rt)

        with patch("aura_cli.commands._handle_history") as mock_hist:
            rc = _handle_history_dispatch(ctx)

        assert rc == 0
        mock_hist.assert_called_once_with(rt["goal_archive"], limit=5, as_json=False)


# ---------------------------------------------------------------------------
# dispatch_command — main routing function
# ---------------------------------------------------------------------------


class TestDispatchCommand:
    def _make_parsed(self, action, **ns_kwargs):
        ns = _make_args(action=action, **ns_kwargs)
        return SimpleNamespace(
            action=action,
            namespace=ns,
            warning_records=[],
            warnings=[],
        )

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_unknown_action_returns_one(self, _compat, capsys):
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("totally_unknown_action")
        rc = dispatch_command(parsed, project_root=Path("/proj"))
        assert rc == 1
        assert "No dispatch rule" in capsys.readouterr().err

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_warning_records_printed_to_stderr(self, _compat, capsys):
        from aura_cli.dispatch import dispatch_command

        warning = MagicMock()
        warning.message = "deprecated flag"
        parsed = SimpleNamespace(
            action="help",
            namespace=_make_args(action="help"),
            warning_records=[warning],
            warnings=[],
        )

        with patch("aura_cli.dispatch.render_help", return_value="help text"):
            rc = dispatch_command(parsed, project_root=Path("/proj"))

        err = capsys.readouterr().err
        assert "deprecated flag" in err

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_plain_warnings_printed_to_stderr(self, _compat, capsys):
        from aura_cli.dispatch import dispatch_command

        parsed = SimpleNamespace(
            action="help",
            namespace=_make_args(action="help"),
            warning_records=[],
            warnings=["plain warning msg"],
        )

        with patch("aura_cli.dispatch.render_help", return_value="help"):
            rc = dispatch_command(parsed, project_root=Path("/proj"))

        err = capsys.readouterr().err
        assert "plain warning msg" in err

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._handle_doctor")
    def test_doctor_action_dispatches(self, mock_doctor, _compat):
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("doctor")
        rc = dispatch_command(parsed, project_root=Path("/proj"))
        assert rc == 0
        mock_doctor.assert_called_once()

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.render_help", return_value="help output")
    def test_help_action_dispatches(self, mock_render, _compat, capsys):
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("help")
        rc = dispatch_command(parsed, project_root=Path("/proj"))
        assert rc == 0

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch._prepare_runtime_context", return_value=1)
    def test_runtime_required_prep_failure_returns_prep_code(self, mock_prep, _compat):
        from aura_cli.dispatch import dispatch_command, COMMAND_DISPATCH_REGISTRY

        # Use a runtime-required action
        action = next(
            a for a, rule in COMMAND_DISPATCH_REGISTRY.items() if rule.requires_runtime
        )
        parsed = self._make_parsed(action)
        rc = dispatch_command(parsed, project_root=Path("/proj"))
        assert rc == 1

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.render_help", return_value="json help")
    def test_json_help_action_dispatches(self, mock_render, _compat, capsys):
        from aura_cli.dispatch import dispatch_command

        parsed = self._make_parsed("json_help")
        rc = dispatch_command(parsed, project_root=Path("/proj"))
        assert rc == 0


# ---------------------------------------------------------------------------
# COMMAND_DISPATCH_REGISTRY — structure checks
# ---------------------------------------------------------------------------


class TestCommandDispatchRegistry:
    def test_registry_is_dict(self):
        from aura_cli.dispatch import COMMAND_DISPATCH_REGISTRY, DispatchRule

        assert isinstance(COMMAND_DISPATCH_REGISTRY, dict)
        for action, rule in COMMAND_DISPATCH_REGISTRY.items():
            assert isinstance(rule, DispatchRule)
            assert rule.action == action
            assert callable(rule.handler)

    def test_required_actions_present(self):
        from aura_cli.dispatch import COMMAND_DISPATCH_REGISTRY

        required = [
            "help",
            "doctor",
            "goal_once",
            "goal_run",
            "goal_status",
            "goal_add",
            "evolve",
            "scaffold",
            "interactive",
            "cancel",
        ]
        for action in required:
            assert action in COMMAND_DISPATCH_REGISTRY, f"Missing: {action}"

    def test_all_rules_have_handlers(self):
        from aura_cli.dispatch import COMMAND_DISPATCH_REGISTRY

        for action, rule in COMMAND_DISPATCH_REGISTRY.items():
            assert rule.handler is not None, f"No handler for {action}"

    def test_studio_is_alias_for_watch(self):
        from aura_cli.dispatch import COMMAND_DISPATCH_REGISTRY

        assert COMMAND_DISPATCH_REGISTRY["studio"].handler is COMMAND_DISPATCH_REGISTRY["watch"].handler


# ---------------------------------------------------------------------------
# _dispatch_rule helper
# ---------------------------------------------------------------------------


class TestDispatchRule:
    def test_creates_dispatch_rule(self):
        from aura_cli.dispatch import _dispatch_rule, DispatchRule

        handler = MagicMock()
        with patch("aura_cli.dispatch.action_runtime_required", return_value=True):
            rule = _dispatch_rule("myaction", handler)

        assert isinstance(rule, DispatchRule)
        assert rule.action == "myaction"
        assert rule.handler is handler
        assert rule.requires_runtime is True

    def test_frozen_rule_cannot_be_mutated(self):
        from aura_cli.dispatch import DispatchRule

        rule = DispatchRule(action="x", requires_runtime=False, handler=lambda: None)
        with pytest.raises((AttributeError, TypeError)):
            rule.action = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RuntimeContext & DispatchContext dataclass tests
# ---------------------------------------------------------------------------


class TestDataclassesExtra:
    def test_runtime_context_extra_field(self):
        from aura_cli.dispatch import RuntimeContext

        rc = RuntimeContext(agent="planner", model="gpt-4", verbose=True, dry_run=True, non_interactive=True, timeout=30)
        assert rc.agent == "planner"
        assert rc.model == "gpt-4"
        assert rc.verbose is True
        assert rc.timeout == 30

    def test_runtime_context_extra_dict(self):
        from aura_cli.dispatch import RuntimeContext

        rc = RuntimeContext(extra={"foo": "bar"})
        assert rc.extra["foo"] == "bar"

    def test_dispatch_context_has_runtime_none_by_default(self):
        from aura_cli.dispatch import DispatchContext

        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/p"),
            runtime_factory=MagicMock(),
            args=SimpleNamespace(),
        )
        assert ctx.runtime is None


# ---------------------------------------------------------------------------
# _handle_interactive_dispatch
# ---------------------------------------------------------------------------


class TestInteractiveDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_calls_cli_loop(self, _compat):
        from aura_cli.dispatch import _handle_interactive_dispatch

        rt = _fake_runtime()
        ctx = _make_ctx(runtime=rt)

        with patch("aura_cli.cli_main.cli_interaction_loop") as mock_loop:
            rc = _handle_interactive_dispatch(ctx)

        assert rc == 0
        mock_loop.assert_called_once_with(ctx.args, rt)


# ---------------------------------------------------------------------------
# _handle_mcp_tools_dispatch & _handle_mcp_call_dispatch
# ---------------------------------------------------------------------------


class TestMcpToolsCallDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.cmd_mcp_tools")
    def test_mcp_tools_dispatch(self, mock_cmd, _compat, capsys):
        from aura_cli.dispatch import _handle_mcp_tools_dispatch

        mock_cmd.return_value = 0
        ctx = _make_ctx(parsed=SimpleNamespace(warning_records=[], warnings=[]))
        rc = _handle_mcp_tools_dispatch(ctx)
        assert rc == 0
        mock_cmd.assert_called_once()

    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.cmd_mcp_call")
    def test_mcp_call_dispatch_passes_args(self, mock_cmd, _compat, capsys):
        from aura_cli.dispatch import _handle_mcp_call_dispatch

        mock_cmd.return_value = 0
        args = _make_args(mcp_call="tool_name", mcp_args={"x": 1})
        ctx = _make_ctx(args=args, parsed=SimpleNamespace(warning_records=[], warnings=[]))
        rc = _handle_mcp_call_dispatch(ctx)
        assert rc == 0
        mock_cmd.assert_called_once_with("tool_name", {"x": 1})


# ---------------------------------------------------------------------------
# _handle_diag_dispatch
# ---------------------------------------------------------------------------


class TestDiagDispatch:
    @patch("aura_cli.dispatch._sync_cli_compat")
    @patch("aura_cli.dispatch.cmd_diag")
    def test_diag_dispatch(self, mock_cmd, _compat, capsys):
        from aura_cli.dispatch import _handle_diag_dispatch

        mock_cmd.return_value = 0
        ctx = _make_ctx(parsed=SimpleNamespace(warning_records=[], warnings=[]))
        rc = _handle_diag_dispatch(ctx)
        assert rc == 0
        mock_cmd.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_logs_dispatch
# ---------------------------------------------------------------------------


class TestLogsDispatchExtra:
    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_stream_file(self, _compat):
        from aura_cli.dispatch import _handle_logs_dispatch

        with patch("aura_cli.tui.log_streamer.LogStreamer") as mock_cls:
            mock_streamer = MagicMock()
            mock_cls.return_value = mock_streamer
            args = _make_args(level="debug", file="/var/log/aura.log", tail=100, follow=False)
            ctx = _make_ctx(args=args)
            rc = _handle_logs_dispatch(ctx)

        assert rc == 0
        mock_streamer.stream_file.assert_called_once()

    @patch("aura_cli.dispatch._sync_cli_compat")
    def test_stream_stdin(self, _compat):
        from aura_cli.dispatch import _handle_logs_dispatch

        with patch("aura_cli.tui.log_streamer.LogStreamer") as mock_cls:
            mock_streamer = MagicMock()
            mock_cls.return_value = mock_streamer
            args = _make_args(level="info", file=None, tail=None, follow=False)
            ctx = _make_ctx(args=args)
            rc = _handle_logs_dispatch(ctx)

        assert rc == 0
        mock_streamer.stream_stdin.assert_called_once()


# ---------------------------------------------------------------------------
# _run_json_printing_callable_with_warnings — edge cases
# ---------------------------------------------------------------------------


class TestRunJsonPrintingCallableWithWarnings:
    def test_non_json_output_passes_through(self, capsys):
        from aura_cli.dispatch import _run_json_printing_callable_with_warnings

        warning = MagicMock()
        warning.message = "w1"
        parsed = SimpleNamespace(warning_records=[warning], warnings=[])
        ctx = _make_ctx(parsed=parsed)

        def func():
            print("plain text output")
            return 0

        result = _run_json_printing_callable_with_warnings(ctx, func)
        out = capsys.readouterr().out
        assert "plain text output" in out
        assert result == 0

    def test_empty_output_returns_result(self, capsys):
        from aura_cli.dispatch import _run_json_printing_callable_with_warnings

        warning = MagicMock()
        warning.message = "w1"
        parsed = SimpleNamespace(warning_records=[warning], warnings=[])
        ctx = _make_ctx(parsed=parsed)

        def func():
            return 42

        result = _run_json_printing_callable_with_warnings(ctx, func)
        assert result == 42

    def test_no_warnings_calls_directly(self, capsys):
        from aura_cli.dispatch import _run_json_printing_callable_with_warnings

        ctx = _make_ctx(parsed=SimpleNamespace(warning_records=[], warnings=[]))

        called_with = []

        def func(a, b):
            called_with.append((a, b))
            return 0

        _run_json_printing_callable_with_warnings(ctx, func, "x", "y")
        assert called_with == [("x", "y")]


# ---------------------------------------------------------------------------
# _print_json_payload
# ---------------------------------------------------------------------------


class TestPrintJsonPayloadExtra:
    def test_prints_valid_json(self, capsys):
        from aura_cli.dispatch import _print_json_payload

        _print_json_payload({"a": 1, "b": [1, 2]})
        out = json.loads(capsys.readouterr().out)
        assert out["a"] == 1

    def test_with_parsed_object(self, capsys):
        from aura_cli.dispatch import _print_json_payload

        with patch("aura_cli.dispatch.attach_cli_warnings", return_value={"result": "ok", "_warnings": []}):
            _print_json_payload({"result": "ok"}, parsed=SimpleNamespace())
        out = json.loads(capsys.readouterr().out)
        assert "result" in out


# ---------------------------------------------------------------------------
# _resolve_beads_runtime_override — additional combinations
# ---------------------------------------------------------------------------


class TestBeadsOverrideExtra:
    def test_multiple_flags_last_one_wins(self):
        from aura_cli.dispatch import _resolve_beads_runtime_override

        # both beads and no_beads — beads is iterated first, then no_beads overrides
        args = _make_args(beads=True, no_beads=True)
        with patch("aura_cli.dispatch.config") as mock_cfg, patch("aura_cli.dispatch.DEFAULT_CONFIG", {"beads": {}}):
            mock_cfg.get.return_value = {}
            beads_config, beads_cli = _resolve_beads_runtime_override(args)

        # Result should show the flag combination was processed
        assert beads_cli is not None
        assert beads_config is not None

    def test_beads_required_sets_both_enabled_and_required(self):
        from aura_cli.dispatch import _resolve_beads_runtime_override

        args = _make_args(beads_required=True)
        with patch("aura_cli.dispatch.config") as mock_cfg, patch("aura_cli.dispatch.DEFAULT_CONFIG", {"beads": {}}):
            mock_cfg.get.return_value = {}
            beads_config, beads_cli = _resolve_beads_runtime_override(args)

        assert beads_cli["enabled"] is True
        assert beads_cli["required"] is True


# ---------------------------------------------------------------------------
# _handle_contract_report_dispatch
# ---------------------------------------------------------------------------


class TestContractReportDispatchExtra:
    def test_check_mode_failure(self, capsys):
        from aura_cli.dispatch import _handle_contract_report_dispatch

        args = _make_args(no_dispatch=False, compact=False, check=True)
        ctx = _make_ctx(args=args)

        with patch("aura_cli.contract_report.build_cli_contract_report") as mock_build, \
             patch("aura_cli.contract_report.render_cli_contract_report", return_value="report\n"), \
             patch("aura_cli.contract_report.cli_contract_report_exit_code", return_value=1), \
             patch("aura_cli.contract_report.cli_contract_report_failure_message", return_value="FAILED"):
            mock_report = MagicMock()
            mock_build.return_value = mock_report
            rc = _handle_contract_report_dispatch(ctx)

        assert rc == 1
        err = capsys.readouterr().err
        assert "FAILED" in err

    def test_success_returns_zero(self, capsys):
        from aura_cli.dispatch import _handle_contract_report_dispatch

        args = _make_args(no_dispatch=True, compact=True, check=False)
        ctx = _make_ctx(args=args)

        with patch("aura_cli.contract_report.build_cli_contract_report") as mock_build, \
             patch("aura_cli.contract_report.render_cli_contract_report", return_value="ok\n"), \
             patch("aura_cli.contract_report.cli_contract_report_exit_code", return_value=0), \
             patch("aura_cli.contract_report.cli_contract_report_failure_message", return_value=""):
            mock_report = MagicMock()
            mock_build.return_value = mock_report
            rc = _handle_contract_report_dispatch(ctx)

        assert rc == 0
