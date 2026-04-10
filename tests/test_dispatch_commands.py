"""Tests for aura_cli/dispatch.py and aura_cli/commands.py."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    """Return a simple namespace with sensible defaults for CLI args."""
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


# ---------------------------------------------------------------------------
# dispatch.py — _resolve_dispatch_action
# ---------------------------------------------------------------------------


class TestResolveDispatchAction:
    def setup_method(self):
        from aura_cli.dispatch import _resolve_dispatch_action

        self._fn = _resolve_dispatch_action

    def test_returns_action_when_set(self):
        parsed = SimpleNamespace(action="goal_once")
        assert self._fn(parsed) == "goal_once"

    def test_returns_interactive_when_no_action(self):
        parsed = SimpleNamespace(action=None)
        assert self._fn(parsed) == "interactive"

    def test_returns_interactive_when_action_missing(self):
        parsed = SimpleNamespace()  # no .action attribute at all
        assert self._fn(parsed) == "interactive"

    def test_various_action_names(self):
        for name in ("goal_run", "evolve", "doctor", "scaffold"):
            assert self._fn(SimpleNamespace(action=name)) == name


# ---------------------------------------------------------------------------
# dispatch.py — _resolve_runtime_mode
# ---------------------------------------------------------------------------


class TestResolveRuntimeMode:
    def setup_method(self):
        from aura_cli.dispatch import _resolve_runtime_mode

        self._fn = _resolve_runtime_mode

    def test_queue_mode_for_goal_status(self):
        assert self._fn("goal_status", _make_args()) == "queue"

    def test_queue_mode_for_goal_add(self):
        assert self._fn("goal_add", _make_args()) == "queue"

    def test_queue_mode_for_interactive(self):
        assert self._fn("interactive", _make_args()) == "queue"

    def test_lean_mode_for_goal_once_dry_run(self):
        assert self._fn("goal_once", _make_args(dry_run=True)) == "lean"

    def test_none_for_goal_once_without_dry_run(self):
        assert self._fn("goal_once", _make_args(dry_run=False)) is None

    def test_none_for_evolve(self):
        assert self._fn("evolve", _make_args()) is None

    def test_none_for_doctor(self):
        assert self._fn("doctor", _make_args()) is None


# ---------------------------------------------------------------------------
# dispatch.py — _resolve_beads_runtime_override
# ---------------------------------------------------------------------------


class TestResolveBeadsRuntimeOverride:
    def setup_method(self):
        from aura_cli.dispatch import _resolve_beads_runtime_override

        self._fn = _resolve_beads_runtime_override

    def _patch_config(self, beads_cfg=None):
        mock_config = MagicMock()
        mock_config.get.return_value = beads_cfg or {}
        mock_default = {"beads": {}}
        return mock_config, mock_default

    def test_no_flag_returns_none_none(self):
        mock_config, mock_default = self._patch_config()
        with patch("aura_cli.dispatch.config", mock_config), patch("aura_cli.dispatch.DEFAULT_CONFIG", mock_default):
            result = self._fn(_make_args())
        assert result == (None, None)

    def test_beads_flag_enables_beads(self):
        mock_config, mock_default = self._patch_config()
        with patch("aura_cli.dispatch.config", mock_config), patch("aura_cli.dispatch.DEFAULT_CONFIG", mock_default):
            cfg, override = self._fn(_make_args(beads=True))
        assert cfg is not None
        assert override["enabled"] is True
        assert override["source"] == "cli"

    def test_no_beads_flag_disables_beads(self):
        mock_config, mock_default = self._patch_config()
        with patch("aura_cli.dispatch.config", mock_config), patch("aura_cli.dispatch.DEFAULT_CONFIG", mock_default):
            cfg, override = self._fn(_make_args(no_beads=True))
        assert override["enabled"] is False

    def test_beads_required_sets_required_true(self):
        mock_config, mock_default = self._patch_config()
        with patch("aura_cli.dispatch.config", mock_config), patch("aura_cli.dispatch.DEFAULT_CONFIG", mock_default):
            cfg, override = self._fn(_make_args(beads_required=True))
        assert override["required"] is True

    def test_beads_optional_sets_required_false(self):
        mock_config, mock_default = self._patch_config()
        with patch("aura_cli.dispatch.config", mock_config), patch("aura_cli.dispatch.DEFAULT_CONFIG", mock_default):
            cfg, override = self._fn(_make_args(beads_optional=True))
        assert override["required"] is False


# ---------------------------------------------------------------------------
# dispatch.py — RuntimeContext / DispatchContext / DispatchRule dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_runtime_context_defaults(self):
        from aura_cli.dispatch import RuntimeContext

        rc = RuntimeContext()
        assert rc.agent is None
        assert rc.model is None
        assert rc.verbose is False
        assert rc.dry_run is False
        assert rc.non_interactive is False
        assert rc.timeout is None
        assert rc.extra == {}

    def test_runtime_context_custom(self):
        from aura_cli.dispatch import RuntimeContext

        rc = RuntimeContext(agent="coder", model="gpt-4", verbose=True, dry_run=True)
        assert rc.agent == "coder"
        assert rc.model == "gpt-4"
        assert rc.verbose is True
        assert rc.dry_run is True

    def test_dispatch_rule_frozen(self):
        from aura_cli.dispatch import DispatchRule

        handler = lambda ctx: 0
        rule = DispatchRule(action="test", requires_runtime=False, handler=handler)
        with pytest.raises((AttributeError, TypeError)):
            rule.action = "other"  # type: ignore[misc]

    def test_dispatch_context_fields(self):
        from aura_cli.dispatch import DispatchContext

        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/tmp"),
            runtime_factory=MagicMock(),
            args=_make_args(),
        )
        assert ctx.runtime is None


# ---------------------------------------------------------------------------
# dispatch.py — _handle_help_dispatch
# ---------------------------------------------------------------------------


class TestHandleHelpDispatch:
    def _make_ctx(self, **kwargs):
        from aura_cli.dispatch import DispatchContext

        return DispatchContext(
            parsed=SimpleNamespace(warning_records=[]),
            project_root=Path("."),
            runtime_factory=MagicMock(),
            args=_make_args(**kwargs),
        )

    def test_success_returns_zero(self, capsys):
        from aura_cli.dispatch import _handle_help_dispatch

        with patch("aura_cli.dispatch.render_help", return_value="help text"):
            code = _handle_help_dispatch(self._make_ctx())
        assert code == 0
        assert "help text" in capsys.readouterr().out

    def test_value_error_returns_two(self, capsys):
        from aura_cli.dispatch import _handle_help_dispatch

        with patch("aura_cli.dispatch.render_help", side_effect=ValueError("bad topic")):
            code = _handle_help_dispatch(self._make_ctx())
        assert code == 2

    def test_value_error_json_mode_prints_json(self, capsys):
        from aura_cli.dispatch import _handle_help_dispatch

        with patch("aura_cli.dispatch.render_help", side_effect=ValueError("bad topic")), patch("aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p), patch("aura_cli.dispatch.unknown_command_help_topic_payload", return_value={"error": "bad topic"}):
            code = _handle_help_dispatch(self._make_ctx(json=True))
        assert code == 2
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert "error" in parsed


# ---------------------------------------------------------------------------
# dispatch.py — _handle_json_help_dispatch
# ---------------------------------------------------------------------------


class TestHandleJsonHelpDispatch:
    def test_calls_render_help_with_json_format(self, capsys):
        from aura_cli.dispatch import _handle_json_help_dispatch, DispatchContext

        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("."),
            runtime_factory=MagicMock(),
            args=_make_args(),
        )
        with patch("aura_cli.dispatch.render_help", return_value='{"help": true}') as mock_render:
            code = _handle_json_help_dispatch(ctx)
        mock_render.assert_called_once_with(format="json")
        assert code == 0


# ---------------------------------------------------------------------------
# dispatch.py — _handle_config_set_dispatch
# ---------------------------------------------------------------------------


class TestHandleConfigSetDispatch:
    def _make_ctx(self, key, value):
        from aura_cli.dispatch import DispatchContext

        args = _make_args()
        args.config_key = key
        args.config_value = value
        return DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("."),
            runtime_factory=MagicMock(),
            args=args,
        )

    def test_plain_key_calls_update_config(self, capsys):
        from aura_cli.dispatch import _handle_config_set_dispatch

        mock_config = MagicMock()
        with patch("aura_cli.dispatch.config", mock_config):
            code = _handle_config_set_dispatch(self._make_ctx("timeout", "30"))
        mock_config.update_config.assert_called_once_with({"timeout": "30"})
        assert code == 0

    def test_model_dot_key_routes_to_model_routing(self, capsys):
        from aura_cli.dispatch import _handle_config_set_dispatch

        mock_config = MagicMock()
        with patch("aura_cli.dispatch.config", mock_config):
            code = _handle_config_set_dispatch(self._make_ctx("model.plan", "claude-3"))
        mock_config.update_config.assert_called_once_with({"model_routing": {"plan": "claude-3"}})
        assert code == 0

    def test_exception_returns_one(self, capsys):
        from aura_cli.dispatch import _handle_config_set_dispatch

        mock_config = MagicMock()
        mock_config.update_config.side_effect = RuntimeError("disk full")
        with patch("aura_cli.dispatch.config", mock_config):
            code = _handle_config_set_dispatch(self._make_ctx("key", "val"))
        assert code == 1


# ---------------------------------------------------------------------------
# dispatch.py — _handle_show_config_dispatch
# ---------------------------------------------------------------------------


class TestHandleShowConfigDispatch:
    def test_prints_json(self, capsys):
        from aura_cli.dispatch import _handle_show_config_dispatch, DispatchContext

        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("."),
            runtime_factory=MagicMock(),
            args=_make_args(),
        )
        mock_config = MagicMock()
        mock_config.show_config.return_value = {"key": "val"}
        with patch("aura_cli.dispatch.config", mock_config):
            code = _handle_show_config_dispatch(ctx)
        out = capsys.readouterr().out
        assert code == 0
        parsed = json.loads(out)
        assert parsed["key"] == "val"


# ---------------------------------------------------------------------------
# dispatch.py — _handle_queue_list_dispatch
# ---------------------------------------------------------------------------


class TestHandleQueueListDispatch:
    def _make_ctx(self, queue_items, json_mode=False):
        from aura_cli.dispatch import DispatchContext

        mock_queue = MagicMock()
        mock_queue.queue = queue_items
        args = _make_args(json=json_mode)
        ctx = DispatchContext(
            parsed=SimpleNamespace(warning_records=[]),
            project_root=Path("."),
            runtime_factory=MagicMock(),
            args=args,
        )
        ctx.runtime = {"goal_queue": mock_queue}
        return ctx

    def test_empty_queue_prints_message(self, capsys):
        from aura_cli.dispatch import _handle_queue_list_dispatch

        ctx = self._make_ctx([])
        code = _handle_queue_list_dispatch(ctx)
        assert code == 0
        assert "empty" in capsys.readouterr().out.lower()

    def test_non_empty_queue_lists_goals(self, capsys):
        from aura_cli.dispatch import _handle_queue_list_dispatch

        ctx = self._make_ctx(["goal A", "goal B"])
        code = _handle_queue_list_dispatch(ctx)
        assert code == 0
        out = capsys.readouterr().out
        assert "goal A" in out
        assert "goal B" in out

    def test_json_mode_outputs_json(self, capsys):
        from aura_cli.dispatch import _handle_queue_list_dispatch

        ctx = self._make_ctx(["goal X"], json_mode=True)
        with patch("aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p):
            code = _handle_queue_list_dispatch(ctx)
        assert code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["count"] == 1
        assert "goal X" in data["queue"]


# ---------------------------------------------------------------------------
# dispatch.py — _handle_queue_clear_dispatch
# ---------------------------------------------------------------------------


class TestHandleQueueClearDispatch:
    def _make_ctx(self, queue_items, json_mode=False):
        from aura_cli.dispatch import DispatchContext

        mock_queue = MagicMock()
        mock_queue.queue = list(queue_items)
        args = _make_args(json=json_mode)
        ctx = DispatchContext(
            parsed=SimpleNamespace(warning_records=[]),
            project_root=Path("."),
            runtime_factory=MagicMock(),
            args=args,
        )
        ctx.runtime = {"goal_queue": mock_queue}
        return ctx

    def test_clears_queue_and_prints_count(self, capsys):
        from aura_cli.dispatch import _handle_queue_clear_dispatch

        ctx = self._make_ctx(["a", "b", "c"])
        code = _handle_queue_clear_dispatch(ctx)
        ctx.runtime["goal_queue"].clear.assert_called_once()
        assert code == 0
        assert "3" in capsys.readouterr().out

    def test_json_mode_returns_json(self, capsys):
        from aura_cli.dispatch import _handle_queue_clear_dispatch

        ctx = self._make_ctx(["x", "y"], json_mode=True)
        with patch("aura_cli.dispatch.attach_cli_warnings", side_effect=lambda p, _: p):
            code = _handle_queue_clear_dispatch(ctx)
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["cleared_count"] == 2


# ---------------------------------------------------------------------------
# dispatch.py — _handle_doctor_dispatch / _handle_readiness_dispatch
# ---------------------------------------------------------------------------


class TestSimpleDispatchHandlers:
    def _make_ctx(self):
        from aura_cli.dispatch import DispatchContext

        return DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/proj"),
            runtime_factory=MagicMock(),
            args=_make_args(),
        )

    def test_doctor_dispatch_returns_zero(self):
        from aura_cli.dispatch import _handle_doctor_dispatch

        with patch("aura_cli.dispatch._handle_doctor") as mock_doc:
            code = _handle_doctor_dispatch(self._make_ctx())
        mock_doc.assert_called_once_with(Path("/proj"))
        assert code == 0

    def test_readiness_dispatch_returns_zero(self):
        from aura_cli.dispatch import _handle_readiness_dispatch

        with patch("aura_cli.dispatch._handle_readiness") as mock_r:
            code = _handle_readiness_dispatch(self._make_ctx())
        mock_r.assert_called_once()
        assert code == 0


# ---------------------------------------------------------------------------
# commands.py — _handle_add
# ---------------------------------------------------------------------------


class TestHandleAdd:
    def setup_method(self):
        from aura_cli.commands import _handle_add

        self._fn = _handle_add

    def _make_queue(self):
        q = MagicMock()
        q.queue = []
        q.__len__ = lambda self: len(self.queue)
        return q

    def test_adds_valid_goal(self, capsys):
        q = self._make_queue()
        with patch("aura_cli.commands.log_json"):
            self._fn(q, "add Build a new feature")
        q.add.assert_called_once_with("Build a new feature")

    def test_empty_goal_does_not_add(self, capsys):
        q = self._make_queue()
        with patch("aura_cli.commands.log_json"):
            self._fn(q, "add   ")
        q.add.assert_not_called()

    def test_goal_too_long_rejected(self, capsys):
        q = self._make_queue()
        long_goal = "x" * 501
        with patch("aura_cli.commands.log_json"):
            self._fn(q, f"add {long_goal}")
        q.add.assert_not_called()
        assert "too long" in capsys.readouterr().out

    def test_forbidden_characters_rejected(self, capsys):
        q = self._make_queue()
        for bad in ("add do; rm -rf", "add a && b", "add $(cmd)"):
            q.reset_mock()
            with patch("aura_cli.commands.log_json"):
                self._fn(q, bad)
            q.add.assert_not_called()

    def test_valid_goal_prints_confirmation(self, capsys):
        q = self._make_queue()
        with patch("aura_cli.commands.log_json"):
            self._fn(q, "add Refactor auth module")
        out = capsys.readouterr().out
        assert "Refactor auth module" in out


# ---------------------------------------------------------------------------
# commands.py — _handle_history
# ---------------------------------------------------------------------------


class TestHandleHistory:
    def setup_method(self):
        from aura_cli.commands import _handle_history

        self._fn = _handle_history

    def _make_archive(self, completed):
        arch = MagicMock()
        arch.completed = completed
        return arch

    def test_empty_archive_prints_message(self, capsys):
        arch = self._make_archive([])
        self._fn(arch)
        assert "No completed" in capsys.readouterr().out

    def test_prints_goals_in_reverse_order(self, capsys):
        arch = self._make_archive([("first goal", 0.9), ("second goal", 0.8)])
        self._fn(arch, limit=10)
        out = capsys.readouterr().out
        assert "second goal" in out
        assert "first goal" in out

    def test_json_mode_outputs_valid_json(self, capsys):
        arch = self._make_archive([("goal A", 0.7)])
        self._fn(arch, limit=10, as_json=True)
        data = json.loads(capsys.readouterr().out)
        assert "history" in data
        assert data["total"] == 1

    def test_limit_respected(self, capsys):
        goals = [(f"goal {i}", 0.5) for i in range(20)]
        arch = self._make_archive(goals)
        self._fn(arch, limit=5)
        out = capsys.readouterr().out
        # Should show "5 of 20" in header
        assert "5" in out and "20" in out

    def test_json_items_missing_score(self, capsys):
        arch = self._make_archive(["bare string goal"])
        self._fn(arch, as_json=True)
        data = json.loads(capsys.readouterr().out)
        assert data["history"][0]["goal"] == "bare string goal"
        assert data["history"][0]["score"] is None


# ---------------------------------------------------------------------------
# commands.py — _compute_global_insights
# ---------------------------------------------------------------------------


class TestComputeGlobalInsights:
    def setup_method(self):
        from aura_cli.commands import _compute_global_insights

        self._fn = _compute_global_insights

    def _make_session(self, ideas_generated=5, ideas_selected=2, techniques=None, status="completed", output=None):
        s = MagicMock()
        s.ideas_generated = ideas_generated
        s.ideas_selected = ideas_selected
        s.techniques = techniques or ["scamper"]
        s.status = status
        s.output = output
        return s

    def test_empty_sessions(self):
        result = self._fn([])
        assert result["total_sessions"] == 0
        assert result["total_ideas"] == 0
        assert result["selection_rate"] == 0

    def test_aggregates_ideas(self):
        sessions = [self._make_session(10, 3), self._make_session(20, 5)]
        result = self._fn(sessions)
        assert result["total_ideas"] == 30
        assert result["total_selected"] == 8

    def test_selection_rate_calculated(self):
        sessions = [self._make_session(10, 5)]
        result = self._fn(sessions)
        assert abs(result["selection_rate"] - 0.5) < 1e-9

    def test_technique_usage_counted(self):
        s1 = self._make_session(techniques=["scamper", "six_hats"])
        s2 = self._make_session(techniques=["scamper"])
        result = self._fn([s1, s2])
        assert result["technique_usage"]["scamper"] == 2
        assert result["technique_usage"]["six_hats"] == 1

    def test_sessions_by_status(self):
        sessions = [
            self._make_session(status="active"),
            self._make_session(status="completed"),
            self._make_session(status="completed"),
        ]
        result = self._fn(sessions)
        assert result["sessions_by_status"]["active"] == 1
        assert result["sessions_by_status"]["completed"] == 2

    def test_average_scores_from_output(self):
        output = MagicMock()
        output.novelty_score = 0.8
        output.feasibility_score = 0.6
        output.diversity_score = 0.7
        s = self._make_session(output=output)
        result = self._fn([s])
        assert abs(result["average_scores"]["novelty"] - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# commands.py — _compute_session_insights
# ---------------------------------------------------------------------------


class TestComputeSessionInsights:
    def setup_method(self):
        from aura_cli.commands import _compute_session_insights

        self._fn = _compute_session_insights

    def _make_idea(self, technique="scamper", novelty=0.7, feasibility=0.6, impact=0.8, description="an idea"):
        idea = MagicMock()
        idea.technique = technique
        idea.novelty = novelty
        idea.feasibility = feasibility
        idea.impact = impact
        idea.description = description
        return idea

    def test_basic_fields_present(self):
        session = MagicMock()
        session.session_id = "sess-1"
        session.problem_statement = "Test problem"
        session.status = "active"
        session.techniques = ["scamper"]
        session.ideas_generated = 3
        session.ideas_selected = 1
        session.output = None
        result = self._fn(session)
        assert result["session_id"] == "sess-1"
        assert result["status"] == "active"
        assert result["ideas_generated"] == 3

    def test_quality_metrics_computed(self):
        idea1 = self._make_idea(novelty=0.8, feasibility=0.6, impact=0.5)
        idea2 = self._make_idea(novelty=0.4, feasibility=0.4, impact=0.9)
        output = MagicMock()
        output.all_ideas = [idea1, idea2]
        output.selected_ideas = [idea1]
        session = MagicMock()
        session.output = output
        session.session_id = "s"
        session.problem_statement = "p"
        session.status = "active"
        session.techniques = []
        session.ideas_generated = 2
        session.ideas_selected = 1
        result = self._fn(session)
        assert "quality_metrics" in result
        assert abs(result["quality_metrics"]["avg_novelty"] - 0.6) < 1e-9

    def test_technique_breakdown(self):
        idea1 = self._make_idea(technique="scamper")
        idea2 = self._make_idea(technique="scamper")
        idea3 = self._make_idea(technique="six_hats")
        output = MagicMock()
        output.all_ideas = [idea1, idea2, idea3]
        output.selected_ideas = []
        session = MagicMock()
        session.output = output
        session.session_id = "s"
        session.problem_statement = "p"
        session.status = "active"
        session.techniques = []
        session.ideas_generated = 3
        session.ideas_selected = 0
        result = self._fn(session)
        assert result["technique_breakdown"]["scamper"] == 2
        assert result["technique_breakdown"]["six_hats"] == 1


# ---------------------------------------------------------------------------
# commands.py — _handle_exit
# ---------------------------------------------------------------------------


class TestHandleExit:
    def test_logs_exit_event(self):
        from aura_cli.commands import _handle_exit

        with patch("aura_cli.commands.log_json") as mock_log:
            _handle_exit()
        mock_log.assert_called_once_with("INFO", "aura_cli_exit")


# ---------------------------------------------------------------------------
# commands.py — _generate_json_export / _generate_csv_export
# ---------------------------------------------------------------------------


class TestGenerateExports:
    def _make_session(self, ideas=None, selected=None):
        from datetime import datetime

        session = MagicMock()
        session.session_id = "test-session-001"
        session.problem_statement = "How to improve?"
        session.status = "completed"
        session.current_phase = MagicMock()
        session.current_phase.value = "synthesis"
        session.phases_completed = []
        session.techniques = ["scamper"]
        session.constraints = []
        session.ideas_generated = len(ideas or [])
        session.ideas_selected = len(selected or [])
        session.created_at = datetime(2024, 1, 1)
        session.updated_at = datetime(2024, 1, 2)
        session.output = None
        return session

    def _make_idea(self, desc="Idea text", technique="scamper"):
        idea = MagicMock()
        idea.description = desc
        idea.technique = technique
        idea.novelty = 0.7
        idea.feasibility = 0.8
        idea.impact = 0.6
        idea.metadata = {}
        return idea

    def test_json_export_is_valid_json(self):
        from aura_cli.commands import _generate_json_export

        idea = self._make_idea()
        session = self._make_session(ideas=[idea])
        result = _generate_json_export(session, [idea], [])
        data = json.loads(result)
        assert data["session_id"] == "test-session-001"
        assert len(data["ideas"]) == 1

    def test_json_export_contains_selected_ideas(self):
        from aura_cli.commands import _generate_json_export

        idea = self._make_idea(desc="Top idea")
        session = self._make_session(ideas=[idea], selected=[idea])
        result = _generate_json_export(session, [idea], [idea])
        data = json.loads(result)
        assert len(data["selected_ideas"]) == 1
        assert data["selected_ideas"][0]["description"] == "Top idea"

    def test_csv_export_has_header(self):
        from aura_cli.commands import _generate_csv_export

        idea = self._make_idea()
        session = self._make_session(ideas=[idea])
        result = _generate_csv_export(session, [idea], [])
        lines = result.strip().split("\n")
        assert "Session ID" in lines[0]
        assert len(lines) >= 2  # header + at least one data row

    def test_csv_marks_selected_ideas(self):
        from aura_cli.commands import _generate_csv_export

        idea = self._make_idea(desc="Selected idea")
        session = self._make_session(ideas=[idea], selected=[idea])
        result = _generate_csv_export(session, [idea], [idea])
        assert "Yes" in result
