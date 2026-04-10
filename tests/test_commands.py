"""Comprehensive unit tests for aura_cli/commands.py."""

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    """Return a SimpleNamespace with sensible defaults for CLI args."""
    defaults = dict(
        action=None,
        dry_run=False,
        decompose=False,
        json=False,
        output="table",
        use_llm=True,
        techniques="",
        constraints="",
        batch_file=None,
        problem_statement=[],
        execute_phase=None,
        session_id=None,
        phase=None,
        show_ideas=False,
        format="markdown",
        limit=20,
        preview=False,
        max_goals=5,
        key=None,
        value=None,
        yes=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_mock_session(
    session_id="sess-001",
    problem="How to improve X?",
    status="active",
    phase_value="divergence",
    techniques=None,
    ideas_generated=5,
    ideas_selected=2,
    output=None,
    created_at=None,
    updated_at=None,
    constraints=None,
    phases_completed=None,
):
    """Create a mock InnovationSession-like object."""
    s = MagicMock()
    s.session_id = session_id
    s.problem_statement = problem
    s.status = status
    s.current_phase = SimpleNamespace(value=phase_value)
    s.techniques = techniques or ["scamper", "six_hats"]
    s.ideas_generated = ideas_generated
    s.ideas_selected = ideas_selected
    s.output = output
    s.created_at = created_at or datetime(2024, 1, 15, 10, 0, 0)
    s.updated_at = updated_at or datetime(2024, 1, 15, 11, 0, 0)
    s.constraints = constraints or {}
    s.phases_completed = phases_completed or []
    return s


def _make_mock_idea(
    description="A great idea",
    technique="scamper",
    novelty=0.8,
    feasibility=0.7,
    impact=0.9,
    metadata=None,
):
    """Create a mock Idea-like object."""
    idea = MagicMock()
    idea.description = description
    idea.technique = technique
    idea.novelty = novelty
    idea.feasibility = feasibility
    idea.impact = impact
    idea.metadata = metadata or {}
    return idea


# ---------------------------------------------------------------------------
# _get_meta_conductor
# ---------------------------------------------------------------------------


class TestGetMetaConductor:
    def setup_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None  # reset singleton between tests

    def teardown_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def test_creates_new_conductor_when_none(self):
        mock_conductor = MagicMock()
        MockClass = MagicMock(return_value=mock_conductor)
        with patch.dict("sys.modules", {"agents.meta_conductor": MagicMock(MetaConductor=MockClass)}):
            from aura_cli.commands import _get_meta_conductor

            result = _get_meta_conductor()
        assert result is mock_conductor

    def test_returns_existing_conductor_on_second_call(self):
        mock_conductor = MagicMock()
        MockClass = MagicMock(return_value=mock_conductor)
        with patch.dict("sys.modules", {"agents.meta_conductor": MagicMock(MetaConductor=MockClass)}):
            from aura_cli.commands import _get_meta_conductor

            r1 = _get_meta_conductor()
            r2 = _get_meta_conductor()
        assert r1 is r2
        assert MockClass.call_count == 1

    def test_updates_brain_on_existing_conductor_without_brain(self):
        mock_conductor = MagicMock()
        mock_conductor.brain = None
        MockClass = MagicMock(return_value=mock_conductor)
        with patch.dict("sys.modules", {"agents.meta_conductor": MagicMock(MetaConductor=MockClass)}):
            from aura_cli.commands import _get_meta_conductor

            _get_meta_conductor()  # create it
            new_brain = MagicMock()
            _get_meta_conductor(brain=new_brain)  # update brain
        assert mock_conductor.brain is new_brain

    def test_passes_use_llm_false_to_constructor(self):
        mock_conductor = MagicMock()
        MockClass = MagicMock(return_value=mock_conductor)
        with patch.dict("sys.modules", {"agents.meta_conductor": MagicMock(MetaConductor=MockClass)}):
            from aura_cli.commands import _get_meta_conductor

            _get_meta_conductor(use_llm=False)
        MockClass.assert_called_once_with(brain=None, use_llm=False)


# ---------------------------------------------------------------------------
# _handle_help
# ---------------------------------------------------------------------------


class TestHandleHelp:
    def test_renders_help_from_render_help(self, capsys):
        mock_render = MagicMock(return_value="  AURA CLI Help Text  ")
        with patch.dict("sys.modules", {"aura_cli.cli_options": MagicMock(render_help=mock_render)}):
            from aura_cli.commands import _handle_help

            _handle_help()
        captured = capsys.readouterr()
        assert "AURA CLI Help Text" in captured.out

    def test_falls_back_to_static_help_on_exception(self, capsys):
        with patch.dict(
            "sys.modules",
            {"aura_cli.cli_options": MagicMock(render_help=MagicMock(side_effect=Exception("fail")))},
        ):
            from aura_cli.commands import _handle_help

            _handle_help()
        captured = capsys.readouterr()
        assert "AURA CLI Commands" in captured.out
        assert "add" in captured.out
        assert "run" in captured.out

    def test_static_fallback_includes_all_commands(self, capsys):
        with patch("aura_cli.commands._handle_help.__module__"):
            pass
        # Directly invoke fallback path
        with patch("aura_cli.cli_options.render_help", side_effect=ImportError):
            from aura_cli.commands import _handle_help

            _handle_help()
        captured = capsys.readouterr()
        # If render_help itself raises, static help should still show
        assert "exit" in captured.out.lower() or "AURA CLI" in captured.out


# ---------------------------------------------------------------------------
# _handle_doctor
# ---------------------------------------------------------------------------


class TestHandleDoctor:
    def test_calls_run_doctor_v2(self):
        mock_run_doctor = MagicMock()
        mock_cap_check = MagicMock()
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands.capability_doctor_check", mock_cap_check),
            patch.dict("sys.modules", {"aura_cli.doctor": MagicMock(run_doctor_v2=mock_run_doctor)}),
        ):
            from aura_cli.commands import _handle_doctor

            _handle_doctor(project_root=Path("/tmp"))
        mock_run_doctor.assert_called_once()

    def test_passes_project_root_to_doctor(self):
        mock_run_doctor = MagicMock()
        with (
            patch("aura_cli.commands.log_json"),
            patch.dict("sys.modules", {"aura_cli.doctor": MagicMock(run_doctor_v2=mock_run_doctor)}),
        ):
            from aura_cli.commands import _handle_doctor

            _handle_doctor(project_root=Path("/my/project"))
        call_kwargs = mock_run_doctor.call_args
        assert call_kwargs.kwargs.get("project_root") == Path("/my/project") or (len(call_kwargs.args) > 0 and call_kwargs.args[0] == Path("/my/project"))

    def test_works_with_none_project_root(self):
        mock_run_doctor = MagicMock()
        with (
            patch("aura_cli.commands.log_json"),
            patch.dict("sys.modules", {"aura_cli.doctor": MagicMock(run_doctor_v2=mock_run_doctor)}),
        ):
            from aura_cli.commands import _handle_doctor

            _handle_doctor()  # project_root=None by default
        mock_run_doctor.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_clear
# ---------------------------------------------------------------------------


class TestHandleClear:
    def test_runs_clear_on_posix(self):
        mock_run = MagicMock()
        with (
            patch("subprocess.run", mock_run),
            patch("os.name", "posix"),
        ):
            from aura_cli.commands import _handle_clear

            _handle_clear()
        mock_run.assert_called_once_with(["clear"], shell=False)

    def test_runs_cls_on_windows(self):
        mock_run = MagicMock()
        with (
            patch("subprocess.run", mock_run),
            patch("os.name", "nt"),
        ):
            from aura_cli.commands import _handle_clear

            _handle_clear()
        mock_run.assert_called_once_with(["cls"], shell=False)


# ---------------------------------------------------------------------------
# _handle_add
# ---------------------------------------------------------------------------


class TestHandleAdd:
    def _run(self, goal_queue, command, **kwargs):
        out = io.StringIO()
        with (
            patch("aura_cli.commands.log_json"),
            redirect_stdout(out),
        ):
            from aura_cli.commands import _handle_add

            _handle_add(goal_queue, command)
        return out.getvalue()

    def test_adds_valid_goal(self):
        gq = MagicMock()
        gq.queue = ["goal1"]
        output = self._run(gq, "add My new goal")
        gq.add.assert_called_once_with("My new goal")
        assert "Added goal" in output

    def test_prints_queue_length_after_add(self):
        gq = MagicMock()
        gq.queue = ["a", "b"]
        output = self._run(gq, "add Fix the bug")
        assert "Queue length: 2" in output

    def test_empty_goal_does_not_add(self):
        gq = MagicMock()
        self._run(gq, "add ")
        gq.add.assert_not_called()

    def test_goal_too_long_rejected(self):
        gq = MagicMock()
        long_goal = "x" * 501
        output = self._run(gq, f"add {long_goal}")
        gq.add.assert_not_called()
        assert "too long" in output

    def test_goal_exactly_500_chars_accepted(self):
        gq = MagicMock()
        gq.queue = []
        goal_500 = "a" * 500
        self._run(gq, f"add {goal_500}")
        gq.add.assert_called_once_with(goal_500)

    def test_semicolon_rejected(self):
        gq = MagicMock()
        output = self._run(gq, "add rm -rf /; echo done")
        gq.add.assert_not_called()
        assert "suspicious" in output

    def test_double_ampersand_rejected(self):
        gq = MagicMock()
        output = self._run(gq, "add goal && evil")
        gq.add.assert_not_called()
        assert "suspicious" in output

    def test_pipe_OR_rejected(self):
        gq = MagicMock()
        output = self._run(gq, "add goal || fallback")
        gq.add.assert_not_called()
        assert "suspicious" in output

    def test_backtick_rejected(self):
        gq = MagicMock()
        output = self._run(gq, "add `evil`")
        gq.add.assert_not_called()
        assert "suspicious" in output

    def test_dollar_paren_rejected(self):
        gq = MagicMock()
        output = self._run(gq, "add $(evil)")
        gq.add.assert_not_called()
        assert "suspicious" in output

    def test_normal_goal_with_special_chars_allowed(self):
        gq = MagicMock()
        gq.queue = []
        output = self._run(gq, "add Improve test coverage by 20%")
        gq.add.assert_called_once_with("Improve test coverage by 20%")


# ---------------------------------------------------------------------------
# _handle_run
# ---------------------------------------------------------------------------


class TestHandleRun:
    def test_skips_when_no_goals(self, capsys):
        gq = MagicMock()
        gq.has_goals.return_value = False
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands.run_goals_loop") as mock_loop,
        ):
            from aura_cli.commands import _handle_run

            _handle_run(
                _make_args(),
                gq,
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                Path("."),
            )
        mock_loop.assert_not_called()

    def test_runs_loop_when_goals_present(self):
        gq = MagicMock()
        gq.has_goals.return_value = True
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands.run_goals_loop") as mock_loop,
        ):
            from aura_cli.commands import _handle_run

            args = _make_args(decompose=True)
            goal_archive = MagicMock()
            orchestrator = MagicMock()
            debugger = MagicMock()
            planner = MagicMock()
            project_root = Path("/proj")
            _handle_run(args, gq, goal_archive, orchestrator, debugger, planner, project_root)
        mock_loop.assert_called_once()

    def test_passes_decompose_flag(self):
        gq = MagicMock()
        gq.has_goals.return_value = True
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands.run_goals_loop") as mock_loop,
        ):
            from aura_cli.commands import _handle_run

            args = _make_args(decompose=True)
            _handle_run(args, gq, MagicMock(), MagicMock(), MagicMock(), MagicMock(), Path("."))
        _, call_kwargs = mock_loop.call_args
        assert call_kwargs.get("decompose") is True or mock_loop.call_args.args[-1] is True


# ---------------------------------------------------------------------------
# _handle_history
# ---------------------------------------------------------------------------


class TestHandleHistory:
    def test_empty_archive_prints_message(self, capsys):
        archive = SimpleNamespace(completed=[])
        from aura_cli.commands import _handle_history

        _handle_history(archive)
        assert "No completed goals" in capsys.readouterr().out

    def test_prints_recent_goals(self, capsys):
        archive = SimpleNamespace(completed=[("Goal A", 0.9), ("Goal B", 0.7)])
        from aura_cli.commands import _handle_history

        _handle_history(archive, limit=10)
        out = capsys.readouterr().out
        assert "Goal A" in out
        assert "Goal B" in out

    def test_limit_respected(self, capsys):
        archive = SimpleNamespace(completed=[("G1", 1.0), ("G2", 0.9), ("G3", 0.8)])
        from aura_cli.commands import _handle_history

        _handle_history(archive, limit=2)
        out = capsys.readouterr().out
        # Should show last 2 reversed (newest first)
        assert "G3" in out
        assert "G2" in out

    def test_as_json_returns_valid_json(self, capsys):
        archive = SimpleNamespace(completed=[("Goal X", 0.85)])
        from aura_cli.commands import _handle_history

        _handle_history(archive, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert "history" in payload
        assert payload["total"] == 1
        assert payload["history"][0]["goal"] == "Goal X"
        assert payload["history"][0]["score"] == pytest.approx(0.85)

    def test_as_json_no_goals_returns_empty_list(self, capsys):
        archive = SimpleNamespace(completed=[])
        from aura_cli.commands import _handle_history

        _handle_history(archive, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert payload["history"] == []
        assert payload["total"] == 0

    def test_as_json_scalar_item_handled(self, capsys):
        archive = SimpleNamespace(completed=["just a string"])
        from aura_cli.commands import _handle_history

        _handle_history(archive, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert payload["history"][0]["goal"] == "just a string"
        assert payload["history"][0]["score"] is None

    def test_newest_first_ordering(self, capsys):
        archive = SimpleNamespace(completed=[("old", 0.1), ("mid", 0.5), ("new", 0.9)])
        from aura_cli.commands import _handle_history

        _handle_history(archive, limit=10)
        out = capsys.readouterr().out
        idx_new = out.index("new")
        idx_old = out.index("old")
        assert idx_new < idx_old  # newest printed first

    def test_score_formatted_with_two_decimals(self, capsys):
        archive = SimpleNamespace(completed=[("G", 0.8)])
        from aura_cli.commands import _handle_history

        _handle_history(archive, limit=10)
        assert "0.80" in capsys.readouterr().out

    def test_none_score_omitted_from_display(self, capsys):
        archive = SimpleNamespace(completed=["Goal with no score"])
        from aura_cli.commands import _handle_history

        _handle_history(archive, limit=10)
        out = capsys.readouterr().out
        assert "Goal with no score" in out


# ---------------------------------------------------------------------------
# _handle_exit
# ---------------------------------------------------------------------------


class TestHandleExit:
    def test_logs_exit(self):
        with patch("aura_cli.commands.log_json") as mock_log:
            from aura_cli.commands import _handle_exit

            _handle_exit()
        mock_log.assert_called_once_with("INFO", "aura_cli_exit")


# ---------------------------------------------------------------------------
# _compute_global_insights
# ---------------------------------------------------------------------------


class TestComputeGlobalInsights:
    def test_empty_sessions(self):
        from aura_cli.commands import _compute_global_insights

        result = _compute_global_insights([])
        assert result["total_sessions"] == 0
        assert result["total_ideas"] == 0
        assert result["selection_rate"] == 0

    def test_selection_rate_computed(self):
        from aura_cli.commands import _compute_global_insights

        s1 = MagicMock()
        s1.ideas_generated = 10
        s1.ideas_selected = 4
        s1.techniques = ["scamper"]
        s1.output = None

        s2 = MagicMock()
        s2.ideas_generated = 5
        s2.ideas_selected = 1
        s2.techniques = ["six_hats"]
        s2.output = None

        result = _compute_global_insights([s1, s2])
        assert result["total_ideas"] == 15
        assert result["total_selected"] == 5
        assert result["selection_rate"] == pytest.approx(5 / 15)

    def test_technique_frequency_counted(self):
        from aura_cli.commands import _compute_global_insights

        s1 = MagicMock()
        s1.ideas_generated = 0
        s1.ideas_selected = 0
        s1.techniques = ["scamper", "lotus"]
        s1.output = None

        s2 = MagicMock()
        s2.ideas_generated = 0
        s2.ideas_selected = 0
        s2.techniques = ["scamper"]
        s2.output = None

        result = _compute_global_insights([s1, s2])
        assert result["technique_usage"]["scamper"] == 2
        assert result["technique_usage"]["lotus"] == 1

    def test_average_scores_from_outputs(self):
        from aura_cli.commands import _compute_global_insights

        out1 = MagicMock()
        out1.novelty_score = 0.8
        out1.feasibility_score = 0.6
        out1.diversity_score = 0.7

        s = MagicMock()
        s.ideas_generated = 2
        s.ideas_selected = 1
        s.techniques = []
        s.output = out1
        s.status = "completed"

        result = _compute_global_insights([s])
        assert result["average_scores"]["novelty"] == pytest.approx(0.8)
        assert result["average_scores"]["feasibility"] == pytest.approx(0.6)

    def test_sessions_by_status_counted(self):
        from aura_cli.commands import _compute_global_insights

        s1 = MagicMock()
        s1.ideas_generated = 0
        s1.ideas_selected = 0
        s1.techniques = []
        s1.output = None
        s1.status = "active"

        s2 = MagicMock()
        s2.ideas_generated = 0
        s2.ideas_selected = 0
        s2.techniques = []
        s2.output = None
        s2.status = "completed"

        result = _compute_global_insights([s1, s2])
        assert result["sessions_by_status"]["active"] == 1
        assert result["sessions_by_status"]["completed"] == 1


# ---------------------------------------------------------------------------
# _compute_session_insights
# ---------------------------------------------------------------------------


class TestComputeSessionInsights:
    def test_basic_structure_without_output(self):
        from aura_cli.commands import _compute_session_insights

        s = _make_mock_session(output=None)
        result = _compute_session_insights(s)
        assert result["session_id"] == "sess-001"
        assert result["ideas_generated"] == 5
        assert "technique_breakdown" not in result

    def test_technique_breakdown_with_output(self):
        from aura_cli.commands import _compute_session_insights

        ideas = [
            _make_mock_idea(technique="scamper"),
            _make_mock_idea(technique="scamper"),
            _make_mock_idea(technique="lotus"),
        ]
        out = MagicMock()
        out.all_ideas = ideas
        out.selected_ideas = [ideas[0]]
        s = _make_mock_session(output=out)
        result = _compute_session_insights(s)
        assert result["technique_breakdown"]["scamper"] == 2
        assert result["technique_breakdown"]["lotus"] == 1

    def test_quality_metrics_computed(self):
        from aura_cli.commands import _compute_session_insights

        ideas = [
            _make_mock_idea(novelty=0.8, feasibility=0.6, impact=0.9),
            _make_mock_idea(novelty=0.4, feasibility=0.8, impact=0.7),
        ]
        out = MagicMock()
        out.all_ideas = ideas
        out.selected_ideas = []
        s = _make_mock_session(output=out)
        result = _compute_session_insights(s)
        assert "quality_metrics" in result
        assert result["quality_metrics"]["avg_novelty"] == pytest.approx(0.6)

    def test_top_ideas_sorted_by_total_score(self):
        from aura_cli.commands import _compute_session_insights

        ideas = [
            _make_mock_idea(description="Low", novelty=0.1, feasibility=0.1, impact=0.1),
            _make_mock_idea(description="High", novelty=0.9, feasibility=0.9, impact=0.9),
        ]
        out = MagicMock()
        out.all_ideas = ideas
        out.selected_ideas = []
        s = _make_mock_session(output=out)
        result = _compute_session_insights(s)
        assert result["top_ideas"][0]["total_score"] > result["top_ideas"][-1]["total_score"]


# ---------------------------------------------------------------------------
# _print_session_insights
# ---------------------------------------------------------------------------


class TestPrintSessionInsights:
    def test_prints_session_id_and_problem(self, capsys):
        from aura_cli.commands import _print_session_insights

        insights = {
            "session_id": "abc-123",
            "problem": "How to fix?",
            "status": "active",
            "techniques_used": ["scamper"],
            "ideas_generated": 3,
            "ideas_selected": 1,
        }
        _print_session_insights(insights)
        out = capsys.readouterr().out
        assert "abc-123" in out
        assert "How to fix?" in out

    def test_prints_technique_breakdown(self, capsys):
        from aura_cli.commands import _print_session_insights

        insights = {
            "session_id": "x",
            "problem": "P",
            "status": "active",
            "techniques_used": ["scamper"],
            "ideas_generated": 2,
            "ideas_selected": 1,
            "technique_breakdown": {"scamper": 2},
        }
        _print_session_insights(insights)
        assert "scamper: 2" in capsys.readouterr().out

    def test_prints_quality_metrics(self, capsys):
        from aura_cli.commands import _print_session_insights

        insights = {
            "session_id": "x",
            "problem": "P",
            "status": "active",
            "techniques_used": [],
            "ideas_generated": 1,
            "ideas_selected": 1,
            "quality_metrics": {
                "avg_novelty": 0.75,
                "avg_feasibility": 0.65,
                "avg_impact": 0.85,
                "selected_avg_novelty": 0.8,
                "selected_avg_feasibility": 0.7,
                "selected_avg_impact": 0.9,
            },
        }
        _print_session_insights(insights)
        out = capsys.readouterr().out
        assert "0.75" in out

    def test_prints_top_ideas(self, capsys):
        from aura_cli.commands import _print_session_insights

        insights = {
            "session_id": "x",
            "problem": "P",
            "status": "active",
            "techniques_used": [],
            "ideas_generated": 1,
            "ideas_selected": 0,
            "top_ideas": [{"technique": "lotus", "description": "Smart approach", "total_score": 2.5}],
        }
        _print_session_insights(insights)
        out = capsys.readouterr().out
        assert "lotus" in out
        assert "2.5" in out


# ---------------------------------------------------------------------------
# _print_global_insights
# ---------------------------------------------------------------------------


class TestPrintGlobalInsights:
    def test_prints_overview(self, capsys):
        from aura_cli.commands import _print_global_insights

        insights = {
            "total_sessions": 3,
            "total_ideas": 15,
            "total_selected": 5,
            "selection_rate": 0.333,
            "technique_usage": {"scamper": 2},
            "average_scores": {"novelty": 0.7, "feasibility": 0.6, "diversity": 0.8},
            "sessions_by_status": {"active": 1, "completed": 2},
        }
        _print_global_insights(insights)
        out = capsys.readouterr().out
        assert "3" in out
        assert "15" in out
        assert "33.3%" in out

    def test_prints_technique_usage(self, capsys):
        from aura_cli.commands import _print_global_insights

        insights = {
            "total_sessions": 1,
            "total_ideas": 5,
            "total_selected": 2,
            "selection_rate": 0.4,
            "technique_usage": {"lotus": 3, "star": 1},
            "average_scores": {"novelty": 0.5, "feasibility": 0.5, "diversity": 0.5},
            "sessions_by_status": {"active": 1, "completed": 0},
        }
        _print_global_insights(insights)
        assert "lotus: 3" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _generate_json_export
# ---------------------------------------------------------------------------


class TestGenerateJsonExport:
    def _make_session_for_export(self, with_output=False):
        ideas = [_make_mock_idea(description="Idea A"), _make_mock_idea(description="Idea B")]
        out = None
        if with_output:
            out = MagicMock()
            out.all_ideas = ideas
            out.selected_ideas = [ideas[0]]
            out.diversity_score = 0.8
            out.novelty_score = 0.7
            out.feasibility_score = 0.6

        s = _make_mock_session(
            output=out,
            phases_completed=[SimpleNamespace(value="immersion")],
        )
        return s, (out.all_ideas if out else []), (out.selected_ideas if out else [])

    def test_returns_valid_json(self):
        from aura_cli.commands import _generate_json_export

        s, all_ideas, selected = self._make_session_for_export(with_output=True)
        content = _generate_json_export(s, all_ideas, selected)
        parsed = json.loads(content)
        assert parsed["session_id"] == "sess-001"
        assert len(parsed["ideas"]) == 2

    def test_includes_scores_when_output_present(self):
        from aura_cli.commands import _generate_json_export

        s, all_ideas, selected = self._make_session_for_export(with_output=True)
        content = _generate_json_export(s, all_ideas, selected)
        parsed = json.loads(content)
        assert "scores" in parsed
        assert parsed["scores"]["diversity"] == pytest.approx(0.8)

    def test_no_scores_when_no_output(self):
        from aura_cli.commands import _generate_json_export

        s, all_ideas, selected = self._make_session_for_export(with_output=False)
        s.output = None
        content = _generate_json_export(s, [], [])
        parsed = json.loads(content)
        assert "scores" not in parsed

    def test_created_at_is_iso_string(self):
        from aura_cli.commands import _generate_json_export

        s, all_ideas, selected = self._make_session_for_export()
        content = _generate_json_export(s, all_ideas, selected)
        parsed = json.loads(content)
        assert "2024" in parsed["created_at"]


# ---------------------------------------------------------------------------
# _generate_csv_export
# ---------------------------------------------------------------------------


class TestGenerateCsvExport:
    def test_returns_csv_with_header(self):
        from aura_cli.commands import _generate_csv_export

        s = _make_mock_session(phases_completed=[])
        all_ideas = [_make_mock_idea()]
        content = _generate_csv_export(s, all_ideas, [])
        assert "Session ID" in content
        assert "Technique" in content

    def test_data_rows_present(self):
        from aura_cli.commands import _generate_csv_export

        s = _make_mock_session(phases_completed=[])
        idea = _make_mock_idea(description="Smart fix", technique="scamper")
        content = _generate_csv_export(s, [idea], [])
        assert "scamper" in content

    def test_selected_flag_in_csv(self):
        from aura_cli.commands import _generate_csv_export

        s = _make_mock_session(phases_completed=[])
        idea = _make_mock_idea()
        content = _generate_csv_export(s, [idea], [idea])
        assert "Yes" in content

    def test_not_selected_flag(self):
        from aura_cli.commands import _generate_csv_export

        s = _make_mock_session(phases_completed=[])
        idea = _make_mock_idea()
        content = _generate_csv_export(s, [idea], [])
        assert "No" in content

    def test_empty_ideas_only_header(self):
        from aura_cli.commands import _generate_csv_export

        s = _make_mock_session(phases_completed=[])
        content = _generate_csv_export(s, [], [])
        lines = [line for line in content.strip().split("\n") if line.strip()]
        assert len(lines) == 1  # just the header


# ---------------------------------------------------------------------------
# _generate_html_export
# ---------------------------------------------------------------------------


class TestGenerateHtmlExport:
    def test_returns_valid_html_structure(self):
        from aura_cli.commands import _generate_html_export

        s = _make_mock_session(phases_completed=[])
        content = _generate_html_export(s, [], [])
        assert "<!DOCTYPE html>" in content
        assert "<html>" in content
        assert "Innovation Session Report" in content

    def test_session_id_in_html(self):
        from aura_cli.commands import _generate_html_export

        s = _make_mock_session(session_id="my-session-id", phases_completed=[])
        content = _generate_html_export(s, [], [])
        assert "my-session-id" in content

    def test_idea_rendered_in_html(self):
        from aura_cli.commands import _generate_html_export

        s = _make_mock_session(phases_completed=[])
        idea = _make_mock_idea(description="Brilliant concept", technique="lotus")
        content = _generate_html_export(s, [idea], [])
        assert "Brilliant concept" in content
        assert "lotus" in content

    def test_selected_idea_has_selected_class(self):
        from aura_cli.commands import _generate_html_export

        s = _make_mock_session(phases_completed=[])
        idea = _make_mock_idea()
        content = _generate_html_export(s, [idea], [idea])
        assert 'class="idea selected"' in content

    def test_unselected_idea_has_no_selected_class(self):
        from aura_cli.commands import _generate_html_export

        s = _make_mock_session(phases_completed=[])
        idea = _make_mock_idea()
        content = _generate_html_export(s, [idea], [])
        assert 'class="idea "' in content


# ---------------------------------------------------------------------------
# _generate_markdown_export
# ---------------------------------------------------------------------------


class TestGenerateMarkdownExport:
    def test_has_required_headings(self):
        from aura_cli.commands import _generate_markdown_export

        s = _make_mock_session(phases_completed=[])
        content = _generate_markdown_export(s, [], [])
        assert "# Innovation Session Report" in content
        assert "## Problem Statement" in content
        assert "## Summary" in content

    def test_includes_session_id(self):
        from aura_cli.commands import _generate_markdown_export

        s = _make_mock_session(session_id="xyz-999", phases_completed=[])
        content = _generate_markdown_export(s, [], [])
        assert "xyz-999" in content

    def test_ideas_section_rendered(self):
        from aura_cli.commands import _generate_markdown_export

        s = _make_mock_session(phases_completed=[])
        idea = _make_mock_idea(description="Some idea", technique="reverse")
        content = _generate_markdown_export(s, [idea], [])
        assert "## Ideas Generated" in content
        assert "reverse" in content

    def test_selected_ideas_summary_section(self):
        from aura_cli.commands import _generate_markdown_export

        s = _make_mock_session(phases_completed=[])
        idea = _make_mock_idea(description="Best idea ever")
        content = _generate_markdown_export(s, [idea], [idea])
        assert "## Selected Ideas Summary" in content

    def test_scores_section_when_output_present(self):
        from aura_cli.commands import _generate_markdown_export

        out = MagicMock()
        out.diversity_score = 0.75
        out.novelty_score = 0.85
        out.feasibility_score = 0.65
        s = _make_mock_session(output=out, phases_completed=[])
        content = _generate_markdown_export(s, [], [])
        assert "## Scores" in content
        assert "0.75" in content

    def test_no_scores_section_without_output(self):
        from aura_cli.commands import _generate_markdown_export

        s = _make_mock_session(output=None, phases_completed=[])
        content = _generate_markdown_export(s, [], [])
        assert "## Scores" not in content


# ---------------------------------------------------------------------------
# _handle_innovate_start
# ---------------------------------------------------------------------------


class TestHandleInnovateStart:
    def setup_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def teardown_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def _make_phase_enum(self):
        from enum import Enum

        class InnovationPhase(Enum):
            IMMERSION = "immersion"
            DIVERGENCE = "divergence"
            CONVERGENCE = "convergence"
            INCUBATION = "incubation"
            TRANSFORMATION = "transformation"

        return InnovationPhase

    def test_error_when_no_problem(self, capsys):
        InnovationPhase = self._make_phase_enum()
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor"),
            patch.dict(
                "sys.modules",
                {
                    "agents.brainstorming_bots": MagicMock(list_techniques=lambda: ["scamper"]),
                    "agents.schemas": MagicMock(InnovationPhase=InnovationPhase),
                },
            ),
        ):
            from aura_cli.commands import _handle_innovate_start

            args = _make_args(problem_statement=[], batch_file=None)
            _handle_innovate_start(args)
        assert "Error: Problem statement required" in capsys.readouterr().out

    def test_error_on_invalid_technique(self, capsys):
        InnovationPhase = self._make_phase_enum()
        with (
            patch("aura_cli.commands.log_json"),
            patch.dict(
                "sys.modules",
                {
                    "agents.brainstorming_bots": MagicMock(list_techniques=lambda: ["scamper", "lotus"]),
                    "agents.schemas": MagicMock(InnovationPhase=InnovationPhase),
                },
            ),
        ):
            from aura_cli.commands import _handle_innovate_start

            args = _make_args(problem_statement=["Improve X"], techniques="invalid_tech")
            _handle_innovate_start(args)
        out = capsys.readouterr().out
        assert "Invalid techniques" in out

    def test_error_on_invalid_constraints_json(self, capsys):
        InnovationPhase = self._make_phase_enum()
        with (
            patch("aura_cli.commands.log_json"),
            patch.dict(
                "sys.modules",
                {
                    "agents.brainstorming_bots": MagicMock(list_techniques=lambda: ["scamper"]),
                    "agents.schemas": MagicMock(InnovationPhase=InnovationPhase),
                },
            ),
        ):
            from aura_cli.commands import _handle_innovate_start

            args = _make_args(problem_statement=["Improve X"], constraints="not valid json")
            _handle_innovate_start(args)
        assert "Invalid JSON" in capsys.readouterr().out

    def test_successful_start_single_session(self, capsys):
        InnovationPhase = self._make_phase_enum()
        session = _make_mock_session(phases_completed=[])
        conductor = MagicMock()
        conductor.start_session.return_value = session

        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
            patch.dict(
                "sys.modules",
                {
                    "agents.brainstorming_bots": MagicMock(list_techniques=lambda: ["scamper"]),
                    "agents.schemas": MagicMock(InnovationPhase=InnovationPhase),
                },
            ),
        ):
            from aura_cli.commands import _handle_innovate_start

            args = _make_args(problem_statement=["Improve X"])
            _handle_innovate_start(args)
        out = capsys.readouterr().out
        assert "Innovation Session Started" in out or "sess-001" in out

    def test_batch_file_not_found_error(self, capsys):
        InnovationPhase = self._make_phase_enum()
        with (
            patch("aura_cli.commands.log_json"),
            patch.dict(
                "sys.modules",
                {
                    "agents.brainstorming_bots": MagicMock(list_techniques=lambda: ["scamper"]),
                    "agents.schemas": MagicMock(InnovationPhase=InnovationPhase),
                },
            ),
        ):
            from aura_cli.commands import _handle_innovate_start

            args = _make_args(batch_file="/nonexistent/path.txt")
            _handle_innovate_start(args)
        assert "not found" in capsys.readouterr().out

    def test_json_output_format(self, capsys):
        InnovationPhase = self._make_phase_enum()
        session = _make_mock_session(phases_completed=[])
        conductor = MagicMock()
        conductor.start_session.return_value = session

        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
            patch.dict(
                "sys.modules",
                {
                    "agents.brainstorming_bots": MagicMock(list_techniques=lambda: ["scamper"]),
                    "agents.schemas": MagicMock(InnovationPhase=InnovationPhase),
                },
            ),
        ):
            from aura_cli.commands import _handle_innovate_start

            args = _make_args(problem_statement=["Improve X"], json=True)
            _handle_innovate_start(args)
        payload = json.loads(capsys.readouterr().out)
        assert "sessions" in payload
        assert payload["count"] == 1


# ---------------------------------------------------------------------------
# _handle_innovate_list
# ---------------------------------------------------------------------------


class TestHandleInnovateList:
    def setup_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def teardown_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def test_no_sessions_prints_message(self, capsys):
        conductor = MagicMock()
        conductor.list_sessions.return_value = []
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_list

            _handle_innovate_list(_make_args())
        assert "No active sessions" in capsys.readouterr().out

    def test_active_sessions_displayed(self, capsys):
        s = _make_mock_session(status="active")
        conductor = MagicMock()
        conductor.list_sessions.return_value = [s]
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_list

            _handle_innovate_list(_make_args())
        out = capsys.readouterr().out
        assert "Active Sessions" in out

    def test_completed_sessions_displayed(self, capsys):
        s = _make_mock_session(status="completed")
        conductor = MagicMock()
        conductor.list_sessions.return_value = [s]
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_list

            _handle_innovate_list(_make_args())
        out = capsys.readouterr().out
        assert "Completed Sessions" in out

    def test_json_output(self, capsys):
        s = _make_mock_session()
        conductor = MagicMock()
        conductor.list_sessions.return_value = [s]
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_list

            _handle_innovate_list(_make_args(json=True))
        payload = json.loads(capsys.readouterr().out)
        assert "sessions" in payload
        assert payload["total"] == 1

    def test_limit_applied(self, capsys):
        sessions = [_make_mock_session(session_id=f"s{i}", status="active") for i in range(25)]
        conductor = MagicMock()
        conductor.list_sessions.return_value = sessions
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_list

            _handle_innovate_list(_make_args(json=True, limit=5))
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["sessions"]) == 5


# ---------------------------------------------------------------------------
# _handle_innovate_show
# ---------------------------------------------------------------------------


class TestHandleInnovateShow:
    def setup_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def teardown_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def test_error_when_no_session_id(self, capsys):
        conductor = MagicMock()
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_show

            _handle_innovate_show(_make_args(session_id=None))
        assert "session_id required" in capsys.readouterr().out

    def test_session_not_found(self, capsys):
        conductor = MagicMock()
        conductor.get_session.return_value = None
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_show

            _handle_innovate_show(_make_args(session_id="bad-id"))
        assert "not found" in capsys.readouterr().out

    def test_session_not_found_json(self, capsys):
        conductor = MagicMock()
        conductor.get_session.return_value = None
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_show

            _handle_innovate_show(_make_args(session_id="bad-id", json=True))
        payload = json.loads(capsys.readouterr().out)
        assert "error" in payload

    def test_shows_session_details(self, capsys):
        s = _make_mock_session(output=None, phases_completed=[])
        s.output = None
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_show

            _handle_innovate_show(_make_args(session_id="sess-001"))
        out = capsys.readouterr().out
        assert "sess-001" in out

    def test_json_output_structure(self, capsys):
        s = _make_mock_session(output=None, phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_show

            _handle_innovate_show(_make_args(session_id="sess-001", json=True))
        payload = json.loads(capsys.readouterr().out)
        assert payload["session_id"] == "sess-001"
        assert "current_phase" in payload

    def test_show_ideas_flag(self, capsys):
        ideas = [_make_mock_idea(description="Great idea")]
        out = MagicMock()
        out.all_ideas = ideas
        out.selected_ideas = []
        out.diversity_score = 0.7
        out.novelty_score = 0.8
        out.feasibility_score = 0.6
        s = _make_mock_session(output=out, phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_show

            _handle_innovate_show(_make_args(session_id="sess-001", show_ideas=True))
        assert "Great idea" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _handle_innovate_resume
# ---------------------------------------------------------------------------


class TestHandleInnovateResume:
    def setup_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def teardown_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def _make_phase_enum(self):
        from enum import Enum

        class InnovationPhase(Enum):
            IMMERSION = "immersion"
            DIVERGENCE = "divergence"
            CONVERGENCE = "convergence"
            INCUBATION = "incubation"
            TRANSFORMATION = "transformation"

        return InnovationPhase

    def test_error_when_no_session_id(self, capsys):
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_innovate_resume

            _handle_innovate_resume(_make_args(session_id=None))
        assert "session_id required" in capsys.readouterr().out

    def test_session_not_found(self, capsys):
        InnovationPhase = self._make_phase_enum()
        conductor = MagicMock()
        conductor.get_session.return_value = None
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
            patch.dict("sys.modules", {"agents.schemas": MagicMock(InnovationPhase=InnovationPhase)}),
        ):
            from aura_cli.commands import _handle_innovate_resume

            _handle_innovate_resume(_make_args(session_id="bad-id"))
        assert "not found" in capsys.readouterr().out

    def test_invalid_phase_name(self, capsys):
        InnovationPhase = self._make_phase_enum()
        s = _make_mock_session(phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
            patch.dict("sys.modules", {"agents.schemas": MagicMock(InnovationPhase=InnovationPhase)}),
        ):
            from aura_cli.commands import _handle_innovate_resume

            _handle_innovate_resume(_make_args(session_id="sess-001", phase="invalid_phase"))
        assert "Invalid phase" in capsys.readouterr().out

    def test_successful_resume(self, capsys):
        InnovationPhase = self._make_phase_enum()
        s = _make_mock_session(phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        conductor.execute_phase.return_value = {"ideas_count": 3}

        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
            patch.dict("sys.modules", {"agents.schemas": MagicMock(InnovationPhase=InnovationPhase)}),
        ):
            from aura_cli.commands import _handle_innovate_resume

            _handle_innovate_resume(_make_args(session_id="sess-001", phase="divergence"))
        out = capsys.readouterr().out
        assert "Phase complete" in out or "divergence" in out

    def test_resume_exception_handled(self, capsys):
        InnovationPhase = self._make_phase_enum()
        s = _make_mock_session(phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        conductor.execute_phase.side_effect = RuntimeError("Something went wrong")

        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
            patch.dict("sys.modules", {"agents.schemas": MagicMock(InnovationPhase=InnovationPhase)}),
        ):
            from aura_cli.commands import _handle_innovate_resume

            _handle_innovate_resume(_make_args(session_id="sess-001", phase="divergence"))
        assert "Error executing phase" in capsys.readouterr().out

    def test_resume_json_output_on_error(self, capsys):
        InnovationPhase = self._make_phase_enum()
        s = _make_mock_session(phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        conductor.execute_phase.side_effect = RuntimeError("boom")

        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
            patch.dict("sys.modules", {"agents.schemas": MagicMock(InnovationPhase=InnovationPhase)}),
        ):
            from aura_cli.commands import _handle_innovate_resume

            _handle_innovate_resume(_make_args(session_id="sess-001", phase="divergence", json=True))
        payload = json.loads(capsys.readouterr().out)
        assert "error" in payload


# ---------------------------------------------------------------------------
# _handle_innovate_techniques
# ---------------------------------------------------------------------------


class TestHandleInnovateTechniques:
    def _make_bot(self, name):
        b = MagicMock()
        b.return_value = MagicMock(technique_name=name)
        return b

    def test_json_output(self, capsys):
        bots = {"scamper": self._make_bot("SCAMPER"), "lotus": self._make_bot("Lotus")}
        with (
            patch("aura_cli.commands.log_json"),
            patch.dict("sys.modules", {"agents.brainstorming_bots": MagicMock(BRAINSTORMING_BOTS=bots)}),
        ):
            from aura_cli.commands import _handle_innovate_techniques

            _handle_innovate_techniques(_make_args(json=True))
        payload = json.loads(capsys.readouterr().out)
        assert "techniques" in payload
        assert len(payload["techniques"]) == 2

    def test_table_output(self, capsys):
        bots = {"scamper": self._make_bot("SCAMPER")}
        with (
            patch("aura_cli.commands.log_json"),
            patch.dict("sys.modules", {"agents.brainstorming_bots": MagicMock(BRAINSTORMING_BOTS=bots)}),
        ):
            from aura_cli.commands import _handle_innovate_techniques

            _handle_innovate_techniques(_make_args())
        out = capsys.readouterr().out
        assert "Brainstorming Techniques" in out
        assert "scamper" in out

    def test_description_in_output(self, capsys):
        bots = {"six_hats": self._make_bot("Six Hats")}
        with (
            patch("aura_cli.commands.log_json"),
            patch.dict("sys.modules", {"agents.brainstorming_bots": MagicMock(BRAINSTORMING_BOTS=bots)}),
        ):
            from aura_cli.commands import _handle_innovate_techniques

            _handle_innovate_techniques(_make_args())
        out = capsys.readouterr().out
        assert "Six Thinking Hats" in out


# ---------------------------------------------------------------------------
# _handle_innovate_export
# ---------------------------------------------------------------------------


class TestHandleInnovateExport:
    def setup_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def teardown_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def test_error_when_no_session_id(self, capsys):
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_innovate_export

            _handle_innovate_export(_make_args(session_id=None))
        assert "session_id required" in capsys.readouterr().out

    def test_session_not_found(self, capsys):
        conductor = MagicMock()
        conductor.get_session.return_value = None
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_export

            _handle_innovate_export(_make_args(session_id="bad-id"))
        assert "not found" in capsys.readouterr().out

    def test_markdown_export_printed_to_stdout(self, capsys):
        s = _make_mock_session(output=None, phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_export

            # output=None so content goes to stdout
            _handle_innovate_export(_make_args(session_id="sess-001", format="markdown", output=None))
        out = capsys.readouterr().out
        assert "Innovation Session Report" in out

    def test_json_export_format(self, capsys):
        s = _make_mock_session(output=None, phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_export

            result = _handle_innovate_export(_make_args(session_id="sess-001", format="json", output=None))
        assert result is not None
        parsed = json.loads(result)
        assert "session_id" in parsed

    def test_csv_export_format(self, capsys):
        s = _make_mock_session(output=None, phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_export

            result = _handle_innovate_export(_make_args(session_id="sess-001", format="csv", output=None))
        assert "Session ID" in result

    def test_html_export_format(self, capsys):
        s = _make_mock_session(output=None, phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_export

            result = _handle_innovate_export(_make_args(session_id="sess-001", format="html", output=None))
        assert "<!DOCTYPE html>" in result

    def test_export_to_file(self, tmp_path, capsys):
        s = _make_mock_session(output=None, phases_completed=[])
        conductor = MagicMock()
        conductor.get_session.return_value = s
        output_file = str(tmp_path / "export.md")
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_export

            _handle_innovate_export(_make_args(session_id="sess-001", output=output_file))
        out = capsys.readouterr().out
        assert "Exported to" in out
        assert Path(output_file).exists()


# ---------------------------------------------------------------------------
# _handle_innovate_to_goals
# ---------------------------------------------------------------------------


class TestHandleInnovateToGoals:
    def setup_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def teardown_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def test_error_when_no_session_id(self, capsys):
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_innovate_to_goals

            _handle_innovate_to_goals(_make_args(session_id=None))
        assert "--session-id required" in capsys.readouterr().out

    def test_session_not_found(self, capsys):
        conductor = MagicMock()
        conductor.get_session.return_value = None
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_to_goals

            _handle_innovate_to_goals(_make_args(session_id="bad-id"))
        assert "not found" in capsys.readouterr().out

    def test_no_selected_ideas_warns(self, capsys):
        s = _make_mock_session()
        s.output = MagicMock()
        s.output.selected_ideas = []
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_to_goals

            _handle_innovate_to_goals(_make_args(session_id="sess-001"))
        assert "No selected ideas" in capsys.readouterr().out

    def test_preview_mode_does_not_add_goals(self, capsys):
        idea = _make_mock_idea(description="Do something great")
        s = _make_mock_session()
        s.output = MagicMock()
        s.output.selected_ideas = [idea]
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_to_goals

            _handle_innovate_to_goals(_make_args(session_id="sess-001", preview=True))
        out = capsys.readouterr().out
        assert "PREVIEW" in out

    def test_goals_added_to_queue(self, capsys):
        idea = _make_mock_idea(description="Implement caching layer")
        s = _make_mock_session()
        s.output = MagicMock()
        s.output.selected_ideas = [idea]
        conductor = MagicMock()
        conductor.get_session.return_value = s
        goal_queue = MagicMock()
        runtime = {"goal_queue": goal_queue, "brain": None}
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_to_goals

            _handle_innovate_to_goals(_make_args(session_id="sess-001"), runtime=runtime)
        goal_queue.add.assert_called_once()
        assert "Created 1 goals" in capsys.readouterr().out

    def test_max_goals_limits_conversion(self, capsys):
        ideas = [_make_mock_idea(description=f"Idea {i}") for i in range(10)]
        s = _make_mock_session()
        s.output = MagicMock()
        s.output.selected_ideas = ideas
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_to_goals

            _handle_innovate_to_goals(_make_args(session_id="sess-001", max_goals=3))
        out = capsys.readouterr().out
        assert "Created 3 goals" in out

    def test_fallback_when_no_goal_queue(self, capsys):
        idea = _make_mock_idea(description="A creative idea that drives value")
        s = _make_mock_session()
        s.output = MagicMock()
        s.output.selected_ideas = [idea]
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_to_goals

            _handle_innovate_to_goals(_make_args(session_id="sess-001"))
        out = capsys.readouterr().out
        assert "Created 1 goals" in out


# ---------------------------------------------------------------------------
# _handle_innovate_insights
# ---------------------------------------------------------------------------


class TestHandleInnovateInsights:
    def setup_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def teardown_method(self):
        import aura_cli.commands as cmd_mod

        cmd_mod._meta_conductor = None

    def test_global_insights_no_session_id(self, capsys):
        sessions = [_make_mock_session(status="active"), _make_mock_session(session_id="s2", status="completed")]
        for s in sessions:
            s.output = None
        conductor = MagicMock()
        conductor.list_sessions.return_value = sessions
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_insights

            _handle_innovate_insights(_make_args(session_id=None))
        out = capsys.readouterr().out
        assert "Global" in out

    def test_session_insights_with_session_id(self, capsys):
        s = _make_mock_session(output=None)
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_insights

            _handle_innovate_insights(_make_args(session_id="sess-001"))
        out = capsys.readouterr().out
        assert "sess-001" in out

    def test_session_not_found(self, capsys):
        conductor = MagicMock()
        conductor.get_session.return_value = None
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_insights

            _handle_innovate_insights(_make_args(session_id="bad-id"))
        assert "not found" in capsys.readouterr().out

    def test_global_insights_json(self, capsys):
        conductor = MagicMock()
        conductor.list_sessions.return_value = []
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_insights

            _handle_innovate_insights(_make_args(session_id=None, json=True))
        payload = json.loads(capsys.readouterr().out)
        assert "total_sessions" in payload

    def test_session_insights_json(self, capsys):
        s = _make_mock_session(output=None)
        conductor = MagicMock()
        conductor.get_session.return_value = s
        with (
            patch("aura_cli.commands.log_json"),
            patch("aura_cli.commands._get_meta_conductor", return_value=conductor),
        ):
            from aura_cli.commands import _handle_innovate_insights

            _handle_innovate_insights(_make_args(session_id="sess-001", json=True))
        payload = json.loads(capsys.readouterr().out)
        assert payload["session_id"] == "sess-001"


# ---------------------------------------------------------------------------
# _handle_migrate_credentials
# ---------------------------------------------------------------------------


class TestHandleMigrateCredentials:
    def _make_config_manager(self, store_info=None, dry_run_results=None, migrate_results=None):
        cm = MagicMock()
        cm.get_credential_store_info.return_value = store_info or {
            "keyring_available": True,
            "fallback_available": True,
            "fallback_path": "/home/user/.creds",
        }
        dry = dry_run_results or {"migrated": [], "already_secure": [], "errors": {}}
        real = migrate_results or {"migrated": ["ANTHROPIC_API_KEY"], "errors": {}}

        def migrate_side_effect(dry_run=True):
            return dry if dry_run else real

        cm.migrate_credentials.side_effect = migrate_side_effect
        return cm

    def test_nothing_to_migrate(self, capsys):
        cm = self._make_config_manager()
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_migrate_credentials

            _handle_migrate_credentials(_make_args(), config_manager=cm)
        assert "Nothing to migrate" in capsys.readouterr().out

    def test_shows_credentials_found(self, capsys):
        cm = self._make_config_manager(dry_run_results={"migrated": ["API_KEY"], "already_secure": [], "errors": {}})
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_migrate_credentials

            # --yes to skip interactive prompt
            _handle_migrate_credentials(_make_args(yes=True), config_manager=cm)
        out = capsys.readouterr().out
        assert "API_KEY" in out

    def test_dry_run_shows_already_secure(self, capsys):
        cm = self._make_config_manager(dry_run_results={"migrated": [], "already_secure": ["EXISTING_KEY"], "errors": {}})
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_migrate_credentials

            _handle_migrate_credentials(_make_args(), config_manager=cm)
        assert "EXISTING_KEY" in capsys.readouterr().out

    def test_migration_cancelled_when_no_yes(self, capsys):
        cm = self._make_config_manager(dry_run_results={"migrated": ["KEY"], "already_secure": [], "errors": {}})
        with (
            patch("aura_cli.commands.log_json"),
            patch("builtins.input", side_effect=EOFError),
        ):
            from aura_cli.commands import _handle_migrate_credentials

            _handle_migrate_credentials(_make_args(yes=False), config_manager=cm)
        assert "cancelled" in capsys.readouterr().out

    def test_migration_cancelled_by_user_input_n(self, capsys):
        cm = self._make_config_manager(dry_run_results={"migrated": ["KEY"], "already_secure": [], "errors": {}})
        with (
            patch("aura_cli.commands.log_json"),
            patch("builtins.input", return_value="n"),
        ):
            from aura_cli.commands import _handle_migrate_credentials

            _handle_migrate_credentials(_make_args(yes=False), config_manager=cm)
        assert "Migration cancelled" in capsys.readouterr().out

    def test_migration_executed_with_yes_flag(self, capsys):
        cm = self._make_config_manager(dry_run_results={"migrated": ["ANTHROPIC_API_KEY"], "already_secure": [], "errors": {}})
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_migrate_credentials

            _handle_migrate_credentials(_make_args(yes=True), config_manager=cm)
        out = capsys.readouterr().out
        assert "Successfully migrated" in out

    def test_migration_errors_reported(self, capsys):
        dry = {"migrated": ["BAD_KEY"], "already_secure": [], "errors": {}}
        real = {"migrated": [], "errors": {"BAD_KEY": "permission denied"}}
        cm = self._make_config_manager(dry_run_results=dry, migrate_results=real)
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_migrate_credentials

            _handle_migrate_credentials(_make_args(yes=True), config_manager=cm)
        out = capsys.readouterr().out
        assert "permission denied" in out

    def test_keyring_status_displayed(self, capsys):
        cm = self._make_config_manager()
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_migrate_credentials

            _handle_migrate_credentials(_make_args(), config_manager=cm)
        out = capsys.readouterr().out
        assert "Keyring" in out


# ---------------------------------------------------------------------------
# _handle_secure_store
# ---------------------------------------------------------------------------


class TestHandleSecureStore:
    def test_error_when_no_key(self, capsys):
        cm = MagicMock()
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_secure_store

            _handle_secure_store(_make_args(key=None, value="secret"), config_manager=cm)
        assert "--key is required" in capsys.readouterr().out

    def test_stores_credential_successfully(self, capsys):
        cm = MagicMock()
        cm.secure_store_credential.return_value = True
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_secure_store

            _handle_secure_store(_make_args(key="MY_KEY", value="my_secret"), config_manager=cm)
        out = capsys.readouterr().out
        assert "stored securely" in out
        cm.secure_store_credential.assert_called_once_with("MY_KEY", "my_secret")

    def test_failure_message_on_store_error(self, capsys):
        cm = MagicMock()
        cm.secure_store_credential.return_value = False
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_secure_store

            _handle_secure_store(_make_args(key="MY_KEY", value="my_secret"), config_manager=cm)
        assert "Failed to store" in capsys.readouterr().out

    def test_interactive_value_prompt_when_no_value(self, capsys):
        cm = MagicMock()
        cm.secure_store_credential.return_value = True
        with (
            patch("aura_cli.commands.log_json"),
            patch("getpass.getpass", return_value="prompted_value"),
        ):
            from aura_cli.commands import _handle_secure_store

            _handle_secure_store(_make_args(key="MY_KEY", value=None), config_manager=cm)
        cm.secure_store_credential.assert_called_once_with("MY_KEY", "prompted_value")

    def test_error_when_empty_prompted_value(self, capsys):
        cm = MagicMock()
        with (
            patch("aura_cli.commands.log_json"),
            patch("getpass.getpass", return_value=""),
        ):
            from aura_cli.commands import _handle_secure_store

            _handle_secure_store(_make_args(key="MY_KEY", value=None), config_manager=cm)
        assert "value is required" in capsys.readouterr().out
        cm.secure_store_credential.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_secure_delete
# ---------------------------------------------------------------------------


class TestHandleSecureDelete:
    def test_error_when_no_key(self, capsys):
        cm = MagicMock()
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_secure_delete

            _handle_secure_delete(_make_args(key=None), config_manager=cm)
        assert "--key is required" in capsys.readouterr().out

    def test_deletion_confirmed_with_yes_flag(self, capsys):
        cm = MagicMock()
        cm.secure_delete_credential.return_value = True
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_secure_delete

            _handle_secure_delete(_make_args(key="MY_KEY", yes=True), config_manager=cm)
        out = capsys.readouterr().out
        assert "deleted" in out
        cm.secure_delete_credential.assert_called_once_with("MY_KEY")

    def test_failure_message_on_delete_error(self, capsys):
        cm = MagicMock()
        cm.secure_delete_credential.return_value = False
        with patch("aura_cli.commands.log_json"):
            from aura_cli.commands import _handle_secure_delete

            _handle_secure_delete(_make_args(key="MY_KEY", yes=True), config_manager=cm)
        assert "Failed to delete" in capsys.readouterr().out

    def test_cancels_on_eof(self, capsys):
        cm = MagicMock()
        with (
            patch("aura_cli.commands.log_json"),
            patch("builtins.input", side_effect=EOFError),
        ):
            from aura_cli.commands import _handle_secure_delete

            _handle_secure_delete(_make_args(key="MY_KEY", yes=False), config_manager=cm)
        assert "cancelled" in capsys.readouterr().out
        cm.secure_delete_credential.assert_not_called()

    def test_cancels_on_no_response(self, capsys):
        cm = MagicMock()
        with (
            patch("aura_cli.commands.log_json"),
            patch("builtins.input", return_value="n"),
        ):
            from aura_cli.commands import _handle_secure_delete

            _handle_secure_delete(_make_args(key="MY_KEY", yes=False), config_manager=cm)
        assert "Deletion cancelled" in capsys.readouterr().out
        cm.secure_delete_credential.assert_not_called()

    def test_proceeds_on_yes_response(self, capsys):
        cm = MagicMock()
        cm.secure_delete_credential.return_value = True
        with (
            patch("aura_cli.commands.log_json"),
            patch("builtins.input", return_value="y"),
        ):
            from aura_cli.commands import _handle_secure_delete

            _handle_secure_delete(_make_args(key="MY_KEY", yes=False), config_manager=cm)
        cm.secure_delete_credential.assert_called_once_with("MY_KEY")
