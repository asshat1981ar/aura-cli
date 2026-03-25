"""
PRD-005: TUI data model tests.
Verifies AuraStudio callbacks and panel builders without live display.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from aura_cli.tui.app import AuraStudio
from memory.store import MemoryStore


def test_tui_instantiation():
    app = AuraStudio()
    assert app is not None
    assert app._cycle_log == []


def test_tui_callbacks():
    app = AuraStudio()

    # 1. Cycle start
    app.on_cycle_start("fix something")
    assert app._current_goal == "fix something"
    assert app._phases_status == {}

    # 2. Phase start
    app.on_phase_start("plan")
    assert app._current_phase == "plan"
    assert app._phases_status["plan"] == "⟳"

    # 3. Phase complete
    app.on_phase_complete("plan", 1200.0, success=True)
    assert app._phases_status["plan"] == "✅"

    # 4. Cycle complete
    app.on_cycle_complete({"goal": "fix something", "outcome": "SUCCESS", "duration_s": 5.2})
    assert len(app._cycle_log) == 1
    assert app._cycle_log[0]["goal"] == "fix something"


def test_tui_context_assembled():
    app = AuraStudio()
    bundle = {"goal": "test", "budget_report": {"total_used": 100}}
    app.on_context_assembled(bundle)
    assert app._last_context_bundle == bundle


def test_tui_pipeline_configured():
    app = AuraStudio()
    app.on_pipeline_configured({"confidence": 0.85})
    assert app._strategy_confidence == 0.85


def test_panel_builders_dont_crash():
    # Verify that build_* functions return Rich objects
    from aura_cli.tui.panels.cycle_panel import build_cycle_panel
    from aura_cli.tui.panels.queue_panel import build_queue_panel
    from rich.panel import Panel

    p = build_cycle_panel("goal", {"ingest": "✅"}, "ingest", 0.9)
    assert isinstance(p, Panel)

    # Mock goal queue
    mock_queue = MagicMock()
    mock_queue.queue = ["goal1", "goal2"]
    mock_archive = MagicMock()
    mock_archive.completed = [("done goal", 8.0)]
    p2 = build_queue_panel(mock_queue, mock_archive)
    assert isinstance(p2, Panel)


def test_cycle_panel_renders_beads_context():
    from aura_cli.tui.panels.cycle_panel import build_cycle_panel
    from rich.console import Console

    panel = build_cycle_panel(
        "",
        {},
        "",
        None,
        last_summary={
            "outcome": "SUCCESS",
            "stop_reason": "PASS",
            "queued_follow_up_goals": [],
            "beads_status": "allow",
            "beads_decision_id": "beads-42",
            "beads_summary": "Proceed with the scoped test fix.",
        },
        beads_runtime={"enabled": True, "required": True, "scope": "goal_run"},
    )

    console = Console(record=True, width=120)
    console.print(panel)
    rendered = console.export_text()

    assert "BEADS" in rendered
    assert "beads-42" in rendered
    assert "goal_run" in rendered


def test_metrics_panel_renders_run_tool_audit_summary():
    from aura_cli.tui.panels.metrics_panel import build_metrics_panel
    from rich.console import Console

    panel = build_metrics_panel(
        cycle_log=[{"outcome": "SUCCESS", "duration_s": 1.2}],
        run_tool_audit={
            "count": 4,
            "success_count": 2,
            "error_count": 1,
            "timeout_count": 1,
            "truncated_count": 0,
            "last_command": "python3 -m pytest -q",
            "recent": [
                {"command": "python3 -m pytest -q", "code": 0, "timed_out": False, "truncated": False},
                {"command": "python3 -c \"import time; time.sleep(1)\"", "code": -15, "timed_out": True, "truncated": False},
                {"command": "python3 -c \"print('x' * 9999)\"", "code": 0, "timed_out": False, "truncated": True},
            ],
        },
    )

    console = Console(record=True, width=140)
    console.print(panel)
    rendered = console.export_text()

    assert "run tool" in rendered
    assert "tracked" in rendered
    assert "last cmd" in rendered
    assert "python3 -m pytest -q" in rendered
    assert "timeout" in rendered
    assert "truncated" in rendered


def test_tui_build_layout_passes_run_tool_audit_to_metrics_panel(monkeypatch):
    captured = {}

    def _fake_metrics_panel(*, cycle_log=None, run_tool_audit=None):
        captured["cycle_log"] = cycle_log
        captured["run_tool_audit"] = run_tool_audit
        from rich.panel import Panel

        return Panel("metrics")

    app = AuraStudio(
        runtime={
            "goal_queue": MagicMock(queue=[]),
            "goal_archive": MagicMock(completed=[]),
            "memory_store": MagicMock(),
            "orchestrator": MagicMock(active_cycle_summary=None, last_cycle_summary=None, current_goal=None),
        }
    )
    app._cycle_log = [{"goal": "test", "outcome": "SUCCESS", "duration_s": 1.0}]

    monkeypatch.setattr("aura_cli.tui.app.build_metrics_panel", _fake_metrics_panel)
    monkeypatch.setattr(
        "aura_cli.tui.app.build_operator_runtime_snapshot",
        lambda *args, **kwargs: {
            "queue": {"pending_count": 0, "pending": [], "completed_count": 0, "completed": [], "active_goal": None, "updated_at": 0.0},
            "active_cycle": None,
            "last_cycle": None,
            "beads_runtime": None,
            "run_tool_audit": {
                "count": 1,
                "success_count": 1,
                "error_count": 0,
                "timeout_count": 0,
                "truncated_count": 0,
                "last_command": "echo ok",
                "recent": [{"command": "echo ok", "code": 0, "timed_out": False, "truncated": False}],
            },
        },
    )

    app._build_layout()

    assert captured["cycle_log"] == app._cycle_log
    assert captured["run_tool_audit"]["last_command"] == "echo ok"


def test_tui_plain_render_includes_run_tool_audit(capsys, tmp_path: Path):
    memory_store = MemoryStore(tmp_path / "memory")
    memory_store.append_log(
        {
            "type": "server_run_tool",
            "command": "echo ok",
            "code": 0,
            "timed_out": False,
            "truncated": False,
            "duration_s": 0.01,
            "output_bytes": 3,
            "timestamp": 1.0,
        }
    )

    app = AuraStudio(runtime={"memory_store": memory_store})
    app._current_goal = "test goal"
    app._current_phase = "verify"
    app._render_plain()

    output = capsys.readouterr().out
    assert "=== AURA Studio ===" in output
    assert "Current goal: test goal" in output
    assert "Run tool audit:" in output
    assert "Tracked: 1" in output
    assert "Last command: echo ok" in output
