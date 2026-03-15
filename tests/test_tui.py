"""
PRD-005: TUI data model tests.
Verifies AuraStudio callbacks and panel builders without live display.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock
from aura_cli.tui.app import AuraStudio

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


def test_cycle_panel_renders_failure_routing_context():
    from aura_cli.tui.panels.cycle_panel import build_cycle_panel
    from rich.console import Console

    panel = build_cycle_panel(
        "",
        {},
        "",
        None,
        last_summary={
            "outcome": "FAILED",
            "stop_reason": "MAX_CYCLES",
            "queued_follow_up_goals": [],
            "failure_routing_decision": "plan",
            "failure_routing_reason": (
                "structural: detected 'design' in failures/logs"
            ),
        },
    )

    console = Console(record=True, width=120)
    console.print(panel)
    rendered = console.export_text()

    assert "Routing" in rendered
    assert "plan" in rendered
    assert "design" in rendered
