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
    app.on_cycle_complete({"goal": "fix something", "success": True, "duration_s": 5.2})
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
    p2 = build_queue_panel(mock_queue)
    assert isinstance(p2, Panel)
