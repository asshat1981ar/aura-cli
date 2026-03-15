"""
Tests for autonomous backfill goal generation in LoopOrchestrator.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.orchestrator import LoopOrchestrator

@pytest.fixture
def mock_agents():
    return {
        "ingest": MagicMock(run=lambda x: {
            "goal": "mock", "snapshot": {}, "memory_summary": "mock", "constraints": [],
            "bundle": {"goal": "mock"}
        }),
        "plan": MagicMock(run=lambda x: {"steps": [{"id": 1, "task": "mock"}]}),
        "act": MagicMock(run=lambda x: {"changes": []}),
        "verify": MagicMock(run=lambda x: {"status": "pass"}),
        "reflect": MagicMock(run=lambda x: {"summary": "mock"}),
        "synthesize": MagicMock(run=lambda x: {"tasks": [{"id": 1, "description": "mock", "tests": []}]})
    }

def test_orchestrator_enqueues_backfill_goals(mock_agents, tmp_path):
    """
    Verify that if coverage gaps are found and auto_backfill_coverage is True,
    the orchestrator adds goals to the queue.
    """
    mock_goal_queue = MagicMock()
    
    orchestrator = LoopOrchestrator(
        agents=mock_agents,
        project_root=tmp_path,
        goal_queue=mock_goal_queue
    )
    # Enable the new feature
    orchestrator.auto_backfill_coverage = True
    
    # Mock StructuralAnalyzerSkill to return gaps
    mock_structural_res = {
        "coverage_gaps": [
            {"file": "core/critical.py", "coverage_pct": 0.0, "risk_priority": "HIGH"}
        ],
        "summary": "Found gaps"
    }
    
    with patch("core.orchestrator.classify_goal", return_value="feat"), \
         patch("core.orchestrator.dispatch_skills", return_value={"structural_analyzer": mock_structural_res}):
        
        orchestrator.run_cycle("Implement new feature")
        
        # Verify that a goal was enqueued for the gap
        mock_goal_queue.add.assert_called()
        args, _ = mock_goal_queue.add.call_args
        assert "backfill" in args[0].lower()
        assert "core/critical.py" in args[0]

def test_orchestrator_skips_backfill_when_disabled(mock_agents, tmp_path):
    mock_goal_queue = MagicMock()
    
    orchestrator = LoopOrchestrator(
        agents=mock_agents,
        project_root=tmp_path,
        goal_queue=mock_goal_queue
    )
    orchestrator.auto_backfill_coverage = False
    
    mock_structural_res = {
        "coverage_gaps": [{"file": "gap.py", "coverage_pct": 0.0}],
        "summary": "Found gaps"
    }
    
    with patch("core.orchestrator.classify_goal", return_value="feat"), \
         patch("core.orchestrator.dispatch_skills", return_value={"structural_analyzer": mock_structural_res}):
        
        orchestrator.run_cycle("Implement new feature")
        
        mock_goal_queue.add.assert_not_called()
