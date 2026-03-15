"""
Integration test for the full autonomous backfill flow.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.orchestrator import LoopOrchestrator
from agents.planner import PlannerAgent

@pytest.fixture
def mock_runtime():
    return {
        "brain": MagicMock(),
        "model": MagicMock(),
        "goal_queue": MagicMock()
    }

def test_full_backfill_flow_integration(mock_runtime, tmp_path):
    """
    Test that a goal on a module with gaps enqueues a backfill goal
    AND the planner for that goal prioritizes the backfill.
    """
    brain = mock_runtime["brain"]
    model = mock_runtime["model"]
    goal_queue = mock_runtime["goal_queue"]
    
    # 1. Setup Orchestrator
    agents = {
        "ingest": MagicMock(run=lambda x: {
            "goal": "Implement feature", "snapshot": {}, "memory_summary": "", 
            "constraints": [], "bundle": {"goal": "Implement feature"}
        }),
        "plan": PlannerAgent(brain, model), # Real planner logic
        "synthesize": MagicMock(run=lambda x: {"tasks": [{"id": 1, "description": "mock", "tests": []}]}),
        "act": MagicMock(run=lambda x: {"changes": []}),
        "verify": MagicMock(run=lambda x: {"status": "pass"}),
        "reflect": MagicMock(run=lambda x: {"summary": "done"})
    }
    
    orchestrator = LoopOrchestrator(
        agents=agents,
        project_root=tmp_path,
        goal_queue=goal_queue
    )
    orchestrator.auto_backfill_coverage = True
    
    # 2. Mock StructuralAnalyzer to report a gap
    mock_structural_res = {
        "coverage_gaps": [{"file": "core/untested.py", "coverage_pct": 0.0, "risk_priority": "HIGH"}],
        "summary": "Found gaps"
    }
    
    # 3. Mock Model response for the plan
    model.respond.return_value = '["Step 1: Test Backfill for core/untested.py", "Step 2: Implement feature"]'
    
    with patch("core.orchestrator.classify_goal", return_value="feat"), \
         patch("core.orchestrator.dispatch_skills", return_value={"structural_analyzer": mock_structural_res}):
        
        # 4. Run Cycle
        orchestrator.run_cycle("Implement feature")
        
        # 5. Assertions
        # Goal enqueued?
        goal_queue.add.assert_called()
        enqueued_goal = goal_queue.add.call_args[0][0]
        assert "test_backfill" in enqueued_goal
        assert "core/untested.py" in enqueued_goal
        
        # Planner prompt included backfill instructions?
        prompt = model.respond.call_args[0][0]
        assert "LOW/ZERO test coverage" in prompt
        assert "core/untested.py" in prompt
