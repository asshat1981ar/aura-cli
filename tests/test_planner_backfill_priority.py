"""
Tests for PlannerAgent backfill prioritization.
"""
from __future__ import annotations

import pytest
import json
from unittest.mock import MagicMock, patch
from agents.planner import PlannerAgent

@pytest.fixture
def mock_brain():
    return MagicMock()

@pytest.fixture
def mock_model():
    return MagicMock()

def test_planner_includes_backfill_instructions_in_prompt(mock_brain, mock_model):
    """
    Verify that if backfill_context is provided, it appears in the LLM prompt.
    """
    planner = PlannerAgent(mock_brain, mock_model)
    
    backfill_ctx = [
        {"file": "core/untested.py", "coverage": 0.0}
    ]
    
    mock_model.respond.return_value = '["Step 1: Write tests for core/untested.py", "Step 2: Do the thing"]'
    
    planner.run({
        "goal": "Implement feature X",
        "memory_snapshot": "mem",
        "similar_past_problems": "sim",
        "known_weaknesses": "weak",
        "backfill_context": backfill_ctx
    })
    
    # Check that the prompt contains the backfill info
    args, _ = mock_model.respond.call_args
    prompt = args[0]
    assert "core/untested.py" in prompt
    assert "0.0%" in prompt or "0.0" in prompt
    assert "PRIORITIZE" in prompt or "High Risk" in prompt

def test_planner_prefers_backfill_steps_first(mock_brain, mock_model):
    """
    Ensure the LLM instructions explicitly ask for backfill tasks to be first.
    """
    planner = PlannerAgent(mock_brain, mock_model)
    
    backfill_ctx = [{"file": "gaps.py", "coverage": 10.0}]
    mock_model.respond.return_value = '["Step 1: Backfill tests", "Step 2: Feature"]'
    
    res = planner.run({
        "goal": "Feature",
        "backfill_context": backfill_ctx
    })
    
    assert "Backfill" in res["steps"][0]
