"""Tests for PlannerAgent L2 vector store context hints (Sprint 6).

Covers:
- Hints are queried and injected into the prompt when vector_store is set
- No hints are fetched when vector_store is None
- A query error does not break planning
- log_json is called with the 'planner_vector_hints' event
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from agents.planner import PlannerAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain() -> MagicMock:
    brain = MagicMock()
    brain.get_memories.return_value = []
    brain.remember.return_value = None
    return brain


def _make_model(response: str = "{}") -> MagicMock:
    """Return a MagicMock model whose .respond() returns *response*.

    Uses spec=["respond"] so inspect.getattr_static won't find
    respond_for_role, exercising the fallback branch of _respond().
    """
    model = MagicMock(spec=["respond"])
    model.respond.return_value = response
    return model


# A minimal valid structured planner response
_VALID_PLAN_JSON = json.dumps({
    "analysis": "Analysing the goal.",
    "gap_assessment": "No gaps found.",
    "approach": "Incremental TDD approach.",
    "risk_assessment": "Low risk.",
    "plan": [
        {
            "step_number": 1,
            "description": "Implement feature",
            "target_file": "src/feature.py",
            "verification": "pytest tests/",
        }
    ],
    "confidence": 0.9,
    "total_steps": 1,
    "estimated_complexity": "low",
})

_VALID_LEGACY_JSON = json.dumps(["Step 1: Do something", "Step 2: Verify it"])


# ---------------------------------------------------------------------------
# test_vector_hints_injected_when_store_set
# ---------------------------------------------------------------------------

class TestVectorHintsInjected:
    """Hints returned by vector_store.query must appear in the LLM prompt."""

    def test_vector_hints_injected_when_store_set(self):
        brain = _make_brain()
        model = _make_model(_VALID_LEGACY_JSON)
        # Force legacy path so we can trivially inspect the prompt string
        agent = PlannerAgent(brain, model, vector_store=None)
        agent.use_structured = False

        vs = MagicMock()
        vs.query.return_value = ["Reflection A", "Reflection B"]
        agent.vector_store = vs

        captured_prompts: list[str] = []
        original_respond = agent._respond

        def capturing_respond(prompt: str) -> str:
            captured_prompts.append(prompt)
            return original_respond(prompt)

        agent._respond = capturing_respond  # type: ignore[method-assign]

        result = agent.run({"goal": "Build a new feature"})

        # vector_store.query must have been called with the goal and top_k=3
        vs.query.assert_called_once_with("Build a new feature", top_k=3)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "Past Reflections" in prompt
        assert "Reflection A" in prompt
        assert "Reflection B" in prompt

        # Planning should still succeed
        assert "steps" in result


# ---------------------------------------------------------------------------
# test_no_hints_when_store_none
# ---------------------------------------------------------------------------

class TestNoHintsWhenStoreNone:
    """When vector_store is None the planner should work normally."""

    def test_no_hints_when_store_none(self):
        brain = _make_brain()
        model = _make_model(_VALID_LEGACY_JSON)
        agent = PlannerAgent(brain, model, vector_store=None)
        agent.use_structured = False

        captured_prompts: list[str] = []
        original_respond = agent._respond

        def capturing_respond(prompt: str) -> str:
            captured_prompts.append(prompt)
            return original_respond(prompt)

        agent._respond = capturing_respond  # type: ignore[method-assign]

        result = agent.run({"goal": "Refactor codebase"})

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "Past Reflections" not in prompt

        assert "steps" in result


# ---------------------------------------------------------------------------
# test_vector_store_error_does_not_break_planner
# ---------------------------------------------------------------------------

class TestVectorStoreErrorHandling:
    """If vector_store.query raises an exception planning must still succeed."""

    def test_vector_store_error_does_not_break_planner(self):
        brain = _make_brain()
        model = _make_model(_VALID_LEGACY_JSON)
        agent = PlannerAgent(brain, model)
        agent.use_structured = False

        vs = MagicMock()
        vs.query.side_effect = RuntimeError("DB connection lost")
        agent.vector_store = vs

        with patch("agents.planner.log_json") as mock_log:
            result = agent.run({"goal": "Fix the bug"})

        # Planning must return a valid result despite the error
        assert "steps" in result
        steps = result["steps"]
        assert isinstance(steps, list)
        assert len(steps) > 0

        # A WARN log must have been emitted for the query failure
        warn_calls = [c for c in mock_log.call_args_list if c.args[0] == "WARN"]
        assert any("planner_vector_hints_failed" in str(c) for c in warn_calls), (
            "Expected a WARN log for planner_vector_hints_failed"
        )


# ---------------------------------------------------------------------------
# test_hint_count_logged
# ---------------------------------------------------------------------------

class TestHintCountLogged:
    """log_json must be called with the 'planner_vector_hints' event."""

    def test_hint_count_logged(self):
        brain = _make_brain()
        model = _make_model(_VALID_LEGACY_JSON)
        agent = PlannerAgent(brain, model)
        agent.use_structured = False

        vs = MagicMock()
        vs.query.return_value = ["hint one", "hint two", "hint three"]
        agent.vector_store = vs

        with patch("agents.planner.log_json") as mock_log:
            agent.run({"goal": "Optimise search"})

        # Find the planner_vector_hints INFO call
        info_calls = [c for c in mock_log.call_args_list if c.args[0] == "INFO"]
        hint_log = next(
            (c for c in info_calls if c.args[1] == "planner_vector_hints"),
            None,
        )
        assert hint_log is not None, "Expected log_json('INFO', 'planner_vector_hints', ...) to be called"

        details = hint_log.kwargs.get("details", {})
        assert details.get("hint_count") == 3
        assert details.get("goal") == "Optimise search"
