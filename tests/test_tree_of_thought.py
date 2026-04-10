"""Tests for core.tree_of_thought — Tree-of-Thought planning."""

import json
import pytest
from unittest.mock import MagicMock

from core.tree_of_thought import (
    TreeOfThoughtPlanner,
    PlanCandidate,
    PLAN_STRATEGIES,
    SCORING_CRITERIA,
)


def _make_model(response="[]"):
    """Create a mock model without respond_for_role (uses respond)."""
    model = MagicMock(spec=["respond"])
    model.respond.return_value = response
    return model


def _make_model_with_role(response="[]"):
    """Create a mock model that supports respond_for_role."""
    model = MagicMock(spec=["respond", "respond_for_role"])
    model.respond_for_role.return_value = response
    return model


class TestPlanGeneration:
    def test_generates_n_candidates(self):
        steps = json.dumps([{"step": 1, "description": "Do X", "files": ["a.py"], "verification": "run tests"}])
        model = _make_model(steps)
        planner = TreeOfThoughtPlanner(n_candidates=3)

        candidates = planner.generate_plans(model, "Fix bug", {})

        assert len(candidates) == 3
        assert model.respond.call_count == 3

    def test_uses_respond_for_role_when_available(self):
        steps = json.dumps([{"step": 1, "description": "Do X"}])
        model = _make_model_with_role(steps)
        planner = TreeOfThoughtPlanner(n_candidates=2)

        candidates = planner.generate_plans(model, "Add feature", {})

        assert len(candidates) == 2
        assert model.respond_for_role.call_count == 2
        model.respond_for_role.assert_called_with("planning", pytest.approx(str, abs=True) if False else model.respond_for_role.call_args[0][1])

    def test_strategy_names_match(self):
        model = _make_model("[]")
        planner = TreeOfThoughtPlanner(n_candidates=3)

        candidates = planner.generate_plans(model, "Goal", {})

        assert candidates[0].strategy == "conservative"
        assert candidates[1].strategy == "aggressive"
        assert candidates[2].strategy == "incremental"

    def test_caps_at_max_strategies(self):
        planner = TreeOfThoughtPlanner(n_candidates=100)
        assert planner.n_candidates == len(PLAN_STRATEGIES)

    def test_context_fields_passed_through(self):
        model = _make_model("[]")
        planner = TreeOfThoughtPlanner(n_candidates=1)
        context = {
            "memory_snapshot": "some memory",
            "known_weaknesses": "weak at tests",
            "skill_context": {"linter": "enabled"},
        }

        planner.generate_plans(model, "Goal", context)

        prompt = model.respond.call_args[0][0]
        assert "some memory" in prompt
        assert "weak at tests" in prompt
        assert "linter" in prompt


class TestStepParsing:
    def test_parse_json_array(self):
        response = json.dumps(
            [
                {"step": 1, "description": "First"},
                {"step": 2, "description": "Second"},
            ]
        )
        planner = TreeOfThoughtPlanner()
        steps = planner._parse_steps(response)

        assert len(steps) == 2
        assert steps[0]["description"] == "First"

    def test_parse_json_object_with_steps_key(self):
        response = json.dumps({"steps": [{"step": 1, "description": "A"}]})
        planner = TreeOfThoughtPlanner()
        steps = planner._parse_steps(response)

        assert len(steps) == 1
        assert steps[0]["description"] == "A"

    def test_parse_numbered_list(self):
        response = "1. Create file\n2. Add tests\n3. Run linter"
        planner = TreeOfThoughtPlanner()
        steps = planner._parse_steps(response)

        assert len(steps) == 3
        assert steps[0]["description"] == "Create file"
        assert steps[2]["description"] == "Run linter"

    def test_parse_numbered_list_parenthesis(self):
        response = "1) Create file\n2) Add tests"
        planner = TreeOfThoughtPlanner()
        steps = planner._parse_steps(response)

        assert len(steps) == 2
        assert steps[0]["description"] == "Create file"

    def test_parse_fallback_raw_text(self):
        response = "Just do the thing"
        planner = TreeOfThoughtPlanner()
        steps = planner._parse_steps(response)

        assert len(steps) == 1
        assert steps[0]["description"] == "Just do the thing"


class TestScoring:
    def test_score_parsing_from_json(self):
        model = _make_model()
        scores_response = json.dumps(
            {
                "scores": {
                    "conservative": {"feasibility": 0.9, "coverage": 0.8, "risk": 0.7, "testability": 0.6, "clarity": 0.5},
                    "aggressive": {"feasibility": 0.5, "coverage": 0.9, "risk": 0.4, "testability": 0.7, "clarity": 0.8},
                }
            }
        )
        model.respond.return_value = scores_response

        candidates = [
            PlanCandidate(strategy="conservative", strategy_description="safe", steps=[{"step": 1, "description": "A"}]),
            PlanCandidate(strategy="aggressive", strategy_description="bold", steps=[{"step": 1, "description": "B"}]),
        ]

        planner = TreeOfThoughtPlanner()
        winner = planner.score_plans(model, candidates, "Fix bug")

        assert winner.strategy == "conservative"
        assert winner.scores["feasibility"] == 0.9
        assert abs(winner.total_score - 0.7) < 0.01  # (0.9+0.8+0.7+0.6+0.5)/5

    def test_single_candidate_auto_wins(self):
        model = _make_model()
        candidate = PlanCandidate(strategy="conservative", strategy_description="safe", steps=[{"step": 1, "description": "A"}])

        planner = TreeOfThoughtPlanner()
        winner = planner.score_plans(model, [candidate], "Goal")

        assert winner is candidate
        assert winner.total_score == 1.0
        model.respond.assert_not_called()

    def test_no_valid_candidates_raises(self):
        model = _make_model()
        empty_candidates = [
            PlanCandidate(strategy="conservative", strategy_description="safe", steps=[]),
            PlanCandidate(strategy="aggressive", strategy_description="bold", steps=[]),
        ]

        planner = TreeOfThoughtPlanner()
        with pytest.raises(ValueError, match="No valid plan candidates"):
            planner.score_plans(model, empty_candidates, "Goal")

    def test_fallback_scoring_on_model_error(self):
        model = _make_model()
        model.respond.side_effect = RuntimeError("LLM down")

        candidates = [
            PlanCandidate(strategy="conservative", strategy_description="safe", steps=[{"step": 1, "description": "A"}, {"step": 2, "description": "B"}]),
            PlanCandidate(strategy="aggressive", strategy_description="bold", steps=[{"step": 1, "description": "C"}]),
        ]

        planner = TreeOfThoughtPlanner()
        winner = planner.score_plans(model, candidates, "Goal")

        # Fallback: conservative has 2 steps * 0.1 + (2-0)*0.2 = 0.6
        # aggressive has 1 step * 0.1 + (2-1)*0.2 = 0.3
        assert winner.strategy == "conservative"
        assert winner.total_score > 0

    def test_score_parsing_with_surrounding_text(self):
        """Model returns JSON embedded in conversational text."""
        model = _make_model()
        model.respond.return_value = 'Here are my scores:\n{"scores": {"conservative": {"feasibility": 0.8, "coverage": 0.7, "risk": 0.6, "testability": 0.9, "clarity": 0.7}}}\nHope this helps!'

        candidates = [
            PlanCandidate(strategy="conservative", strategy_description="safe", steps=[{"step": 1, "description": "A"}]),
        ]

        planner = TreeOfThoughtPlanner()
        # Single candidate auto-wins, so test _parse_scores directly
        planner._parse_scores(model.respond.return_value, candidates)

        assert candidates[0].scores["feasibility"] == 0.8
        assert candidates[0].scores["testability"] == 0.9


class TestErrorHandling:
    def test_model_exception_during_plan(self):
        model = _make_model()
        model.respond.side_effect = RuntimeError("API error")

        planner = TreeOfThoughtPlanner(n_candidates=2)
        candidates = planner.generate_plans(model, "Goal", {})

        assert len(candidates) == 2
        assert all("ERROR" in c.raw_response for c in candidates)
        assert all(len(c.steps) == 0 for c in candidates)

    def test_empty_context(self):
        steps = json.dumps([{"step": 1, "description": "Do X"}])
        model = _make_model(steps)
        planner = TreeOfThoughtPlanner(n_candidates=1)

        candidates = planner.generate_plans(model, "Goal", {})

        assert len(candidates) == 1
        assert len(candidates[0].steps) == 1


class TestPlanCandidate:
    def test_default_fields(self):
        pc = PlanCandidate(strategy="test", strategy_description="desc")
        assert pc.steps == []
        assert pc.raw_response == ""
        assert pc.scores == {}
        assert pc.total_score == 0.0
        assert pc.reasoning == ""
