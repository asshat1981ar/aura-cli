"""Tests for core/weakness_remediator.py — WeaknessRemediator.

Tests the main weaknesses → goals conversion pipeline, including:
- Deduplication via SHA-256 hashing
- Severity-based scoring
- Goal template formatting
- Queue marking
"""

import hashlib
import json
from unittest.mock import Mock, MagicMock

import pytest

from core.weakness_remediator import (
    WeaknessRemediator,
    _GOAL_TEMPLATES,
    _SEVERITY_WEIGHT,
)


class TestWeaknessRemediatorRun:
    """Test the main public run() method."""

    def test_run_empty_weaknesses(self):
        """When brain returns no weaknesses, returns skipped result."""
        remediator = WeaknessRemediator()
        mock_brain = Mock()
        mock_brain.recall_weaknesses.return_value = []
        mock_goal_queue = Mock()

        result = remediator.run(mock_brain, mock_goal_queue, limit=3)

        assert result["goals_generated"] == 0
        assert result["goals"] == []
        assert result.get("skipped") is True

    def test_run_exception_handling(self):
        """When an exception occurs, returns error dict with 0 goals."""
        remediator = WeaknessRemediator()
        mock_brain = Mock()
        mock_brain.recall_weaknesses.side_effect = RuntimeError("Database error")
        mock_goal_queue = Mock()

        result = remediator.run(mock_brain, mock_goal_queue, limit=3)

        assert "error" in result
        assert result["goals_generated"] == 0
        assert result["goals"] == []

    def test_run_with_valid_weaknesses(self):
        """Process valid weaknesses and return generated goals."""
        remediator = WeaknessRemediator()
        mock_brain = Mock()
        mock_brain.recall_weaknesses.return_value = [
            json.dumps({"type": "phase_failure", "phase": "reasoning", "failure_rate": 0.5, "severity": "HIGH"}),
        ]
        mock_brain.recall_queued_weakness_hashes.return_value = []
        mock_brain.mark_weakness_queued = Mock()
        mock_goal_queue = Mock()

        result = remediator.run(mock_brain, mock_goal_queue, limit=3)

        assert result["goals_generated"] >= 1
        assert len(result["goals"]) >= 1
        mock_goal_queue.add.assert_called_once()
        mock_brain.mark_weakness_queued.assert_called_once()


class TestWeaknessRemediatorInternal:
    """Test internal _run method and helper methods."""

    def test_internal_run_skips_queued_hashes(self):
        """Already-queued weaknesses (by hash) are skipped."""
        remediator = WeaknessRemediator()
        raw = "Some weakness text"
        w_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

        mock_brain = Mock()
        mock_brain.recall_weaknesses.return_value = [raw]
        mock_brain.recall_queued_weakness_hashes.return_value = [w_hash]
        mock_goal_queue = Mock()

        result = remediator._run(mock_brain, mock_goal_queue, limit=3)

        assert result["goals_generated"] == 0
        assert result["goals"] == []
        mock_goal_queue.add.assert_not_called()

    def test_internal_run_respects_limit(self):
        """Only generates up to `limit` goals."""
        remediator = WeaknessRemediator()
        weaknesses = [json.dumps({"type": "negative_sentiment", "description": f"issue {i}", "severity": "HIGH"}) for i in range(5)]
        mock_brain = Mock()
        mock_brain.recall_weaknesses.return_value = weaknesses
        mock_brain.recall_queued_weakness_hashes.return_value = []
        mock_brain.mark_weakness_queued = Mock()
        mock_goal_queue = Mock()

        result = remediator._run(mock_brain, mock_goal_queue, limit=2)

        assert result["goals_generated"] == 2
        assert mock_goal_queue.add.call_count == 2

    def test_internal_run_deduplicates_similar_goals(self):
        """Similar goal texts (same first 60 chars) are deduplicated."""
        remediator = WeaknessRemediator()
        # Two weaknesses that will produce goals with same prefix
        w1 = json.dumps({"type": "negative_sentiment", "description": "Fix issue A" * 20})
        w2 = json.dumps({"type": "negative_sentiment", "description": "Fix issue A" * 20})

        mock_brain = Mock()
        mock_brain.recall_weaknesses.return_value = [w1, w2]
        mock_brain.recall_queued_weakness_hashes.return_value = []
        mock_brain.mark_weakness_queued = Mock()
        mock_goal_queue = Mock()

        result = remediator._run(mock_brain, mock_goal_queue, limit=10)

        # Should skip the second one due to dedup
        assert result["goals_generated"] == 1
        assert mock_goal_queue.add.call_count == 1

    def test_internal_run_sorts_by_score_descending(self):
        """Goals are generated in score-descending order."""
        remediator = WeaknessRemediator()
        # High severity, high failure rate
        w_high = json.dumps(
            {
                "type": "phase_failure",
                "phase": "reasoning",
                "failure_rate": 0.8,
                "severity": "HIGH",
            }
        )
        # Low severity
        w_low = json.dumps(
            {
                "type": "phase_failure",
                "phase": "planning",
                "failure_rate": 0.1,
                "severity": "LOW",
            }
        )

        mock_brain = Mock()
        mock_brain.recall_weaknesses.return_value = [w_low, w_high]
        mock_brain.recall_queued_weakness_hashes.return_value = []
        mock_brain.mark_weakness_queued = Mock()
        mock_goal_queue = Mock()

        result = remediator._run(mock_brain, mock_goal_queue, limit=2)

        # Both should be queued, but in the correct order
        calls = mock_goal_queue.add.call_args_list
        assert len(calls) == 2
        # High-score goal should be queued first
        assert "reasoning" in calls[0][0][0]


class TestParseWeakness:
    """Test _parse_weakness method."""

    def test_parse_valid_json_dict(self):
        """Parse valid JSON dict structure."""
        remediator = WeaknessRemediator()
        raw = json.dumps({"type": "phase_failure", "phase": "reasoning", "severity": "HIGH"})

        result = remediator._parse_weakness(raw)

        assert result["type"] == "phase_failure"
        assert result["phase"] == "reasoning"
        assert result["severity"] == "HIGH"

    def test_parse_invalid_json_with_error_keyword(self):
        """Non-JSON text with 'error' keyword becomes keyword type."""
        remediator = WeaknessRemediator()
        raw = "A critical error occurred in the system"

        result = remediator._parse_weakness(raw)

        assert result["type"] == "keyword"
        assert "error" in result["description"].lower()

    def test_parse_invalid_json_with_fail_keyword(self):
        """Non-JSON text with 'fail' keyword becomes keyword type."""
        remediator = WeaknessRemediator()
        raw = "Phase 2 fail: reasoning module crashed"

        result = remediator._parse_weakness(raw)

        assert result["type"] == "keyword"

    def test_parse_invalid_json_with_crash_keyword(self):
        """Non-JSON text with 'crash' keyword becomes keyword type."""
        remediator = WeaknessRemediator()
        raw = "System crash detected"

        result = remediator._parse_weakness(raw)

        assert result["type"] == "keyword"

    def test_parse_invalid_json_with_exception_keyword(self):
        """Non-JSON text with 'exception' keyword becomes keyword type."""
        remediator = WeaknessRemediator()
        raw = "exception: IndexError in array bounds"

        result = remediator._parse_weakness(raw)

        assert result["type"] == "keyword"

    def test_parse_plain_text_fallback(self):
        """Plain text without keywords becomes negative_sentiment type."""
        remediator = WeaknessRemediator()
        raw = "User feedback suggests the output quality is low"

        result = remediator._parse_weakness(raw)

        assert result["type"] == "negative_sentiment"
        assert "feedback" in result["description"]

    def test_parse_truncates_long_description(self):
        """Descriptions longer than 200 chars are truncated."""
        remediator = WeaknessRemediator()
        raw = "x" * 500

        result = remediator._parse_weakness(raw)

        assert len(result["description"]) <= 200

    def test_parse_json_non_dict_fallback(self):
        """JSON that's not a dict is treated as plain text."""
        remediator = WeaknessRemediator()
        raw = json.dumps(["list", "not", "dict"])

        result = remediator._parse_weakness(raw)

        assert result["type"] == "negative_sentiment"


class TestScoreWeakness:
    """Test _score_weakness method."""

    def test_score_base_high_severity(self):
        """HIGH severity gets weight 3.0."""
        remediator = WeaknessRemediator()
        w = {"severity": "HIGH"}

        score = remediator._score_weakness(w)

        assert score >= 3.0

    def test_score_base_medium_severity(self):
        """MEDIUM severity gets weight 2.0."""
        remediator = WeaknessRemediator()
        w = {"severity": "MEDIUM"}

        score = remediator._score_weakness(w)

        assert score >= 2.0

    def test_score_base_low_severity(self):
        """LOW severity gets weight 1.0."""
        remediator = WeaknessRemediator()
        w = {"severity": "LOW"}

        score = remediator._score_weakness(w)

        assert score >= 1.0

    def test_score_default_when_severity_missing(self):
        """Missing severity defaults to LOW (1.0)."""
        remediator = WeaknessRemediator()
        w = {}

        score = remediator._score_weakness(w)

        assert score >= 1.0

    def test_score_boost_by_failure_rate(self):
        """failure_rate boosts score by rate * 2.0."""
        remediator = WeaknessRemediator()
        w = {"severity": "LOW", "failure_rate": 0.5}

        score = remediator._score_weakness(w)

        # 1.0 (LOW) + 0.5 * 2.0 = 2.0
        assert score >= 2.0

    def test_score_reduce_by_actionable_rate(self):
        """Low actionable_rate reduces score."""
        remediator = WeaknessRemediator()
        w = {"severity": "HIGH", "actionable_rate": 0.8}

        score = remediator._score_weakness(w)

        # 3.0 (HIGH) - 0.8 = 2.2
        assert score >= 2.0

    def test_score_boost_by_low_success_rate(self):
        """Low success_rate boosts score."""
        remediator = WeaknessRemediator()
        w = {"severity": "MEDIUM", "success_rate": 0.1}

        score = remediator._score_weakness(w)

        # 2.0 (MEDIUM) + (1.0 - 0.1) * 2.0 = 2.0 + 1.8 = 3.8
        assert score >= 3.0

    def test_score_high_success_rate_minimal_boost(self):
        """High success_rate gives minimal boost."""
        remediator = WeaknessRemediator()
        w = {"severity": "LOW", "success_rate": 0.95}

        score = remediator._score_weakness(w)

        # 1.0 (LOW) + (1.0 - 0.95) * 2.0 = 1.0 + 0.1 = 1.1
        assert 1.0 <= score <= 1.2

    def test_score_combined_factors(self):
        """Multiple factors combine additively."""
        remediator = WeaknessRemediator()
        w = {
            "severity": "HIGH",
            "failure_rate": 0.6,
            "success_rate": 0.2,
        }

        score = remediator._score_weakness(w)

        # 3.0 + 0.6*2.0 + (1.0-0.2)*2.0 = 3.0 + 1.2 + 1.6 = 5.8
        assert score >= 5.0


class TestCraftGoal:
    """Test _craft_goal method."""

    def test_craft_goal_phase_failure_template(self):
        """phase_failure uses correct template with substitution."""
        remediator = WeaknessRemediator()
        w = {
            "type": "phase_failure",
            "phase": "reasoning",
            "failure_rate": 0.75,
            "severity": "HIGH",
        }

        goal = remediator._craft_goal(w)

        assert "reasoning" in goal
        assert "75%" in goal
        assert "HIGH" in goal

    def test_craft_goal_low_value_skill_template(self):
        """low_value_skill uses correct template."""
        remediator = WeaknessRemediator()
        w = {
            "type": "low_value_skill",
            "skill": "summarizer",
            "actionable_rate": 0.2,
            "runs": 100,
        }

        goal = remediator._craft_goal(w)

        assert "summarizer" in goal
        assert "20%" in goal
        assert "100" in goal

    def test_craft_goal_goal_type_struggling_template(self):
        """goal_type_struggling uses correct template."""
        remediator = WeaknessRemediator()
        w = {
            "type": "goal_type_struggling",
            "goal_type": "research",
            "success_rate": 0.35,
        }

        goal = remediator._craft_goal(w)

        assert "research" in goal
        assert "35%" in goal

    def test_craft_goal_negative_sentiment_template(self):
        """negative_sentiment uses simple template."""
        remediator = WeaknessRemediator()
        w = {
            "type": "negative_sentiment",
            "description": "Output quality is consistently poor",
        }

        goal = remediator._craft_goal(w)

        assert "Output quality is consistently poor" in goal

    def test_craft_goal_keyword_type(self):
        """keyword type uses simple template."""
        remediator = WeaknessRemediator()
        w = {
            "type": "keyword",
            "description": "System error in caching layer",
        }

        goal = remediator._craft_goal(w)

        assert "System error in caching layer" in goal

    def test_craft_goal_unknown_type_fallback(self):
        """Unknown weakness type uses default template."""
        remediator = WeaknessRemediator()
        w = {
            "type": "unknown_type",
            "description": "Some new weakness type",
        }

        goal = remediator._craft_goal(w)

        assert "Fix weakness" in goal

    def test_craft_goal_missing_template_fields(self):
        """Missing required fields in template substitution falls back gracefully."""
        remediator = WeaknessRemediator()
        w = {
            "type": "phase_failure",
            "phase": "reasoning",
            # Missing failure_rate, severity
        }

        goal = remediator._craft_goal(w)

        # Should return fallback format, not raise
        assert isinstance(goal, str)
        assert len(goal) > 0

    def test_craft_goal_truncates_description(self):
        """Description is truncated to 120 chars in goal text."""
        remediator = WeaknessRemediator()
        w = {
            "type": "keyword",
            "description": "x" * 500,
        }

        goal = remediator._craft_goal(w)

        # Full description would be 500 chars, but should be truncated in formatting
        assert isinstance(goal, str)
        # The goal might exceed 120 if the template adds text, but description should be limited
        assert "x" * 500 not in goal

    def test_craft_goal_handles_malformed_input(self):
        """Handles malformed input gracefully."""
        remediator = WeaknessRemediator()
        w = {}

        goal = remediator._craft_goal(w)

        # Should not raise, returns some fallback
        assert isinstance(goal, str)
        assert len(goal) > 0


class TestGoalTemplatesAndWeights:
    """Test module-level constants."""

    def test_goal_templates_dict_structure(self):
        """All templates in _GOAL_TEMPLATES are strings."""
        for key, template in _GOAL_TEMPLATES.items():
            assert isinstance(template, str)

    def test_severity_weights_dict_structure(self):
        """All weights in _SEVERITY_WEIGHT are floats or ints."""
        for key, weight in _SEVERITY_WEIGHT.items():
            assert isinstance(weight, (int, float))
            assert weight > 0

    def test_expected_template_keys(self):
        """Expected template keys exist."""
        expected = ["phase_failure", "low_value_skill", "goal_type_struggling", "negative_sentiment", "keyword"]
        for key in expected:
            assert key in _GOAL_TEMPLATES

    def test_expected_severity_keys(self):
        """Expected severity levels exist."""
        expected = ["HIGH", "MEDIUM", "LOW"]
        for key in expected:
            assert key in _SEVERITY_WEIGHT
