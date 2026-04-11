"""Tests for pattern recognition."""

import pytest

from aura.learning.feedback import ExecutionOutcome, ExecutionStatus
from aura.learning.patterns import PatternRecognizer, SuccessPattern


class TestPatternRecognizer:
    @pytest.fixture
    def recognizer(self):
        return PatternRecognizer()
    
    def test_analyze_outcomes_no_data(self, recognizer):
        outcomes = []
        patterns = recognizer.analyze_outcomes(outcomes)
        
        assert len(patterns) == 0
    
    def test_analyze_outcomes_insufficient_data(self, recognizer):
        # Less than 3 successes - not enough
        outcomes = [
            ExecutionOutcome(
                agent_name="planner",
                goal="Plan something",
                status=ExecutionStatus.SUCCESS,
                duration_ms=100.0,
                output_quality=0.9,
            ),
            ExecutionOutcome(
                agent_name="planner",
                goal="Plan another",
                status=ExecutionStatus.FAILURE,
                duration_ms=200.0,
                output_quality=0.0,
            ),
        ]
        
        patterns = recognizer.analyze_outcomes(outcomes)
        
        assert len(patterns) == 0
    
    def test_analyze_outcomes_success_pattern(self, recognizer):
        outcomes = [
            ExecutionOutcome(
                agent_name="planner",
                goal="Refactor code",
                status=ExecutionStatus.SUCCESS,
                duration_ms=100.0,
                output_quality=0.9,
            ),
            ExecutionOutcome(
                agent_name="planner",
                goal="Refactor module",
                status=ExecutionStatus.SUCCESS,
                duration_ms=110.0,
                output_quality=0.85,
            ),
            ExecutionOutcome(
                agent_name="planner",
                goal="Refactor class",
                status=ExecutionStatus.SUCCESS,
                duration_ms=105.0,
                output_quality=0.9,
            ),
            ExecutionOutcome(
                agent_name="planner",
                goal="Add feature",
                status=ExecutionStatus.FAILURE,
                duration_ms=200.0,
                output_quality=0.0,
            ),
        ]
        
        patterns = recognizer.analyze_outcomes(outcomes)
        
        assert len(patterns) >= 1
        # Should identify "refactor" as success pattern
        refactor_patterns = [p for p in patterns if "refactor" in p.keywords]
        assert len(refactor_patterns) > 0
    
    def test_find_matching_patterns(self, recognizer):
        # First add a pattern
        pattern = SuccessPattern(
            pattern_id="test_pattern",
            description="Test pattern",
            keywords=["refactor", "cleanup"],
            agent_name="planner",
            success_count=5,
            total_count=5,
            avg_quality=0.9,
        )
        recognizer.patterns["test_pattern"] = pattern
        recognizer._keyword_index["refactor"].append("test_pattern")
        
        # Find matching patterns
        matches = recognizer.find_matching_patterns("refactor the code")
        
        assert len(matches) == 1
        assert matches[0].pattern_id == "test_pattern"
    
    def test_get_recommendations(self, recognizer):
        # Add a high-success pattern
        pattern = SuccessPattern(
            pattern_id="refactor_pattern",
            description="refactoring tasks",
            keywords=["refactor"],
            agent_name="planner",
            success_count=10,
            total_count=10,
            avg_quality=0.95,
        )
        recognizer.patterns["refactor_pattern"] = pattern
        recognizer._keyword_index["refactor"].append("refactor_pattern")
        
        recommendations = recognizer.get_recommendations("refactor the module")
        
        assert len(recommendations) > 0
        assert "refactoring tasks" in recommendations[0]
    
    def test_categorize_goals(self, recognizer):
        goals = [
            "Refactor the code",
            "Add a new feature",
            "Fix the bug",
            "Write tests",
            "Optimize performance",
        ]
        
        categories = recognizer._categorize_goals(goals)
        
        assert "refactor" in categories
        assert "add" in categories
        assert "fix" in categories
        assert "test" in categories
        assert "optimize" in categories


class TestSuccessPattern:
    def test_success_rate_calculation(self):
        pattern = SuccessPattern(
            pattern_id="test",
            description="Test",
            keywords=["test"],
            agent_name="agent",
            success_count=8,
            total_count=10,
            avg_quality=0.8,
        )
        
        assert pattern.success_rate == 0.8
    
    def test_to_dict(self):
        pattern = SuccessPattern(
            pattern_id="test",
            description="Test pattern",
            keywords=["test"],
            agent_name="agent",
            success_count=5,
            total_count=5,
            avg_quality=0.9,
            examples=["Example 1", "Example 2"],
        )
        
        data = pattern.to_dict()
        
        assert data["pattern_id"] == "test"
        assert data["success_rate"] == 1.0
        assert "examples" in data
