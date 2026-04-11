"""Tests for prompt optimization."""

import pytest

from aura.learning.feedback import ExecutionOutcome, ExecutionStatus
from aura.learning.optimization import PromptOptimizer, PromptTemplate


class TestPromptOptimizer:
    @pytest.fixture
    def optimizer(self):
        return PromptOptimizer()

    def test_default_templates_loaded(self, optimizer):
        assert "plan" in optimizer.templates
        assert "code" in optimizer.templates

    def test_record_template_performance(self, optimizer):
        outcome = ExecutionOutcome(
            agent_name="test",
            goal="Test goal",
            status=ExecutionStatus.SUCCESS,
            duration_ms=100.0,
            output_quality=0.9,
        )

        optimizer.record_template_performance("plan", outcome)
        template = optimizer.templates["plan"]

        assert template.total_count == 1
        assert template.success_count == 1
        assert template.avg_quality == 0.9

    def test_suggest_template_for_goal(self, optimizer):
        # Should suggest "plan" for planning-related goals
        suggestion = optimizer.suggest_template_for_goal("Create a plan for refactoring")
        assert suggestion == "plan"

        # Should suggest "code" for coding-related goals
        suggestion = optimizer.suggest_template_for_goal("Implement a function")
        assert suggestion == "code"

    def test_optimize_prompt_insufficient_data(self, optimizer):
        # Less than 5 outcomes
        outcomes = [
            ExecutionOutcome(
                agent_name="test",
                goal="G1",
                status=ExecutionStatus.SUCCESS,
                duration_ms=100.0,
                output_quality=0.9,
            ),
        ]

        result = optimizer.optimize_prompt("plan", outcomes)

        assert result is None

    def test_optimize_prompt_with_failures(self, optimizer):
        outcomes = [
            ExecutionOutcome(
                agent_name="test",
                goal="G1",
                status=ExecutionStatus.SUCCESS,
                duration_ms=100.0,
                output_quality=0.9,
                metadata={"template": "plan"},
            ),
            ExecutionOutcome(
                agent_name="test",
                goal="G2",
                status=ExecutionStatus.SUCCESS,
                duration_ms=100.0,
                output_quality=0.9,
                metadata={"template": "plan"},
            ),
            ExecutionOutcome(
                agent_name="test",
                goal="G3",
                status=ExecutionStatus.SUCCESS,
                duration_ms=100.0,
                output_quality=0.9,
                metadata={"template": "plan"},
            ),
            ExecutionOutcome(
                agent_name="test",
                goal="G4",
                status=ExecutionStatus.FAILURE,
                duration_ms=200.0,
                output_quality=0.0,
                metadata={"template": "plan"},
                error_message="Response too long",
            ),
            ExecutionOutcome(
                agent_name="test",
                goal="G5",
                status=ExecutionStatus.FAILURE,
                duration_ms=200.0,
                output_quality=0.0,
                metadata={"template": "plan"},
                error_message="Response too long",
            ),
        ]

        result = optimizer.optimize_prompt("plan", outcomes)

        # Should suggest reducing length
        assert result is not None
        assert "concise" in result.lower()

    def test_export_templates(self, optimizer):
        # Add some performance data
        outcome = ExecutionOutcome(
            agent_name="test",
            goal="Goal",
            status=ExecutionStatus.SUCCESS,
            duration_ms=100.0,
            output_quality=0.9,
        )
        optimizer.record_template_performance("plan", outcome)

        exported = optimizer.export_templates()

        assert "plan" in exported
        assert exported["plan"]["total_count"] == 1


class TestPromptTemplate:
    def test_success_rate_calculation(self):
        template = PromptTemplate(
            name="test",
            template="Test template",
            success_count=8,
            total_count=10,
        )

        assert template.success_rate == 0.8

    def test_initial_values(self):
        template = PromptTemplate(
            name="test",
            template="Test template",
        )

        assert template.version == 1
        assert template.success_count == 0
        assert template.total_count == 0
        assert template.variations == []
