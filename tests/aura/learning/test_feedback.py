"""Tests for feedback collection."""

import pytest
from datetime import datetime

from aura.learning.feedback import (
    ExecutionOutcome,
    ExecutionStatus,
    FeedbackCollector,
)


class TestExecutionOutcome:
    def test_outcome_creation(self):
        outcome = ExecutionOutcome(
            agent_name="test_agent",
            goal="Test goal",
            status=ExecutionStatus.SUCCESS,
            duration_ms=100.0,
            output_quality=0.9,
        )
        
        assert outcome.agent_name == "test_agent"
        assert outcome.status == ExecutionStatus.SUCCESS
        assert outcome.output_quality == 0.9
        assert outcome.id is not None
    
    def test_outcome_to_dict(self):
        outcome = ExecutionOutcome(
            agent_name="test_agent",
            goal="Test goal",
            status=ExecutionStatus.FAILURE,
            duration_ms=200.0,
            output_quality=0.0,
            error_message="Test error",
        )
        
        data = outcome.to_dict()
        
        assert data["agent_name"] == "test_agent"
        assert data["status"] == "failure"
        assert data["error_message"] == "Test error"
        assert "id" in data
    
    def test_outcome_from_dict(self):
        data = {
            "id": "abc123",
            "agent_name": "test_agent",
            "goal": "Test goal",
            "status": "success",
            "duration_ms": 150.0,
            "output_quality": 0.85,
            "error_message": None,
            "metadata": {},
            "timestamp": "2024-01-01T00:00:00",
        }
        
        outcome = ExecutionOutcome.from_dict(data)
        
        assert outcome.id == "abc123"
        assert outcome.status == ExecutionStatus.SUCCESS
        assert outcome.output_quality == 0.85


class TestFeedbackCollector:
    @pytest.fixture
    def collector(self, tmp_path):
        db_path = tmp_path / "test_feedback.db"
        return FeedbackCollector(db_path=str(db_path))
    
    def test_record_and_get(self, collector):
        outcome = ExecutionOutcome(
            agent_name="planner",
            goal="Create a plan",
            status=ExecutionStatus.SUCCESS,
            duration_ms=100.0,
            output_quality=0.9,
        )
        
        outcome_id = collector.record(outcome)
        recent = collector.get_recent(limit=10)
        
        assert len(recent) == 1
        assert recent[0].agent_name == "planner"
        assert recent[0].id == outcome_id
    
    def test_get_recent_by_agent(self, collector):
        collector.record(ExecutionOutcome(
            agent_name="planner",
            goal="Goal 1",
            status=ExecutionStatus.SUCCESS,
            duration_ms=100.0,
            output_quality=0.9,
        ))
        collector.record(ExecutionOutcome(
            agent_name="coder",
            goal="Goal 2",
            status=ExecutionStatus.SUCCESS,
            duration_ms=200.0,
            output_quality=0.8,
        ))
        
        planner_outcomes = collector.get_recent(agent_name="planner")
        
        assert len(planner_outcomes) == 1
        assert planner_outcomes[0].agent_name == "planner"
    
    def test_get_stats(self, collector):
        collector.record(ExecutionOutcome(
            agent_name="planner",
            goal="Goal 1",
            status=ExecutionStatus.SUCCESS,
            duration_ms=100.0,
            output_quality=0.9,
        ))
        collector.record(ExecutionOutcome(
            agent_name="planner",
            goal="Goal 2",
            status=ExecutionStatus.FAILURE,
            duration_ms=200.0,
            output_quality=0.0,
        ))
        
        stats = collector.get_stats(agent_name="planner")
        
        assert stats["total"] == 2
        assert "by_status" in stats
        assert stats["by_status"]["success"]["count"] == 1
        assert stats["by_status"]["failure"]["count"] == 1
    
    def test_success_rate(self, collector):
        # 2 successes, 1 failure
        collector.record(ExecutionOutcome(
            agent_name="planner", goal="G1",
            status=ExecutionStatus.SUCCESS, duration_ms=100.0, output_quality=0.9,
        ))
        collector.record(ExecutionOutcome(
            agent_name="planner", goal="G2",
            status=ExecutionStatus.SUCCESS, duration_ms=100.0, output_quality=0.9,
        ))
        collector.record(ExecutionOutcome(
            agent_name="planner", goal="G3",
            status=ExecutionStatus.FAILURE, duration_ms=100.0, output_quality=0.0,
        ))
        
        rate = collector.get_success_rate(agent_name="planner")
        
        assert rate == 2 / 3
    
    def test_clear_old(self, collector):
        import time
        
        # Add outcome
        collector.record(ExecutionOutcome(
            agent_name="planner", goal="Goal",
            status=ExecutionStatus.SUCCESS, duration_ms=100.0, output_quality=0.9,
        ))
        
        # Wait briefly then clear with 0 days
        time.sleep(0.1)
        cutoff_before = time.time()
        collector.clear_old(days=0)
        
        recent = collector.get_recent()
        # All entries older than cutoff should be cleared
        assert all(o.timestamp.timestamp() >= cutoff_before for o in recent) or len(recent) == 0


class TestExecutionStatus:
    def test_status_values(self):
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.FAILURE.value == "failure"
        assert ExecutionStatus.PARTIAL.value == "partial"
        assert ExecutionStatus.TIMEOUT.value == "timeout"
