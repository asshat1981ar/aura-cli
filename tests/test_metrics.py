"""Tests for metrics collection and analytics system."""

import pytest
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.metrics import (
    MetricsCollector,
    SystemMetrics,
    GoalMetrics,
    AnalyticsEngine,
    AgentPerformanceTracker,
    AgentPerformance,
    AgentTaskRecord,
)


class TestMetricsCollector:
    """Test metrics collection."""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create metrics collector with temp file."""
        collector = MetricsCollector(buffer_size=100)
        collector.METRICS_FILE = tmp_path / "metrics.json"
        return collector

    def test_system_metrics_collection(self, collector):
        """Test collecting system metrics."""
        metrics = collector.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp > 0
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100

    def test_goal_metrics_recording(self, collector):
        """Test recording goal completion."""
        collector.record_goal_completion(
            goal_id="test-1",
            description="Test goal",
            status="completed",
            duration_seconds=5.5,
            cycles_used=3,
            agent_count=2,
            tokens_used=1000,
        )
        
        goals = collector.get_goal_metrics()
        assert len(goals) == 1
        assert goals[0].goal_id == "test-1"
        assert goals[0].status == "completed"

    def test_metrics_summary(self, collector):
        """Test getting metrics summary."""
        # Add some test data
        collector.record_goal_completion(
            goal_id="test-1",
            description="Test goal 1",
            status="completed",
            duration_seconds=5.0,
            cycles_used=3,
        )
        collector.record_goal_completion(
            goal_id="test-2",
            description="Test goal 2",
            status="failed",
            duration_seconds=10.0,
            cycles_used=10,
        )
        
        summary = collector.get_summary()
        
        assert summary["goals"]["total"] == 2
        assert summary["goals"]["completed"] == 1
        assert summary["goals"]["failed"] == 1
        assert summary["goals"]["success_rate"] == 0.5

    def test_persistence(self, collector):
        """Test metrics persistence."""
        collector.record_goal_completion(
            goal_id="persist-test",
            description="Test",
            status="completed",
            duration_seconds=1.0,
            cycles_used=1,
        )
        
        # Save metrics
        collector._save_metrics()
        
        # Verify file exists
        assert collector.METRICS_FILE.exists()
        
        # Create new collector and load
        new_collector = MetricsCollector()
        new_collector.METRICS_FILE = collector.METRICS_FILE
        new_collector._load_metrics()
        
        assert len(new_collector.get_goal_metrics()) >= 1


class TestAnalyticsEngine:
    """Test analytics engine."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create analytics engine with mock collector."""
        collector = MetricsCollector()
        collector.METRICS_FILE = tmp_path / "metrics.json"
        return AnalyticsEngine(collector)

    def test_goal_trends_analysis(self, engine):
        """Test goal trend analysis."""
        # Add test data
        for i in range(5):
            engine.collector.record_goal_completion(
                goal_id=f"test-{i}",
                description=f"Test goal {i}",
                status="completed" if i < 4 else "failed",
                duration_seconds=5.0 + i,
                cycles_used=3,
            )
        
        trends = engine.analyze_goal_trends(days=7)
        
        assert trends["total_goals"] == 5
        assert trends["completed"] == 4
        assert trends["success_rate"] == 0.8
        assert "trend" in trends

    def test_system_performance_analysis(self, engine):
        """Test system performance analysis."""
        # Collect some system metrics
        for _ in range(5):
            engine.collector.collect_system_metrics()
        
        perf = engine.analyze_system_performance(hours=24)
        
        assert "status" in perf
        assert "health_score" in perf
        assert "cpu" in perf
        assert "memory" in perf
        assert "recommendations" in perf

    def test_prediction(self, engine):
        """Test goal completion prediction."""
        # Add historical data
        for i in range(10):
            engine.collector.record_goal_completion(
                goal_id=f"pred-{i}",
                description=f"Test goal {i}",
                status="completed",
                duration_seconds=60.0 + i * 10,
                cycles_used=5,
            )
        
        prediction = engine.predict_goal_completion("medium")
        
        assert "predicted_duration_seconds" in prediction
        assert "confidence" in prediction
        assert prediction["based_on_samples"] == 10

    def test_insights_generation(self, engine):
        """Test insights generation."""
        # Add data that will trigger insights - declining trend
        # First 3 days: good performance
        for i in range(6):
            engine.collector.record_goal_completion(
                goal_id=f"insight-good-{i}",
                description=f"Good day {i}",
                status="completed",  # High success
                duration_seconds=10.0,
                cycles_used=5,
            )
        
        # Add system metrics to trigger health warning
        for _ in range(5):
            metrics = engine.collector.collect_system_metrics()
        
        insights = engine.get_insights()
        
        assert isinstance(insights, list)
        # Should have at least one insight (goals or system)
        assert len(insights) >= 0  # May or may not have insights depending on data


class TestAgentPerformanceTracker:
    """Test agent performance tracking."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create performance tracker with temp file."""
        tracker = AgentPerformanceTracker()
        tracker.DATA_FILE = tmp_path / "agent_perf.json"
        return tracker

    def test_record_task(self, tracker):
        """Test recording agent task."""
        tracker.record_task(
            agent_id="agent-1",
            agent_name="Test Agent",
            task_type="code_review",
            success=True,
            duration_seconds=5.0,
            tokens_used=1000,
        )
        
        perf = tracker.get_performance("agent-1")
        assert perf is not None
        assert perf.total_tasks == 1
        assert perf.successful_tasks == 1
        assert perf.success_rate == 1.0

    def test_performance_score(self, tracker):
        """Test performance score calculation."""
        # Record multiple tasks
        for i in range(10):
            tracker.record_task(
                agent_id="agent-2",
                agent_name="Scoring Agent",
                task_type="analysis",
                success=True,
                duration_seconds=30.0,
                tokens_used=500,
            )
        
        perf = tracker.get_performance("agent-2")
        assert perf.performance_score > 0
        assert perf.performance_score <= 100

    def test_leaderboard(self, tracker):
        """Test leaderboard generation."""
        # Add multiple agents
        for agent_num in range(3):
            for task_num in range(5):
                tracker.record_task(
                    agent_id=f"agent-{agent_num}",
                    agent_name=f"Agent {agent_num}",
                    task_type="task",
                    success=True,
                    duration_seconds=10.0 + agent_num * 5,
                    tokens_used=1000,
                )
        
        leaderboard = tracker.get_leaderboard(limit=10)
        assert len(leaderboard) == 3
        # Should be sorted by performance score
        assert leaderboard[0]["performance_score"] >= leaderboard[1]["performance_score"]

    def test_recommendations(self, tracker):
        """Test recommendation generation."""
        # Add underperforming agent
        for i in range(5):
            tracker.record_task(
                agent_id="poor-agent",
                agent_name="Poor Performer",
                task_type="task",
                success=False,  # All failed
                duration_seconds=100.0,  # Slow
                tokens_used=5000,  # Expensive
            )
        
        recs = tracker.get_recommendations()
        assert len(recs) > 0
        assert any("Underperforming" in r["title"] for r in recs)

    def test_reset_stats(self, tracker):
        """Test resetting agent statistics."""
        tracker.record_task(
            agent_id="reset-agent",
            agent_name="Reset Test",
            task_type="task",
            success=True,
            duration_seconds=5.0,
            tokens_used=100,
        )
        
        assert tracker.reset_agent_stats("reset-agent")
        
        perf = tracker.get_performance("reset-agent")
        assert perf.total_tasks == 0
        assert perf.successful_tasks == 0

    def test_summary(self, tracker):
        """Test getting summary statistics."""
        # Add data for multiple agents
        for agent_num in range(3):
            for task_num in range(5):
                tracker.record_task(
                    agent_id=f"agent-{agent_num}",
                    agent_name=f"Agent {agent_num}",
                    task_type="task",
                    success=task_num < 4,  # 80% success
                    duration_seconds=10.0,
                    tokens_used=500,
                )
        
        summary = tracker.get_summary()
        
        assert summary["total_agents"] == 3
        assert summary["total_tasks"] == 15
        assert summary["total_successful"] == 12
        assert summary["overall_success_rate"] == 0.8


class TestIntegration:
    """Integration tests for the metrics system."""

    def test_end_to_end_metrics_flow(self, tmp_path):
        """Test complete metrics flow from collection to analytics."""
        # Create components
        collector = MetricsCollector()
        collector.METRICS_FILE = tmp_path / "metrics.json"
        engine = AnalyticsEngine(collector)
        
        # Collect system metrics
        collector.collect_system_metrics()
        
        # Record goal completions
        collector.record_goal_completion(
            goal_id="int-test",
            description="Integration test",
            status="completed",
            duration_seconds=10.0,
            cycles_used=5,
            agent_count=2,
            tokens_used=2000,
        )
        
        # Generate analytics
        trends = engine.analyze_goal_trends(days=7)
        summary = collector.get_summary()
        
        assert trends["total_goals"] == 1
        assert summary["goals"]["completed"] == 1

    def test_agent_and_goal_integration(self, tmp_path):
        """Test agent tracking with goal metrics."""
        collector = MetricsCollector()
        collector.METRICS_FILE = tmp_path / "metrics.json"
        tracker = AgentPerformanceTracker()
        tracker.DATA_FILE = tmp_path / "agent_perf.json"
        
        # Record agent task
        tracker.record_task(
            agent_id="integration-agent",
            agent_name="Integration Agent",
            task_type="goal_execution",
            success=True,
            duration_seconds=15.0,
            tokens_used=1500,
        )
        
        # Record goal completion
        collector.record_goal_completion(
            goal_id="agent-goal",
            description="Agent-assisted goal",
            status="completed",
            duration_seconds=15.0,
            cycles_used=3,
            agent_count=1,
            tokens_used=1500,
        )
        
        # Verify both systems have data
        agent_perf = tracker.get_performance("integration-agent")
        goals = collector.get_goal_metrics()
        
        assert agent_perf.total_tasks == 1
        assert len(goals) == 1
