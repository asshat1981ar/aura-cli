"""Tests for scheduler models."""

import pytest
from datetime import datetime, timedelta

from aura.scheduler.models import (
    ScheduleType,
    ScheduledTask,
    SchedulerConfig,
    TaskResult,
    TaskStatus,
)


class TestTaskStatus:
    def test_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestScheduleType:
    def test_type_values(self):
        assert ScheduleType.ONCE.value == "once"
        assert ScheduleType.INTERVAL.value == "interval"
        assert ScheduleType.CRON.value == "cron"
        assert ScheduleType.DELAYED.value == "delayed"


class TestTaskResult:
    def test_success_result(self):
        result = TaskResult(success=True, output="done")
        
        assert result.success is True
        assert result.output == "done"
        assert result.error is None
        assert result.duration_ms == 0.0
    
    def test_failure_result(self):
        result = TaskResult(success=False, error="failed")
        
        assert result.success is False
        assert result.error == "failed"
        assert result.output is None
    
    def test_duration_calculation(self):
        started = datetime.utcnow()
        completed = started + timedelta(seconds=2)
        
        result = TaskResult(
            success=True,
            started_at=started,
            completed_at=completed,
        )
        
        assert result.duration_ms == 2000.0


class TestScheduledTask:
    def test_once_task(self):
        run_at = datetime.utcnow() + timedelta(hours=1)
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            schedule_type=ScheduleType.ONCE,
            schedule_value=run_at,
        )
        
        assert task.schedule_type == ScheduleType.ONCE
        assert task.run_at == run_at
        assert task.next_run == run_at
        assert task.status == TaskStatus.PENDING
        assert task.run_count == 0
    
    def test_delayed_task(self):
        task = ScheduledTask(
            name="delayed",
            func=lambda: None,
            schedule_type=ScheduleType.DELAYED,
            schedule_value=300,  # 5 minutes
        )
        
        assert task.schedule_type == ScheduleType.DELAYED
        # run_at should be ~5 minutes in the future
        assert task.run_at > datetime.utcnow()
        assert task.run_at <= datetime.utcnow() + timedelta(seconds=301)
    
    def test_interval_task(self):
        task = ScheduledTask(
            name="interval",
            func=lambda: None,
            schedule_type=ScheduleType.INTERVAL,
            schedule_value=60,  # 60 seconds
            max_runs=5,
        )
        
        assert task.schedule_type == ScheduleType.INTERVAL
        assert task.max_runs == 5
        assert task.next_run is not None
    
    def test_task_with_args(self):
        task = ScheduledTask(
            name="with_args",
            func=lambda x, y: x + y,
            schedule_type=ScheduleType.ONCE,
            schedule_value=datetime.utcnow(),
            args=(1, 2),
            kwargs={"z": 3},
        )
        
        assert task.args == (1, 2)
        assert task.kwargs == {"z": 3}
    
    def test_to_dict(self):
        run_at = datetime.utcnow()
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            schedule_type=ScheduleType.ONCE,
            schedule_value=run_at,
        )
        
        data = task.to_dict()
        
        assert data["name"] == "test"
        assert data["schedule_type"] == "once"
        assert data["status"] == "pending"
        assert "id" in data


class TestSchedulerConfig:
    def test_default_config(self):
        config = SchedulerConfig()
        
        assert config.max_workers == 4
        assert config.default_timeout == 300
        assert config.retry_failed is True
        assert config.max_retries == 3
        assert config.retry_delay == 60
        assert config.timezone == "UTC"
    
    def test_custom_config(self):
        config = SchedulerConfig(
            max_workers=8,
            default_timeout=600,
            retry_failed=False,
        )
        
        assert config.max_workers == 8
        assert config.default_timeout == 600
        assert config.retry_failed is False
