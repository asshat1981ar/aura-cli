"""Tests for task scheduler engine."""

import asyncio
import pytest
from datetime import datetime, timedelta

from aura.scheduler.engine import TaskScheduler
from aura.scheduler.models import ScheduleType, TaskStatus


class TestTaskScheduler:
    @pytest.fixture
    def scheduler(self):
        return TaskScheduler()
    
    def test_schedule_once(self, scheduler):
        run_at = datetime.utcnow() + timedelta(hours=1)
        
        task = scheduler.schedule_once("test", lambda: "done", run_at)
        
        assert task.name == "test"
        assert task.schedule_type == ScheduleType.ONCE
        assert task.run_at == run_at
        assert task.id in scheduler._tasks
    
    def test_schedule_delayed(self, scheduler):
        task = scheduler.schedule_delayed("delayed", lambda: "done", 300)
        
        assert task.schedule_type == ScheduleType.DELAYED
        assert task.run_at > datetime.utcnow()
        assert task.run_at <= datetime.utcnow() + timedelta(seconds=301)
    
    def test_schedule_interval(self, scheduler):
        task = scheduler.schedule_interval("interval", lambda: "done", 60, max_runs=5)
        
        assert task.schedule_type == ScheduleType.INTERVAL
        assert task.max_runs == 5
        assert task.next_run is not None
    
    def test_cancel_task(self, scheduler):
        task = scheduler.schedule_delayed("test", lambda: "done", 300)
        
        result = scheduler.cancel(task.id)
        
        assert result is True
        assert task.status == TaskStatus.CANCELLED
    
    def test_cancel_nonexistent_task(self, scheduler):
        result = scheduler.cancel("nonexistent")
        assert result is False
    
    def test_cancel_running_task(self, scheduler):
        task = scheduler.schedule_delayed("test", lambda: "done", 300)
        task.status = TaskStatus.RUNNING
        
        result = scheduler.cancel(task.id)
        
        assert result is False  # Can't cancel running tasks
    
    def test_get_task(self, scheduler):
        task = scheduler.schedule_delayed("test", lambda: "done", 300)
        
        retrieved = scheduler.get_task(task.id)
        
        assert retrieved == task
    
    def test_get_tasks(self, scheduler):
        task1 = scheduler.schedule_delayed("task1", lambda: "done", 300)
        task2 = scheduler.schedule_delayed("task2", lambda: "done", 600)
        task2.status = TaskStatus.CANCELLED
        
        all_tasks = scheduler.get_tasks()
        pending_tasks = scheduler.get_tasks(TaskStatus.PENDING)
        
        assert len(all_tasks) == 2
        assert len(pending_tasks) == 1
        assert pending_tasks[0] == task1
    
    @pytest.mark.asyncio
    async def test_run_sync_task(self, scheduler):
        def sync_func(x, y):
            return x + y
        
        task = scheduler.schedule_delayed("test", sync_func, 0, 1, 2)
        result = await scheduler.run_task(task)
        
        assert result.success is True
        assert result.output == 3
        assert task.status == TaskStatus.COMPLETED
        assert task.run_count == 1
    
    @pytest.mark.asyncio
    async def test_run_async_task(self, scheduler):
        async def async_func(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        task = scheduler.schedule_delayed("test", async_func, 0, 5)
        result = await scheduler.run_task(task)
        
        assert result.success is True
        assert result.output == 10
    
    @pytest.mark.asyncio
    async def test_run_failing_task(self, scheduler):
        def failing_func():
            raise ValueError("Test error")
        
        task = scheduler.schedule_delayed("test", failing_func, 0)
        result = await scheduler.run_task(task)
        
        assert result.success is False
        assert "Test error" in result.error
        assert task.status == TaskStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_run_interval_task_updates_next_run(self, scheduler):
        def func():
            return "done"
        
        task = scheduler.schedule_interval("test", func, 60)
        before = datetime.utcnow()
        
        await scheduler.run_task(task)
        
        assert task.next_run is not None
        assert task.next_run > before
        assert task.status == TaskStatus.PENDING  # Reset to pending for next run
    
    @pytest.mark.asyncio
    async def test_run_interval_task_max_runs(self, scheduler):
        def func():
            return "done"
        
        task = scheduler.schedule_interval("test", func, 60, max_runs=1)
        
        await scheduler.run_task(task)
        
        assert task.run_count == 1
        assert task.next_run is None  # No more runs
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_scheduler_loop_runs_pending_tasks(self, scheduler):
        results = []
        
        def collect():
            results.append(1)
            return "done"
        
        # Schedule task to run immediately
        scheduler.schedule_delayed("test", collect, 0)
        
        await scheduler.start()
        await asyncio.sleep(0.5)  # Wait for scheduler to run
        await scheduler.stop()
        
        assert len(results) >= 1
    
    @pytest.mark.asyncio
    async def test_clear_completed(self, scheduler):
        task1 = scheduler.schedule_delayed("task1", lambda: "done", 0)
        task2 = scheduler.schedule_delayed("task2", lambda: "done", 300)
        
        # Mark task1 as completed
        task1.status = TaskStatus.COMPLETED
        
        scheduler.clear_completed()
        
        assert task1.id not in scheduler._tasks
        assert task2.id in scheduler._tasks
    
    def test_schedule_with_args_and_kwargs(self, scheduler):
        def func(a, b, c=None):
            return (a, b, c)
        
        task = scheduler.schedule_delayed("test", func, 0, 1, 2, c=3)
        
        assert task.args == (1, 2)
        assert task.kwargs == {"c": 3}
