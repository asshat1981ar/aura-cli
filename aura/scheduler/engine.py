"""Task scheduling engine."""

import asyncio
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .models import (
    ScheduleType,
    ScheduledTask,
    SchedulerConfig,
    TaskResult,
    TaskStatus,
)


class TaskScheduler:
    """Task scheduler with support for various scheduling types."""
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def schedule_once(
        self,
        name: str,
        func: Callable,
        run_at: datetime,
        *args,
        **kwargs,
    ) -> ScheduledTask:
        """Schedule a task to run once at a specific time."""
        task = ScheduledTask(
            name=name,
            func=func,
            schedule_type=ScheduleType.ONCE,
            schedule_value=run_at,
            args=args,
            kwargs=kwargs,
        )
        self._tasks[task.id] = task
        return task
    
    def schedule_delayed(
        self,
        name: str,
        func: Callable,
        delay_seconds: float,
        *args,
        **kwargs,
    ) -> ScheduledTask:
        """Schedule a task to run after a delay."""
        task = ScheduledTask(
            name=name,
            func=func,
            schedule_type=ScheduleType.DELAYED,
            schedule_value=delay_seconds,
            args=args,
            kwargs=kwargs,
        )
        self._tasks[task.id] = task
        return task
    
    def schedule_interval(
        self,
        name: str,
        func: Callable,
        interval_seconds: float,
        max_runs: Optional[int] = None,
        *args,
        **kwargs,
    ) -> ScheduledTask:
        """Schedule a task to run at intervals."""
        task = ScheduledTask(
            name=name,
            func=func,
            schedule_type=ScheduleType.INTERVAL,
            schedule_value=interval_seconds,
            args=args,
            kwargs=kwargs,
            max_runs=max_runs,
        )
        self._tasks[task.id] = task
        return task
    
    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_tasks(self, status: Optional[TaskStatus] = None) -> List[ScheduledTask]:
        """Get all tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks
    
    async def run_task(self, task: ScheduledTask) -> TaskResult:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.utcnow()
        task.run_count += 1
        
        started_at = datetime.utcnow()
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=self.config.default_timeout,
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, task.func, *task.args, **task.kwargs),
                    timeout=self.config.default_timeout,
                )
            
            task.result = TaskResult(
                success=True,
                output=result,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
            task.status = TaskStatus.COMPLETED
            
        except Exception as e:
            task.result = TaskResult(
                success=False,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
            task.status = TaskStatus.FAILED
        
        # Update next_run for interval tasks
        if task.schedule_type == ScheduleType.INTERVAL:
            if task.max_runs is None or task.run_count < task.max_runs:
                task.next_run = datetime.utcnow() + task.schedule_value
                task.status = TaskStatus.PENDING
            else:
                task.next_run = None
        
        return task.result
    
    async def start(self):
        """Start the scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.utcnow()
                
                # Find tasks that should run
                pending_tasks = [
                    t for t in self._tasks.values()
                    if t.status == TaskStatus.PENDING
                    and t.next_run
                    and t.next_run <= now
                ]
                
                # Run tasks concurrently
                if pending_tasks:
                    await asyncio.gather(
                        *[self.run_task(t) for t in pending_tasks],
                        return_exceptions=True,
                    )
                
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
    
    def clear_completed(self):
        """Remove completed and cancelled tasks."""
        to_remove = [
            task_id for task_id, task in self._tasks.items()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
        ]
        for task_id in to_remove:
            del self._tasks[task_id]
