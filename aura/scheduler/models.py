"""Data models for task scheduler."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional


class TaskStatus(Enum):
    """Status of a scheduled task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduleType(Enum):
    """Type of schedule."""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    DELAYED = "delayed"


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0


@dataclass
class ScheduledTask:
    """A scheduled task."""
    name: str
    func: Callable
    schedule_type: ScheduleType
    # For ONCE: run_at datetime
    # For INTERVAL: interval in seconds
    # For DELAYED: delay in seconds
    # For CRON: cron expression string
    schedule_value: Any
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    run_at: Optional[datetime] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    result: Optional[TaskResult] = None
    
    def __post_init__(self):
        if self.schedule_type == ScheduleType.ONCE and isinstance(self.schedule_value, datetime):
            self.run_at = self.schedule_value
            self.next_run = self.schedule_value
        elif self.schedule_type == ScheduleType.DELAYED and isinstance(self.schedule_value, (int, float)):
            self.run_at = datetime.utcnow() + timedelta(seconds=self.schedule_value)
            self.next_run = self.run_at
        elif self.schedule_type == ScheduleType.INTERVAL and isinstance(self.schedule_value, (int, float)):
            # Store as timedelta for consistency
            self.schedule_value = timedelta(seconds=self.schedule_value)
            self.next_run = datetime.utcnow() + self.schedule_value
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "schedule_type": self.schedule_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
        }


@dataclass
class SchedulerConfig:
    """Configuration for the task scheduler."""
    max_workers: int = 4
    default_timeout: int = 300
    retry_failed: bool = True
    max_retries: int = 3
    retry_delay: int = 60
    timezone: str = "UTC"
