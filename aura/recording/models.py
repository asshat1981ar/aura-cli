"""Data models for command recording and replay."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class StepStatus(Enum):
    """Status of a recording step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RecordingStep:
    """A single step in a recording."""

    command: str
    args: List[str] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: int = 60
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "args": self.args,
            "kwargs": self.kwargs,
            "condition": self.condition,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class Recording:
    """A recording session with steps."""

    name: str
    description: str = ""
    steps: List[RecordingStep] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def add_step(self, step: RecordingStep):
        """Add a step to the recording."""
        self.steps.append(step)

    @property
    def duration_ms(self) -> int:
        """Estimated duration based on retry delays."""
        return int(sum(s.retry_delay * s.retry_count for s in self.steps) * 1000)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "variables": self.variables,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Recording":
        """Create Recording from dictionary."""
        recording = cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            variables=data.get("variables", {}),
        )
        for step_data in data.get("steps", []):
            step = RecordingStep(
                command=step_data["command"],
                args=step_data.get("args", []),
                kwargs=step_data.get("kwargs", {}),
                condition=step_data.get("condition"),
                retry_count=step_data.get("retry_count", 3),
                retry_delay=step_data.get("retry_delay", 1.0),
                timeout=step_data.get("timeout", 60),
            )
            recording.add_step(step)
        return recording


@dataclass
class ReplayResult:
    """Result of replaying a recording."""

    recording_name: str
    success: bool
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    variables: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.step_results if r.get("success"))

    @property
    def failed_count(self) -> int:
        return len(self.step_results) - self.success_count

    def to_dict(self) -> dict:
        return {
            "recording_name": self.recording_name,
            "success": self.success,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "step_results": self.step_results,
            "variables": self.variables,
        }
