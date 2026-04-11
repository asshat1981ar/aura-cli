"""Data models for offline mode."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ConnectivityStatus(Enum):
    """Network connectivity status."""

    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class CommandPriority(Enum):
    """Command execution priority."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class CommandStatus(Enum):
    """Status of a queued command."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedCommand:
    """A command in the offline queue."""

    command: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: CommandPriority = CommandPriority.NORMAL
    status: CommandStatus = CommandStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "command": self.command,
            "args": list(self.args),
            "kwargs": self.kwargs,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "retry_count": self.retry_count,
        }


@dataclass
class CommandResult:
    """Result of executing a command."""

    command: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    queued_id: Optional[str] = None
