"""A2A Task lifecycle with state machine.

Tasks are the core unit of work in A2A. They progress through a well-defined
state machine: submitted -> working -> completed/failed/canceled.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class A2AMessage:
    """A message in the A2A task conversation."""

    role: str  # "user" or "agent"
    content: str
    parts: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class A2ATask:
    """An A2A task with full lifecycle management."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capability: str = ""
    state: TaskState = TaskState.SUBMITTED
    messages: list[A2AMessage] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Valid state transitions
    _TRANSITIONS: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._TRANSITIONS = {
            TaskState.SUBMITTED: {
                TaskState.WORKING,
                TaskState.CANCELED,
                TaskState.FAILED,
            },
            TaskState.WORKING: {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.INPUT_REQUIRED,
                TaskState.CANCELED,
            },
            TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELED},
        }

    def transition(self, new_state: TaskState):
        """Transition task to new state with validation."""
        allowed = self._TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(f"Invalid transition: {self.state.value} -> {new_state.value}")
        self.state = new_state
        self.updated_at = time.time()

    def add_message(self, role: str, content: str, parts: list[dict] | None = None):
        self.messages.append(A2AMessage(role=role, content=content, parts=parts or []))
        self.updated_at = time.time()

    def add_artifact(self, name: str, content: Any, mime_type: str = "application/json"):
        self.artifacts.append(
            {
                "name": name,
                "content": content,
                "mime_type": mime_type,
            }
        )
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "capability": self.capability,
            "state": self.state.value,
            "messages": [{"role": m.role, "content": m.content, "parts": m.parts, "timestamp": m.timestamp} for m in self.messages],
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
