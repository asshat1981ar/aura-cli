"""Structured result of each orchestration cycle for learning/adaptation."""
from __future__ import annotations
import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CycleOutcome:
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    goal_type: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    phases_completed: List[str] = field(default_factory=list)
    changes_applied: int = 0
    tests_before: int = 0
    tests_after: int = 0
    tests_delta: int = 0
    lint_score_before: float = 0.0
    lint_score_after: float = 0.0
    strategy_used: str = ""
    success: bool = False
    failure_phase: Optional[str] = None
    failure_reason: Optional[str] = None
    brain_entries_added: int = 0

    def duration_s(self) -> float:
        return self.completed_at - self.started_at

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "CycleOutcome":
        d = json.loads(s)
        return cls(**d)

    def mark_complete(
        self,
        success: bool,
        failure_phase: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        self.completed_at = time.time()
        self.success = success
        self.failure_phase = failure_phase
        self.failure_reason = failure_reason
        self.tests_delta = self.tests_after - self.tests_before
