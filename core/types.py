from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Context:
    goal: str
    snapshot: str
    memory_summary: str
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    steps: List[str]
    risks: List[str] = field(default_factory=list)


@dataclass
class Critique:
    issues: List[str]
    fixes: List[str] = field(default_factory=list)


@dataclass
class TaskSpec:
    id: str
    title: str
    intent: str
    files: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)


@dataclass
class TaskBundle:
    tasks: List[TaskSpec]


@dataclass
class Change:
    file_path: str
    old_code: str
    new_code: str
    overwrite_file: bool = False


@dataclass
class ChangeSet:
    changes: List[Change]


@dataclass
class Verification:
    status: str  # "pass" | "fail" | "skip"
    failures: List[str] = field(default_factory=list)
    logs: str = ""


@dataclass
class Reflection:
    summary: str
    learnings: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)


@dataclass
class DecisionLogEntry:
    cycle_id: str
    phase_outputs: Dict[str, Any]
    stop_reason: Optional[str] = None
