"""SADD type definitions — dataclasses and validation for the SADD system."""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Workstream specification (parsed from design doc)
# ---------------------------------------------------------------------------


@dataclass
class WorkstreamSpec:
    """A single workstream extracted from a design spec."""

    id: str
    title: str
    goal_text: str
    priority: int = 1
    estimated_cycles: int = 5
    depends_on: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    execution_mode: Literal["dry_run", "automatic", "approval_required"] = "automatic"


# ---------------------------------------------------------------------------
# Workstream result (normalized contract for dependency handoff)
# ---------------------------------------------------------------------------


@dataclass
class WorkstreamResult:
    """Normalized output from a completed workstream execution."""

    ws_id: str
    status: Literal["completed", "failed", "skipped"] = "completed"
    cycles_used: int = 0
    stop_reason: Optional[str] = None
    changed_files: List[str] = field(default_factory=list)
    verification_summary: str = ""
    reflector_output: str = ""
    elapsed_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkstreamResult:
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}})


# ---------------------------------------------------------------------------
# Design spec (top-level parsed document)
# ---------------------------------------------------------------------------


@dataclass
class DesignSpec:
    """Parsed design specification containing workstreams."""

    title: str
    summary: str
    workstreams: List[WorkstreamSpec]
    raw_markdown: str = ""
    parse_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DesignSpec:
        ws_data = data.pop("workstreams", [])
        workstreams = [WorkstreamSpec(**w) for w in ws_data]
        return cls(workstreams=workstreams, **{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)} and k != "workstreams"})


# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Configuration for a SADD session."""

    max_parallel: int = 3
    max_cycles_per_workstream: int = 5
    dry_run: bool = False
    fail_fast: bool = False
    retry_failed: bool = True


# ---------------------------------------------------------------------------
# Session report and workstream outcome
# ---------------------------------------------------------------------------


@dataclass
class WorkstreamOutcome:
    """Result summary for a single workstream in a session report."""

    id: str
    title: str
    status: str
    cycles_used: int = 0
    stop_reason: Optional[str] = None
    elapsed_s: float = 0.0
    artifacts: List[str] = field(default_factory=list)


@dataclass
class SessionReport:
    """Aggregated report for a completed SADD session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    design_title: str = ""
    total_workstreams: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    outcomes: List[WorkstreamOutcome] = field(default_factory=list)
    elapsed_s: float = 0.0
    learnings: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        """Human-readable summary of the session."""
        lines = [
            f"SADD Session: {self.design_title}",
            f"  Workstreams: {self.total_workstreams} total, {self.completed} completed, {self.failed} failed, {self.skipped} skipped",
            f"  Elapsed: {self.elapsed_s:.1f}s",
        ]
        if self.outcomes:
            lines.append("  Outcomes:")
            for o in self.outcomes:
                lines.append(f"    [{o.status}] {o.title} ({o.cycles_used} cycles, {o.elapsed_s:.1f}s)")
        if self.learnings:
            lines.append("  Learnings:")
            for l in self.learnings:
                lines.append(f"    - {l}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class SADDValidationError(Exception):
    """Raised when SADD types fail validation."""

    pass


def validate_spec(spec: DesignSpec) -> List[str]:
    """Validate a DesignSpec, returning a list of error messages (empty = valid)."""
    errors: List[str] = []

    if not spec.title:
        errors.append("DesignSpec.title is required")

    if not spec.workstreams:
        errors.append("DesignSpec must contain at least one workstream")

    ws_ids = set()
    for ws in spec.workstreams:
        if not ws.id:
            errors.append(f"WorkstreamSpec.id is required (title={ws.title!r})")
        if ws.id in ws_ids:
            errors.append(f"Duplicate workstream ID: {ws.id!r}")
        ws_ids.add(ws.id)

        if not ws.title:
            errors.append(f"WorkstreamSpec.title is required (id={ws.id!r})")

        if not ws.goal_text:
            errors.append(f"WorkstreamSpec.goal_text is required (id={ws.id!r})")

        if ws.priority < 1:
            errors.append(f"WorkstreamSpec.priority must be >= 1 (id={ws.id!r})")

        if ws.estimated_cycles < 1:
            errors.append(f"WorkstreamSpec.estimated_cycles must be >= 1 (id={ws.id!r})")

        for dep_id in ws.depends_on:
            if dep_id not in ws_ids and dep_id not in {w.id for w in spec.workstreams}:
                errors.append(f"Workstream {ws.id!r} depends on unknown ID {dep_id!r}")

    return errors


def validate_result(result: WorkstreamResult) -> List[str]:
    """Validate a WorkstreamResult, returning a list of error messages."""
    errors: List[str] = []

    if not result.ws_id:
        errors.append("WorkstreamResult.ws_id is required")

    if result.status not in ("completed", "failed", "skipped"):
        errors.append(f"Invalid status: {result.status!r}")

    if result.cycles_used < 0:
        errors.append("WorkstreamResult.cycles_used must be >= 0")

    if result.elapsed_s < 0:
        errors.append("WorkstreamResult.elapsed_s must be >= 0")

    return errors
