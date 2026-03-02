"""Canonical BEADS bridge contracts used by the Python runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


BEADS_SCHEMA_VERSION = 1

BeadsDecisionStatus = Literal["allow", "block", "revise"]
BeadsBridgeStatus = Literal["ok", "error"]


class BeadsInput(TypedDict):
    schema_version: int
    goal: str
    goal_type: str | None
    runtime_mode: str
    project_root: str
    queue_summary: dict[str, Any]
    active_context: dict[str, Any]
    prd_context: dict[str, Any] | None
    conductor_track: dict[str, Any] | None


class BeadsDecision(TypedDict):
    schema_version: int
    decision_id: str
    status: BeadsDecisionStatus
    summary: str
    rationale: list[str]
    required_constraints: list[str]
    required_skills: list[str]
    required_tests: list[str]
    follow_up_goals: list[str]
    stop_reason: str | None


class BeadsResult(TypedDict):
    schema_version: int
    ok: bool
    status: BeadsBridgeStatus
    decision: BeadsDecision | None
    error: str | None
    stderr: str | None
    duration_ms: int


@dataclass(frozen=True)
class BeadsBridgeConfig:
    """Runtime configuration for BEADS bridge execution."""

    command: tuple[str, ...]
    timeout_seconds: float = 20.0
    enabled: bool = True
    required: bool = True
    persist_artifacts: bool = True
    scope: str = "goal_run"
    env: dict[str, str] = field(default_factory=dict)
