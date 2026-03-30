"""Shared models for the hierarchical swarm workflow."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SwarmTopology(str, Enum):
    """Supported swarm topologies."""

    HIERARCHICAL = "hierarchical"
    MESH = "mesh"


class AgentRole(str, Enum):
    """Known swarm agent roles."""

    COORDINATOR = "coordinator"
    ARCHITECT = "architect"
    CODER = "coder"
    TESTER = "tester"
    DEBUGGER = "debugger"


class TaskState(str, Enum):
    """Lifecycle state for swarm tasks."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"


class SDLCLens(str, Enum):
    """Root-cause analysis lenses spanning the SDLC."""

    REQUIREMENTS = "requirements"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    INTEGRATION = "integration"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    OPERATIONS = "operations"
    DX = "developer_experience"
    DELIVERY = "delivery"


class SupervisorConfig(BaseModel):
    """Configuration for the hierarchical supervisor runtime."""

    topology: SwarmTopology = SwarmTopology.HIERARCHICAL
    learning_interval: int = 5
    retros_dir: str = ".aura_forge/retros"
    max_parallel_tasks: int = 3
    github_delivery_enabled: bool = False
    mcp_ports: Dict[str, int] = Field(
        default_factory=lambda: {
            "aura": 8001,
            "skills": 8002,
            "planning": 8003,
            "memory": 8004,
            "testing": 8005,
            "diagnostics": 8006,
            "github": 8007,
        }
    )
    github_delivery_enabled: bool = False


class SwarmTask(BaseModel):
    """A single unit of work routed to a specialized agent."""

    task_id: str
    title: str
    role: AgentRole
    story_id: str
    description: str
    acceptance_criteria: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    state: TaskState = TaskState.PENDING


class TaskResult(BaseModel):
    """Normalized result emitted by any worker agent."""

    task_id: str
    role: AgentRole
    state: TaskState
    summary: str
    evidence: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    tests_passed: bool = False
    error_message: Optional[str] = None
    output: Dict[str, Any] = Field(default_factory=dict)


class CycleLesson(BaseModel):
    """A reusable lesson derived from a prior cycle."""

    cycle_number: int
    lesson: str
    source_task_id: str
    confidence: float = 0.8


class SDLCFinding(BaseModel):
    """A single failure observation mapped to an SDLC lens."""

    lens: SDLCLens
    severity: str
    observation: str
    probable_cause: str
    recommended_action: str


class DebugReport(BaseModel):
    """Structured root-cause report used to create debug follow-up tasks."""

    task_id: str
    failure_summary: str
    findings: List[SDLCFinding] = Field(default_factory=list)
    recovery_plan: List[str] = Field(default_factory=list)
    should_retry: bool = True


class PRGateDecision(BaseModel):
    """Decision returned by the PR gate after a cycle finishes."""

    should_open_pr: bool
    reason: str
    github_server_port: int = 8007


class CycleReport(BaseModel):
    """Top-level cycle outcome persisted to memory and retros."""

    cycle_number: int
    story_id: str
    tasks: List[SwarmTask] = Field(default_factory=list)
    results: List[TaskResult] = Field(default_factory=list)
    lessons_injected: List[CycleLesson] = Field(default_factory=list)
    debug_report: Optional[DebugReport] = None
    pr_gate: Optional[PRGateDecision] = None
