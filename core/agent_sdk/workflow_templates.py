"""Workflow templates and executor for the Agent SDK meta-controller.

Defines predefined phase sequences for common goal types (bug_fix, feature,
refactor) with retry policies and escalation rules.
"""
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowPhase:
    """A single phase in a workflow template."""

    tool_name: str
    required: bool = True
    retry_on_fail: int = 0
    escalate_on_fail: bool = False


@dataclass
class WorkflowTemplate:
    """Ordered phase sequence for a goal type."""

    name: str
    goal_types: List[str]
    phases: List[WorkflowPhase]
    max_retries_total: int = 3
    verification_mode: str = "post"  # "post" | "pre_and_post" | "none"


@dataclass
class PhaseResult:
    """Result of executing a single workflow phase."""

    tool_name: str
    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    model_used: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class WorkflowResult:
    """Result of executing a complete workflow."""

    success: bool
    phases_completed: int
    phase_results: List[PhaseResult]
    total_cost_usd: float = 0.0
    model_escalations: int = 0
    error_summary: Optional[str] = None


class FailureAction(enum.Enum):
    """What to do when a phase fails."""

    RETRY_PHASE = "retry_phase"
    ESCALATE_AND_RETRY = "escalate"
    REPLAN = "replan"
    ABORT = "abort"


def get_builtin_templates() -> Dict[str, WorkflowTemplate]:
    """Return the three built-in workflow templates."""
    return {
        "bug_fix": WorkflowTemplate(
            name="bug_fix",
            goal_types=["bug_fix"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="search_memory", required=False),
                WorkflowPhase(tool_name="dispatch_skills", required=False),
                WorkflowPhase(tool_name="create_plan"),
                WorkflowPhase(tool_name="generate_code", retry_on_fail=3, escalate_on_fail=True),
                WorkflowPhase(tool_name="run_sandbox", retry_on_fail=2),
                WorkflowPhase(tool_name="apply_changes"),
                WorkflowPhase(tool_name="verify_changes"),
                WorkflowPhase(tool_name="reflect_on_outcome", required=False),
                WorkflowPhase(tool_name="store_memory", required=False),
            ],
            verification_mode="post",
        ),
        "feature": WorkflowTemplate(
            name="feature",
            goal_types=["feature", "default"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="search_memory", required=False),
                WorkflowPhase(tool_name="dispatch_skills", required=False),
                WorkflowPhase(tool_name="create_plan"),
                WorkflowPhase(tool_name="critique_plan"),
                WorkflowPhase(tool_name="generate_code", retry_on_fail=2, escalate_on_fail=True),
                WorkflowPhase(tool_name="run_sandbox", retry_on_fail=2),
                WorkflowPhase(tool_name="apply_changes"),
                WorkflowPhase(tool_name="verify_changes"),
                WorkflowPhase(tool_name="reflect_on_outcome", required=False),
                WorkflowPhase(tool_name="store_memory", required=False),
            ],
            verification_mode="post",
        ),
        "refactor": WorkflowTemplate(
            name="refactor",
            goal_types=["refactor"],
            phases=[
                WorkflowPhase(tool_name="analyze_goal"),
                WorkflowPhase(tool_name="search_memory", required=False),
                WorkflowPhase(tool_name="dispatch_skills", required=False),
                WorkflowPhase(tool_name="verify_changes"),  # baseline
                WorkflowPhase(tool_name="create_plan"),
                WorkflowPhase(tool_name="generate_code", retry_on_fail=2),
                WorkflowPhase(tool_name="apply_changes"),
                WorkflowPhase(tool_name="verify_changes"),  # regression check
                WorkflowPhase(tool_name="reflect_on_outcome", required=False),
                WorkflowPhase(tool_name="store_memory", required=False),
            ],
            verification_mode="pre_and_post",
        ),
    }


class WorkflowExecutor:
    """Execute workflow templates by dispatching tool handlers in order."""

    def __init__(
        self,
        templates: Dict[str, WorkflowTemplate],
        tool_handlers: Dict[str, Callable[[Dict], Dict]],
    ) -> None:
        self._templates = templates
        self._handlers = tool_handlers

    def select_workflow(self, goal_type: str) -> WorkflowTemplate:
        """Select workflow by goal type, falling back to feature."""
        for wf in self._templates.values():
            if goal_type in wf.goal_types:
                return wf
        return self._templates.get("feature", list(self._templates.values())[0])

    def execute(
        self,
        workflow: WorkflowTemplate,
        goal: str,
        context: Dict[str, Any],
    ) -> WorkflowResult:
        """Execute all phases in order with retry support."""
        results: List[PhaseResult] = []
        total_retries = 0

        for phase in workflow.phases:
            pr = self._run_phase(phase, goal, context)
            results.append(pr)

            if pr.success:
                continue

            # Phase failed — try retries
            if not phase.required:
                continue  # skip optional failures

            retried = False
            for attempt in range(phase.retry_on_fail):
                if total_retries >= workflow.max_retries_total:
                    break
                total_retries += 1
                pr = self._run_phase(phase, goal, context)
                results.append(pr)
                if pr.success:
                    retried = True
                    break

            if not pr.success and phase.required:
                return WorkflowResult(
                    success=False,
                    phases_completed=len([r for r in results if r.success]),
                    phase_results=results,
                    error_summary=pr.error or f"Phase {phase.tool_name} failed",
                )

        return WorkflowResult(
            success=True,
            phases_completed=len([r for r in results if r.success]),
            phase_results=results,
        )

    def _run_phase(
        self, phase: WorkflowPhase, goal: str, context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute a single phase via its tool handler."""
        handler = self._handlers.get(phase.tool_name)
        if handler is None:
            return PhaseResult(
                tool_name=phase.tool_name,
                success=not phase.required,
                output={},
                error=f"No handler for {phase.tool_name}",
            )

        start = time.monotonic()
        try:
            output = handler({"goal": goal, **context})
            elapsed = (time.monotonic() - start) * 1000
            has_error = "error" in output and output["error"]
            return PhaseResult(
                tool_name=phase.tool_name,
                success=not has_error,
                output=output,
                error=output.get("error"),
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return PhaseResult(
                tool_name=phase.tool_name,
                success=False,
                output={},
                error=str(exc),
                elapsed_ms=elapsed,
            )
