from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

@dataclass
class RetryPolicy:
    """Exponential-backoff retry configuration for a single step."""
    max_attempts: int = 3
    backoff_base: float = 0.5   # seconds; sleep = backoff_base * 2^attempt
    max_backoff: float = 30.0   # cap on sleep duration

    def sleep_for(self, attempt: int) -> float:
        return min(self.backoff_base * (2 ** attempt), self.max_backoff)


@dataclass
class WorkflowStep:
    """
    One step in a workflow.

    Args:
        name:           Unique label within the workflow.
        skill_name:     Name of an AURA skill (from agents/skills/registry) to run,
                        OR None if `fn` is provided.
        fn:             Callable(inputs: dict) -> dict — used when skill_name is None.
        static_inputs:  Hard-coded inputs merged before calling the step.
        inputs_from:    Wire previous step outputs: {"my_key": "step_name.output_key"}.
                        "step_name.output_key" means step_outputs[step_name][output_key].
                        Use "step_name.*" to forward the entire output dict.
        skip_if_false:  Name of an earlier step's boolean output key.  If that
                        key is falsy the step is skipped (status → "skipped").
        retry:          Retry/backoff policy (default: 3 attempts).
        timeout_s:      Hard wall-clock timeout in seconds (0 = no limit).
    """
    name: str
    skill_name: Optional[str] = None
    fn: Optional[Callable[[Dict], Dict]] = None
    static_inputs: Dict[str, Any] = field(default_factory=dict)
    inputs_from: Dict[str, str] = field(default_factory=dict)
    skip_if_false: Optional[str] = None   # e.g. "check_security.critical_count"
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_s: float = 120.0


@dataclass
class WorkflowDefinition:
    """
    Named, ordered list of steps.  Steps execute sequentially; parallel
    execution can be composed by grouping steps with a fan-out/fan-in pattern
    via the `inputs_from` wiring.
    """
    name: str
    steps: List[WorkflowStep]
    description: str = ""
    max_retries_total: int = 0  # extra safety: abort after N total retries across all steps


@dataclass
class StepResult:
    step_name: str
    status: Literal["ok", "skipped", "failed", "timeout"]
    output: Dict[str, Any]
    attempts: int
    elapsed_ms: float
    error: Optional[str] = None


@dataclass
class WorkflowExecution:
    id: str
    workflow_name: str
    status: Literal["pending", "running", "paused", "completed", "failed", "cancelled"]
    current_step_index: int
    step_outputs: Dict[str, Dict]   # {step_name: output_dict}
    history: List[StepResult]
    initial_inputs: Dict[str, Any]
    error: Optional[str]
    started_at: float
    updated_at: float
    total_retries_used: int = 0

    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")


@dataclass
class LoopCycle:
    cycle_number: int
    status: Literal["ok", "failed", "skipped"]
    phase_outputs: Dict[str, Any]
    elapsed_ms: float
    error: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class AgenticLoop:
    id: str
    goal: str
    max_cycles: int
    current_cycle: int
    status: Literal["running", "paused", "completed", "failed", "stopped"]
    history: List[LoopCycle]
    stop_reason: Optional[str]
    score: float
    started_at: float
    updated_at: float

    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "stopped")
