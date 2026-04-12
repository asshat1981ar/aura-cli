"""
ReAct Loop Engine — interleaved Think-Act-Observe execution loop.

The core agent runtime for AURA. Every subsequent capability hooks into
this loop via the pre_think / post_act / post_observe hook system.

Usage:
    from core.react_engine import ReActLoop, Tool, ToolSchema

    def search(query: str) -> str:
        return f"Results for: {query}"

    loop = ReActLoop(
        llm=my_llm_provider,
        tools=[Tool(name="search", description="Search the web",
                    fn=search, schema=ToolSchema(parameters={"query": {"type": "string"}}))],
        max_steps=10,
    )
    result = loop.run("Find the capital of France")
"""
from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol

from core.logging_utils import log_json


# ---------------------------------------------------------------------------
# Tool schema & registration
# ---------------------------------------------------------------------------

@dataclass
class ToolSchema:
    """JSON-Schema-style parameter description for a tool."""
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def validate(self, args: Dict[str, Any]) -> List[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: List[str] = []
        for key in self.required:
            if key not in args:
                errors.append(f"Missing required parameter: {key}")
        for key, value in args.items():
            if key in self.parameters:
                param_spec = self.parameters[key]
                expected_type = param_spec.get("type") if isinstance(param_spec, dict) else param_spec
                if expected_type and not _type_matches(value, expected_type):
                    errors.append(f"Parameter '{key}' expected type '{expected_type}', got '{type(value).__name__}'")
        return errors


def _type_matches(value: Any, type_str: str) -> bool:
    """Check if a value roughly matches a JSON-Schema type string.

    Supported subset: string, integer, number, boolean, array, object.
    Per JSON-Schema semantics, booleans are NOT valid integers or numbers.
    Unknown type strings pass through (return True).
    """
    # JSON-Schema treats bool as distinct from integer/number
    if type_str in ("integer", "number") and isinstance(value, bool):
        return False
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = mapping.get(type_str)
    if expected is None:
        return True  # unknown type — pass through
    return isinstance(value, expected)


@dataclass
class Tool:
    """A tool that the ReAct agent can invoke."""
    name: str
    description: str
    fn: Callable[..., Any]
    schema: ToolSchema = field(default_factory=ToolSchema)


# ---------------------------------------------------------------------------
# Trace types
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """One step in the reasoning trace."""
    step_number: int
    phase: Literal["think", "act", "observe"]
    timestamp: str
    content: Any
    duration_ms: float = 0.0


@dataclass
class ReActTrace:
    """Complete execution trace for a ReAct run."""
    run_id: str
    goal: str
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    status: Literal["running", "completed", "max_steps", "aborted", "error"] = "running"
    total_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "steps": [
                {
                    "step_number": s.step_number,
                    "phase": s.phase,
                    "timestamp": s.timestamp,
                    "content": s.content,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
        }


# ---------------------------------------------------------------------------
# LLM Provider abstraction
# ---------------------------------------------------------------------------

class LLMProvider(Protocol):
    """Pluggable interface for LLM backends.

    Implementations must provide a ``think`` method that accepts the current
    conversation context and returns a structured decision dict.
    """

    def think(
        self,
        goal: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a decision dict with keys:
        - "thought": str  — the reasoning text
        - "action": str | None  — tool name, or None for final answer
        - "action_input": dict | None  — tool arguments
        - "final_answer": str | None  — set when the agent is done
        """
        ...


# ---------------------------------------------------------------------------
# Hook system
# ---------------------------------------------------------------------------

HookFn = Callable[[Dict[str, Any]], None]


@dataclass
class HookRegistry:
    """Manages pre_think / post_act / post_observe hooks."""
    pre_think: List[HookFn] = field(default_factory=list)
    post_act: List[HookFn] = field(default_factory=list)
    post_observe: List[HookFn] = field(default_factory=list)

    def fire(self, phase: str, payload: Dict[str, Any]) -> None:
        hooks = getattr(self, phase, [])
        for hook in hooks:
            try:
                hook(payload)
            except Exception:
                log_json("WARN", "hook_error", details={
                    "phase": phase,
                    "error": traceback.format_exc(),
                })


# ---------------------------------------------------------------------------
# ReAct Loop
# ---------------------------------------------------------------------------

@dataclass
class ReActResult:
    """Result of a ReAct loop execution."""
    answer: Optional[str]
    trace: ReActTrace
    status: str


class ReActLoop:
    """Core Think-Act-Observe execution loop.

    Args:
        llm: An LLM provider implementing the ``LLMProvider`` protocol.
        tools: List of ``Tool`` objects the agent can call.
        max_steps: Maximum think-act-observe iterations.
        timeout_s: Wall-clock timeout in seconds (0 = no limit).
        stop_condition: Optional callable that receives the trace and returns
            True to stop early.
        hooks: Optional pre-configured hook registry.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: Optional[List[Tool]] = None,
        max_steps: int = 10,
        timeout_s: float = 0,
        stop_condition: Optional[Callable[[ReActTrace], bool]] = None,
        hooks: Optional[HookRegistry] = None,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.stop_condition = stop_condition
        self.hooks = hooks or HookRegistry()
        self._tools: Dict[str, Tool] = {}
        for tool in (tools or []):
            self.register_tool(tool)

    # -- Tool management ----------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def _tool_specs(self) -> List[Dict[str, Any]]:
        """Return tool descriptions for the LLM prompt."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema.parameters,
                "required": t.schema.required,
            }
            for t in self._tools.values()
        ]

    # -- Execution ----------------------------------------------------------

    def run(self, goal: str) -> ReActResult:
        """Execute the ReAct loop for a given goal.

        Returns a ``ReActResult`` with the final answer, trace, and status.
        """
        run_id = uuid.uuid4().hex[:12]
        trace = ReActTrace(run_id=run_id, goal=goal)
        start_time = time.monotonic()
        step_num = 0

        log_json("INFO", "react_loop_start", goal=goal, details={
            "run_id": run_id,
            "max_steps": self.max_steps,
            "tools": list(self._tools.keys()),
        })

        history: List[Dict[str, Any]] = []

        try:
            while step_num < self.max_steps:
                elapsed = (time.monotonic() - start_time) * 1000
                if self.timeout_s > 0 and elapsed > self.timeout_s * 1000:
                    trace.status = "aborted"
                    log_json("WARN", "react_loop_timeout", goal=goal, details={
                        "run_id": run_id, "elapsed_ms": elapsed,
                    })
                    break

                step_num += 1

                # --- THINK ---
                self.hooks.fire("pre_think", {
                    "run_id": run_id, "step": step_num, "goal": goal,
                    "history": history,
                })

                think_start = time.monotonic()
                decision = self.llm.think(
                    goal=goal,
                    tools=self._tool_specs(),
                    history=history,
                )
                think_ms = (time.monotonic() - think_start) * 1000

                thought = decision.get("thought", "")
                trace.steps.append(TraceStep(
                    step_number=step_num, phase="think",
                    timestamp=_now_iso(), content=thought,
                    duration_ms=think_ms,
                ))
                history.append({"role": "think", "content": thought})

                # Check for final answer
                final_answer = decision.get("final_answer")
                if final_answer is not None:
                    trace.final_answer = final_answer
                    trace.status = "completed"
                    log_json("INFO", "react_loop_final_answer", goal=goal, details={
                        "run_id": run_id, "step": step_num, "answer": final_answer,
                    })
                    break

                # --- ACT ---
                action_name = decision.get("action")
                action_input = decision.get("action_input")
                # Normalize: tool execution requires a dict; non-dict/None from LLM → {}
                if not isinstance(action_input, dict):
                    action_input = {}

                if action_name is None:
                    # No action and no final answer — treat as malformed; observe error
                    observation = "Error: LLM returned neither an action nor a final answer."
                else:
                    observation = self._execute_tool(action_name, action_input)

                trace.steps.append(TraceStep(
                    step_number=step_num, phase="act",
                    timestamp=_now_iso(),
                    content={"action": action_name, "input": action_input},
                ))
                history.append({
                    "role": "act",
                    "content": {"action": action_name, "input": action_input},
                })

                self.hooks.fire("post_act", {
                    "run_id": run_id, "step": step_num,
                    "action": action_name, "input": action_input,
                })

                # --- OBSERVE ---
                trace.steps.append(TraceStep(
                    step_number=step_num, phase="observe",
                    timestamp=_now_iso(), content=observation,
                ))
                history.append({"role": "observe", "content": observation})

                self.hooks.fire("post_observe", {
                    "run_id": run_id, "step": step_num,
                    "observation": observation,
                })

                # Check stop condition
                if self.stop_condition and self.stop_condition(trace):
                    trace.status = "completed"
                    break
            else:
                # Exhausted max_steps
                trace.status = "max_steps"
                log_json("WARN", "react_loop_max_steps", goal=goal, details={
                    "run_id": run_id, "max_steps": self.max_steps,
                })

        except Exception:
            trace.status = "error"
            log_json("ERROR", "react_loop_error", goal=goal, details={
                "run_id": run_id, "error": traceback.format_exc(),
            })

        trace.total_duration_ms = (time.monotonic() - start_time) * 1000

        log_json("INFO", "react_loop_end", goal=goal, details={
            "run_id": run_id, "status": trace.status,
            "steps": step_num, "duration_ms": trace.total_duration_ms,
        })

        return ReActResult(
            answer=trace.final_answer,
            trace=trace,
            status=trace.status,
        )

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name, returning the result as a string observation.

        Tool failures are captured as error observations, not exceptions.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'. Available tools: {list(self._tools.keys())}"

        # Validate arguments
        errors = tool.schema.validate(args)
        if errors:
            return f"Error: Invalid arguments for tool '{name}': {'; '.join(errors)}"

        try:
            # Filter to schema-defined parameters to guard against LLM hallucinations
            filtered_args = {k: v for k, v in args.items() if k in tool.schema.parameters}
            result = tool.fn(**filtered_args)
            return str(result)
        except Exception as exc:
            log_json("WARN", "tool_execution_error", details={
                "tool": name, "error": str(exc),
            })
            return f"Error executing tool '{name}': {exc}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
