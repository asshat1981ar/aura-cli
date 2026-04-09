"""Handler wrapper for DebuggerAgent.

Provides the standard ``handle(task, context) -> dict`` dispatch interface.

``task`` keys used:
    error_message (str): The error text to diagnose.
    goal (str, optional): The goal being pursued when the error occurred.
    context_text (str, optional): Relevant code or log context.
    improve_plan (str, optional): Previous IMPROVE plan, if any.
    implement_details (dict, optional): Implementation step that triggered the error.

``context`` keys used:
    agent (DebuggerAgent, optional): Pre-initialised agent instance.
    brain (optional): Brain instance used to construct a fresh agent.
    model (optional): Model instance used to construct a fresh agent.
"""

from __future__ import annotations

from core.logging_utils import log_json


def handle(task: dict, context: dict) -> dict:
    """Dispatch a debugging task to DebuggerAgent.

    Args:
        task: Task payload — see module docstring.
        context: Execution context.  Must contain either ``"agent"`` or
            ``"brain"`` + ``"model"``.

    Returns:
        dict with ``"summary"``, ``"diagnosis"``, ``"fix_strategy"``,
        and ``"severity"`` keys on success, or ``{"error": "<message>"}``
        on failure.
    """
    try:
        agent = _resolve_agent(context)
        error_message = task.get("error_message", "")
        goal = task.get("goal", "")
        context_text = task.get("context_text", "")
        improve_plan = task.get("improve_plan", "")
        implement_details = task.get("implement_details") or {}

        log_json("INFO", "handler_debugger_start", details={"error_snippet": error_message[:80]})
        result = agent.diagnose(
            error_message=error_message,
            current_goal=goal,
            context=context_text,
            improve_plan=improve_plan,
            implement_details=implement_details,
        )
        log_json("INFO", "handler_debugger_done", details={"severity": result.get("severity")})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "handler_debugger_failed", details={"error": str(exc)})
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_agent(context: dict):
    """Return a DebuggerAgent from context, constructing one if needed."""
    agent = context.get("agent")
    if agent is not None:
        return agent

    brain = context.get("brain")
    model = context.get("model")
    if brain is None or model is None:
        raise ValueError(
            "handlers/debugger: context must contain 'agent' or both 'brain' and 'model'"
        )

    from agents.debugger import DebuggerAgent  # deferred import
    return DebuggerAgent(brain=brain, model=model)
