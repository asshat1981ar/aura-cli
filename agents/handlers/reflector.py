"""Handler wrapper for ReflectorAgent.

Provides the standard ``handle(task, context) -> dict`` dispatch interface.

``task`` keys used:
    verification (dict, optional): Verification phase output
        (``status``, ``failures``).
    next_actions (list, optional): Proposed next actions.
    skill_context (dict, optional): Skill dispatch outputs.
    pipeline_run_id (str, optional): ID of the current pipeline run.

``context`` keys used:
    agent (ReflectorAgent, optional): Pre-initialised agent instance.
    brain (optional): Unused — ReflectorAgent takes no brain argument;
        included for interface consistency.
    model (optional): Unused — see above.
"""

from __future__ import annotations

from core.logging_utils import log_json

# Module-level singleton — ReflectorAgent is stateless and has no
# constructor arguments, so we can share one instance.
_SHARED_AGENT = None


def handle(task: dict, context: dict) -> dict:
    """Dispatch a reflection task to ReflectorAgent.

    Args:
        task: Task payload — see module docstring.
        context: Execution context.  An ``"agent"`` key may be provided for
            injection; otherwise the module-level shared instance is used.

    Returns:
        dict with ``"summary"``, ``"learnings"``, ``"next_actions"``, and
        ``"skill_summary"`` keys on success, or ``{"error": "<message>"}``
        on failure.
    """
    try:
        agent = _resolve_agent(context)
        input_data = {
            "verification": task.get("verification", {}),
            "next_actions": task.get("next_actions", []),
            "skill_context": task.get("skill_context", {}),
            "pipeline_run_id": task.get("pipeline_run_id"),
        }
        log_json(
            "INFO",
            "handler_reflector_start",
            details={"verify_status": input_data["verification"].get("status", "unknown")},
        )
        result = agent.run(input_data)
        log_json(
            "INFO",
            "handler_reflector_done",
            details={"learnings_count": len(result.get("learnings", []))},
        )
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "handler_reflector_failed", details={"error": str(exc)})
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_agent(context: dict):
    """Return a ReflectorAgent from context, using a shared instance as fallback."""
    global _SHARED_AGENT
    agent = context.get("agent")
    if agent is not None:
        return agent

    if _SHARED_AGENT is None:
        from agents.reflector import ReflectorAgent  # deferred import

        _SHARED_AGENT = ReflectorAgent()

    return _SHARED_AGENT
