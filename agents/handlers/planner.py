"""Handler wrapper for PlannerAgent.

Provides the standard ``handle(task, context) -> dict`` dispatch interface.

``task`` keys used:
    goal (str): The goal to plan for.
    memory_snapshot (str, optional): Serialised memory context.
    similar_past_problems (str, optional): Prior similar problems.
    known_weaknesses (str, optional): Documented agent weaknesses.
    backfill_context (list, optional): Low-coverage modules.

``context`` keys used (at least one required):
    agent (PlannerAgent, optional): Pre-initialised agent instance.
    brain (optional): Brain instance used to construct a fresh agent.
    model (optional): Model instance used to construct a fresh agent.
"""

from __future__ import annotations

from core.logging_utils import log_json


def handle(task: dict, context: dict) -> dict:
    """Dispatch a planning task to PlannerAgent.

    Args:
        task: Task payload — see module docstring for supported keys.
        context: Execution context.  Must contain either an ``"agent"`` key
            holding a pre-initialised :class:`~agents.planner.PlannerAgent`,
            or ``"brain"`` + ``"model"`` keys used to construct one on-demand.

    Returns:
        dict with ``"steps"`` key (list of plan strings) on success, or
        ``{"error": "<message>"}`` on failure.
    """
    try:
        agent = _resolve_agent(context)
        input_data = {
            "goal": task.get("goal", ""),
            "memory_snapshot": task.get("memory_snapshot", ""),
            "similar_past_problems": task.get("similar_past_problems", ""),
            "known_weaknesses": task.get("known_weaknesses", ""),
            "backfill_context": task.get("backfill_context", []),
        }
        log_json("INFO", "handler_planner_start", details={"goal_snippet": input_data["goal"][:80]})
        result = agent.run(input_data)
        log_json("INFO", "handler_planner_done", details={"step_count": len(result.get("steps", []))})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "handler_planner_failed", details={"error": str(exc)})
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_agent(context: dict):
    """Return a PlannerAgent from context, constructing one if needed."""
    agent = context.get("agent")
    if agent is not None:
        return agent

    brain = context.get("brain")
    model = context.get("model")
    if brain is None or model is None:
        raise ValueError("handlers/planner: context must contain 'agent' or both 'brain' and 'model'")

    from agents.planner import PlannerAgent  # deferred import

    return PlannerAgent(brain=brain, model=model)
