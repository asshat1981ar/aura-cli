"""Handler wrapper for CriticAgent.

Provides the standard ``handle(task, context) -> dict`` dispatch interface.

``task`` keys used:
    mode (str): ``"plan"`` (default), ``"code"``, or ``"mutation"``.
    goal (str): The original goal / task description.
    plan (list[str], optional): Plan steps — required when mode is ``"plan"``.
    code (str, optional): Source code to critique — required when mode is ``"code"``.
    requirements (str, optional): Requirements text for code critique.
    mutation_proposal (str, optional): Mutation description — required when
        mode is ``"mutation"``.

``context`` keys used:
    agent (CriticAgent, optional): Pre-initialised agent instance.
    brain (optional): Brain instance used to construct a fresh agent.
    model (optional): Model instance used to construct a fresh agent.
"""

from __future__ import annotations

from core.logging_utils import log_json

_VALID_MODES = {"plan", "code", "mutation"}


def handle(task: dict, context: dict) -> dict:
    """Dispatch a critique task to CriticAgent.

    Args:
        task: Task payload — see module docstring.
        context: Execution context.  Must contain either ``"agent"`` or
            ``"brain"`` + ``"model"``.

    Returns:
        dict with critique result keys on success, or
        ``{"error": "<message>"}`` on failure.
    """
    try:
        agent = _resolve_agent(context)
        mode = task.get("mode", "plan")
        if mode not in _VALID_MODES:
            raise ValueError(f"handlers/critic: unknown mode '{mode}'. Valid: {_VALID_MODES}")

        goal = task.get("goal", "")
        log_json("INFO", "handler_critic_start", details={"mode": mode, "goal_snippet": goal[:80]})

        if mode == "plan":
            plan = task.get("plan", [])
            result = agent.critique_plan(task=goal, plan=plan)
        elif mode == "code":
            code = task.get("code", "")
            requirements = task.get("requirements", "")
            result = agent.critique_code(task=goal, code=code, requirements=requirements)
        else:  # mutation
            proposal = task.get("mutation_proposal", "")
            result = agent.validate_mutation(mutation_proposal=proposal)

        log_json("INFO", "handler_critic_done", details={"mode": mode, "has_result": bool(result)})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "handler_critic_failed", details={"error": str(exc)})
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_agent(context: dict):
    """Return a CriticAgent from context, constructing one if needed."""
    agent = context.get("agent")
    if agent is not None:
        return agent

    brain = context.get("brain")
    model = context.get("model")
    if brain is None or model is None:
        raise ValueError("handlers/critic: context must contain 'agent' or both 'brain' and 'model'")

    from agents.critic import CriticAgent  # deferred import

    return CriticAgent(brain=brain, model=model)
