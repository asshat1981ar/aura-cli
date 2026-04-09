"""Handler wrapper for CoderAgent.

Provides the standard ``handle(task, context) -> dict`` dispatch interface.

``task`` keys used:
    task (str): The implementation task description.
    goal (str, optional): Alias for ``task`` — used if ``task`` is absent.

``context`` keys used:
    agent (CoderAgent, optional): Pre-initialised agent instance.
    brain (optional): Brain instance used to construct a fresh agent.
    model (optional): Model instance used to construct a fresh agent.
    tester (optional): Optional tester passed when constructing a fresh agent.
"""

from __future__ import annotations

from core.logging_utils import log_json


def handle(task: dict, context: dict) -> dict:
    """Dispatch a code-generation task to CoderAgent.

    Args:
        task: Task payload.  The ``"task"`` key (or ``"goal"`` fallback) is
            passed to :meth:`~agents.coder.CoderAgent.implement`.
        context: Execution context.  Must contain either an ``"agent"`` key
            holding a pre-initialised :class:`~agents.coder.CoderAgent`, or
            ``"brain"`` + ``"model"`` keys used to construct one on-demand.

    Returns:
        ``{"code": "<raw LLM output string>"}`` on success, or
        ``{"error": "<message>"}`` on failure.  The code string may contain
        markdown fences; pass it to :mod:`agents.handlers.applicator` to
        extract and write the file.
    """
    try:
        agent = _resolve_agent(context)
        task_text = task.get("task") or task.get("goal", "")
        log_json("INFO", "handler_coder_start", details={"task_snippet": task_text[:80]})
        raw_code = agent.implement(task_text)
        log_json("INFO", "handler_coder_done", details={"output_chars": len(raw_code or "")})
        return {"code": raw_code}
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "handler_coder_failed", details={"error": str(exc)})
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_agent(context: dict):
    """Return a CoderAgent from context, constructing one if needed."""
    agent = context.get("agent")
    if agent is not None:
        return agent

    brain = context.get("brain")
    model = context.get("model")
    if brain is None or model is None:
        raise ValueError(
            "handlers/coder: context must contain 'agent' or both 'brain' and 'model'"
        )

    from agents.coder import CoderAgent  # deferred import
    tester = context.get("tester")
    return CoderAgent(brain=brain, model=model, tester=tester)
