"""Handler wrapper for ApplicatorAgent.

Provides the standard ``handle(task, context) -> dict`` dispatch interface.

``task`` keys used:
    llm_output (str): Raw LLM text containing a ```python``` code block.
    target_path (str, optional): Explicit destination file path.  When
        omitted the agent looks for a ``# AURA_TARGET:`` directive in the
        code block.
    allow_overwrite (bool, optional): Whether overwriting an existing file is
        permitted.  Defaults to ``True``.
    action (str, optional): ``"apply"`` (default) or ``"rollback"``.
    apply_result (dict, optional): For ``"rollback"`` action — serialised
        :class:`~agents.applicator.ApplyResult` from a previous apply call.

``context`` keys used:
    agent (ApplicatorAgent, optional): Pre-initialised agent instance.
    brain (optional): Brain instance used to construct a fresh agent.
    backup_dir (str, optional): Backup directory passed to a freshly
        constructed agent.  Defaults to ``".aura/backups"``.
"""

from __future__ import annotations

from core.logging_utils import log_json


def handle(task: dict, context: dict) -> dict:
    """Dispatch an apply/rollback task to ApplicatorAgent.

    Args:
        task: Task payload — see module docstring.
        context: Execution context.  Must contain either ``"agent"`` or
            ``"brain"``.

    Returns:
        For ``"apply"``: dict serialisation of
        :class:`~agents.applicator.ApplyResult` (``success``, ``target_path``,
        ``backup_path``, ``code``, ``error``, ``metadata``).
        For ``"rollback"``: ``{"rolled_back": True/False}``.
        On unexpected failure: ``{"error": "<message>"}``.
    """
    try:
        agent = _resolve_agent(context)
        action = task.get("action", "apply")

        if action == "rollback":
            return _handle_rollback(agent, task)

        return _handle_apply(agent, task)
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "handler_applicator_failed", details={"error": str(exc)})
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _handle_apply(agent, task: dict) -> dict:
    llm_output = task.get("llm_output", "")
    target_path = task.get("target_path")
    allow_overwrite = task.get("allow_overwrite", True)

    log_json(
        "INFO",
        "handler_applicator_apply_start",
        details={"target_path": target_path, "allow_overwrite": allow_overwrite},
    )
    result = agent.apply(
        llm_output=llm_output,
        target_path=target_path,
        allow_overwrite=allow_overwrite,
    )
    log_json(
        "INFO",
        "handler_applicator_apply_done",
        details={"success": result.success, "target_path": result.target_path},
    )
    return {
        "success": result.success,
        "target_path": result.target_path,
        "backup_path": result.backup_path,
        "code": result.code,
        "error": result.error,
        "metadata": result.metadata,
    }


def _handle_rollback(agent, task: dict) -> dict:
    from agents.applicator import ApplyResult  # deferred import

    raw = task.get("apply_result", {})
    apply_result = ApplyResult(
        success=raw.get("success", False),
        target_path=raw.get("target_path"),
        backup_path=raw.get("backup_path"),
        code=raw.get("code"),
        error=raw.get("error"),
        metadata=raw.get("metadata", {}),
    )

    log_json(
        "INFO",
        "handler_applicator_rollback_start",
        details={"target_path": apply_result.target_path},
    )
    rolled_back = agent.rollback(apply_result)
    log_json(
        "INFO",
        "handler_applicator_rollback_done",
        details={"rolled_back": rolled_back},
    )
    return {"rolled_back": rolled_back}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_agent(context: dict):
    """Return an ApplicatorAgent from context, constructing one if needed."""
    agent = context.get("agent")
    if agent is not None:
        return agent

    brain = context.get("brain")
    if brain is None:
        raise ValueError(
            "handlers/applicator: context must contain 'agent' or 'brain'"
        )

    from agents.applicator import ApplicatorAgent  # deferred import
    backup_dir = context.get("backup_dir", ".aura/backups")
    return ApplicatorAgent(brain=brain, backup_dir=backup_dir)
