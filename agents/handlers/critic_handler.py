"""Critic phase handler for the AURA agent pipeline.

Exposes :func:`run_critic_phase` — the standard entry point called by
``aura_cli/dispatch.py`` and other orchestration layers.

This module is a thin adapter over :mod:`agents.handlers.critic`, adding the
``run_<name>_phase`` naming convention expected by dispatch.py.

Usage::

    from agents.handlers.critic_handler import run_critic_phase

    result = run_critic_phase(
        context={"brain": brain, "model": model},
        mode="plan",
        goal="Refactor caching",
        plan=["step 1", "step 2"],
    )
"""

from __future__ import annotations

from core.logging_utils import log_json
from agents.handlers import critic as _critic_handler


def run_critic_phase(context: dict, **kwargs) -> dict:
    """Run the critique phase for a plan, code block, or mutation proposal.

    Delegates to :func:`agents.handlers.critic.handle` with ``**kwargs``
    assembled into a ``task`` dict.

    Args:
        context: Execution context.  Must contain either an ``"agent"`` key
            (a pre-initialised :class:`~agents.critic.CriticAgent`) or both
            ``"brain"`` and ``"model"`` keys to construct one on demand.
        **kwargs: Task payload fields forwarded verbatim to the underlying
            handler.  Common keys: ``mode`` (``"plan"`` / ``"code"`` /
            ``"mutation"``), ``goal``, ``plan``, ``code``, ``requirements``,
            ``mutation_proposal``.

    Returns:
        Critique result dict on success, or
        ``{"error": "<message>", "phase": "critic"}`` on failure.
    """
    log_json("INFO", "phase_start", details={"phase": "critic"})
    try:
        result = _critic_handler.handle(task=kwargs, context=context)
        log_json("INFO", "phase_end", details={"phase": "critic", "ok": "error" not in result})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "phase_error", details={"phase": "critic", "error": str(exc)})
        return {"error": str(exc), "phase": "critic"}
