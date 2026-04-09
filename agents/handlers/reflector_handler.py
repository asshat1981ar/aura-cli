"""Reflector phase handler for the AURA agent pipeline.

Exposes :func:`run_reflector_phase` — the standard entry point called by
``aura_cli/dispatch.py`` and other orchestration layers.

This module is a thin adapter over :mod:`agents.handlers.reflector`, adding the
``run_<name>_phase`` naming convention expected by dispatch.py.

Usage::

    from agents.handlers.reflector_handler import run_reflector_phase

    result = run_reflector_phase(
        context={},
        verification={"status": "pass", "failures": []},
        next_actions=["commit", "bump version"],
    )
    # result["learnings"] contains post-cycle insights
"""

from __future__ import annotations

from core.logging_utils import log_json
from agents.handlers import reflector as _reflector_handler


def run_reflector_phase(context: dict, **kwargs) -> dict:
    """Run the reflection phase after a pipeline cycle completes.

    Delegates to :func:`agents.handlers.reflector.handle` with ``**kwargs``
    assembled into a ``task`` dict.

    Args:
        context: Execution context.  An ``"agent"`` key may supply a
            pre-initialised :class:`~agents.reflector.ReflectorAgent`;
            otherwise a module-level shared instance is used (the agent is
            stateless).
        **kwargs: Task payload fields forwarded verbatim to the underlying
            handler.  Common keys: ``verification``, ``next_actions``,
            ``skill_context``, ``pipeline_run_id``.

    Returns:
        ``{"summary": ..., "learnings": [...], "next_actions": [...],
        "skill_summary": ...}`` on success, or
        ``{"error": "<message>", "phase": "reflector"}`` on failure.
    """
    log_json("INFO", "phase_start", details={"phase": "reflector"})
    try:
        result = _reflector_handler.handle(task=kwargs, context=context)
        log_json("INFO", "phase_end", details={"phase": "reflector", "ok": "error" not in result})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "phase_error", details={"phase": "reflector", "error": str(exc)})
        return {"error": str(exc), "phase": "reflector"}
