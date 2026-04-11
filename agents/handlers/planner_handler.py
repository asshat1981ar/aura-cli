"""Planner phase handler for the AURA agent pipeline.

Exposes :func:`run_planner_phase` — the standard entry point called by
``aura_cli/dispatch.py`` and other orchestration layers.

This module is a thin adapter over :mod:`agents.handlers.planner`, adding the
``run_<name>_phase`` naming convention expected by dispatch.py.

Usage::

    from agents.handlers.planner_handler import run_planner_phase

    result = run_planner_phase(
        context={"brain": brain, "model": model},
        goal="Refactor the caching layer",
        memory_snapshot="...",
    )
    # result["steps"] contains the generated plan
"""

from __future__ import annotations

from core.logging_utils import log_json
from agents.handlers import planner as _planner_handler


def run_planner_phase(context: dict, **kwargs) -> dict:
    """Run the planning phase for a given goal.

    Delegates to :func:`agents.handlers.planner.handle` with ``**kwargs``
    assembled into a ``task`` dict.

    Args:
        context: Execution context.  Must contain either an ``"agent"`` key
            (a pre-initialised :class:`~agents.planner.PlannerAgent`) or both
            ``"brain"`` and ``"model"`` keys to construct one on demand.
        **kwargs: Task payload fields forwarded verbatim to the underlying
            handler.  Common keys: ``goal``, ``memory_snapshot``,
            ``similar_past_problems``, ``known_weaknesses``,
            ``backfill_context``.

    Returns:
        ``{"steps": [...]}`` on success, or
        ``{"error": "<message>", "phase": "planner"}`` on failure.
    """
    log_json("INFO", "phase_start", details={"phase": "planner"})
    try:
        result = _planner_handler.handle(task=kwargs, context=context)
        log_json("INFO", "phase_end", details={"phase": "planner", "ok": "error" not in result})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "phase_error", details={"phase": "planner", "error": str(exc)})
        return {"error": str(exc), "phase": "planner"}
