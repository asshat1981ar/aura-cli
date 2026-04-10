"""Debugger phase handler for the AURA agent pipeline.

Exposes :func:`run_debugger_phase` — the standard entry point called by
``aura_cli/dispatch.py`` and other orchestration layers.

This module is a thin adapter over :mod:`agents.handlers.debugger`, adding the
``run_<name>_phase`` naming convention expected by dispatch.py.

Usage::

    from agents.handlers.debugger_handler import run_debugger_phase

    result = run_debugger_phase(
        context={"brain": brain, "model": model},
        error_message="AttributeError: 'NoneType' object has no attribute 'run'",
        goal="Implement LRU cache",
    )
    # result["fix_strategy"] contains the diagnosis
"""

from __future__ import annotations

from core.logging_utils import log_json
from agents.handlers import debugger as _debugger_handler


def run_debugger_phase(context: dict, **kwargs) -> dict:
    """Run the debugging phase to diagnose an error.

    Delegates to :func:`agents.handlers.debugger.handle` with ``**kwargs``
    assembled into a ``task`` dict.

    Args:
        context: Execution context.  Must contain either an ``"agent"`` key
            (a pre-initialised :class:`~agents.debugger.DebuggerAgent`) or
            both ``"brain"`` and ``"model"`` keys to construct one on demand.
        **kwargs: Task payload fields forwarded verbatim to the underlying
            handler.  Common keys: ``error_message``, ``goal``,
            ``context_text``, ``improve_plan``, ``implement_details``.

    Returns:
        ``{"summary": ..., "diagnosis": ..., "fix_strategy": ...,
        "severity": ...}`` on success, or
        ``{"error": "<message>", "phase": "debugger"}`` on failure.
    """
    log_json("INFO", "phase_start", details={"phase": "debugger"})
    try:
        result = _debugger_handler.handle(task=kwargs, context=context)
        log_json("INFO", "phase_end", details={"phase": "debugger", "ok": "error" not in result})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "phase_error", details={"phase": "debugger", "error": str(exc)})
        return {"error": str(exc), "phase": "debugger"}
