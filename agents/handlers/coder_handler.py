"""Coder phase handler for the AURA agent pipeline.

Exposes :func:`run_coder_phase` — the standard entry point called by
``aura_cli/dispatch.py`` and other orchestration layers.

This module is a thin adapter over :mod:`agents.handlers.coder`, adding the
``run_<name>_phase`` naming convention expected by dispatch.py.

Usage::

    from agents.handlers.coder_handler import run_coder_phase

    result = run_coder_phase(
        context={"brain": brain, "model": model},
        task="Implement LRU cache in core/cache.py",
    )
    # result["code"] contains the raw LLM output
"""

from __future__ import annotations

from core.logging_utils import log_json
from agents.handlers import coder as _coder_handler


def run_coder_phase(context: dict, **kwargs) -> dict:
    """Run the code-generation phase for a given task.

    Delegates to :func:`agents.handlers.coder.handle` with ``**kwargs``
    assembled into a ``task`` dict.

    Args:
        context: Execution context.  Must contain either an ``"agent"`` key
            (a pre-initialised :class:`~agents.coder.CoderAgent`) or both
            ``"brain"`` and ``"model"`` keys to construct one on demand.
            An optional ``"tester"`` key may also be provided.
        **kwargs: Task payload fields forwarded verbatim to the underlying
            handler.  Common keys: ``task``, ``goal``.

    Returns:
        ``{"code": "<raw LLM output>"}`` on success, or
        ``{"error": "<message>", "phase": "coder"}`` on failure.
    """
    log_json("INFO", "phase_start", details={"phase": "coder"})
    try:
        result = _coder_handler.handle(task=kwargs, context=context)
        log_json("INFO", "phase_end", details={"phase": "coder", "ok": "error" not in result})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "phase_error", details={"phase": "coder", "error": str(exc)})
        return {"error": str(exc), "phase": "coder"}
