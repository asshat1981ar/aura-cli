"""Applicator phase handler for the AURA agent pipeline.

Exposes :func:`run_applicator_phase` — the standard entry point called by
``aura_cli/dispatch.py`` and other orchestration layers.

This module is a thin adapter over :mod:`agents.handlers.applicator`, adding
the ``run_<name>_phase`` naming convention expected by dispatch.py.

Usage::

    from agents.handlers.applicator_handler import run_applicator_phase

    result = run_applicator_phase(
        context={"brain": brain},
        llm_output="```python\\n# AURA_TARGET: src/cache.py\\n...\\n```",
    )
    # result["success"] is True if the file was written
"""

from __future__ import annotations

from core.logging_utils import log_json
from agents.handlers import applicator as _applicator_handler


def run_applicator_phase(context: dict, **kwargs) -> dict:
    """Run the application phase to write LLM-generated code to disk.

    Delegates to :func:`agents.handlers.applicator.handle` with ``**kwargs``
    assembled into a ``task`` dict.

    Args:
        context: Execution context.  Must contain either an ``"agent"`` key
            (a pre-initialised :class:`~agents.applicator.ApplicatorAgent`)
            or a ``"brain"`` key to construct one on demand.  An optional
            ``"backup_dir"`` key overrides the default backup location.
        **kwargs: Task payload fields forwarded verbatim to the underlying
            handler.  Common keys: ``llm_output``, ``target_path``,
            ``allow_overwrite``, ``action``, ``apply_result``.

    Returns:
        For ``action="apply"``: ``{"success": bool, "target_path": ...,
        "backup_path": ..., "code": ..., "error": ..., "metadata": ...}``.
        For ``action="rollback"``: ``{"rolled_back": bool}``.
        On unexpected failure: ``{"error": "<message>", "phase": "applicator"}``.
    """
    log_json("INFO", "phase_start", details={"phase": "applicator"})
    try:
        result = _applicator_handler.handle(task=kwargs, context=context)
        log_json("INFO", "phase_end", details={"phase": "applicator", "ok": "error" not in result})
        return result
    except Exception as exc:  # pragma: no cover
        log_json("ERROR", "phase_error", details={"phase": "applicator", "error": str(exc)})
        return {"error": str(exc), "phase": "applicator"}
