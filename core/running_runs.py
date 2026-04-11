"""Registry of currently active pipeline runs.

Each run is registered here when it starts and deregistered when it completes.
The :func:`register_run` / :func:`deregister_run` helpers are called by the
orchestrator; the ``aura cancel <run-id>`` command calls :func:`cancel_run` to
signal a stop.

Run IDs are arbitrary strings (typically UUID-based). A ``threading.Event``
is stored per run so that any thread can signal cancellation without
sending OS signals.
"""

from __future__ import annotations

import threading
from typing import Any

# Module-level registry: run_id -> {"stop_event": threading.Event, "pid": int | None}
_REGISTRY: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def register_run(run_id: str, pid: int | None = None) -> threading.Event:
    """Register a new active run and return its stop :class:`threading.Event`.

    The orchestrator should call this at the start of each pipeline run and
    periodically check ``stop_event.is_set()`` to honour cancellation requests.

    Args:
        run_id: Unique identifier for the run (e.g. a UUID string).
        pid:    Optional OS PID of the worker process (used for SIGTERM fallback).

    Returns:
        A :class:`threading.Event` that will be set when :func:`cancel_run` is
        called for this *run_id*.
    """
    stop_event = threading.Event()
    with _LOCK:
        _REGISTRY[run_id] = {"stop_event": stop_event, "pid": pid}
    return stop_event


def deregister_run(run_id: str) -> None:
    """Remove *run_id* from the registry (call when a run completes or fails)."""
    with _LOCK:
        _REGISTRY.pop(run_id, None)


def cancel_run(run_id: str) -> bool:
    """Signal cancellation for *run_id*.

    Sets the stop event so the orchestrator loop exits cleanly on the next
    iteration.  As a best-effort fallback, also sends ``SIGTERM`` to the
    registered PID (if any).

    Args:
        run_id: The run to cancel.

    Returns:
        ``True`` if the run was found and signalled; ``False`` if unknown.
    """
    import os
    import signal

    with _LOCK:
        entry = _REGISTRY.get(run_id)

    if entry is None:
        return False

    entry["stop_event"].set()

    pid = entry.get("pid")
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass  # Process already gone or no permission — stop_event is enough.

    return True


def list_runs() -> list[dict[str, Any]]:
    """Return a snapshot of all active runs.

    Returns:
        List of dicts with keys ``run_id``, ``pid``, and ``stop_requested``.
    """
    with _LOCK:
        return [
            {
                "run_id": run_id,
                "pid": entry.get("pid"),
                "stop_requested": entry["stop_event"].is_set(),
            }
            for run_id, entry in _REGISTRY.items()
        ]


def get_stop_event(run_id: str) -> "threading.Event | None":
    """Return the stop event for *run_id*, or ``None`` if not registered."""
    with _LOCK:
        entry = _REGISTRY.get(run_id)
    return entry["stop_event"] if entry is not None else None
