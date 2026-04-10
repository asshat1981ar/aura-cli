"""Shared mutable server state accessible by all routers.

This module is the single source of truth for runtime state set by the
lifespan handler in server.py.  Routers import the *module* reference and
access attributes through it so they always see the latest values.

Usage (in a router)::

    from aura_cli.api import state as _state

    async def my_endpoint():
        orch = _state.orchestrator
        if orch is None:
            raise HTTPException(503, detail="orchestrator not initialised")
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Mutable runtime singletons — set by server._apply_runtime_state()
# ---------------------------------------------------------------------------

runtime: Dict[str, Any] = {}
orchestrator: Any = None
model_adapter: Any = None
memory_store: Any = None
_runtime_init_error: Optional[str] = None

# Resolved once at process start; stays constant thereafter.
PROJECT_ROOT: Path = Path.cwd()


async def resolve_runtime_component(name: str) -> Any:
    """Return a named runtime component; raise 503 if unavailable.

    Defers to server._ensure_runtime_initialized() the first time so that
    routers never need to import from server.py directly.
    """
    # Check module-level state first (set by _apply_runtime_state).
    component = globals().get(name)
    if component is not None:
        return component

    # Lazy-init: trigger runtime bootstrap via the server module.
    if not runtime:
        try:
            from aura_cli import server as _server  # type: ignore[attr-defined]

            await _server._ensure_runtime_initialized()
        except Exception:
            pass
        component = globals().get(name)
        if component is not None:
            return component

    detail = f"{name} is not configured"
    if _runtime_init_error and not runtime:
        detail = f"{detail}: {_runtime_init_error}"
    raise HTTPException(status_code=503, detail=detail)
