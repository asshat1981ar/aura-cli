"""AURA API routers — endpoint definitions by domain."""

from __future__ import annotations

__all__ = ["health_router", "runs_router", "ws_router"]

from aura_cli.api.routers.health import router as health_router
from aura_cli.api.routers.runs import router as runs_router
from aura_cli.api.routers.ws import router as ws_router
