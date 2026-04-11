"""AURA API application composition root.

Thin composition root that wires together all routers and middleware.
Extracted from api_server.py as part of Sprint 1 server decomposition.

Usage:
    from aura_cli.api.app import create_app
    app = create_app()
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aura_cli.api.routers import health_router, runs_router, ws_router
from core.logging_utils import log_json

# Try to import AURA components
try:
    from core.auth import init_auth, UserRole, AUTH_AVAILABLE
except ImportError:
    AUTH_AVAILABLE = False


def _get_goal_queue_data() -> Dict[str, Any]:
    """Read goal queue from the JSON file."""
    queue_path = Path("memory/goal_queue.json")
    if not queue_path.exists():
        return {"queue": [], "in_flight": {}}

    try:
        with open(queue_path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return {"queue": raw, "in_flight": {}}
        return raw
    except (json.JSONDecodeError, IOError):
        return {"queue": [], "in_flight": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager — startup and shutdown."""
    # Startup
    log_json("INFO", "aura_api_startup", details={"timestamp": datetime.utcnow().isoformat()})

    # Initialize auth if available
    if AUTH_AVAILABLE:
        init_auth(secret_key=os.getenv("AURA_SECRET_KEY", "aura-secret-key-change-in-production"))
        from core.auth import get_auth_manager

        auth = get_auth_manager()
        # Create default admin user if none exists
        try:
            auth.create_user("admin", password="admin", role=UserRole.ADMIN)
        except Exception:
            pass  # User may already exist

    yield

    # Shutdown
    log_json("INFO", "aura_api_shutdown", details={"timestamp": datetime.utcnow().isoformat()})


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application with all routers mounted.
    """
    app = FastAPI(
        title="AURA API",
        description="API for AURA autonomous development platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(runs_router)
    app.include_router(ws_router)

    return app


# Legacy compatibility: create app instance for import
# This allows: from aura_cli.api.app import app
app = create_app()
