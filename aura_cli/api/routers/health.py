"""Comprehensive health check system for AURA CLI."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class ComponentStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class OverallStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


COMPONENT_SLAs: dict[str, int] = {
    "memory_db": 100,
    "auth_db": 50,
    "redis_cache": 50,
}


def _check_sqlite_sync(db_path: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        path = Path(db_path)
        if not path.exists():
            return {"status": ComponentStatus.DEGRADED, "error": "db not found", "latency_ms": 0}
        conn = sqlite3.connect(db_path, timeout=1.0)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        latency_ms = (time.perf_counter() - start) * 1000
        sla_ms = COMPONENT_SLAs.get("memory_db", 100)
        return {
            "status": ComponentStatus.OK if latency_ms < sla_ms else ComponentStatus.DEGRADED,
            "latency_ms": round(latency_ms, 2),
            "sla_ms": sla_ms,
            "sla_met": latency_ms < sla_ms,
        }
    except Exception as exc:
        return {
            "status": ComponentStatus.ERROR,
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }


@router.get("/health")
async def liveness() -> dict:
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/ready")
async def readiness() -> JSONResponse:
    """Deep readiness check — verifies all critical dependencies."""
    checks: dict[str, Any] = {}
    overall = OverallStatus.HEALTHY

    # Check brain DB
    brain_db = os.environ.get("AURA_BRAIN_DB_PATH", "memory/brain.db")
    brain_check = await asyncio.to_thread(_check_sqlite_sync, brain_db)
    checks["memory_db"] = brain_check
    if brain_check["status"] == ComponentStatus.ERROR:
        overall = OverallStatus.UNHEALTHY
    elif brain_check["status"] == ComponentStatus.DEGRADED and overall == OverallStatus.HEALTHY:
        overall = OverallStatus.DEGRADED

    # Check auth DB
    try:
        from core.auth import _default_auth_db_path  # noqa: PLC0415
    except ImportError:

        def _default_auth_db_path():
            return Path(os.environ.get("AURA_AUTH_DB_PATH", "aura_auth.db"))

    auth_db = str(_default_auth_db_path())
    auth_check = await asyncio.to_thread(_check_sqlite_sync, auth_db)
    checks["auth_db"] = auth_check
    if auth_check["status"] == ComponentStatus.ERROR:
        overall = OverallStatus.UNHEALTHY

    # Check Redis (optional — degraded only if REDIS_URL set but unreachable)
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            import redis as redis_lib  # noqa: PLC0415

            r = redis_lib.from_url(redis_url, socket_connect_timeout=1)
            start = time.perf_counter()
            r.ping()
            latency_ms = (time.perf_counter() - start) * 1000
            checks["redis"] = {"status": ComponentStatus.OK, "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            checks["redis"] = {"status": ComponentStatus.DEGRADED, "error": str(exc)}
            if overall == OverallStatus.HEALTHY:
                overall = OverallStatus.DEGRADED

    status_code = 200 if overall in (OverallStatus.HEALTHY, OverallStatus.DEGRADED) else 503
    return JSONResponse(
        {
            "status": overall,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        status_code=status_code,
    )
