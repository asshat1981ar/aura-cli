"""Health check endpoints for AURA API.

Extracted from api_server.py as part of Sprint 1 server decomposition.
Provides /health and /api/health endpoints for monitoring and load balancers.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["health"])


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


def _get_agents_from_registry() -> List[Dict[str, Any]]:
    """Get agents from registry for health checks."""
    try:
        from core.mcp_agent_registry import list_registered_services
        services = list_registered_services()
        return [{"name": s.get("name", "unknown"), "status": "ok"} for s in services]
    except Exception:
        return []


def _check_sqlite_health() -> Dict[str, Any]:
    """Check SQLite database connectivity."""
    db_path = Path("memory/brain_v2.db")
    if not db_path.exists():
        return {"status": "unknown", "error": "Database file not found"}

    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "ok"}
    except sqlite3.Error as exc:
        return {"status": "error", "error": str(exc)}


@router.get("/health", include_in_schema=False)
@router.get("/api/health", include_in_schema=True)
async def health_check() -> Dict[str, Any]:
    """Get system health status.

    Returns:
        Dict with status (healthy/degraded/critical), checks, and timestamp.
    """
    queue_data = _get_goal_queue_data()
    agents = _get_agents_from_registry()
    sqlite_health = _check_sqlite_health()

    # Check if system is healthy
    failed_agents = len([a for a in agents if a.get("status") == "error"])
    queued_goals = len(queue_data.get("queue", [])) if isinstance(queue_data, dict) else 0

    status = "healthy"
    if failed_agents > 0 or sqlite_health["status"] != "ok":
        status = "degraded"
    if failed_agents > len(agents) / 2:
        status = "critical"

    return {
        "status": status,
        "version": "1.0.0",
        "checks": {
            "agents": "pass" if failed_agents == 0 else "warn",
            "queue": "pass" if queued_goals < 100 else "warn",
            "api": "pass",
            "database": sqlite_health["status"],
        },
        "metrics": {
            "goals_queued": queued_goals,
            "goals_in_flight": len(queue_data.get("in_flight", {})),
            "agents_registered": len(agents),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness probe for Kubernetes.

    Checks that the database is accessible before reporting ready.
    """
    sqlite_health = _check_sqlite_health()

    if sqlite_health["status"] != "ok":
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": {"database": sqlite_health}},
        )

    return {"status": "ready", "checks": {"database": sqlite_health}}


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness probe for Kubernetes.

    Simple check that the process is running.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
