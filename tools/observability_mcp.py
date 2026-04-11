#!/usr/bin/env python3
"""
tools/observability_mcp.py — Observability MCP Server for AURA CLI.

Provides tools for AI agents to inspect system health, metrics, quality trends,
and loop progress.

Port: 8030 (configurable via OBSERVABILITY_MCP_PORT env var)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("observability_mcp")

REPO_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = REPO_ROOT / "memory"
LOGS_DIR = REPO_ROOT / "logs"
AUTH_DB_PATH = os.environ.get(
    "AURA_AUTH_DB_PATH",
    str(Path.home() / ".local" / "share" / "aura" / "auth.db"),
)

app = FastAPI(
    title="AURA Observability MCP",
    description="Observability tools for AI agents monitoring AURA CLI",
    version="1.0.0",
)


class ComponentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    tool: str
    status: str
    data: Any
    timestamp: str


TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="health_summary",
        description="Get aggregated system health across all AURA subsystems",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolDefinition(
        name="quality_trends",
        description="Get quality trend data from recent autonomous loop cycles",
        parameters={
            "type": "object",
            "properties": {"cycles": {"type": "integer", "description": "Number of cycles", "default": 10}},
            "required": [],
        },
    ),
    ToolDefinition(
        name="cache_stats",
        description="Get memory cache hit/miss statistics",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolDefinition(
        name="loop_progress",
        description="Get current autonomous loop state and progress",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolDefinition(
        name="recent_decisions",
        description="Get recent decisions from the JSONL decision log",
        parameters={
            "type": "object",
            "properties": {"count": {"type": "integer", "description": "Number of entries", "default": 20}},
            "required": [],
        },
    ),
    ToolDefinition(
        name="error_summary",
        description="Summarize recent errors and warnings from logs",
        parameters={
            "type": "object",
            "properties": {"minutes": {"type": "integer", "description": "Look back window", "default": 60}},
            "required": [],
        },
    ),
]


def _make_result(tool_name: str, data: Any, status: str = "success") -> ToolResult:
    return ToolResult(
        tool=tool_name,
        status=status,
        data=data,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _check_sqlite(db_path: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        path = Path(db_path)
        if not path.exists():
            return {"status": ComponentStatus.UNKNOWN, "error": "db not found", "latency_ms": 0}
        conn = sqlite3.connect(db_path, timeout=1.0)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "status": ComponentStatus.HEALTHY if latency_ms < 100 else ComponentStatus.DEGRADED,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        return {
            "status": ComponentStatus.UNHEALTHY,
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }


def tool_health_summary(**_kwargs: Any) -> ToolResult:
    checks: dict[str, Any] = {}
    overall = ComponentStatus.HEALTHY

    # Config file
    config_path = REPO_ROOT / "aura.config.json"
    if config_path.exists():
        try:
            json.loads(config_path.read_text())
            checks["config"] = {"status": ComponentStatus.HEALTHY}
        except Exception as e:
            checks["config"] = {"status": ComponentStatus.UNHEALTHY, "error": str(e)}
            overall = ComponentStatus.DEGRADED
    else:
        checks["config"] = {"status": ComponentStatus.UNKNOWN, "error": "aura.config.json not found"}

    # Auth DB
    checks["auth_db"] = _check_sqlite(AUTH_DB_PATH)
    if checks["auth_db"]["status"] == ComponentStatus.UNHEALTHY:
        overall = ComponentStatus.DEGRADED

    # Memory store
    jsonl_files = list(MEMORY_DIR.glob("*.jsonl")) if MEMORY_DIR.exists() else []
    checks["memory_store"] = {
        "status": ComponentStatus.HEALTHY if jsonl_files else ComponentStatus.UNKNOWN,
        "jsonl_files": len(jsonl_files),
    }

    # Disk space
    try:
        import shutil

        usage = shutil.disk_usage(str(REPO_ROOT))
        free_gb = usage.free / (1024**3)
        checks["disk"] = {
            "status": ComponentStatus.HEALTHY if free_gb > 1.0 else ComponentStatus.DEGRADED,
            "free_gb": round(free_gb, 2),
        }
    except Exception as e:
        checks["disk"] = {"status": ComponentStatus.UNKNOWN, "error": str(e)}

    return _make_result(
        "health_summary",
        {
            "overall": overall,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def tool_quality_trends(cycles: int = 10, **_kwargs: Any) -> ToolResult:
    decision_log = MEMORY_DIR / "decision_log.jsonl"
    if not decision_log.exists():
        return _make_result(
            "quality_trends",
            {
                "cycles_analyzed": 0,
                "note": "No decision log found — no cycles recorded yet",
            },
        )

    entries = []
    try:
        with open(decision_log) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        return _make_result("quality_trends", {"error": str(e)}, status="error")

    verify_events = [e for e in entries if e.get("event") == "verify_result"]
    pass_rate = None
    if verify_events:
        passes = sum(1 for e in verify_events if e.get("passed", False))
        pass_rate = round(passes / len(verify_events) * 100, 1)

    return _make_result(
        "quality_trends",
        {
            "total_events": len(entries),
            "verify_events": len(verify_events),
            "pass_rate_pct": pass_rate,
            "note": f"Analyzed {len(entries)} log entries",
        },
    )


def tool_cache_stats(**_kwargs: Any) -> ToolResult:
    stats: dict[str, Any] = {}

    local_cache = MEMORY_DIR / "local_cache.db"
    if local_cache.exists():
        try:
            conn = sqlite3.connect(str(local_cache))
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            conn.close()
            stats["local_cache"] = {
                "status": ComponentStatus.HEALTHY,
                "entries": count,
                "size_bytes": local_cache.stat().st_size,
            }
        except Exception as e:
            stats["local_cache"] = {"status": ComponentStatus.UNKNOWN, "error": str(e)}
    else:
        stats["local_cache"] = {"status": ComponentStatus.UNKNOWN, "note": "No local cache db"}

    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            import redis

            r = redis.from_url(redis_url, socket_timeout=2)
            info = r.info("memory")
            stats["redis"] = {
                "status": ComponentStatus.HEALTHY,
                "used_memory_human": info.get("used_memory_human"),
            }
        except Exception as e:
            stats["redis"] = {"status": ComponentStatus.DEGRADED, "error": str(e)}
    else:
        stats["redis"] = {"status": ComponentStatus.UNKNOWN, "note": "REDIS_URL not set"}

    return _make_result("cache_stats", stats)


def tool_loop_progress(**_kwargs: Any) -> ToolResult:
    state_file = REPO_ROOT / ".aura_loop_state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            return _make_result("loop_progress", {"active_loop": True, **state})
        except Exception:
            pass
    return _make_result("loop_progress", {"active_loop": False, "current_phase": None})


def tool_recent_decisions(count: int = 20, **_kwargs: Any) -> ToolResult:
    decision_log = MEMORY_DIR / "decision_log.jsonl"
    if not decision_log.exists():
        return _make_result("recent_decisions", {"decisions": [], "note": "No decision log"})

    entries = []
    try:
        with open(decision_log) as f:
            lines = f.readlines()
        for line in lines[-count:]:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entry.pop("api_key", None)
                    entry.pop("token", None)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return _make_result("recent_decisions", {"error": str(e)}, status="error")

    return _make_result("recent_decisions", {"count": len(entries), "decisions": entries})


def tool_error_summary(minutes: int = 60, **_kwargs: Any) -> ToolResult:
    errors = []
    warnings = []
    log_patterns = (list(LOGS_DIR.glob("*.jsonl")) if LOGS_DIR.exists() else []) + list(MEMORY_DIR.glob("*.jsonl"))

    for log_file in log_patterns:
        try:
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    level = entry.get("level", "").lower()
                    msg = entry.get("message", entry.get("msg", ""))[:200]
                    if level in ("error", "critical"):
                        errors.append({"level": level, "message": msg, "source": log_file.name})
                    elif level == "warning":
                        warnings.append({"message": msg, "source": log_file.name})
        except Exception:
            continue

    return _make_result(
        "error_summary",
        {
            "window_minutes": minutes,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "recent_errors": errors[-10:],
            "recent_warnings": warnings[-5:],
            "log_files_scanned": len(log_patterns),
        },
    )


TOOL_FUNCTIONS = {
    "health_summary": tool_health_summary,
    "quality_trends": tool_quality_trends,
    "cache_stats": tool_cache_stats,
    "loop_progress": tool_loop_progress,
    "recent_decisions": tool_recent_decisions,
    "error_summary": tool_error_summary,
}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "observability_mcp", "version": "1.0.0"}


@app.get("/tools")
async def list_tools() -> dict:
    return {"tools": [t.model_dump() for t in TOOLS]}


class ExecuteRequest(BaseModel):
    tool: str
    parameters: dict[str, Any] = {}


@app.post("/execute")
async def execute_tool(request: ExecuteRequest) -> dict:
    if request.tool not in TOOL_FUNCTIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown tool: {request.tool}. Available: {list(TOOL_FUNCTIONS)}",
        )
    try:
        result = TOOL_FUNCTIONS[request.tool](**request.parameters)
        return result.model_dump()
    except Exception as e:
        logger.exception("Tool execution failed: %s", request.tool)
        return _make_result(request.tool, {"error": str(e)}, status="error").model_dump()


@app.get("/discovery")
async def discovery() -> dict:
    return {
        "name": "aura-observability",
        "version": "1.0.0",
        "description": "Observability MCP server for AURA CLI",
        "port": int(os.environ.get("OBSERVABILITY_MCP_PORT", "8030")),
        "tools": [t.model_dump() for t in TOOLS],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("OBSERVABILITY_MCP_PORT", "8030"))
    print(f"🔭 Starting Observability MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
