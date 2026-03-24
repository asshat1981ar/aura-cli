"""
Monitoring MCP Server — metrics collection, health scanning, and alerting for AURA services.

Tools exposed:
  Metrics : metric_push, metric_query
  Health  : health_scan, server_status
  Alerts  : alert_list

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check
  GET  /metrics        → Prometheus text format metrics

Start:
  uvicorn tools.monitoring_mcp:app --port 8016

Auth (optional):
  Set MONITORING_MCP_TOKEN env var
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

import requests

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from tools.mcp_types import ToolCallRequest, ToolResult
from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Monitoring MCP", version="1.0.0")
_TOKEN = os.getenv("MONITORING_MCP_TOKEN", "")
_SERVER_START = time.time()
_call_counts: Dict[str, int] = {}
_call_errors: Dict[str, int] = {}

# In-memory metric storage: {name: [(timestamp, value, labels), ...]}
_metric_store: Dict[str, List[tuple]] = defaultdict(list)
_MAX_METRIC_HISTORY = 10000  # max data points per metric name

# Alert thresholds: {metric_name: {"max": float, "min": float}}
_alert_thresholds: Dict[str, Dict[str, float]] = {}

# Default known AURA MCP server ports
_DEFAULT_SERVERS: Dict[str, int] = {
    "mcp_server": 8001,
    "agentic_loop_mcp": 8002,
    "aura_control_mcp": 8003,
    "aura_mcp_skills": 8004,
    "github_copilot_mcp": 8005,
    "sequential_thinking_mcp": 8006,
    "docker_mcp": 8011,
    "kubernetes_mcp": 8012,
    "neo4j_mcp": 8013,
    "redis_mcp": 8014,
    "notification_mcp": 8015,
    "monitoring_mcp": 8016,
    "weaviate_mcp": 8017,
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _TOKEN:
        return
    if authorization != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Tool descriptors (MCP schema format)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: Dict[str, Dict] = {
    "metric_push": {
        "description": "Push a metric data point to the in-memory store.",
        "input": {
            "name": {"type": "string", "description": "Metric name", "required": True},
            "value": {"type": "number", "description": "Metric value", "required": True},
            "labels": {"type": "object", "description": "Optional label key-value pairs"},
        },
    },
    "metric_query": {
        "description": "Query stored metrics by name within a time window.",
        "input": {
            "name": {"type": "string", "description": "Metric name", "required": True},
            "minutes": {"type": "integer", "description": "Time window in minutes", "default": 60},
        },
    },
    "health_scan": {
        "description": "Scan registered MCP servers for health by hitting their /health endpoints.",
        "input": {
            "servers": {"type": "object", "description": "Optional name→port mapping to override defaults"},
        },
    },
    "alert_list": {
        "description": "List active alerts based on metric thresholds.",
        "input": {},
    },
    "server_status": {
        "description": "Return status of all known AURA MCP servers.",
        "input": {},
    },
}


def _build_descriptor(name: str) -> Dict:
    schema = _TOOL_SCHEMAS[name]
    return {
        "name": name,
        "description": schema["description"],
        "inputSchema": {
            "type": "object",
            "properties": schema.get("input", {}),
            "required": [k for k, v in schema.get("input", {}).items() if v.get("required")],
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _metric_push(args: Dict) -> Any:
    name = args.get("name", "").strip()
    if not name:
        raise ValueError("'name' is required.")
    value = args.get("value")
    if value is None:
        raise ValueError("'value' is required.")
    value = float(value)
    labels = args.get("labels", {}) or {}

    now = time.time()
    _metric_store[name].append((now, value, labels))

    # Trim history if too large
    if len(_metric_store[name]) > _MAX_METRIC_HISTORY:
        _metric_store[name] = _metric_store[name][-_MAX_METRIC_HISTORY:]

    return {"name": name, "value": value, "labels": labels, "stored": True}


def _metric_query(args: Dict) -> Any:
    name = args.get("name", "").strip()
    if not name:
        raise ValueError("'name' is required.")
    minutes = int(args.get("minutes", 60))
    cutoff = time.time() - (minutes * 60)

    points = _metric_store.get(name, [])
    filtered = [(ts, val, lbl) for ts, val, lbl in points if ts >= cutoff]

    values = [val for _, val, _ in filtered]
    result: Dict[str, Any] = {
        "name": name,
        "minutes": minutes,
        "count": len(filtered),
        "points": [{"timestamp": ts, "value": val, "labels": lbl} for ts, val, lbl in filtered[-100:]],
    }
    if values:
        result["min"] = min(values)
        result["max"] = max(values)
        result["avg"] = round(sum(values) / len(values), 4)
    return result


def _health_scan(args: Dict) -> Any:
    servers = args.get("servers") or _DEFAULT_SERVERS
    if not isinstance(servers, dict):
        raise ValueError("'servers' must be a dict of name→port.")

    results: Dict[str, Any] = {}
    for name, port in servers.items():
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                results[name] = {"status": "healthy", "port": port, "data": resp.json()}
            else:
                results[name] = {"status": "unhealthy", "port": port, "status_code": resp.status_code}
        except requests.ConnectionError:
            results[name] = {"status": "unreachable", "port": port}
        except Exception as exc:
            results[name] = {"status": "error", "port": port, "error": str(exc)}

    healthy = sum(1 for r in results.values() if r["status"] == "healthy")
    return {"servers": results, "total": len(results), "healthy": healthy}


def _alert_list(args: Dict) -> Any:
    alerts: List[Dict[str, Any]] = []
    now = time.time()
    # Check last 5 minutes of data against thresholds
    cutoff = now - 300

    for metric_name, thresholds in _alert_thresholds.items():
        points = _metric_store.get(metric_name, [])
        recent = [val for ts, val, _ in points if ts >= cutoff]
        if not recent:
            continue

        sum(recent) / len(recent)
        max_val = max(recent)
        min_val = min(recent)

        if "max" in thresholds and max_val > thresholds["max"]:
            alerts.append({
                "metric": metric_name,
                "type": "threshold_exceeded",
                "threshold_max": thresholds["max"],
                "current_max": max_val,
                "severity": "warning",
            })
        if "min" in thresholds and min_val < thresholds["min"]:
            alerts.append({
                "metric": metric_name,
                "type": "below_minimum",
                "threshold_min": thresholds["min"],
                "current_min": min_val,
                "severity": "warning",
            })

    # Also check tool error rates
    for tool_name in _call_counts:
        calls = _call_counts.get(tool_name, 0)
        errors = _call_errors.get(tool_name, 0)
        if calls > 10 and errors / calls > 0.5:
            alerts.append({
                "metric": f"tool_error_rate.{tool_name}",
                "type": "high_error_rate",
                "error_rate": round(errors / calls, 4),
                "severity": "critical",
            })

    return {"alerts": alerts, "count": len(alerts)}


def _server_status(args: Dict) -> Any:
    results: Dict[str, Any] = {}
    for name, port in _DEFAULT_SERVERS.items():
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=3)
            if resp.status_code == 200:
                results[name] = {"status": "running", "port": port}
            else:
                results[name] = {"status": "error", "port": port, "status_code": resp.status_code}
        except requests.ConnectionError:
            results[name] = {"status": "stopped", "port": port}
        except Exception as exc:
            results[name] = {"status": "unknown", "port": port, "error": str(exc)}

    running = sum(1 for r in results.values() if r["status"] == "running")
    return {"servers": results, "total": len(results), "running": running}


# Map tool names → handler functions
_TOOL_HANDLERS = {
    "metric_push": _metric_push,
    "metric_query": _metric_query,
    "health_scan": _health_scan,
    "alert_list": _alert_list,
    "server_status": _server_status,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "monitoring_mcp",
        "version": "1.0.0",
    }


@app.get("/tools")
async def list_tools(_: None = Depends(_check_auth)) -> List[Dict]:
    return [_build_descriptor(name) for name in _TOOL_SCHEMAS]


@app.get("/tool/{name}")
async def get_tool(name: str, _: None = Depends(_check_auth)) -> Dict:
    if name not in _TOOL_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")
    return _build_descriptor(name)


@app.post("/call")
async def call_tool(request: ToolCallRequest, _: None = Depends(_check_auth)) -> ToolResult:
    name = request.tool_name
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")

    _call_counts[name] = _call_counts.get(name, 0) + 1
    t0 = time.time()
    try:
        result = handler(request.args)
        elapsed = round((time.time() - t0) * 1000, 2)
        log_json("INFO", "monitoring_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "monitoring_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "monitoring_mcp_tool_error", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=f"Internal error: {exc}", elapsed_ms=elapsed)


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics(_: None = Depends(_check_auth)) -> str:
    """Return metrics in Prometheus text exposition format."""
    lines: List[str] = []
    uptime_s = round(time.time() - _SERVER_START, 1)

    # Server uptime
    lines.append("# HELP aura_mcp_uptime_seconds Server uptime in seconds")
    lines.append("# TYPE aura_mcp_uptime_seconds gauge")
    lines.append(f'aura_mcp_uptime_seconds{{server="monitoring_mcp"}} {uptime_s}')
    lines.append("")

    # Tool call counts
    lines.append("# HELP aura_tool_calls_total Total tool calls")
    lines.append("# TYPE aura_tool_calls_total counter")
    for name in _TOOL_SCHEMAS:
        count = _call_counts.get(name, 0)
        lines.append(f'aura_tool_calls_total{{tool="{name}"}} {count}')
    lines.append("")

    # Tool error counts
    lines.append("# HELP aura_tool_errors_total Total tool errors")
    lines.append("# TYPE aura_tool_errors_total counter")
    for name in _TOOL_SCHEMAS:
        count = _call_errors.get(name, 0)
        lines.append(f'aura_tool_errors_total{{tool="{name}"}} {count}')
    lines.append("")

    # Custom pushed metrics (latest value per metric)
    if _metric_store:
        lines.append("# HELP aura_custom_metric Custom pushed metrics (latest value)")
        lines.append("# TYPE aura_custom_metric gauge")
        for metric_name, points in _metric_store.items():
            if points:
                _, latest_val, latest_labels = points[-1]
                label_str = ",".join(f'{k}="{v}"' for k, v in latest_labels.items())
                if label_str:
                    lines.append(f'aura_custom_metric{{name="{metric_name}",{label_str}}} {latest_val}')
                else:
                    lines.append(f'aura_custom_metric{{name="{metric_name}"}} {latest_val}')
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from core.config_manager import config as _cfg
    port = int(os.getenv("MONITORING_MCP_PORT", _cfg.get_mcp_server_port("monitoring", default=8016)))
    uvicorn.run("tools.monitoring_mcp:app", host="0.0.0.0", port=port, reload=False)
