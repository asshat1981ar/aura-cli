"""
Redis MCP Server — cache, pub/sub, and queue operations via Redis.

Tools exposed:
  Cache  : cache_get, cache_set, cache_delete, cache_keys
  PubSub : pubsub_publish
  Queue  : queue_push, queue_pop

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check
  GET  /metrics        → uptime and per-tool call/error counts

Start:
  uvicorn tools.redis_mcp:app --port 8014

Auth (optional):
  Set REDIS_MCP_TOKEN env var
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi import Depends, FastAPI, Header, HTTPException
from tools.mcp_types import ToolCallRequest, ToolResult
from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Lazy-loaded Redis client
# ---------------------------------------------------------------------------

_redis_client = None
_redis_available = True

try:
    import redis as _redis_mod
except ImportError:
    _redis_mod = None  # type: ignore[assignment]
    _redis_available = False


def _get_client():
    """Lazy-load and return the Redis client singleton."""
    global _redis_client
    if not _redis_available:
        raise RuntimeError("redis Python package is not installed. Install it with: pip install redis")
    if _redis_client is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis_client = _redis_mod.from_url(url, decode_responses=True)
    return _redis_client


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Redis MCP", version="1.0.0")
_TOKEN = os.getenv("REDIS_MCP_TOKEN", "")
_SERVER_START = time.time()
_call_counts: Dict[str, int] = {}
_call_errors: Dict[str, int] = {}


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
    "cache_get": {
        "description": "Get a value from Redis by key.",
        "input": {
            "key": {"type": "string", "description": "Cache key", "required": True},
        },
    },
    "cache_set": {
        "description": "Set a value in Redis with optional TTL.",
        "input": {
            "key": {"type": "string", "description": "Cache key", "required": True},
            "value": {"type": "string", "description": "Value to store", "required": True},
            "ttl": {"type": "integer", "description": "Time-to-live in seconds (optional)"},
        },
    },
    "cache_delete": {
        "description": "Delete a key from Redis.",
        "input": {
            "key": {"type": "string", "description": "Cache key", "required": True},
        },
    },
    "cache_keys": {
        "description": "List keys matching a pattern.",
        "input": {
            "pattern": {"type": "string", "description": "Key pattern (glob-style)", "default": "*"},
            "limit": {"type": "integer", "description": "Max keys to return", "default": 100},
        },
    },
    "pubsub_publish": {
        "description": "Publish a message to a Redis pub/sub channel.",
        "input": {
            "channel": {"type": "string", "description": "Channel name", "required": True},
            "message": {"type": "string", "description": "Message to publish", "required": True},
        },
    },
    "queue_push": {
        "description": "Push a value to a Redis list (LPUSH).",
        "input": {
            "queue": {"type": "string", "description": "Queue (list) name", "required": True},
            "value": {"type": "string", "description": "Value to push", "required": True},
        },
    },
    "queue_pop": {
        "description": "Pop a value from a Redis list (RPOP).",
        "input": {
            "queue": {"type": "string", "description": "Queue (list) name", "required": True},
            "timeout": {"type": "integer", "description": "Block timeout in seconds (0 = non-blocking)", "default": 0},
        },
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


def _cache_get(args: Dict) -> Any:
    key = args.get("key", "").strip()
    if not key:
        raise ValueError("'key' is required.")
    client = _get_client()
    value = client.get(key)
    return {"key": key, "value": value, "found": value is not None}


def _cache_set(args: Dict) -> Any:
    key = args.get("key", "").strip()
    if not key:
        raise ValueError("'key' is required.")
    value = args.get("value")
    if value is None:
        raise ValueError("'value' is required.")
    value = str(value)
    ttl = args.get("ttl")
    client = _get_client()
    if ttl is not None:
        client.setex(key, int(ttl), value)
    else:
        client.set(key, value)
    return {"key": key, "stored": True, "ttl": ttl}


def _cache_delete(args: Dict) -> Any:
    key = args.get("key", "").strip()
    if not key:
        raise ValueError("'key' is required.")
    client = _get_client()
    deleted = client.delete(key)
    return {"key": key, "deleted": deleted > 0}


def _cache_keys(args: Dict) -> Any:
    pattern = args.get("pattern", "*").strip() or "*"
    limit = int(args.get("limit", 100))
    client = _get_client()
    keys: List[str] = []
    cursor = 0
    while len(keys) < limit:
        cursor, batch = client.scan(cursor=cursor, match=pattern, count=min(limit, 100))
        keys.extend(batch)
        if cursor == 0:
            break
    keys = keys[:limit]
    return {"pattern": pattern, "keys": keys, "count": len(keys)}


def _pubsub_publish(args: Dict) -> Any:
    channel = args.get("channel", "").strip()
    if not channel:
        raise ValueError("'channel' is required.")
    message = args.get("message", "").strip()
    if not message:
        raise ValueError("'message' is required.")
    client = _get_client()
    receivers = client.publish(channel, message)
    return {"channel": channel, "receivers": receivers}


def _queue_push(args: Dict) -> Any:
    queue = args.get("queue", "").strip()
    if not queue:
        raise ValueError("'queue' is required.")
    value = args.get("value")
    if value is None:
        raise ValueError("'value' is required.")
    value = str(value)
    client = _get_client()
    length = client.lpush(queue, value)
    return {"queue": queue, "length": length}


def _queue_pop(args: Dict) -> Any:
    queue = args.get("queue", "").strip()
    if not queue:
        raise ValueError("'queue' is required.")
    timeout = int(args.get("timeout", 0))
    client = _get_client()
    if timeout > 0:
        result = client.brpop(queue, timeout=timeout)
        value = result[1] if result else None
    else:
        value = client.rpop(queue)
    return {"queue": queue, "value": value, "found": value is not None}


# Map tool names → handler functions
_TOOL_HANDLERS = {
    "cache_get": _cache_get,
    "cache_set": _cache_set,
    "cache_delete": _cache_delete,
    "cache_keys": _cache_keys,
    "pubsub_publish": _pubsub_publish,
    "queue_push": _queue_push,
    "queue_pop": _queue_pop,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "redis_mcp",
        "version": "1.0.0",
        "redis_available": _redis_available,
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
        log_json("INFO", "redis_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "redis_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "redis_mcp_tool_error", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=f"Internal error: {exc}", elapsed_ms=elapsed)


@app.get("/metrics")
async def get_metrics(_: None = Depends(_check_auth)) -> Dict:
    uptime_s = round(time.time() - _SERVER_START, 1)
    total_calls = sum(_call_counts.values())
    total_errors = sum(_call_errors.values())
    per_tool = {
        name: {
            "calls": _call_counts.get(name, 0),
            "errors": _call_errors.get(name, 0),
        }
        for name in _TOOL_SCHEMAS
    }
    return {
        "uptime_seconds": uptime_s,
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_calls, 1), 4),
        "tools": per_tool,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from core.config_manager import config as _cfg

    port = int(os.getenv("REDIS_MCP_PORT", _cfg.get_mcp_server_port("redis", default=8014)))
    uvicorn.run("tools.redis_mcp:app", host="0.0.0.0", port=port, reload=False)
