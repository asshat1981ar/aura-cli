"""SADD MCP Server — expose SADD workstream operations as MCP tools.

Tools exposed:
  sadd_parse_spec        : parse markdown spec text into workstreams
  sadd_session_status    : get session status by ID
  sadd_list_sessions     : list recent sessions
  sadd_session_events    : get session event log
  sadd_session_artifacts : get artifacts per workstream

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check
  GET  /metrics        → uptime and per-tool call/error counts

Start:
  uvicorn tools.sadd_mcp_server:app --port 8020
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
from core.sadd.design_spec_parser import DesignSpecParser
from core.sadd.session_store import SessionStore
from core.sadd.types import validate_spec
from core.sadd.workstream_graph import WorkstreamGraph

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="SADD MCP", version="1.0.0")
_TOKEN = os.getenv("SADD_MCP_TOKEN", "")
_SERVER_START = time.time()
_call_counts: Dict[str, int] = {}
_call_errors: Dict[str, int] = {}

_store = SessionStore()
_parser = DesignSpecParser()


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
    "sadd_parse_spec": {
        "description": "Parse markdown spec text into workstreams.",
        "input": {
            "spec_text": {"type": "string", "description": "Markdown design spec text", "required": True},
        },
    },
    "sadd_session_status": {
        "description": "Get status of a SADD session by ID.",
        "input": {
            "session_id": {"type": "string", "description": "Session ID", "required": True},
        },
    },
    "sadd_list_sessions": {
        "description": "List recent SADD sessions.",
        "input": {
            "limit": {"type": "integer", "description": "Max sessions to return", "default": 10},
        },
    },
    "sadd_session_events": {
        "description": "Get event log for a SADD session.",
        "input": {
            "session_id": {"type": "string", "description": "Session ID", "required": True},
            "limit": {"type": "integer", "description": "Max events to return", "default": 50},
        },
    },
    "sadd_session_artifacts": {
        "description": "Get artifacts per workstream for a SADD session.",
        "input": {
            "session_id": {"type": "string", "description": "Session ID", "required": True},
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


def _sadd_parse_spec(args: Dict) -> Any:
    spec_text = args.get("spec_text", "").strip()
    if not spec_text:
        raise ValueError("'spec_text' is required.")
    spec = _parser.parse(spec_text)
    errors = validate_spec(spec)
    graph = WorkstreamGraph(spec.workstreams) if not errors else None
    return {
        "title": spec.title,
        "workstream_count": len(spec.workstreams),
        "parse_confidence": spec.parse_confidence,
        "execution_waves": graph.execution_waves() if graph else [],
        "validation_errors": errors,
        "workstreams": [
            {
                "id": ws.id,
                "title": ws.title,
                "dependencies": ws.depends_on,
            }
            for ws in spec.workstreams
        ],
    }


def _sadd_session_status(args: Dict) -> Any:
    session_id = args.get("session_id", "").strip()
    if not session_id:
        raise ValueError("'session_id' is required.")
    session = _store.get_session(session_id)
    if session is None:
        return {"found": False, "session_id": session_id}
    return {"found": True, **session}


def _sadd_list_sessions(args: Dict) -> Any:
    limit = int(args.get("limit", 10))
    sessions = _store.list_sessions(limit=limit)
    return {"sessions": sessions, "count": len(sessions)}


def _sadd_session_events(args: Dict) -> Any:
    session_id = args.get("session_id", "").strip()
    if not session_id:
        raise ValueError("'session_id' is required.")
    limit = int(args.get("limit", 50))
    events = _store.get_events(session_id, limit=limit)
    return {"session_id": session_id, "events": events, "count": len(events)}


def _sadd_session_artifacts(args: Dict) -> Any:
    session_id = args.get("session_id", "").strip()
    if not session_id:
        raise ValueError("'session_id' is required.")
    artifacts = _store.get_artifacts(session_id)
    by_workstream: Dict[str, List] = {}
    for artifact in artifacts:
        ws_id = artifact.get("ws_id", "unknown")
        by_workstream.setdefault(ws_id, []).append(artifact.get("file_path"))
    return {"session_id": session_id, "by_workstream": by_workstream, "total": len(artifacts)}


_TOOL_HANDLERS = {
    "sadd_parse_spec": _sadd_parse_spec,
    "sadd_session_status": _sadd_session_status,
    "sadd_list_sessions": _sadd_list_sessions,
    "sadd_session_events": _sadd_session_events,
    "sadd_session_artifacts": _sadd_session_artifacts,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "sadd_mcp",
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
        log_json("INFO", "sadd_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "sadd_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "sadd_mcp_tool_error", details={"tool": name, "error": str(exc)})
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

    port = int(os.getenv("SADD_MCP_PORT", _cfg.get_mcp_server_port("sadd", default=8020)))
    uvicorn.run("tools.sadd_mcp_server:app", host="0.0.0.0", port=port, reload=False)
