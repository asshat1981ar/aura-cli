"""Unified MCP server — mounts all tool namespaces (E3).

Consolidates the three separate MCP servers into a single FastAPI
application with namespaced tool routing:

- ``devtools/*``  — file I/O, linting, git, search (from mcp_server.py)
- ``workflow/*``  — workflow/loop management (from agentic_loop_mcp.py)
- ``skills/*``    — AURA skills as tools (from aura_mcp_skills_server.py)

The individual servers continue to work standalone; this module provides
an optional single-port entry point.

Usage::

    python -m tools.mcp_unified          # starts on port 8000
    UNIFIED_MCP_PORT=9000 python -m tools.mcp_unified
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from tools.mcp_common import (
    CallRequest,
    RateLimiter,
    ToolResult,
    make_auth_dependency,
)
from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# App & auth
# ---------------------------------------------------------------------------

app = FastAPI(title="AURA Unified MCP Server", version="1.0.0")

_rate_limiter = RateLimiter(limit_per_min=int(os.getenv("MCP_RATE_LIMIT_PER_MIN", "0")))
_require_auth = make_auth_dependency(token_env_var="MCP_API_TOKEN", rate_limiter=_rate_limiter)

_start_time = time.time()

# ---------------------------------------------------------------------------
# Namespace registries (lazy-loaded)
# ---------------------------------------------------------------------------

_namespaces: Dict[str, Any] = {}  # name -> ToolRegistry
_loaded = False


def _ensure_loaded() -> None:
    """Lazy-load all three namespaces on first request."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    # 1. DevTools namespace — import the existing mcp_server tools
    try:
        from tools.mcp_server import TOOLS_MANIFEST, app as _devtools_app
        from tools.mcp_common import ToolRegistry
        devtools = ToolRegistry(namespace="devtools")
        # We can't easily extract the handlers from the if/elif chain,
        # so we proxy to the existing /call endpoint logic
        _namespaces["devtools"] = {
            "type": "proxy",
            "app": _devtools_app,
            "descriptors": TOOLS_MANIFEST,
        }
        log_json("INFO", "unified_mcp_namespace_loaded", details={"namespace": "devtools", "tools": len(TOOLS_MANIFEST)})
    except Exception as exc:
        log_json("ERROR", "unified_mcp_namespace_failed", details={"namespace": "devtools", "error": str(exc)})

    # 2. Workflow namespace — import the agentic loop tools
    try:
        from tools.agentic_loop_mcp import _TOOL_SCHEMAS, _TOOL_HANDLERS
        from tools.mcp_common import ToolRegistry
        workflow = ToolRegistry(namespace="workflow")
        workflow.register_batch(
            descriptors=[{"name": k, **v} for k, v in _TOOL_SCHEMAS.items()],
            handlers=_TOOL_HANDLERS,
        )
        _namespaces["workflow"] = {"type": "registry", "registry": workflow}
        log_json("INFO", "unified_mcp_namespace_loaded", details={"namespace": "workflow", "tools": len(_TOOL_HANDLERS)})
    except Exception as exc:
        log_json("ERROR", "unified_mcp_namespace_failed", details={"namespace": "workflow", "error": str(exc)})

    # 3. Skills namespace — dynamic from skill registry
    try:
        from tools.aura_mcp_skills_server import _SKILL_SCHEMAS
        from agents.skills.registry import all_skills
        from tools.mcp_common import ToolRegistry
        skills_reg = ToolRegistry(namespace="skills")
        loaded_skills = all_skills()
        descriptors = []
        handlers = {}
        for name, schema in _SKILL_SCHEMAS.items():
            if name in loaded_skills:
                descriptors.append({
                    "name": name,
                    "description": schema.get("description", ""),
                    "inputSchema": schema.get("inputSchema", {}),
                })
                skill = loaded_skills[name]
                handlers[name] = lambda args, s=skill: s.run(args)
        skills_reg.register_batch(descriptors=descriptors, handlers=handlers)
        _namespaces["skills"] = {"type": "registry", "registry": skills_reg}
        log_json("INFO", "unified_mcp_namespace_loaded", details={"namespace": "skills", "tools": len(handlers)})
    except Exception as exc:
        log_json("ERROR", "unified_mcp_namespace_failed", details={"namespace": "skills", "error": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: str = Depends(_require_auth)):
    _ensure_loaded()
    ns_status = {}
    total_tools = 0
    for name, ns in _namespaces.items():
        if ns["type"] == "proxy":
            count = len(ns.get("descriptors", []))
        else:
            count = len(ns["registry"].tool_names)
        ns_status[name] = {"tools": count, "status": "ok"}
        total_tools += count
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _start_time, 1),
        "namespaces": ns_status,
        "total_tools": total_tools,
    }


@app.get("/tools")
async def list_tools(namespace: Optional[str] = None, _: str = Depends(_require_auth)):
    _ensure_loaded()
    tools = []
    for name, ns in _namespaces.items():
        if namespace and name != namespace:
            continue
        if ns["type"] == "proxy":
            for desc in ns.get("descriptors", []):
                tools.append({**desc, "namespace": name})
        else:
            for desc in ns["registry"].descriptors:
                tools.append({**desc, "namespace": name})
    return {"tools": tools, "count": len(tools)}


@app.post("/call")
async def call_tool(request: CallRequest, _: str = Depends(_require_auth)):
    _ensure_loaded()
    tool_name = request.tool_name

    # Try namespace-prefixed name first: "skills/security_scanner"
    ns_name = None
    local_name = tool_name
    if "/" in tool_name:
        ns_name, local_name = tool_name.split("/", 1)

    # Search for the tool
    if ns_name and ns_name in _namespaces:
        return _dispatch_in_namespace(ns_name, local_name, request.args)

    # Search all namespaces
    for name, ns in _namespaces.items():
        if ns["type"] == "registry" and local_name in ns["registry"]._handlers:
            return _dispatch_in_namespace(name, local_name, request.args)
        if ns["type"] == "proxy":
            desc_names = [d["name"] for d in ns.get("descriptors", [])]
            if local_name in desc_names:
                return _dispatch_in_namespace(name, local_name, request.args)

    raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name!r}")


def _dispatch_in_namespace(ns_name: str, tool_name: str, args: Dict[str, Any]) -> dict:
    ns = _namespaces[ns_name]
    if ns["type"] == "registry":
        result = ns["registry"].dispatch(tool_name, args)
        return result.model_dump()
    elif ns["type"] == "proxy":
        # For proxy namespaces, forward to the original app's call_tool
        from tools.mcp_server import CallRequest as DevCallRequest
        from starlette.testclient import TestClient
        client = TestClient(ns["app"])
        resp = client.post("/call", json={"tool_name": tool_name, "args": args})
        return resp.json()
    raise HTTPException(status_code=500, detail=f"Unknown namespace type for {ns_name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from core.config_manager import config as _cfg

    port = int(os.getenv("UNIFIED_MCP_PORT", "8000"))
    log_json("INFO", "unified_mcp_server_starting", details={"port": port})
    uvicorn.run(app, host="0.0.0.0", port=port)
