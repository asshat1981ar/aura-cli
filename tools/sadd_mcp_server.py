"""SADD MCP Server — expose SADD workstream operations as MCP tools.

Port: 8020 (configurable via SADD_MCP_PORT)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.sadd.design_spec_parser import DesignSpecParser
from core.sadd.session_store import SessionStore
from core.sadd.types import validate_spec
from core.sadd.workstream_graph import WorkstreamGraph


def create_app() -> FastAPI:
    app = FastAPI(title="AURA SADD MCP Server")
    store = SessionStore()
    parser = DesignSpecParser()

    TOOLS = [
        {"name": "sadd_parse_spec", "description": "Parse a markdown design spec into workstreams",
         "input_schema": {"type": "object", "properties": {"markdown": {"type": "string"}}, "required": ["markdown"]}},
        {"name": "sadd_session_status", "description": "Get status of a SADD session",
         "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
        {"name": "sadd_list_sessions", "description": "List recent SADD sessions",
         "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}},
        {"name": "sadd_session_events", "description": "Get events for a SADD session",
         "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
        {"name": "sadd_session_artifacts", "description": "Get artifacts for a SADD session",
         "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}, "ws_id": {"type": "string"}}}},
    ]

    @app.get("/health")
    async def health():
        return {"server": "aura-sadd", "status": "ok", "tools": len(TOOLS), "uptime": time.time()}

    @app.get("/tools")
    async def tools():
        return TOOLS

    @app.post("/call")
    async def call_tool(request: Request):
        body = await request.json()
        tool_name = body.get("tool_name", "")
        args = body.get("args", {})
        t0 = time.time()

        try:
            if tool_name == "sadd_parse_spec":
                spec = parser.parse(args["markdown"])
                errors = validate_spec(spec)
                graph = WorkstreamGraph(spec.workstreams) if not errors else None
                result = {
                    "title": spec.title,
                    "workstreams": len(spec.workstreams),
                    "parse_confidence": spec.parse_confidence,
                    "waves": graph.execution_waves() if graph else [],
                    "errors": errors,
                }
            elif tool_name == "sadd_session_status":
                session = store.get_session(args["session_id"])
                result = session if session else {"error": "not found"}
            elif tool_name == "sadd_list_sessions":
                result = store.list_sessions(limit=args.get("limit", 20))
            elif tool_name == "sadd_session_events":
                result = store.get_events(args["session_id"])
            elif tool_name == "sadd_session_artifacts":
                result = store.get_artifacts(args["session_id"], args.get("ws_id"))
            else:
                return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)

            return {"data": result, "elapsed_ms": (time.time() - t0) * 1000}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SADD_MCP_PORT", "8020"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port)
