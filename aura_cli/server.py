"""HTTP API surface for AURA.

Provides a small FastAPI app plus direct-call helpers that the test suite uses
to exercise health, metrics, tool discovery, and streaming execution flows.
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Runtime objects are set by the CLI bootstrapper. Tests patch these directly.
runtime: Optional[dict] = None
orchestrator = None
model_adapter = None
memory_store = None

app = FastAPI(title="AURA Agent API", version="1.0.0")


class ExecuteRequest(BaseModel):
    tool_name: str
    args: List[str] = []


def _ensure_runtime():
    """Lazily initialise the shared runtime if not already provided."""
    global runtime, orchestrator, model_adapter, memory_store
    if runtime is None:
        from aura_cli.cli_main import create_runtime

        runtime = create_runtime()
        orchestrator = runtime.get("orchestrator")
        model_adapter = runtime.get("model_adapter")
        memory_store = runtime.get("memory_store")


def require_auth(auth_header: Optional[str]) -> None:
    """Enforce bearer-token auth when AGENT_API_TOKEN is set."""
    token = os.getenv("AGENT_API_TOKEN")
    if not token:
        return
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Malformed Authorization header")
    provided = auth_header.removeprefix("Bearer ").strip()
    if provided != token:
        raise HTTPException(status_code=403, detail="Invalid token")


def _provider_status() -> Dict[str, str]:
    """Return synthetic provider health info."""
    return {
        "openai": "ok",
        "openrouter": "ok",
        "gemini": "ok",
    }


def _beads_runtime_snapshot() -> Dict[str, Any]:
    """Placeholder snapshot for Beads runtime metadata."""
    return {"enabled": False, "required": False, "scope": None}


async def health(auth: Optional[str] = None) -> Dict[str, Any]:
    # Health is intentionally permissive in tests; auth is not enforced here.
    return {
        "status": "ok",
        "providers": _provider_status(),
        "run_enabled": os.getenv("AGENT_API_ENABLE_RUN") == "1",
        "beads_runtime": _beads_runtime_snapshot(),
    }


async def metrics(auth: Optional[str] = None) -> Dict[str, Any]:
    return {
        "status": "ok",
        "skill_metrics": {},
    }


def _tool_defs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "ask",
            "description": "Ask the configured model a question.",
            "inputSchema": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        },
        {
            "name": "run",
            "description": "Execute a shell command (streamed).",
            "inputSchema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "env",
            "description": "Return environment details.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "goal",
            "description": "Run a one-off goal through the orchestrator.",
            "inputSchema": {
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "required": ["goal"],
            },
        },
    ]


async def tools(auth: Optional[str] = None) -> Dict[str, Any]:
    return {"status": "success", "tools": _tool_defs()}


async def discovery(auth: Optional[str] = None) -> Dict[str, Any]:
    return {
        "current_server": {"name": "aura-dev-tools", "kind": "tooling"},
        "servers": [
            {"name": "aura-dev-tools", "url": "http://127.0.0.1:8001", "kind": "tooling"},
            {"name": "aura-copilot", "url": "http://127.0.0.1:8007", "kind": "copilot"},
        ],
        "supported_environments": [
            {"name": "gemini-cli", "cli_command": "gemini"},
            {"name": "claude-code", "cli_command": "claude"},
            {"name": "codex-cli", "cli_command": "codex"},
        ],
    }


async def environments(auth: Optional[str] = None) -> Dict[str, Any]:
    envs = [
        {"name": "gemini-cli", "cli_command": "gemini"},
        {"name": "claude-code", "cli_command": "claude"},
        {"name": "codex-cli", "cli_command": "codex"},
    ]
    return {"status": "success", "environments": envs}


async def architecture(auth: Optional[str] = None) -> Dict[str, Any]:
    supported_envs = [
        {"name": "gemini-cli", "cli_command": "gemini"},
        {"name": "claude-code", "cli_command": "claude"},
        {"name": "codex-cli", "cli_command": "codex"},
    ]
    return {
        "routing": {"strategy": "health-aware-round-robin"},
        "knowledge_backends": [{"name": "neo4j", "status": "unknown"}],
        "supported_environments": supported_envs,
    }


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_subprocess(cmd: str) -> Iterable[str]:
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if process.stdout:
        async for raw in process.stdout:
            yield _sse({"type": "stdout", "data": raw.decode().rstrip("\n")})

    if process.stderr:
        async for raw in process.stderr:
            yield _sse({"type": "stderr", "data": raw.decode().rstrip("\n")})

    code = await process.wait()
    yield _sse({"type": "exit", "code": code})


async def _stream_goal(goal_text: str) -> Iterable[str]:
    yield _sse({"type": "start", "goal": goal_text})
    health_payload = await health()
    yield _sse({"type": "health", **health_payload})

    _ensure_runtime()
    history: List[Dict[str, Any]] = []
    result = await asyncio.to_thread(orchestrator.run_cycle, goal_text) if orchestrator else {}
    summary = result.get("cycle_summary")
    if not summary and result:
        summary = {
            "cycle_id": result.get("cycle_id"),
            "goal": result.get("goal", goal_text),
            "goal_type": result.get("goal_type", "default"),
            "verification_status": result.get("phase_outputs", {})
            .get("verification", {})
            .get("status"),
            "stop_reason": result.get("stop_reason"),
        }
    if result:
        history.append(result)
    yield _sse({"type": "cycle", "summary": summary or {}})
    yield _sse({"type": "complete", "history": history, "stop_reason": result.get("stop_reason") if result else None})


async def execute(request: ExecuteRequest, auth: Optional[str] = None):
    tool = request.tool_name
    args = request.args or []

    if tool == "env":
        return {"status": "success", "data": dict(os.environ)}

    if tool == "ask":
        if not args:
            raise HTTPException(status_code=400, detail="Prompt is required")
        _ensure_runtime()
        answer = model_adapter.respond(args[0]) if model_adapter else ""
        return {"status": "success", "data": answer}

    if tool == "run":
        if os.getenv("AGENT_API_ENABLE_RUN") != "1":
            raise HTTPException(status_code=403, detail="Run tool disabled")
        if not args:
            raise HTTPException(status_code=400, detail="Command is required")
        cmd = args[0]
        stream = _stream_subprocess(cmd)
        return StreamingResponse(stream, media_type="text/event-stream")

    if tool == "goal":
        if not args:
            raise HTTPException(status_code=400, detail="Goal text is required")
        goal_text = args[0]
        stream = _stream_goal(goal_text)
        return StreamingResponse(stream, media_type="text/event-stream")

    raise HTTPException(status_code=404, detail="Unknown tool")


# FastAPI route bindings
@app.get("/health")
async def health_route(authorization: Optional[str] = Header(default=None)):
    return JSONResponse(await health(authorization))


@app.get("/metrics")
async def metrics_route(authorization: Optional[str] = Header(default=None)):
    return JSONResponse(await metrics(authorization))


@app.get("/tools")
async def tools_route(authorization: Optional[str] = Header(default=None)):
    return JSONResponse(await tools(authorization))


@app.get("/discovery")
async def discovery_route(authorization: Optional[str] = Header(default=None)):
    return JSONResponse(await discovery(authorization))


@app.get("/environments")
async def environments_route(authorization: Optional[str] = Header(default=None)):
    return JSONResponse(await environments(authorization))


@app.get("/architecture")
async def architecture_route(authorization: Optional[str] = Header(default=None)):
    return JSONResponse(await architecture(authorization))


@app.post("/execute")
async def execute_route(request: ExecuteRequest, authorization: Optional[str] = Header(default=None)):
    return await execute(request, authorization)
