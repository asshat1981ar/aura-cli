from __future__ import annotations

import asyncio
import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config_manager import config
from core.logging_utils import log_json
from core.mcp_contracts import (
    build_discovery_payload,
    build_health_payload,
    build_tool_descriptor,
)
from core.mcp_registry import get_registered_service, list_registered_services
from core.ai_environment_registry import list_ai_environments
from core.mcp_architecture import default_routing_profile

# Project root for environment listing
PROJECT_ROOT = Path.cwd()

# Global runtime state (set by CLI entrypoint or tests)
runtime: Dict[str, Any] = {}
orchestrator = None
model_adapter = None
memory_store = None

RUN_TOOL_TIMEOUT_S = float(os.getenv("AURA_RUN_TOOL_TIMEOUT_S", "15"))
RUN_TOOL_MAX_OUTPUT_BYTES = int(os.getenv("AURA_RUN_TOOL_MAX_OUTPUT_BYTES", str(64 * 1024)))
RUN_TOOL_READ_CHUNK_BYTES = int(os.getenv("AURA_RUN_TOOL_READ_CHUNK_BYTES", "1024"))
RUN_TOOL_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONPATH",
    "TERM",
    "TMPDIR",
    "USER",
)


def _beads_runtime_snapshot():
    """Helper for TUI/Studio metadata (used in SSE)."""
    return {"enabled": False, "required": False, "scope": "none"}


app = FastAPI(
    title="AURA Dev Tools MCP",
    description="Main entry point for AURA developer tools and orchestration.",
    version="1.0.0",
)


class ExecuteRequest(BaseModel):
    tool_name: str
    args: List[Any] = []


def _require_runtime_component(name: str, component: Any) -> Any:
    if component is None:
        raise HTTPException(status_code=503, detail=f"{name} is not configured")
    return component


def _runtime_metrics_snapshot() -> Dict[str, Any]:
    entries: list[Any] = []
    if memory_store is not None and hasattr(memory_store, "read_log"):
        try:
            entries = list(memory_store.read_log(limit=1000) or [])
        except Exception:
            entries = []
    return {
        "total_calls": len(entries),
        "registered_services": len(list_registered_services()),
        "environment_count": len(list_ai_environments(PROJECT_ROOT)),
    }


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _run_tool_env() -> Dict[str, str]:
    return {key: value for key, value in os.environ.items() if key in RUN_TOOL_ENV_ALLOWLIST and value}


async def _enqueue_stream(stream: asyncio.StreamReader | None, event_type: str, queue: asyncio.Queue[tuple[str, bytes | None]]) -> None:
    if stream is None:
        await queue.put((event_type, None))
        return
    while True:
        chunk = await stream.read(RUN_TOOL_READ_CHUNK_BYTES)
        if not chunk:
            break
        await queue.put((event_type, chunk))
    await queue.put((event_type, None))


async def _terminate_process(process: asyncio.subprocess.Process) -> int:
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
    return int(process.returncode or 0)


async def _execute_ask(req: ExecuteRequest):
    prompt = req.args[0] if req.args else ""
    adapter = _require_runtime_component("model_adapter", model_adapter)
    res = adapter.respond(prompt)
    return {"status": "success", "data": res}


async def _execute_env(_: ExecuteRequest):
    raise HTTPException(status_code=501, detail="The 'env' tool is currently disabled due to security concerns.")


async def _execute_run(req: ExecuteRequest):
    if os.getenv("AGENT_API_ENABLE_RUN") != "1":
        raise HTTPException(status_code=403, detail="Run tool is disabled")
    if not req.args:
        raise HTTPException(status_code=400, detail="Missing command in args")
    command = str(req.args[0]).strip()
    if not command:
        raise HTTPException(status_code=400, detail="Missing command in args")

    async def run_generator():
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(PROJECT_ROOT),
            env=_run_tool_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        yield _sse_event({"type": "start", "command": command, "pid": process.pid})

        queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()
        readers = [
            asyncio.create_task(_enqueue_stream(process.stdout, "stdout", queue)),
            asyncio.create_task(_enqueue_stream(process.stderr, "stderr", queue)),
        ]
        deadline = asyncio.get_running_loop().time() + RUN_TOOL_TIMEOUT_S
        total_output = 0
        closed_streams = 0

        try:
            while closed_streams < len(readers):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    code = await _terminate_process(process)
                    yield _sse_event({"type": "error", "error": "Command timed out", "timeout_s": RUN_TOOL_TIMEOUT_S})
                    yield _sse_event({"type": "exit", "code": code, "timed_out": True})
                    return

                event_type, chunk = await asyncio.wait_for(queue.get(), timeout=remaining)
                if chunk is None:
                    closed_streams += 1
                    continue

                remaining_bytes = RUN_TOOL_MAX_OUTPUT_BYTES - total_output
                if remaining_bytes <= 0:
                    code = await _terminate_process(process)
                    yield _sse_event({"type": "truncated", "limit_bytes": RUN_TOOL_MAX_OUTPUT_BYTES})
                    yield _sse_event({"type": "exit", "code": code, "truncated": True})
                    return

                emitted = chunk[:remaining_bytes]
                total_output += len(emitted)
                text = emitted.decode("utf-8", errors="replace")
                if text:
                    yield _sse_event({"type": event_type, "data": text})

                if len(chunk) > len(emitted) or total_output >= RUN_TOOL_MAX_OUTPUT_BYTES:
                    code = await _terminate_process(process)
                    yield _sse_event({"type": "truncated", "limit_bytes": RUN_TOOL_MAX_OUTPUT_BYTES})
                    yield _sse_event({"type": "exit", "code": code, "truncated": True})
                    return

            remaining = max(0.1, deadline - asyncio.get_running_loop().time())
            try:
                code = await asyncio.wait_for(process.wait(), timeout=remaining)
                yield _sse_event({"type": "exit", "code": int(code or 0)})
            except asyncio.TimeoutError:
                code = await _terminate_process(process)
                yield _sse_event({"type": "error", "error": "Command timed out", "timeout_s": RUN_TOOL_TIMEOUT_S})
                yield _sse_event({"type": "exit", "code": code, "timed_out": True})
        finally:
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    return StreamingResponse(run_generator(), media_type="text/event-stream")


async def _execute_goal(req: ExecuteRequest):
    if not req.args:
        raise HTTPException(status_code=400, detail="Missing goal text in args")
    active_orchestrator = _require_runtime_component("orchestrator", orchestrator)

    async def goal_generator():
        yield f"data: {json.dumps({'type': 'start', 'goal': req.args[0]})}\n\n"
        await asyncio.sleep(0.1)

        health_info = {"type": "health", "status": "ok", "providers": {"openai": "connected", "openrouter": "connected", "gemini": "connected"}, "beads_runtime": _beads_runtime_snapshot()}
        yield f"data: {json.dumps(health_info)}\n\n"
        await asyncio.sleep(0.1)

        cycle_data = await asyncio.to_thread(active_orchestrator.run_cycle, req.args[0])
        yield f"data: {json.dumps({'type': 'cycle', 'summary': cycle_data.get('cycle_summary', cycle_data)})}\n\n"
        yield f"data: {json.dumps({'type': 'complete', 'status': 'success', 'stop_reason': cycle_data.get('stop_reason', 'done'), 'history': [cycle_data]})}\n\n"

    return StreamingResponse(goal_generator(), media_type="text/event-stream")


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    token = os.getenv("AGENT_API_TOKEN")
    if not token:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not secrets.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(status_code=403, detail="Invalid token")


@app.get("/health")
async def health(_: None = Depends(require_auth)) -> Dict:
    return build_health_payload(
        status="ok",
        server="aura-dev-tools",
        version="1.0.0",
        providers={"chat": "connected", "embeddings": "enabled", "openai": "connected", "openrouter": "connected", "gemini": "connected"},
        run_enabled=os.getenv("AGENT_API_ENABLE_RUN") == "1",
    )


@app.get("/metrics")
async def metrics(_: None = Depends(require_auth)) -> Dict:
    return {
        "status": "ok",
        "skill_metrics": _runtime_metrics_snapshot(),
    }


@app.get("/tools")
async def tools(_: None = Depends(require_auth)) -> Dict:
    return {
        "status": "success",
        "tools": [
            build_tool_descriptor("ask", "Ask AURA a question"),
            build_tool_descriptor("run", "Run a shell command"),
            build_tool_descriptor("env", "Get environment variables"),
            build_tool_descriptor("goal", "Execute an autonomous goal"),
        ],
    }


@app.get("/discovery")
async def discovery(_: None = Depends(require_auth)) -> Dict:
    return build_discovery_payload(
        current_server={"name": "aura-dev-tools", "port": 8001},
        servers=list_registered_services(),
        supported_environments=list_ai_environments(PROJECT_ROOT),
    )


@app.get("/environments")
async def environments(_: None = Depends(require_auth)) -> Dict:
    return {
        "status": "success",
        "environments": list_ai_environments(PROJECT_ROOT),
    }


@app.get("/architecture")
async def architecture(_: None = Depends(require_auth)) -> Dict:
    return {
        "routing": default_routing_profile(),
        "knowledge_backends": [{"name": "neo4j"}, {"name": "weaviate"}],
        "supported_environments": list_ai_environments(PROJECT_ROOT),
    }


@app.post("/execute")
async def execute(req: ExecuteRequest, _: None = Depends(require_auth)):
    handlers = {
        "ask": _execute_ask,
        "env": _execute_env,
        "run": _execute_run,
        "goal": _execute_goal,
    }
    handler = handlers.get(req.tool_name)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found")
    return await handler(req)
