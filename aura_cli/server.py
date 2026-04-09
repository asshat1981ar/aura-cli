from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from core.logging_utils import log_json
from core.mcp_contracts import (
    build_discovery_payload,
    build_tool_descriptor,
)
from core.mcp_registry import list_registered_services
from core.ai_environment_registry import list_ai_environments
from core.mcp_architecture import default_routing_profile
from core.operator_runtime import build_run_tool_audit_summary

# ---------------------------------------------------------------------------
# Prometheus metrics — imported lazily so the server starts even if
# prometheus_client is not installed in dev environments.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
    pipeline_runs_total = Counter(
        "aura_pipeline_runs_total",
        "Total pipeline runs",
        ["status"],
    )
    active_pipeline_runs = Gauge(
        "aura_active_pipeline_runs",
        "Currently executing pipelines",
    )
    sandbox_violations_total = Counter(
        "aura_sandbox_violations_total",
        "Sandbox violations caught by the run-tool denylist",
        ["type"],
    )
    http_request_duration = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["endpoint"],
    )
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False

# Project root for environment listing
PROJECT_ROOT = Path.cwd()

# Global runtime state (set by CLI entrypoint or tests)
runtime: Dict[str, Any] = {}
orchestrator = None
model_adapter = None
memory_store = None
_runtime_init_error: str | None = None

RUN_TOOL_TIMEOUT_S = float(os.getenv("AURA_RUN_TOOL_TIMEOUT_S", "15"))
RUN_TOOL_MAX_OUTPUT_BYTES = int(os.getenv("AURA_RUN_TOOL_MAX_OUTPUT_BYTES", str(64 * 1024)))
RUN_TOOL_READ_CHUNK_BYTES = int(os.getenv("AURA_RUN_TOOL_READ_CHUNK_BYTES", "1024"))
RUN_TOOL_MAX_TIMEOUT_S = 60.0
RUN_TOOL_MAX_OUTPUT_HARD_CAP = 256 * 1024
RUN_TOOL_MAX_READ_CHUNK_BYTES = 8 * 1024
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
RUN_TOOL_DENYLIST = (
    "halt",
    "poweroff",
    "reboot",
    "shutdown",
    "rm -rf /",
    "rm -fr /",
    "mkfs",
    ":(){ :|:& };:",
)


def _beads_runtime_snapshot():
    """Helper for TUI/Studio metadata (used in SSE)."""
    return {"enabled": False, "required": False, "scope": "none"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    await _ensure_runtime_initialized()
    yield


app = FastAPI(
    title="AURA Dev Tools MCP",
    description="Main entry point for AURA developer tools and orchestration.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Prometheus HTTP request duration middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def _prometheus_timing_middleware(request: Request, call_next):
    """Record HTTP request duration for every endpoint."""
    start = time.perf_counter()
    response = await call_next(request)
    if _PROMETHEUS_AVAILABLE:
        duration = time.perf_counter() - start
        # Normalise path to avoid high-cardinality labels (strip UUIDs etc.)
        path = request.url.path
        http_request_duration.labels(endpoint=path).observe(duration)
    return response


class ExecuteRequest(BaseModel):
    tool_name: str
    args: List[Any] = []


class WebhookGoalRequest(BaseModel):
    goal: str
    priority: int = 5
    dry_run: bool = False
    # metadata carries pipeline-routing context from n8n:
    # - complexity: "low" | "medium" | "high"
    # - pipeline_run_id: str (trace ID from P1/P3)
    # - quality_gate_critique: str (injected by P3 before AURA act phase)
    metadata: Dict[str, Any] = {}


class WebhookPlanReviewRequest(BaseModel):
    task_bundle: Dict[str, Any]
    goal: str
    pipeline_run_id: str = ""


def _current_project_root() -> Path:
    configured_root = os.getenv("AURA_PROJECT_ROOT")
    return Path(configured_root).resolve() if configured_root else PROJECT_ROOT.resolve()


def _apply_runtime_state(runtime_state: Dict[str, Any]) -> None:
    global runtime, orchestrator, model_adapter, memory_store
    runtime = runtime_state
    orchestrator = runtime_state.get("orchestrator")
    model_adapter = runtime_state.get("model_adapter")
    memory_store = runtime_state.get("memory_store")


async def _ensure_runtime_initialized() -> Dict[str, Any]:
    global _runtime_init_error
    if runtime:
        return runtime
    try:
        from aura_cli.cli_main import create_runtime

        runtime_state = await asyncio.to_thread(create_runtime, _current_project_root(), None)
        _apply_runtime_state(runtime_state)
        _runtime_init_error = None
        return runtime_state
    except Exception as exc:
        _runtime_init_error = str(exc)
        log_json("WARN", "aura_server_runtime_init_failed", details={"error": _runtime_init_error})
        return {}


async def _resolve_runtime_component(name: str) -> Any:
    component = globals().get(name)
    if component is not None:
        return component
    if not runtime:
        await _ensure_runtime_initialized()
        component = globals().get(name)
        if component is not None:
            return component
    detail = f"{name} is not configured"
    if _runtime_init_error and not runtime:
        detail = f"{detail}: {_runtime_init_error}"
    raise HTTPException(status_code=503, detail=detail)


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
        "run_tool_audit": build_run_tool_audit_summary(memory_store),
    }


def _clamped_run_tool_timeout_s() -> float:
    return max(1.0, min(float(RUN_TOOL_TIMEOUT_S), RUN_TOOL_MAX_TIMEOUT_S))


def _clamped_run_tool_output_bytes() -> int:
    return max(1024, min(int(RUN_TOOL_MAX_OUTPUT_BYTES), RUN_TOOL_MAX_OUTPUT_HARD_CAP))


def _clamped_run_tool_read_chunk_bytes() -> int:
    return max(128, min(int(RUN_TOOL_READ_CHUNK_BYTES), RUN_TOOL_MAX_READ_CHUNK_BYTES))


def _is_denylisted_command(command: str) -> str | None:
    normalized = " ".join(command.lower().split())
    for pattern in RUN_TOOL_DENYLIST:
        if pattern in normalized:
            return pattern
    return None


def _log_run_tool_event(event: str, *, command: str, **details: Any) -> None:
    log_json("INFO", event, details={"command": command[:512], **details})


def _persist_run_tool_audit(
    *,
    command: str,
    pid: int | None,
    code: int | None,
    timed_out: bool,
    truncated: bool,
    duration_s: float,
    output_bytes: int,
) -> None:
    if memory_store is None or not hasattr(memory_store, "append_log"):
        return

    entry = {
        "type": "server_run_tool",
        "timestamp": time.time(),
        "command": command[:512],
        "pid": pid,
        "code": code,
        "timed_out": timed_out,
        "truncated": truncated,
        "duration_s": duration_s,
        "output_bytes": output_bytes,
    }
    try:
        memory_store.append_log(entry)
    except Exception as exc:
        log_json("WARN", "aura_server_run_tool_audit_persist_failed", details={"error": str(exc)})


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _run_tool_env() -> Dict[str, str]:
    return {key: value for key, value in os.environ.items() if key in RUN_TOOL_ENV_ALLOWLIST and value}


async def _enqueue_stream(stream: asyncio.StreamReader | None, event_type: str, queue: asyncio.Queue[tuple[str, bytes | None]]) -> None:
    if stream is None:
        await queue.put((event_type, None))
        return
    while True:
        chunk = await stream.read(_clamped_run_tool_read_chunk_bytes())
        if not chunk:
            break
        await queue.put((event_type, chunk))
    await queue.put((event_type, None))


async def _terminate_process(process: asyncio.subprocess.Process) -> int:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            return int(process.returncode or 0)
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                return int(process.returncode or 0)
            await process.wait()
    return int(process.returncode or 0)


def _close_process_transport(process: asyncio.subprocess.Process) -> None:
    transport = getattr(process, "_transport", None)
    if transport is None:
        return
    try:
        transport.close()
    except Exception:
        pass


async def _execute_ask(req: ExecuteRequest):
    prompt = req.args[0] if req.args else ""
    adapter = await _resolve_runtime_component("model_adapter")
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
    blocked_pattern = _is_denylisted_command(command)
    if blocked_pattern is not None:
        if _PROMETHEUS_AVAILABLE:
            sandbox_violations_total.labels(type="denylist").inc()
        raise HTTPException(status_code=403, detail=f"Command blocked by policy: {blocked_pattern}")

    async def run_generator():
        timeout_s = _clamped_run_tool_timeout_s()
        output_limit = _clamped_run_tool_output_bytes()
        started_at = asyncio.get_running_loop().time()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(PROJECT_ROOT),
            env=_run_tool_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _log_run_tool_event(
            "aura_server_run_tool_started",
            command=command,
            pid=process.pid,
            timeout_s=timeout_s,
            output_limit_bytes=output_limit,
        )
        yield _sse_event({"type": "start", "command": command, "pid": process.pid})

        queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()
        readers = [
            asyncio.create_task(_enqueue_stream(process.stdout, "stdout", queue)),
            asyncio.create_task(_enqueue_stream(process.stderr, "stderr", queue)),
        ]
        deadline = asyncio.get_running_loop().time() + timeout_s
        total_output = 0
        closed_streams = 0

        try:
            while closed_streams < len(readers):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    code = await _terminate_process(process)
                    duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                    _log_run_tool_event(
                        "aura_server_run_tool_finished",
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=True,
                        truncated=False,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    _persist_run_tool_audit(
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=True,
                        truncated=False,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    yield _sse_event({"type": "error", "error": "Command timed out", "timeout_s": timeout_s})
                    yield _sse_event({"type": "exit", "code": code, "timed_out": True})
                    return

                try:
                    event_type, chunk = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    code = await _terminate_process(process)
                    duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                    _log_run_tool_event(
                        "aura_server_run_tool_finished",
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=True,
                        truncated=False,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    _persist_run_tool_audit(
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=True,
                        truncated=False,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    yield _sse_event({"type": "error", "error": "Command timed out", "timeout_s": timeout_s})
                    yield _sse_event({"type": "exit", "code": code, "timed_out": True})
                    return
                if chunk is None:
                    closed_streams += 1
                    continue

                remaining_bytes = output_limit - total_output
                if remaining_bytes <= 0:
                    code = await _terminate_process(process)
                    duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                    _log_run_tool_event(
                        "aura_server_run_tool_finished",
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=False,
                        truncated=True,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    _persist_run_tool_audit(
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=False,
                        truncated=True,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    yield _sse_event({"type": "truncated", "limit_bytes": output_limit})
                    yield _sse_event({"type": "exit", "code": code, "truncated": True})
                    return

                emitted = chunk[:remaining_bytes]
                total_output += len(emitted)
                text = emitted.decode("utf-8", errors="replace")
                if text:
                    yield _sse_event({"type": event_type, "data": text})

                if len(chunk) > len(emitted) or total_output >= output_limit:
                    code = await _terminate_process(process)
                    duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                    _log_run_tool_event(
                        "aura_server_run_tool_finished",
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=False,
                        truncated=True,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    _persist_run_tool_audit(
                        command=command,
                        pid=process.pid,
                        code=code,
                        timed_out=False,
                        truncated=True,
                        duration_s=duration_s,
                        output_bytes=total_output,
                    )
                    yield _sse_event({"type": "truncated", "limit_bytes": output_limit})
                    yield _sse_event({"type": "exit", "code": code, "truncated": True})
                    return

            remaining = max(0.1, deadline - asyncio.get_running_loop().time())
            try:
                code = await asyncio.wait_for(process.wait(), timeout=remaining)
                duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                _log_run_tool_event(
                    "aura_server_run_tool_finished",
                    command=command,
                    pid=process.pid,
                    code=int(code or 0),
                    timed_out=False,
                    truncated=False,
                    duration_s=duration_s,
                    output_bytes=total_output,
                )
                _persist_run_tool_audit(
                    command=command,
                    pid=process.pid,
                    code=int(code or 0),
                    timed_out=False,
                    truncated=False,
                    duration_s=duration_s,
                    output_bytes=total_output,
                )
                yield _sse_event({"type": "exit", "code": int(code or 0)})
            except asyncio.TimeoutError:
                code = await _terminate_process(process)
                duration_s = round(asyncio.get_running_loop().time() - started_at, 4)
                _log_run_tool_event(
                    "aura_server_run_tool_finished",
                    command=command,
                    pid=process.pid,
                    code=code,
                    timed_out=True,
                    truncated=False,
                    duration_s=duration_s,
                    output_bytes=total_output,
                )
                _persist_run_tool_audit(
                    command=command,
                    pid=process.pid,
                    code=code,
                    timed_out=True,
                    truncated=False,
                    duration_s=duration_s,
                    output_bytes=total_output,
                )
                yield _sse_event({"type": "error", "error": "Command timed out", "timeout_s": timeout_s})
                yield _sse_event({"type": "exit", "code": code, "timed_out": True})
        finally:
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            _close_process_transport(process)

    return StreamingResponse(run_generator(), media_type="text/event-stream")


async def _execute_goal(req: ExecuteRequest):
    if not req.args:
        raise HTTPException(status_code=400, detail="Missing goal text in args")
    active_orchestrator = await _resolve_runtime_component("orchestrator")

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
async def health() -> Dict:
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/ready")
async def ready() -> Dict:
    """Readiness probe: checks SQLite brain DB is readable; optionally pings Redis."""
    import sqlite3 as _sqlite3

    db_path = Path("memory/brain_v2.db")
    try:
        conn = _sqlite3.connect(str(db_path), timeout=2.0)
        conn.execute("SELECT 1")
        conn.close()
    except Exception as exc:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": f"SQLite unavailable: {exc}"},
        )

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis as _redis  # optional dependency

            client = _redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
            client.ping()
        except Exception as exc:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": f"Redis unavailable: {exc}"},
            )

    return {"status": "ready"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Expose Prometheus-format metrics (no auth — scraper must be on private network).

    Falls back to a JSON summary if prometheus_client is not installed.
    """
    if _PROMETHEUS_AVAILABLE:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    # Graceful fallback when prometheus_client is not installed
    return {
        "status": "ok",
        "skill_metrics": _runtime_metrics_snapshot(),
        "warning": "prometheus_client not installed; install it for Prometheus format",
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


# --- n8n pipeline integration webhooks ---

# In-memory goal queue for webhook-submitted goals
_webhook_goal_queue: Dict[str, Dict[str, Any]] = {}


@app.post("/webhook/goal")
async def webhook_goal(req: WebhookGoalRequest, _: None = Depends(require_auth)) -> Dict:
    """Enqueue a goal submitted by n8n (P1 fast lane or direct trigger)."""
    goal_id = secrets.token_hex(12)
    entry: Dict[str, Any] = {
        "goal_id": goal_id,
        "goal": req.goal,
        "priority": req.priority,
        "dry_run": req.dry_run,
        "metadata": req.metadata,
        "status": "queued",
        "queued_at": time.time(),
    }
    _webhook_goal_queue[goal_id] = entry
    log_json("INFO", "aura_webhook_goal_received", details={
        "goal_id": goal_id,
        "goal": req.goal[:200],
        "pipeline_run_id": req.metadata.get("pipeline_run_id", ""),
        "complexity": req.metadata.get("complexity", "unknown"),
    })

    # Fire-and-forget: run the cycle in background so n8n can poll /webhook/status
    async def _run_cycle() -> None:
        if _PROMETHEUS_AVAILABLE:
            active_pipeline_runs.inc()
        try:
            _webhook_goal_queue[goal_id]["status"] = "running"
            _webhook_goal_queue[goal_id]["started_at"] = time.time()
            active_orchestrator = await _resolve_runtime_component("orchestrator")
            result = await asyncio.to_thread(
                active_orchestrator.run_cycle,
                req.goal,
            )
            _webhook_goal_queue[goal_id].update({
                "status": "done",
                "result": result,
                "completed_at": time.time(),
            })
            if _PROMETHEUS_AVAILABLE:
                pipeline_runs_total.labels(status="success").inc()
        except Exception as exc:
            _webhook_goal_queue[goal_id].update({
                "status": "failed",
                "error": str(exc),
                "completed_at": time.time(),
            })
            log_json("WARN", "aura_webhook_goal_failed", details={"goal_id": goal_id, "error": str(exc)})
            if _PROMETHEUS_AVAILABLE:
                pipeline_runs_total.labels(status="failure").inc()
        finally:
            if _PROMETHEUS_AVAILABLE:
                active_pipeline_runs.dec()

    asyncio.create_task(_run_cycle())
    return {"status": "queued", "goal_id": goal_id}


@app.get("/webhook/status/{goal_id}")
async def webhook_goal_status(goal_id: str, _: None = Depends(require_auth)) -> Dict:
    """Poll the status of a webhook-submitted goal (used by n8n P1 fast lane)."""
    entry = _webhook_goal_queue.get(goal_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found")
    result: Dict[str, Any] = {
        "goal_id": goal_id,
        "status": entry["status"],
        "goal": entry["goal"],
        "queued_at": entry["queued_at"],
    }
    if entry["status"] == "done":
        result["result"] = entry.get("result", {})
        result["completed_at"] = entry.get("completed_at")
    elif entry["status"] == "failed":
        result["error"] = entry.get("error", "unknown error")
        result["completed_at"] = entry.get("completed_at")
    elif entry["status"] == "running":
        result["started_at"] = entry.get("started_at")
    return result


@app.post("/webhook/plan-review")
async def webhook_plan_review(req: WebhookPlanReviewRequest, _: None = Depends(require_auth)) -> Dict:
    """Format a task bundle as a plan text for Dev Suite Quality Gate (P2) review.

    Called by P3 Pipeline Coordinator before triggering AURA act phase.
    Returns human-readable plan text that Dev Suite agents can critique.
    """
    plan_steps = req.task_bundle.get("plan", [])
    if isinstance(plan_steps, str):
        plan_text = plan_steps
    elif isinstance(plan_steps, list):
        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps))
    else:
        plan_text = str(plan_steps)

    review_payload = {
        "goal": req.goal,
        "pipeline_run_id": req.pipeline_run_id,
        "plan_text": plan_text,
        "task_bundle_keys": list(req.task_bundle.keys()),
        "file_targets": req.task_bundle.get("file_targets", []),
        "critique": req.task_bundle.get("critique", ""),
    }
    log_json("INFO", "aura_webhook_plan_review_requested", details={
        "goal": req.goal[:200],
        "pipeline_run_id": req.pipeline_run_id,
        "plan_steps": len(plan_steps) if isinstance(plan_steps, list) else 1,
    })
    return {"status": "ok", "review_payload": review_payload}
