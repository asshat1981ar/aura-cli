"""FastAPI server exposing a minimal Agent API for the Expo app."""
from __future__ import annotations
import asyncio
import json
import os
import shlex
import time
from pathlib import Path
from typing import List

SUBPROCESS_TIMEOUT_S = int(os.getenv("AGENT_RUN_TIMEOUT", os.getenv("AURA_SUBPROCESS_TIMEOUT_S", "30")))

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aura_cli.cli_main import create_runtime
from core.sanitizer import sanitize_command
from core.logging_utils import log_json
from core.operator_runtime import (
    build_beads_runtime_metadata,
    build_cycle_summary,
    build_fot_runtime_metadata,
    build_ralph_runtime_metadata,
)
from core.runtime_auth import resolve_openai_api_key, resolve_openrouter_api_key, resolve_gemini_cli_path
from core.skill_dispatcher import SKILL_METRICS
from memory.store import MemoryStore

# Initialise runtime without changing cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
runtime = create_runtime(PROJECT_ROOT, overrides=None)
model_adapter = runtime["model_adapter"]
orchestrator = runtime["orchestrator"]
memory_store: MemoryStore = runtime.get("memory_store")


def _agent_run_enabled() -> bool:
    return os.getenv("AGENT_API_ENABLE_RUN") == "1"


def _agent_run_allow() -> set[str]:
    raw = os.getenv("AGENT_RUN_ALLOW", "python,python3,pytest,ls,cat,echo,pwd")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _enforce_agent_run_command(parts: List[str]) -> None:
    if not parts:
        raise HTTPException(status_code=400, detail="Empty command")
    sanitize_command(parts)
    binary = Path(parts[0]).name
    if binary not in _agent_run_allow():
        raise HTTPException(status_code=403, detail=f"Command not allowed for Agent API run: {binary}")


def _provider_snapshot() -> dict[str, bool]:
    config_api_key = runtime.get("config_api_key")
    return {
        "openai": bool(resolve_openai_api_key()),
        "openrouter": bool(resolve_openrouter_api_key(config_api_key=config_api_key)),
        "gemini": bool(resolve_gemini_cli_path() or os.getenv("GEMINI_API_KEY")),
    }


def _beads_runtime_snapshot() -> dict | None:
    return build_beads_runtime_metadata(orchestrator)


def _ralph_runtime_snapshot() -> dict | None:
    return build_ralph_runtime_metadata(orchestrator)


def _fot_runtime_snapshot() -> dict | None:
    return build_fot_runtime_metadata(orchestrator)

def require_auth(authorization: str | None = Header(default=None)):
    """Bearer-token auth; requires AGENT_API_TOKEN unless explicitly opted out."""
    token = os.getenv("AGENT_API_TOKEN")
    if not token:
        if os.getenv("AGENT_API_ALLOW_UNAUTHENTICATED") == "1":
            return
        raise HTTPException(
            status_code=500,
            detail="AGENT_API_TOKEN not configured. Set it or set AGENT_API_ALLOW_UNAUTHENTICATED=1 to allow open access.",
        )
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=403, detail="Invalid token")

app = FastAPI(title="AURA Agent API", version="0.1.0")


class ExecuteRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    args: List[str] = Field(default_factory=list)


TOOLS = [
    {"name": "ask", "description": "Ask a question via LLM"},
    {"name": "run", "description": "Run a shell command (allowlist, streaming, disabled unless AGENT_API_ENABLE_RUN=1)"},
    {"name": "env", "description": "Return environment snapshot"},
    {"name": "goal", "description": "Run a single goal via orchestrator (streaming)"},
]


@app.get("/health")
async def health(auth=Depends(require_auth)):
    return {
        "status": "ok",
        "providers": _provider_snapshot(),
        "run_enabled": _agent_run_enabled(),
        "run_allow": sorted(_agent_run_allow()),
    }


@app.get("/metrics")
async def metrics(auth=Depends(require_auth)):
    skill_data = SKILL_METRICS.snapshot()
    return {
        "status": "ok",
        "skill_metrics": skill_data,
    }


@app.get("/tools")
async def tools(auth=Depends(require_auth)):
    return {"status": "success", "tools": TOOLS}


async def _run_shell_async(command_str: str) -> dict:
    parts = shlex.split(command_str)
    _enforce_agent_run_command(parts)
    proc = await asyncio.create_subprocess_exec(
        *parts,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=SUBPROCESS_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Process timed out after {SUBPROCESS_TIMEOUT_S}s",
        }
    return {
        "returncode": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


async def _run_goal(goal: str, max_cycles: int = 5, dry_run: bool = False) -> dict:
    result = await asyncio.to_thread(
        orchestrator.run_loop,
        goal,
        max_cycles=max_cycles,
        dry_run=dry_run,
    )
    return result


@app.post("/execute")
async def execute(req: ExecuteRequest, auth=Depends(require_auth)):
    tool = req.tool_name.lower()
    args = req.args or []

    try:
        if tool == "ask":
            if not args:
                raise HTTPException(status_code=400, detail="ask requires a question argument")
            answer = await asyncio.to_thread(model_adapter.respond, args[0])
            return {"status": "success", "data": answer}

        if tool == "run":
            if not _agent_run_enabled():
                raise HTTPException(status_code=403, detail="run tool disabled (set AGENT_API_ENABLE_RUN=1)")
            if not args:
                raise HTTPException(status_code=400, detail="run requires a command string")
            command = args[0]

            async def run_stream():
                parts = shlex.split(command)
                _enforce_agent_run_command(parts)
                proc = await asyncio.create_subprocess_exec(
                    *parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                queue: asyncio.Queue[dict] = asyncio.Queue()
                deadline = time.monotonic() + SUBPROCESS_TIMEOUT_S

                async def pump(stream, kind: str):
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        await queue.put({"type": kind, "data": line.decode(errors="replace")})

                tasks = [
                    asyncio.create_task(pump(proc.stdout, "stdout")),
                    asyncio.create_task(pump(proc.stderr, "stderr")),
                ]

                timed_out = False
                while any(not t.done() for t in tasks) or not queue.empty():
                    if time.monotonic() > deadline:
                        proc.kill()
                        timed_out = True
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.1)
                        yield f"data: {json.dumps(item)}\n\n"
                    except asyncio.TimeoutError:
                        pass

                await proc.wait()
                for t in tasks:
                    t.cancel()
                exit_code = -1 if timed_out else proc.returncode
                yield f"data: {json.dumps({'type': 'exit', 'code': exit_code})}\n\n"

            return StreamingResponse(run_stream(), media_type="text/event-stream")

        if tool == "env":
            env_subset = {k: v for k, v in os.environ.items() if k.startswith("TERMUX_") or k in ["PATH", "HOME"]}
            return {"status": "success", "data": env_subset}

        if tool == "goal":
            if not args:
                raise HTTPException(status_code=400, detail="goal requires a goal string")

            async def goal_stream():
                yield "data: {\"type\":\"start\"}\n\n"
                providers = _provider_snapshot()
                yield f"data: {json.dumps({'type': 'health', 'status': 'ok', 'providers': providers, 'beads_runtime': _beads_runtime_snapshot(), 'ralph_runtime': _ralph_runtime_snapshot(), 'fot_runtime': _fot_runtime_snapshot(), 'ts': time.time()})}\n\n"
                goal = args[0]
                max_cycles = 5
                history = []
                stop_reason = ""
                started_at = time.time()

                for _ in range(max_cycles):
                    entry = await asyncio.to_thread(orchestrator.run_cycle, goal, False)
                    summary = dict(entry.get("cycle_summary") or build_cycle_summary(entry))
                    verification = entry.get("phase_outputs", {}).get("verification", {})
                    history.append(summary)
                    stop_reason = summary.get("stop_reason") or orchestrator.policy.evaluate(history, verification, started_at=started_at)
                    if stop_reason:
                        summary["stop_reason"] = stop_reason
                        history[-1] = summary

                    # Persist compact summary if memory_store available
                    if memory_store:
                        try:
                            memory_store.put(
                                "cycle_summaries",
                                {**summary, "goal": goal, "timestamp": time.time()},
                            )
                        except Exception as err:  # noqa: BLE001
                            log_json("WARN", "memory_store_put_failed", details={"error": str(err)})

                    yield f"data: {json.dumps({'type': 'cycle', 'summary': summary})}\n\n"

                    if stop_reason:
                        break

                if not stop_reason:
                    stop_reason = "MAX_CYCLES"

                yield f"data: {json.dumps({'type': 'complete', 'stop_reason': stop_reason, 'history': history})}\n\n"

            return StreamingResponse(goal_stream(), media_type="text/event-stream")

        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool}")

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log_json("ERROR", "execute_failed", details={"tool": tool, "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("AGENT_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 8080)),
    )
