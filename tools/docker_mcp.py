"""
Docker MCP Server — manages Docker containers, images, and compose services.

Tools exposed:
  Containers : container_list, container_run, container_stop, container_logs
  Images     : image_list, image_build
  Compose    : compose_up, compose_down

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check
  GET  /metrics        → uptime and per-tool call/error counts

Start:
  uvicorn tools.docker_mcp:app --port 8011

Auth (optional):
  Set DOCKER_MCP_TOKEN env var
"""
from __future__ import annotations

import os
import re
import subprocess
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
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Docker MCP", version="1.0.0")
_TOKEN = os.getenv("DOCKER_MCP_TOKEN", "")
_SERVER_START = time.time()
_call_counts: Dict[str, int] = {}
_call_errors: Dict[str, int] = {}

_SAFE_VALUE_RE = re.compile(r"^[a-zA-Z0-9_\-./: ]*$")


def _sanitize(value: str, field_name: str = "value") -> str:
    """Validate that a user-provided value contains only safe characters."""
    if not _SAFE_VALUE_RE.match(value):
        raise ValueError(
            f"Invalid characters in '{field_name}': only alphanumeric, dash, "
            "underscore, dot, colon, and forward slash are allowed."
        )
    return value


def _run_cmd(cmd: List[str], timeout: int = 120) -> Dict[str, Any]:
    """Run a subprocess command and return structured output."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
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
    "container_list": {
        "description": "List running Docker containers.",
        "input": {},
    },
    "container_run": {
        "description": "Run a Docker container from an image.",
        "input": {
            "image": {"type": "string", "description": "Docker image to run", "required": True},
            "name": {"type": "string", "description": "Container name (optional)"},
            "ports": {"type": "string", "description": "Port mapping e.g. '8080:80' (optional)"},
            "detach": {"type": "boolean", "description": "Run in detached mode", "default": True},
        },
    },
    "container_stop": {
        "description": "Stop a running Docker container.",
        "input": {
            "container_id": {"type": "string", "description": "Container ID or name", "required": True},
        },
    },
    "container_logs": {
        "description": "Get logs from a Docker container.",
        "input": {
            "container_id": {"type": "string", "description": "Container ID or name", "required": True},
            "tail": {"type": "integer", "description": "Number of lines to tail", "default": 100},
        },
    },
    "image_list": {
        "description": "List local Docker images.",
        "input": {},
    },
    "image_build": {
        "description": "Build a Docker image from a Dockerfile.",
        "input": {
            "path": {"type": "string", "description": "Path to build context", "required": True},
            "tag": {"type": "string", "description": "Image tag", "required": True},
        },
    },
    "compose_up": {
        "description": "Start docker-compose services.",
        "input": {
            "file": {"type": "string", "description": "Compose file path", "default": "docker-compose.yml"},
            "detach": {"type": "boolean", "description": "Run in detached mode", "default": True},
        },
    },
    "compose_down": {
        "description": "Stop docker-compose services.",
        "input": {
            "file": {"type": "string", "description": "Compose file path", "default": "docker-compose.yml"},
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

def _container_list(args: Dict) -> Any:
    out = _run_cmd(["docker", "ps", "--format", "json"])
    if out["returncode"] != 0:
        raise RuntimeError(f"docker ps failed: {out['stderr']}")
    lines = [line for line in out["stdout"].strip().splitlines() if line]
    return {"containers": lines, "count": len(lines)}


def _container_run(args: Dict) -> Any:
    image = args.get("image", "").strip()
    if not image:
        raise ValueError("'image' is required.")
    _sanitize(image, "image")

    cmd = ["docker", "run"]
    detach = args.get("detach", True)
    if detach:
        cmd.append("-d")

    name = args.get("name", "").strip()
    if name:
        _sanitize(name, "name")
        cmd.extend(["--name", name])

    ports = args.get("ports", "").strip()
    if ports:
        _sanitize(ports, "ports")
        cmd.extend(["-p", ports])

    cmd.append(image)
    out = _run_cmd(cmd)
    if out["returncode"] != 0:
        raise RuntimeError(f"docker run failed: {out['stderr']}")
    return {"container_id": out["stdout"].strip(), "image": image}


def _container_stop(args: Dict) -> Any:
    container_id = args.get("container_id", "").strip()
    if not container_id:
        raise ValueError("'container_id' is required.")
    _sanitize(container_id, "container_id")
    out = _run_cmd(["docker", "stop", container_id])
    if out["returncode"] != 0:
        raise RuntimeError(f"docker stop failed: {out['stderr']}")
    return {"stopped": container_id}


def _container_logs(args: Dict) -> Any:
    container_id = args.get("container_id", "").strip()
    if not container_id:
        raise ValueError("'container_id' is required.")
    _sanitize(container_id, "container_id")
    tail = int(args.get("tail", 100))
    out = _run_cmd(["docker", "logs", "--tail", str(tail), container_id])
    if out["returncode"] != 0:
        raise RuntimeError(f"docker logs failed: {out['stderr']}")
    return {"container_id": container_id, "logs": out["stdout"]}


def _image_list(args: Dict) -> Any:
    out = _run_cmd(["docker", "images", "--format", "json"])
    if out["returncode"] != 0:
        raise RuntimeError(f"docker images failed: {out['stderr']}")
    lines = [line for line in out["stdout"].strip().splitlines() if line]
    return {"images": lines, "count": len(lines)}


def _image_build(args: Dict) -> Any:
    path = args.get("path", "").strip()
    tag = args.get("tag", "").strip()
    if not path:
        raise ValueError("'path' is required.")
    if not tag:
        raise ValueError("'tag' is required.")
    _sanitize(path, "path")
    _sanitize(tag, "tag")
    out = _run_cmd(["docker", "build", "-t", tag, path])
    if out["returncode"] != 0:
        raise RuntimeError(f"docker build failed: {out['stderr']}")
    return {"tag": tag, "path": path, "output": out["stdout"][-2000:]}


def _compose_up(args: Dict) -> Any:
    file = args.get("file", "docker-compose.yml").strip()
    _sanitize(file, "file")
    detach = args.get("detach", True)
    cmd = ["docker", "compose", "-f", file, "up"]
    if detach:
        cmd.append("-d")
    out = _run_cmd(cmd)
    if out["returncode"] != 0:
        raise RuntimeError(f"docker compose up failed: {out['stderr']}")
    return {"file": file, "output": out["stdout"][-2000:]}


def _compose_down(args: Dict) -> Any:
    file = args.get("file", "docker-compose.yml").strip()
    _sanitize(file, "file")
    out = _run_cmd(["docker", "compose", "-f", file, "down"])
    if out["returncode"] != 0:
        raise RuntimeError(f"docker compose down failed: {out['stderr']}")
    return {"file": file, "output": out["stdout"][-2000:]}


# Map tool names → handler functions
_TOOL_HANDLERS = {
    "container_list": _container_list,
    "container_run": _container_run,
    "container_stop": _container_stop,
    "container_logs": _container_logs,
    "image_list": _image_list,
    "image_build": _image_build,
    "compose_up": _compose_up,
    "compose_down": _compose_down,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "docker_mcp",
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
        log_json("INFO", "docker_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "docker_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "docker_mcp_tool_error", details={"tool": name, "error": str(exc)})
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
    port = int(os.getenv("DOCKER_MCP_PORT", _cfg.get_mcp_server_port("docker", default=8011)))
    uvicorn.run("tools.docker_mcp:app", host="0.0.0.0", port=port, reload=False)
