"""
Kubernetes MCP Server — manages Kubernetes resources via kubectl.

Tools exposed:
  Pods        : pod_list, pod_logs
  Deployments : deployment_scale, rollout_status
  Namespaces  : namespace_list
  Manifests   : apply_manifest

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check
  GET  /metrics        → uptime and per-tool call/error counts

Start:
  uvicorn tools.kubernetes_mcp:app --port 8012

Auth (optional):
  Set K8S_MCP_TOKEN env var
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
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

app = FastAPI(title="Kubernetes MCP", version="1.0.0")
_TOKEN = os.getenv("K8S_MCP_TOKEN", "")
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
    "pod_list": {
        "description": "List pods in a Kubernetes namespace.",
        "input": {
            "namespace": {"type": "string", "description": "Kubernetes namespace", "default": "default"},
        },
    },
    "pod_logs": {
        "description": "Get logs from a Kubernetes pod.",
        "input": {
            "pod": {"type": "string", "description": "Pod name", "required": True},
            "namespace": {"type": "string", "description": "Kubernetes namespace", "default": "default"},
            "tail": {"type": "integer", "description": "Number of lines to tail", "default": 100},
        },
    },
    "deployment_scale": {
        "description": "Scale a Kubernetes deployment.",
        "input": {
            "deployment": {"type": "string", "description": "Deployment name", "required": True},
            "replicas": {"type": "integer", "description": "Number of replicas", "required": True},
            "namespace": {"type": "string", "description": "Kubernetes namespace", "default": "default"},
        },
    },
    "namespace_list": {
        "description": "List all Kubernetes namespaces.",
        "input": {},
    },
    "apply_manifest": {
        "description": "Apply a YAML manifest to the cluster.",
        "input": {
            "manifest": {"type": "string", "description": "YAML manifest content", "required": True},
        },
    },
    "rollout_status": {
        "description": "Check rollout status of a deployment.",
        "input": {
            "deployment": {"type": "string", "description": "Deployment name", "required": True},
            "namespace": {"type": "string", "description": "Kubernetes namespace", "default": "default"},
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

def _pod_list(args: Dict) -> Any:
    namespace = args.get("namespace", "default").strip()
    _sanitize(namespace, "namespace")
    out = _run_cmd(["kubectl", "get", "pods", "-n", namespace, "-o", "json"])
    if out["returncode"] != 0:
        raise RuntimeError(f"kubectl get pods failed: {out['stderr']}")
    return {"namespace": namespace, "output": out["stdout"]}


def _pod_logs(args: Dict) -> Any:
    pod = args.get("pod", "").strip()
    if not pod:
        raise ValueError("'pod' is required.")
    _sanitize(pod, "pod")
    namespace = args.get("namespace", "default").strip()
    _sanitize(namespace, "namespace")
    tail = int(args.get("tail", 100))
    out = _run_cmd(["kubectl", "logs", pod, "-n", namespace, "--tail", str(tail)])
    if out["returncode"] != 0:
        raise RuntimeError(f"kubectl logs failed: {out['stderr']}")
    return {"pod": pod, "namespace": namespace, "logs": out["stdout"]}


def _deployment_scale(args: Dict) -> Any:
    deployment = args.get("deployment", "").strip()
    if not deployment:
        raise ValueError("'deployment' is required.")
    _sanitize(deployment, "deployment")
    replicas = args.get("replicas")
    if replicas is None:
        raise ValueError("'replicas' is required.")
    replicas = int(replicas)
    namespace = args.get("namespace", "default").strip()
    _sanitize(namespace, "namespace")
    out = _run_cmd([
        "kubectl", "scale", f"deployment/{deployment}",
        f"--replicas={replicas}", "-n", namespace,
    ])
    if out["returncode"] != 0:
        raise RuntimeError(f"kubectl scale failed: {out['stderr']}")
    return {"deployment": deployment, "replicas": replicas, "namespace": namespace, "output": out["stdout"]}


def _namespace_list(args: Dict) -> Any:
    out = _run_cmd(["kubectl", "get", "namespaces", "-o", "json"])
    if out["returncode"] != 0:
        raise RuntimeError(f"kubectl get namespaces failed: {out['stderr']}")
    return {"output": out["stdout"]}


def _apply_manifest(args: Dict) -> Any:
    manifest = args.get("manifest", "").strip()
    if not manifest:
        raise ValueError("'manifest' is required.")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(manifest)
        tmp_path = f.name
    try:
        out = _run_cmd(["kubectl", "apply", "-f", tmp_path])
        if out["returncode"] != 0:
            raise RuntimeError(f"kubectl apply failed: {out['stderr']}")
        return {"output": out["stdout"]}
    finally:
        os.unlink(tmp_path)


def _rollout_status(args: Dict) -> Any:
    deployment = args.get("deployment", "").strip()
    if not deployment:
        raise ValueError("'deployment' is required.")
    _sanitize(deployment, "deployment")
    namespace = args.get("namespace", "default").strip()
    _sanitize(namespace, "namespace")
    out = _run_cmd(["kubectl", "rollout", "status", f"deployment/{deployment}", "-n", namespace])
    if out["returncode"] != 0:
        raise RuntimeError(f"kubectl rollout status failed: {out['stderr']}")
    return {"deployment": deployment, "namespace": namespace, "output": out["stdout"]}


# Map tool names → handler functions
_TOOL_HANDLERS = {
    "pod_list": _pod_list,
    "pod_logs": _pod_logs,
    "deployment_scale": _deployment_scale,
    "namespace_list": _namespace_list,
    "apply_manifest": _apply_manifest,
    "rollout_status": _rollout_status,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "kubernetes_mcp",
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
        log_json("INFO", "k8s_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "k8s_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "k8s_mcp_tool_error", details={"tool": name, "error": str(exc)})
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
    port = int(os.getenv("K8S_MCP_PORT", _cfg.get_mcp_server_port("kubernetes", default=8012)))
    uvicorn.run("tools.kubernetes_mcp:app", host="0.0.0.0", port=port, reload=False)
