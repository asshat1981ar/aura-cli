"""Comprehensive MCP tool server for AURA CLI.

Exposes a FastAPI application with a /call endpoint that dispatches to a
rich set of developer tools: file I/O, linting, formatting, searching,
compression, git utilities, and more.

Module-level flags (``ENABLE_WRITE``, ``ENABLE_RUN``, ``RUN_ALLOW``, etc.)
can be overridden at runtime by tests via ``monkeypatch.setattr``.
"""
from __future__ import annotations

import ast
import base64
import io
import json
import os
import re
import shlex
import subprocess
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Module-level configuration — tests may monkeypatch these at any time.
# ---------------------------------------------------------------------------

#: Absolute path to the repository root used as a filesystem jail.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Maximum bytes read by ``read_file`` tool.
MAX_READ_BYTES: int = int(os.getenv("MCP_MAX_READ_BYTES", "100000"))

#: Maximum file size (bytes) accepted by the ``compress`` tool.
COMPRESS_MAX_BYTES: int = int(os.getenv("MCP_COMPRESS_MAX_BYTES", "1000000"))

#: When False, any write-mutating tool returns HTTP 403.
ENABLE_WRITE: bool = os.getenv("MCP_ENABLE_WRITE", "") == "1"

#: When False, any shell-execution tool returns HTTP 403.
ENABLE_RUN: bool = os.getenv("MCP_ENABLE_RUN", "") == "1"

#: Allowed executable names for shell-execution tools.
RUN_ALLOW: set = {
    "python",
    "python3",
    "ruff",
    "pip",
    "pip-audit",
    "npm",
    "npx",
    "prettier",
    "semgrep",
    "echo",
}

#: Rate-limit ceiling (calls per minute, per token).  0 = disabled.
RATE_LIMIT_PER_MIN: int = int(os.getenv("MCP_RATE_LIMIT_PER_MIN", "0"))

# ---------------------------------------------------------------------------
# Rate-limit state (in-process, per-token sliding window)
# ---------------------------------------------------------------------------
_rate_state: Dict[str, deque] = {}


def _check_rate_limit(token: str) -> None:
    """Raise HTTP 429 if *token* has exceeded ``RATE_LIMIT_PER_MIN``."""
    limit = RATE_LIMIT_PER_MIN
    if limit <= 0:
        return
    now = time.time()
    window = 60.0
    timestamps = _rate_state.setdefault(token, deque())
    while timestamps and now - timestamps[0] > window:
        timestamps.popleft()
    if len(timestamps) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    timestamps.append(now)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="AURA MCP Server", version="0.2.0")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _require_auth(authorization: Optional[str] = Header(default=None)) -> str:
    """Validate bearer token; return token string (or ``'anon'`` if auth disabled)."""
    token = os.getenv("MCP_API_TOKEN", "")
    if not token:
        return "anon"
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != token:
        raise HTTPException(status_code=403, detail="Invalid token")
    _check_rate_limit(parts[1])
    return parts[1]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CallRequest(BaseModel):
    """Generic tool-invocation payload."""

    tool_name: str
    args: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jail(path_str: str) -> Path:
    """Resolve *path_str* relative to ``PROJECT_ROOT``; raise 400 if outside."""
    resolved = (PROJECT_ROOT / path_str).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Path outside project root: {path_str}")
    return resolved


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Run *cmd* and return ``{"returncode", "stdout", "stderr"}``.

    Raises HTTP 403 when ``ENABLE_RUN`` is False or the binary is not in
    ``RUN_ALLOW``.  Returns ``returncode=127`` when the binary is not found
    on the system (mirrors shell behaviour).
    """
    if not ENABLE_RUN:
        raise HTTPException(status_code=403, detail="Run tools disabled (set MCP_ENABLE_RUN=1)")
    if not cmd:
        raise HTTPException(status_code=400, detail="Empty command")
    binary = Path(cmd[0]).name
    if binary not in RUN_ALLOW:
        raise HTTPException(status_code=403, detail=f"Command not allowed: {cmd[0]}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except FileNotFoundError:
        return {"returncode": 127, "stdout": "", "stderr": f"{cmd[0]}: command not found"}


# ---------------------------------------------------------------------------
# Tools manifest (returned by GET /tools)
# ---------------------------------------------------------------------------

TOOLS_MANIFEST: List[Dict[str, str]] = [
    {"name": "read_file", "description": "Read a file within the project"},
    {"name": "write_file_safe", "description": "Write a file (requires ENABLE_WRITE)"},
    {"name": "run_sandboxed", "description": "Run an allowlisted command"},
    {"name": "dependency_audit", "description": "Run dependency audit command"},
    {"name": "format", "description": "Run code formatter command"},
    {"name": "docstring_fill", "description": "Fill missing docstrings"},
    {"name": "code_intel_xref", "description": "Cross-reference a symbol across Python files"},
    {"name": "recent_files", "description": "List recently modified files"},
    {"name": "env_snapshot", "description": "Capture environment variable snapshot"},
    {"name": "tail_logs", "description": "Tail a log file (binary-safe)"},
    {"name": "limits", "description": "Show current limits and flags"},
    {"name": "structured_search", "description": "Search for a text pattern across files"},
    {"name": "secrets_scan", "description": "Scan files for common secret patterns"},
    {"name": "package_scripts", "description": "List npm/package.json scripts"},
    {"name": "apply_replacements", "description": "Apply search-replace patches (requires ENABLE_WRITE)"},
    {"name": "json_patch", "description": "JSON Pointer patch (requires ENABLE_WRITE)"},
    {"name": "yaml_patch", "description": "YAML patch (requires ENABLE_WRITE)"},
    {"name": "compress", "description": "Compress files into a base64-encoded zip"},
    {"name": "decompress", "description": "Decompress a base64-encoded zip archive"},
    {"name": "git_blame_snippet", "description": "Show git blame for a line range"},
    {"name": "git_file_history", "description": "Show git log for a file"},
    {"name": "semgrep_scan", "description": "Run semgrep static analysis"},
    {"name": "quick_fix", "description": "Run ruff --fix on files"},
    {"name": "linter_capabilities", "description": "List available linter binaries"},
    {"name": "refactor_plan", "description": "Analyse a Python file and produce a refactor plan"},
    {"name": "lint_files", "description": "Lint specific files with ruff"},
    {"name": "lint_all", "description": "Lint entire project with ruff"},
    {"name": "debug_trace", "description": "Run a Python module for debug tracing"},
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(auth: str = Depends(_require_auth)):
    """Return server health, current limits, and runtime metrics."""
    return {
        "status": "ok",
        "limits": {
            "max_read_bytes": MAX_READ_BYTES,
            "rate_limit_per_min": RATE_LIMIT_PER_MIN,
            "compress_max_bytes": COMPRESS_MAX_BYTES,
        },
        "metrics": {
            "enable_write": ENABLE_WRITE,
            "enable_run": ENABLE_RUN,
        },
    }


@app.get("/tools")
async def tools_list(auth: str = Depends(_require_auth)):
    """Return the full list of available tools."""
    return {"data": {"tools": TOOLS_MANIFEST}}


@app.post("/call")
async def call_tool(req: CallRequest, auth: str = Depends(_require_auth)):  # noqa: C901
    """Dispatch a tool call by name and return its result under ``data``."""
    name = req.tool_name
    args = req.args

    # ------------------------------------------------------------------ #
    # read_file
    # ------------------------------------------------------------------ #
    if name == "read_file":
        path_str = args.get("path", "")
        p = _jail(path_str)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
        content = p.read_bytes()[:MAX_READ_BYTES].decode(errors="replace")
        return {"data": {"content": content}}

    # ------------------------------------------------------------------ #
    # write_file_safe
    # ------------------------------------------------------------------ #
    if name == "write_file_safe":
        if not ENABLE_WRITE:
            raise HTTPException(status_code=403, detail="Write disabled (set MCP_ENABLE_WRITE=1)")
        path_str = args.get("path", "")
        content = args.get("content", "")
        p = _jail(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"data": {"written": True, "path": path_str}}

    # ------------------------------------------------------------------ #
    # tail_logs
    # ------------------------------------------------------------------ #
    if name == "tail_logs":
        path_str = args.get("path", "")
        lines_count = int(args.get("lines", 50))
        # Support both absolute and relative paths
        p = Path(path_str)
        if not p.is_absolute():
            p = _jail(path_str)
        else:
            try:
                p.resolve().relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                raise HTTPException(status_code=400, detail="Path outside project root")
        if not p.exists():
            return {"data": {"lines": []}}
        raw = p.read_bytes()
        if b"\x00" in raw:
            return {"data": {"lines": ["[binary content skipped]"]}}
        text = raw.decode(errors="replace")
        all_lines = text.splitlines()
        return {"data": {"lines": all_lines[-lines_count:]}}

    # ------------------------------------------------------------------ #
    # limits
    # ------------------------------------------------------------------ #
    if name == "limits":
        return {
            "data": {
                "limits": {
                    "max_read_bytes": MAX_READ_BYTES,
                    "rate_limit_per_min": RATE_LIMIT_PER_MIN,
                    "compress_max_bytes": COMPRESS_MAX_BYTES,
                },
                "flags": {
                    "enable_write": ENABLE_WRITE,
                    "enable_run": ENABLE_RUN,
                },
            }
        }

    # ------------------------------------------------------------------ #
    # compress
    # ------------------------------------------------------------------ #
    if name == "compress":
        paths = args.get("paths", [])
        buf = io.BytesIO()
        skipped: List[str] = []
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in paths:
                p = _jail(rel)
                if not p.exists():
                    skipped.append(rel)
                    continue
                size = p.stat().st_size
                if size > COMPRESS_MAX_BYTES:
                    skipped.append(p.name)
                    continue
                zf.write(p, arcname=rel)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"data": {"base64": b64, "skipped": skipped}}

    # ------------------------------------------------------------------ #
    # decompress
    # ------------------------------------------------------------------ #
    if name == "decompress":
        b64_str = args.get("base64", "")
        dest_str = args.get("dest", "tmp_out")
        dest = PROJECT_ROOT / dest_str
        dest.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(b64_str)
        skipped: List[str] = []
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for member in zf.namelist():
                out = (dest / member).resolve()
                try:
                    out.relative_to(dest.resolve())
                except ValueError:
                    skipped.append(member)
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(member))
        return {"data": {"dest": str(dest_str), "skipped": skipped}}

    # ------------------------------------------------------------------ #
    # structured_search
    # ------------------------------------------------------------------ #
    if name == "structured_search":
        query = args.get("query", "")
        paths_arg = args.get("paths", [])
        results: List[Dict[str, Any]] = []
        for rel in paths_arg:
            p = _jail(rel)
            if not p.exists():
                continue
            try:
                text = p.read_text(errors="replace")
            except (OSError, IsADirectoryError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if query in line:
                    results.append({"path": str(p), "line": lineno, "text": line})
        return {"data": {"count": len(results), "results": results}}

    # ------------------------------------------------------------------ #
    # secrets_scan
    # ------------------------------------------------------------------ #
    if name == "secrets_scan":
        paths_arg = args.get("paths", [])
        patterns = [
            r"sk-[a-zA-Z0-9]{32,}",
            r"ghp_[a-zA-Z0-9]{36}",
            r"(?i)api[_\-]?key\s*[:=]\s*['\"]?\w{16,}",
            r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
        ]
        results: List[Dict[str, Any]] = []
        for rel in paths_arg:
            p = _jail(rel)
            if not p.exists():
                continue
            try:
                text = p.read_text(errors="replace")
            except (OSError, IsADirectoryError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                for pat in patterns:
                    if re.search(pat, line):
                        results.append({"path": str(p), "line": lineno, "match": line.strip()})
                        break
        return {"data": {"count": len(results), "results": results}}

    # ------------------------------------------------------------------ #
    # json_patch
    # ------------------------------------------------------------------ #
    if name == "json_patch":
        if not ENABLE_WRITE:
            raise HTTPException(status_code=403, detail="Write disabled")
        path_str = args.get("path", "")
        pointer = args.get("pointer", "")
        value = args.get("value")
        p = _jail(path_str)
        obj = json.loads(p.read_text())
        keys = [k for k in pointer.lstrip("/").split("/") if k]
        target = obj
        for k in keys[:-1]:
            target = target[k] if isinstance(target, dict) else target[int(k)]
        if keys:
            last = keys[-1]
            if isinstance(target, dict):
                target[last] = value
            else:
                target[int(last)] = value
        p.write_text(json.dumps(obj))
        return {"data": {"patched": True}}

    # ------------------------------------------------------------------ #
    # yaml_patch
    # ------------------------------------------------------------------ #
    if name == "yaml_patch":
        if not ENABLE_WRITE:
            raise HTTPException(status_code=403, detail="Write disabled")
        try:
            import yaml  # type: ignore
        except ImportError:
            raise HTTPException(status_code=500, detail="PyYAML not installed")
        path_str = args.get("path", "")
        pointer = args.get("pointer", "")
        value = args.get("value")
        p = _jail(path_str)
        obj = yaml.safe_load(p.read_text())
        keys = [k for k in pointer.lstrip("/").split("/") if k]
        target = obj
        for k in keys[:-1]:
            target = target[k]
        if keys:
            target[keys[-1]] = value
        p.write_text(yaml.dump(obj))
        return {"data": {"patched": True}}

    # ------------------------------------------------------------------ #
    # apply_replacements
    # ------------------------------------------------------------------ #
    if name == "apply_replacements":
        if not ENABLE_WRITE:
            raise HTTPException(status_code=403, detail="Write disabled")
        replacements = args.get("replacements", [])
        changes: List[Dict[str, Any]] = []
        for rep in replacements:
            rel = rep.get("path", "")
            search = rep.get("search", "")
            replace = rep.get("replace", "")
            p = _jail(rel)
            if not p.exists():
                changes.append({"path": rel, "changed": False, "reason": "not found"})
                continue
            text = p.read_text()
            new_text = text.replace(search, replace)
            changed = new_text != text
            if changed:
                p.write_text(new_text)
            changes.append({"path": rel, "changed": changed})
        return {"data": {"changes": changes}}

    # ------------------------------------------------------------------ #
    # run_sandboxed
    # ------------------------------------------------------------------ #
    if name == "run_sandboxed":
        cmd = args.get("cmd", [])
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        result = _run_cmd(cmd)
        return {"data": result}

    # ------------------------------------------------------------------ #
    # format
    # ------------------------------------------------------------------ #
    if name == "format":
        cmd_str = args.get("cmd", "")
        cmd = shlex.split(cmd_str) if isinstance(cmd_str, str) else list(cmd_str)
        result = _run_cmd(cmd)
        return {"data": result}

    # ------------------------------------------------------------------ #
    # dependency_audit
    # ------------------------------------------------------------------ #
    if name == "dependency_audit":
        cmd = args.get("cmd", ["pip-audit"])
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        result = _run_cmd(cmd)
        return {"data": {**result, "report": result["stdout"] or result["stderr"]}}

    # ------------------------------------------------------------------ #
    # docstring_fill
    # ------------------------------------------------------------------ #
    if name == "docstring_fill":
        path_str = args.get("path", "")
        p = _jail(path_str)
        content = p.read_text(errors="replace")
        return {"data": {"content": content, "filled": 0}}

    # ------------------------------------------------------------------ #
    # code_intel_xref
    # ------------------------------------------------------------------ #
    if name == "code_intel_xref":
        symbol = args.get("symbol", "")
        refs: List[Dict[str, str]] = []
        for py_file in PROJECT_ROOT.rglob("*.py"):
            try:
                if symbol in py_file.read_text(errors="replace"):
                    refs.append({"file": str(py_file.relative_to(PROJECT_ROOT))})
            except (OSError, IsADirectoryError):
                pass
        return {"data": {"symbol": symbol, "refs": refs}}

    # ------------------------------------------------------------------ #
    # recent_files
    # ------------------------------------------------------------------ #
    if name == "recent_files":
        n = int(args.get("n", 10))
        files = sorted(
            (f for f in PROJECT_ROOT.rglob("*") if f.is_file()),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        recent = [str(f.relative_to(PROJECT_ROOT)) for f in files][:n]
        return {"data": {"files": recent}}

    # ------------------------------------------------------------------ #
    # env_snapshot
    # ------------------------------------------------------------------ #
    if name == "env_snapshot":
        keys = args.get("keys", None)
        snapshot = {k: os.environ.get(k, "") for k in keys} if keys else dict(os.environ)
        return {"data": {"env": snapshot}}

    # ------------------------------------------------------------------ #
    # package_scripts
    # ------------------------------------------------------------------ #
    if name == "package_scripts":
        pkg = PROJECT_ROOT / "package.json"
        scripts = json.loads(pkg.read_text()).get("scripts", {}) if pkg.exists() else {}
        return {"data": {"scripts": scripts}}

    # ------------------------------------------------------------------ #
    # git_blame_snippet
    # ------------------------------------------------------------------ #
    if name == "git_blame_snippet":
        path_str = args.get("path", "")
        start = args.get("start", 1)
        end = args.get("end", 10)
        result = subprocess.run(
            ["git", "blame", f"-L{start},{end}", path_str],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        return {"data": {"blame": result.stdout, "returncode": result.returncode}}

    # ------------------------------------------------------------------ #
    # git_file_history
    # ------------------------------------------------------------------ #
    if name == "git_file_history":
        path_str = args.get("path", "")
        result = subprocess.run(
            ["git", "log", "--oneline", "--", path_str],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        return {"data": {"history": result.stdout.splitlines(), "returncode": result.returncode}}

    # ------------------------------------------------------------------ #
    # semgrep_scan
    # ------------------------------------------------------------------ #
    if name == "semgrep_scan":
        paths_arg = args.get("paths", ["."])
        config = args.get("config", "auto")
        cmd = ["semgrep", "--config", config, "--json", *paths_arg]
        result = _run_cmd(cmd)
        try:
            findings = json.loads(result["stdout"])
        except (json.JSONDecodeError, ValueError):
            findings = {}
        return {"data": {**result, "findings": findings}}

    # ------------------------------------------------------------------ #
    # quick_fix
    # ------------------------------------------------------------------ #
    if name == "quick_fix":
        paths_arg = args.get("paths", ["."])
        result = _run_cmd(["ruff", "check", "--fix", *paths_arg])
        return {"data": result}

    # ------------------------------------------------------------------ #
    # linter_capabilities
    # ------------------------------------------------------------------ #
    if name == "linter_capabilities":
        capabilities: List[str] = []
        for tool in ["ruff", "flake8", "pylint", "prettier", "semgrep"]:
            probe = subprocess.run(["which", tool], capture_output=True)
            if probe.returncode == 0:
                capabilities.append(tool)
        return {"data": {"linters": capabilities}}

    # ------------------------------------------------------------------ #
    # refactor_plan
    # ------------------------------------------------------------------ #
    if name == "refactor_plan":
        path_str = args.get("path", "")
        p = _jail(path_str)
        items: List[str] = []
        try:
            tree = ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    items.append(f"function:{node.name}:{node.lineno}")
                elif isinstance(node, ast.AsyncFunctionDef):
                    items.append(f"async_function:{node.name}:{node.lineno}")
                elif isinstance(node, ast.ClassDef):
                    items.append(f"class:{node.name}:{node.lineno}")
        except SyntaxError as exc:
            items.append(f"syntax_error:{exc}")
        return {"data": {"items": items}}

    # ------------------------------------------------------------------ #
    # lint_files
    # ------------------------------------------------------------------ #
    if name == "lint_files":
        paths_arg = args.get("paths", [])
        result = _run_cmd(["ruff", "check", *paths_arg])
        return {"data": result}

    # ------------------------------------------------------------------ #
    # lint_all
    # ------------------------------------------------------------------ #
    if name == "lint_all":
        result = _run_cmd(["ruff", "check", str(PROJECT_ROOT)])
        return {"data": result}

    # ------------------------------------------------------------------ #
    # debug_trace
    # ------------------------------------------------------------------ #
    if name == "debug_trace":
        module = args.get("module", "")
        result = _run_cmd(["python3", "-m", module])
        return {"data": result}

    raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))