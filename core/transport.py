"""
JSON-RPC 2.0 over stdio transport for AURA.

Reads newline-delimited JSON-RPC requests from stdin and writes responses to
stdout.  During dispatch any rogue ``print()`` calls made by handlers are
captured so they never corrupt the JSON stream.

Usage
-----
    python main.py --stdio-rpc

Method mapping
--------------
RPC method names use dot-notation families that map directly to the action
keys inside ``COMMAND_DISPATCH_REGISTRY``:

    goal.add       -> goal_add
    goal.run       -> goal_run
    goal.once      -> goal_once
    goal.status    -> goal_status
    queue.list     -> queue_list
    queue.clear    -> queue_clear
    memory.search  -> memory_search
    memory.reindex -> memory_reindex
    mcp.tools      -> mcp_tools
    mcp.call       -> mcp_call
    system.doctor  -> doctor
    system.config  -> show_config
    system.metrics -> metrics_show
    system.diag    -> diag

JSON-RPC 2.0 request schema
----------------------------
    {
        "jsonrpc": "2.0",
        "method": "goal.add",
        "params": {"add_goal": "Refactor core/transport.py", "run_goals": false},
        "id": "req-1"
    }

JSON-RPC 2.0 success response schema
-------------------------------------
    {
        "jsonrpc": "2.0",
        "result": {
            "code": 0,
            "cli_output": "Added goal: ...\nQueue length: 1\n"
        },
        "id": "req-1"
    }

JSON-RPC 2.0 error response schema
------------------------------------
    {
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": "Method not found: unknown.method"},
        "id": "req-1"
    }
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.logging_utils import log_json


# ---------------------------------------------------------------------------
# Error codes (JSON-RPC 2.0 spec)
# ---------------------------------------------------------------------------

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# Method → dispatch-action mapping
# ---------------------------------------------------------------------------

#: Maps JSON-RPC method names (dot-notation) to COMMAND_DISPATCH_REGISTRY keys.
METHOD_MAP: dict[str, str] = {
    "goal.add": "goal_add",
    "goal.run": "goal_run",
    "goal.once": "goal_once",
    "goal.status": "goal_status",
    "queue.list": "queue_list",
    "queue.clear": "queue_clear",
    "memory.search": "memory_search",
    "memory.reindex": "memory_reindex",
    "mcp.tools": "mcp_tools",
    "mcp.call": "mcp_call",
    "system.doctor": "doctor",
    "system.config": "show_config",
    "system.metrics": "metrics_show",
    "system.diag": "diag",
}


DEFAULT_ARGS: dict[str, Any] = {
    "json": True,
    "dry_run": False,
    "decompose": False,
    "model": None,
    "beads": False,
    "no_beads": False,
    "beads_required": False,
    "beads_optional": False,
    "explain": False,
    "max_cycles": None,
    "add_goal": None,
    "run_goals": False,
    "status": False,
    "goal": None,
    "mcp_tools": False,
    "mcp_call": None,
    "mcp_args": None,
    "diag": False,
    "query": None,
    "limit": 5,
    "memory_reindex": False,
}


METHOD_ARG_DEFAULTS: dict[str, dict[str, Any]] = {
    "goal.run": {"run_goals": True},
    "goal.status": {"status": True},
    "mcp.tools": {"mcp_tools": True},
    "memory.reindex": {"memory_reindex": True},
    "system.diag": {"diag": True},
}


class JSONRPCError(Exception):
    """Raised to signal a JSON-RPC protocol-level error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.rpc_code = code
        self.message = message


# ---------------------------------------------------------------------------
# Transport implementation
# ---------------------------------------------------------------------------

class StdioTransport:
    """JSON-RPC 2.0 server that reads from stdin and writes to stdout.

    Stdout/stderr are redirected during every handler invocation so that
    ``print()`` calls from handlers are captured and returned inside the
    ``result.cli_output`` field instead of being written directly to the
    transport stream.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        # Keep a reference to the real stdout so we can always write JSON to it.
        self._real_stdout = sys.stdout
        self._project_root = project_root or Path(".").resolve()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> None:
        """Write *payload* as a single newline-terminated JSON line to the real stdout."""
        line = json.dumps(payload, default=str)
        self._real_stdout.write(line + "\n")
        self._real_stdout.flush()

    def _error_response(self, req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id,
        }

    def _success_response(self, req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "result": result, "id": req_id}

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _validate_request(self, req: Any) -> None:
        """Raise JSONRPCError for structurally invalid requests."""
        if not isinstance(req, dict):
            raise JSONRPCError(INVALID_REQUEST, "Request must be a JSON object")
        if req.get("jsonrpc") != "2.0":
            raise JSONRPCError(INVALID_REQUEST, "jsonrpc field must be '2.0'")
        if "method" not in req or not isinstance(req["method"], str):
            raise JSONRPCError(INVALID_REQUEST, "method field is required and must be a string")
        if "id" in req and req["id"] is not None and not isinstance(req["id"], (str, int, float)):
            raise JSONRPCError(INVALID_REQUEST, "id must be a string, number, or null")

    def _build_args(self, method: str, params: dict[str, Any]) -> SimpleNamespace:
        values = dict(DEFAULT_ARGS)
        values.update(METHOD_ARG_DEFAULTS.get(method, {}))
        values.update(params)
        return SimpleNamespace(**values)

    def _dispatch(self, method: str, params: dict[str, Any]) -> tuple[int, str]:
        """Dispatch *method* with *params* and return (exit_code, captured_output).

        Imports are deferred to avoid circular-import issues at module level.
        """
        from aura_cli.cli_main import (  # pylint: disable=import-outside-toplevel
            COMMAND_DISPATCH_REGISTRY,
            DispatchContext,
            _prepare_runtime_context,
            create_runtime,
        )

        if method not in METHOD_MAP:
            raise JSONRPCError(METHOD_NOT_FOUND, f"Method not found: {method}")

        action = METHOD_MAP[method]
        rule = COMMAND_DISPATCH_REGISTRY.get(action)
        if rule is None:
            raise JSONRPCError(METHOD_NOT_FOUND, f"No handler registered for action: {action}")

        # Build a namespace with CLI-compatible defaults so handlers can rely
        # on the same optional attributes they receive from argparse.
        args = self._build_args(method, params)

        # Build a synthetic parsed object that _resolve_dispatch_action understands.
        parsed = SimpleNamespace(action=action, namespace=args, warnings=[], warning_records=[])

        ctx = DispatchContext(
            parsed=parsed,
            project_root=self._project_root,
            runtime_factory=create_runtime,
            args=args,
            runtime=None,
        )

        if rule.requires_runtime:
            rc = _prepare_runtime_context(ctx)
            if rc is not None:
                return rc, ""

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            exit_code = rule.handler(ctx)

        return exit_code, buf.getvalue()

    def handle_raw_line(self, line: str) -> dict[str, Any] | None:
        """Parse and handle a single JSON-RPC request line.

        Returns a well-formed JSON-RPC response dict, or ``None`` for a
        notification that should not receive any response.
        """
        req_id: Any = None
        is_notification = False
        try:
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JSONRPCError(PARSE_ERROR, f"Parse error: {exc}") from exc

            # We can now extract the id (may be None / absent for notifications).
            if isinstance(req, dict):
                if "id" in req:
                    candidate_id = req.get("id")
                    if candidate_id is None or isinstance(candidate_id, (str, int, float)):
                        req_id = candidate_id
                    else:
                        req_id = None
                else:
                    is_notification = True

            self._validate_request(req)

            method: str = req["method"]
            params: dict[str, Any] = req.get("params") or {}
            if not isinstance(params, dict):
                raise JSONRPCError(INVALID_PARAMS, "params must be a JSON object")

            exit_code, cli_output = self._dispatch(method, params)
            if is_notification:
                return None
            return self._success_response(req_id, {"code": exit_code, "cli_output": cli_output})

        except JSONRPCError as exc:
            log_json("WARN", "stdio_rpc_error", details={"code": exc.rpc_code, "message": exc.message})
            return self._error_response(req_id, exc.rpc_code, exc.message)
        except Exception as exc:  # pylint: disable=broad-except
            log_json("ERROR", "stdio_rpc_internal_error", details={"error": str(exc)})
            return self._error_response(req_id, INTERNAL_ERROR, str(exc))

    # ------------------------------------------------------------------
    # Async event loop
    # ------------------------------------------------------------------

    async def _read_lines(self) -> asyncio.StreamReader:
        """Attach an asyncio StreamReader to sys.stdin."""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        return reader

    async def run_async(self) -> None:
        """Run the server: read newline-delimited JSON from stdin, write responses to stdout."""
        reader = await self._read_lines()
        while True:
            raw = await reader.readline()
            if not raw:
                break  # EOF
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            response = self.handle_raw_line(line)
            if response is not None:
                self._send(response)

    def run(self) -> None:
        """Synchronous entry point (wraps the async loop)."""
        asyncio.run(self.run_async())
