"""
JSON-RPC 2.0 stdio transport for AURA CLI.

Reads newline-delimited JSON-RPC 2.0 requests from stdin, dispatches them
through the AURA COMMAND_DISPATCH_REGISTRY, and writes JSON-RPC 2.0 responses
to stdout.  Designed for third-party integrations (e.g. VS Code extensions)
that want to communicate with AURA over a subprocess pipe.

Protocol reference: docs/JSON_RPC_PROTOCOL.md
"""

import json
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

from core.logging_utils import log_json
from aura_cli.options import action_default_canonical_path
from aura_cli.cli_options import CLIParseError, parse_cli_args
from aura_cli.cli_main import dispatch_command, create_runtime as _default_create_runtime

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 error codes
# ---------------------------------------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# ---------------------------------------------------------------------------
# Method map: JSON-RPC method name  →  COMMAND_DISPATCH_REGISTRY action key
# ---------------------------------------------------------------------------
# Each entry maps a dot-namespaced JSON-RPC method to the corresponding action
# registered in aura_cli.cli_main.COMMAND_DISPATCH_REGISTRY.
METHOD_MAP: dict[str, str] = {
    "goal.add": "goal_add",
    "goal.run": "goal_run",
    "goal.once": "goal_once",
    "goal.status": "goal_status",
    "queue.list": "queue_list",
    "queue.clear": "queue_clear",
    "doctor": "doctor",
    "bootstrap": "bootstrap",
    "config.show": "show_config",
    "diag": "diag",
    "logs": "logs",
    "memory.search": "memory_search",
    "memory.reindex": "memory_reindex",
    "metrics": "metrics_show",
    "workflow.run": "workflow_run",
    "scaffold": "scaffold",
    "evolve": "evolve",
    "mcp.tools": "mcp_tools",
    "mcp.call": "mcp_call",
    "help": "help",
}


def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "error": err, "id": request_id}


def _jsonrpc_result(request_id: Any, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "result": result, "id": request_id}


def _params_to_argv(method: str, action: str, params: dict) -> list[str]:
    """
    Convert JSON-RPC params dict to an argv list suitable for parse_cli_args.

    The conversion follows simple rules:
    - For goal.add / goal.once: params["goal"] → positional argument appended after
      the canonical CLI path.
    - For goal.run: no positional argument required; optional flags like
      "max_cycles" and "dry_run" are translated.
    - For memory.search: params["query"] → positional argument.
    - For mcp.call: params["name"] and params["input"] → --name / --input flags.
    - All other methods: boolean keys become --<key> flags; string/int keys
      become --<key> <value> pairs.

    The returned list does NOT include the executable name.
    """
    canonical = action_default_canonical_path(action)
    argv: list[str] = list(canonical) if canonical else []

    if method in ("goal.add", "goal.once", "goal.add_run"):
        goal_text = params.get("goal", "")
        if goal_text:
            argv.append(str(goal_text))

    elif method == "goal.run":
        if params.get("dry_run"):
            argv.append("--dry-run")
        if "max_cycles" in params:
            argv.extend(["--max-cycles", str(params["max_cycles"])])

    elif method == "memory.search":
        query = params.get("query", "")
        if query:
            argv.append(str(query))

    elif method == "mcp.call":
        if "name" in params:
            argv.extend(["--name", str(params["name"])])
        if "input" in params:
            input_val = params["input"]
            argv.extend(["--input", json.dumps(input_val) if isinstance(input_val, dict) else str(input_val)])

    elif method == "workflow.run":
        goal_text = params.get("goal", "")
        if goal_text:
            argv.append(str(goal_text))

    else:
        # Generic fallback: booleans → flags, scalars → --key value
        for key, value in params.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            else:
                argv.extend([flag, str(value)])

    # Always add --json so dispatch handlers return machine-readable output
    if "--json" not in argv:
        argv.append("--json")

    return argv


class StdioTransport:
    """
    JSON-RPC 2.0 transport over stdin/stdout.

    Usage::

        transport = StdioTransport(project_root=Path("."))
        transport.run()   # blocks, reading lines from stdin

    Each newline-terminated line on stdin must be a JSON-RPC 2.0 request
    object.  The transport writes a JSON-RPC 2.0 response object (plus a
    newline) to stdout for every request.  Batch requests are not supported.

    ``StdioTransport`` can also be used in unit tests via
    :meth:`handle_request`, which processes a single request dict and returns
    the response dict without touching real stdin/stdout.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        runtime_factory=None,
        method_map: dict[str, str] | None = None,
        in_stream=None,
        out_stream=None,
    ) -> None:
        self.project_root = project_root or Path(".")
        self._method_map = method_map if method_map is not None else METHOD_MAP
        self._in = in_stream or sys.stdin
        self._out = out_stream or sys.stdout

        # Lazily resolve runtime_factory to avoid circular imports at module
        # load time.
        self._runtime_factory = runtime_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Read newline-delimited JSON-RPC requests from stdin until EOF,
        writing responses to stdout.
        """
        log_json("INFO", "stdio_transport_started", details={"project_root": str(self.project_root)})
        for raw_line in self._in:
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip():
                continue
            response = self._process_raw(raw_line)
            print(json.dumps(response), file=self._out, flush=True)

    def handle_request(self, request: dict) -> dict:
        """
        Process a single JSON-RPC request dict and return the response dict.
        Does not read from or write to stdin/stdout; useful for unit tests.
        """
        return self._dispatch(request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_raw(self, raw: str) -> dict:
        """Parse a raw JSON string and dispatch."""
        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            return _jsonrpc_error(None, PARSE_ERROR, f"Parse error: {exc}")
        return self._dispatch(request)

    def _dispatch(self, request: dict) -> dict:
        """Validate and execute a single parsed JSON-RPC request."""
        # Validate structure
        if not isinstance(request, dict):
            return _jsonrpc_error(None, INVALID_REQUEST, "Request must be a JSON object")
        if request.get("jsonrpc") != "2.0":
            req_id = request.get("id")
            return _jsonrpc_error(req_id, INVALID_REQUEST, 'Missing or invalid "jsonrpc" field (must be "2.0")')

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if not isinstance(method, str) or not method:
            return _jsonrpc_error(req_id, INVALID_REQUEST, '"method" must be a non-empty string')
        if not isinstance(params, dict):
            return _jsonrpc_error(req_id, INVALID_PARAMS, '"params" must be an object (dict)')

        # Resolve method → action
        action = self._method_map.get(method)
        if action is None:
            return _jsonrpc_error(req_id, METHOD_NOT_FOUND, f'Method not found: "{method}"')

        # Execute via CLI dispatch
        try:
            result = self._invoke_action(method, action, params)
        except Exception as exc:
            log_json("ERROR", "stdio_transport_dispatch_error", details={"method": method, "error": str(exc)})
            return _jsonrpc_error(req_id, INTERNAL_ERROR, f"Internal error: {exc}")

        return _jsonrpc_result(req_id, result)

    def _invoke_action(self, method: str, action: str, params: dict) -> Any:
        """
        Convert params to argv, parse, dispatch, and return the captured output.
        """
        runtime_factory = self._runtime_factory or _default_create_runtime

        argv = _params_to_argv(method, action, params)

        try:
            parsed = parse_cli_args(argv)
        except CLIParseError as exc:
            raise ValueError(f"Invalid params for method '{method}': {exc}") from exc

        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with redirect_stdout(captured_out), redirect_stderr(captured_err):
            exit_code = dispatch_command(
                parsed,
                project_root=self.project_root,
                runtime_factory=runtime_factory,
            )

        stdout_text = captured_out.getvalue().strip()
        stderr_text = captured_err.getvalue().strip()

        # Try to parse stdout as JSON for a richer response
        output: Any
        if stdout_text:
            try:
                output = json.loads(stdout_text)
            except json.JSONDecodeError:
                output = stdout_text
        else:
            output = None

        return {
            "exit_code": exit_code,
            "output": output,
            **({"stderr": stderr_text} if stderr_text else {}),
        }
