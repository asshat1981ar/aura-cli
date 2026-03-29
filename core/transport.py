"""JSON-RPC 2.0 over stdio transport for the AURA orchestrator.

Reads line-delimited JSON-RPC 2.0 requests from stdin and writes responses to
stdout.  Each line is a complete JSON object; responses are also written one
per line.

JSON-RPC 2.0 quick reference:
  Request:  {"jsonrpc": "2.0", "id": 1, "method": "goal/run", "params": {...}}
  Response: {"jsonrpc": "2.0", "id": 1, "result": {...}}
  Error:    {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "..."}}
  Notification (no response): {"jsonrpc": "2.0", "method": "...", "params": {...}}

Supported methods:
  goal/run       — run a goal through the orchestrator
  goal/add       — add a goal to the queue
  goal/status    — return current orchestrator status
  system/health  — health check

Standard error codes:
  -32700  Parse error
  -32600  Invalid request
  -32601  Method not found
  -32000  Server error (orchestrator or handler exception)
"""

import json
import sys
import datetime
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Error code constants (JSON-RPC 2.0)
# ---------------------------------------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
SERVER_ERROR = -32000


class StdioTransport:
    """JSON-RPC 2.0 over stdio transport for AURA orchestrator.

    Reads newline-delimited JSON from stdin and writes newline-delimited JSON
    responses to stdout.  Notifications (requests without an ``id`` field) are
    processed but do not produce a response.

    Args:
        orchestrator: An optional :class:`~core.orchestrator.LoopOrchestrator`
            instance.  When *None* the transport still handles ``system/health``
            and ``goal/status`` but will return a server error for methods that
            require an active orchestrator.
        stdin: Readable stream to consume requests from.  Defaults to
            ``sys.stdin``.
        stdout: Writable stream to emit responses to.  Defaults to
            ``sys.stdout``.
    """

    def __init__(
        self,
        orchestrator=None,
        stdin=None,
        stdout=None,
    ) -> None:
        self.orchestrator = orchestrator
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        self._methods: Dict[str, Callable[[Any], Any]] = {
            "goal/run": self._handle_goal_run,
            "goal/add": self._handle_goal_add,
            "goal/status": self._handle_goal_status,
            "system/health": self._handle_system_health,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Read JSON-RPC requests from stdin, write responses to stdout.

        Runs until EOF.  Each line is processed independently; a malformed line
        produces a parse-error response but does not stop the loop.
        """
        for raw_line in self._stdin:
            raw_line = raw_line.rstrip("\n")
            if not raw_line:
                continue
            response = self._process_line(raw_line)
            if response is not None:
                self._write_response(response)

    def dispatch(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single parsed request dict and return the response dict.

        Returns *None* for notifications (requests without an ``id``).

        This is a lower-level entry point useful for testing individual
        requests without going through the full stdin/stdout loop.
        """
        return self._dispatch(request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_line(self, raw_line: str) -> Optional[Dict[str, Any]]:
        """Parse *raw_line* as JSON and dispatch the resulting request.

        Returns a JSON-RPC response dict, or *None* for notifications.
        On parse failure returns an error response (with ``id: null``).
        """
        try:
            data = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError) as exc:
            return self._error_response(None, PARSE_ERROR, f"Parse error: {exc}")

        return self._dispatch(data)

    def _dispatch(self, data: Any) -> Optional[Dict[str, Any]]:
        """Validate and route a decoded request object.

        Returns a response dict, or *None* for notifications.
        """
        # Spec: id may be absent (notification), string, number, or null.
        request_id = data.get("id") if isinstance(data, dict) else None
        is_notification = isinstance(data, dict) and "id" not in data

        # --- validate structure ------------------------------------------------
        if not isinstance(data, dict):
            return self._error_response(None, INVALID_REQUEST, "Request must be a JSON object")

        if data.get("jsonrpc") != "2.0":
            if not is_notification:
                return self._error_response(request_id, INVALID_REQUEST, "jsonrpc field must be '2.0'")
            return None

        method = data.get("method")
        if not method or not isinstance(method, str):
            if not is_notification:
                return self._error_response(request_id, INVALID_REQUEST, "method field is required and must be a string")
            return None

        params = data.get("params", {})
        if not isinstance(params, (dict, list)):
            params = {}

        # --- route to handler -------------------------------------------------
        handler = self._methods.get(method)
        if handler is None:
            if is_notification:
                # Unknown notifications are silently ignored per spec.
                return None
            return self._error_response(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        try:
            result = handler(params)
        except Exception as exc:  # noqa: BLE001
            if is_notification:
                return None
            return self._error_response(request_id, SERVER_ERROR, f"Server error: {exc}")

        # Notifications: process but do not respond.
        if is_notification:
            return None

        return self._success_response(request_id, result)

    def _write_response(self, response: Dict[str, Any]) -> None:
        """Serialise *response* to a single JSON line on stdout."""
        self._stdout.write(json.dumps(response) + "\n")
        self._stdout.flush()

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    @staticmethod
    def _success_response(request_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error_response(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    # ------------------------------------------------------------------
    # Method handlers
    # ------------------------------------------------------------------

    def _handle_goal_run(self, params: Any) -> Dict[str, Any]:
        """Run a goal through the orchestrator.

        Expected params::

            {
                "goal": "Refactor the auth module",
                "max_cycles": 5,   # optional, default 5
                "dry_run": false   # optional, default false
            }

        Returns the dict produced by :meth:`~core.orchestrator.LoopOrchestrator.run_loop`.
        """
        if not isinstance(params, dict):
            raise ValueError("params must be an object with at least a 'goal' key")

        goal = params.get("goal")
        if not goal or not isinstance(goal, str):
            raise ValueError("'goal' param is required and must be a non-empty string")

        if self.orchestrator is None:
            raise RuntimeError("No orchestrator configured")

        max_cycles = int(params.get("max_cycles", 5))
        dry_run = bool(params.get("dry_run", False))

        return self.orchestrator.run_loop(goal, max_cycles=max_cycles, dry_run=dry_run)

    def _handle_goal_add(self, params: Any) -> Dict[str, Any]:
        """Add a goal to the orchestrator's goal queue.

        Expected params::

            {"goal": "Add unit tests for the auth module"}

        Returns ``{"queued": true}``.
        """
        if not isinstance(params, dict):
            raise ValueError("params must be an object with a 'goal' key")

        goal = params.get("goal")
        if not goal or not isinstance(goal, str):
            raise ValueError("'goal' param is required and must be a non-empty string")

        if self.orchestrator is None:
            raise RuntimeError("No orchestrator configured")

        if self.orchestrator.goal_queue is None:
            raise RuntimeError("Orchestrator has no goal queue")

        self.orchestrator.goal_queue.add(goal)
        return {"queued": True}

    def _handle_goal_status(self, params: Any) -> Dict[str, Any]:
        """Return the current orchestrator status.

        Returns a dict with:

        * ``cycle_count``  — number of cycles completed (if available)
        * ``current_goal`` — the goal currently being processed, or *null*
        * ``queue_size``   — number of goals waiting in the queue
        """
        if self.orchestrator is None:
            return {
                "cycle_count": 0,
                "current_goal": None,
                "queue_size": 0,
            }

        cycle_count = 0
        if hasattr(self.orchestrator, "last_cycle_summary") and self.orchestrator.last_cycle_summary:
            # Best-effort: extract the cycle index if present.
            cycle_count = self.orchestrator.last_cycle_summary.get("cycle_index", 0)

        queue_size = 0
        if self.orchestrator.goal_queue is not None:
            try:
                queue_size = len(self.orchestrator.goal_queue)
            except TypeError:
                queue_size = 0

        return {
            "cycle_count": cycle_count,
            "current_goal": self.orchestrator.current_goal,
            "queue_size": queue_size,
        }

    def _handle_system_health(self, params: Any) -> Dict[str, Any]:
        """Return a simple health-check response.

        Returns::

            {"status": "ok", "timestamp": "<ISO-8601 UTC timestamp>"}
        """
        return {
            "status": "ok",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
