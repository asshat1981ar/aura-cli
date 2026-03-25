"""JSON-RPC 2.0 over stdio transport for AURA CLI.

Enables IDE extensions and TUI clients to talk to the orchestrator without HTTP.
"""
import asyncio
import json
import sys
from typing import Any, Dict

JSONRPC_VERSION = "2.0"

class StdioTransport:
    """Reads JSON-RPC requests from stdin, writes responses to stdout."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self._handlers = {
            "goal": self._handle_goal,
            "status": self._handle_status,
            "ask": self._handle_ask,
            "cancel": self._handle_cancel,
            "ping": self._handle_ping,
        }

    async def serve_forever(self) -> None:
        """Read newline-delimited JSON-RPC messages from stdin."""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
        while True:
            line = await reader.readline()
            if not line:
                break
            await self._handle_line(line.decode().strip())

    async def _handle_line(self, line: str) -> None:
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            self._write_error(None, -32700, f"Parse error: {exc}")
            return
        req_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})
        handler = self._handlers.get(method)
        if handler is None:
            self._write_error(req_id, -32601, f"Method not found: {method}")
            return
        try:
            result = await handler(params)
            self._write_result(req_id, result)
        except Exception as exc:
            self._write_error(req_id, -32603, str(exc))

    async def _handle_goal(self, params: Dict[str, Any]) -> Dict:
        goal = params.get("goal", "")
        dry_run = params.get("dry_run", False)
        context_injection = params.get("context_injection")
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.orchestrator.run_loop(goal, dry_run=dry_run, context_injection=context_injection)
        )
        return {"status": "ok", "result": str(result)}

    async def _handle_status(self, params: Dict[str, Any]) -> Dict:
        return {"status": "ok", "agent_count": len(getattr(self.orchestrator, "agents", {}))}

    async def _handle_ask(self, params: Dict[str, Any]) -> Dict:
        return {"status": "ok", "message": "ask not yet implemented"}

    async def _handle_cancel(self, params: Dict[str, Any]) -> Dict:
        return {"status": "ok", "cancelled": False}

    async def _handle_ping(self, params: Dict[str, Any]) -> Dict:
        return {"pong": True}

    def _write_result(self, req_id: Any, result: Any) -> None:
        resp = {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    def _write_error(self, req_id: Any, code: int, message: str) -> None:
        resp = {"jsonrpc": JSONRPC_VERSION, "id": req_id, "error": {"code": code, "message": message}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
