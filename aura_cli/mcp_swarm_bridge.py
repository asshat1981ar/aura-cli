"""
HTTP bridge: exposes Swarm MCP server (stdio/JSON-RPC) as HTTP at localhost:8050.

AURA's model_adapter expects:
  GET  /tools          → list available tools
  POST /call           → {"tool_name": ..., "args": {...}}
  GET  /health         → {"status": "ok"}

This bridge spawns `node swarm-mcp-server/dist/index.js` as a subprocess,
speaks JSON-RPC 2.0 over its stdin/stdout, and wraps responses as HTTP JSON.

Usage:
  python3 -m aura_cli.mcp_swarm_bridge
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from core.logging_utils import log_json

PORT = int(os.getenv("MCP_SERVER_PORT", "8050"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MCPProcess:
    """Manages the Node.js Swarm MCP server subprocess and JSON-RPC communication."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._req_id = 0
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, Any] = {}
        self._tools_cache: list | None = None

    def start(self):
        env = {**os.environ}
        # Run from project root
        server_path = os.path.join(PROJECT_ROOT, "swarm-mcp-server", "dist", "index.js")
        self._proc = subprocess.Popen(
            ["node", server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT
        )
        # Start reader thread
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        # Initialize MCP session
        self._initialize()
        log_json("INFO", "mcp_swarm_process_started", details={"pid": self._proc.pid})

    def _initialize(self):
        """Send MCP initialize handshake."""
        resp = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "aura-mcp-swarm-bridge", "version": "1.0"},
        })
        if resp:
            self._rpc_notify("notifications/initialized", {})
            log_json("INFO", "mcp_swarm_initialized", details={"server_info": resp.get("result", {}).get("serverInfo", {})})

    def _reader(self):
        """Background thread: reads stdout lines and dispatches responses."""
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                req_id = msg.get("id")
                if req_id is not None and req_id in self._pending:
                    self._results[req_id] = msg
                    self._pending[req_id].set()
            except json.JSONDecodeError:
                pass  # stderr/startup noise

    def _rpc(self, method: str, params: dict, timeout: float = 15.0) -> dict | None:
        with self._lock:
            self._req_id += 1
            req_id = self._req_id
            event = threading.Event()
            self._pending[req_id] = event

        payload = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            log_json("ERROR", "mcp_swarm_pipe_broken")
            return None

        if event.wait(timeout):
            result = self._results.pop(req_id, None)
            self._pending.pop(req_id, None)
            return result
        else:
            self._pending.pop(req_id, None)
            log_json("WARN", "mcp_swarm_rpc_timeout", details={"method": method, "req_id": req_id})
            return None

    def _rpc_notify(self, method: str, params: dict):
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            pass

    def list_tools(self) -> list:
        if self._tools_cache is not None:
            return self._tools_cache
        resp = self._rpc("tools/list", {})
        if resp and "result" in resp:
            tools = resp["result"].get("tools", [])
            self._tools_cache = tools
            return tools
        return []

    def call_tool(self, tool_name: str, args: dict) -> Any:
        resp = self._rpc("tools/call", {"name": tool_name, "arguments": args})
        if resp is None:
            return {"error": "timeout or process failure"}
        if "error" in resp:
            return {"error": resp["error"]}
        return resp.get("result", {})

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


_mcp = MCPProcess()


class BridgeHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that bridges AURA's MCP HTTP calls to the stdio process."""

    def log_message(self, fmt, *args):
        log_json("INFO", "mcp_swarm_request", details={"fmt": fmt % args if args else fmt})

    def _send_json(self, code: int, data: Any):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "mcp_alive": _mcp.alive()})
        elif self.path == "/tools":
            tools = _mcp.list_tools()
            self._send_json(200, {"tools": tools})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/call":
            body = self._read_body()
            tool_name = body.get("tool_name") or body.get("name", "")
            args = body.get("args") or body.get("arguments") or {}
            if not tool_name:
                self._send_json(400, {"error": "tool_name required"})
                return
            result = _mcp.call_tool(tool_name, args)
            self._send_json(200, {"data": result})
        else:
            self._send_json(404, {"error": "not found"})


def main():
    # Start HTTP server first
    server = HTTPServer(("127.0.0.1", PORT), BridgeHandler)
    log_json("INFO", "mcp_swarm_bridge_listening", details={"port": PORT})

    # Initialize MCP process in background thread
    def _init_mcp():
        log_json("INFO", "mcp_swarm_starting")
        _mcp.start()
        log_json("INFO", "mcp_swarm_ready", details={"tool_count": len(_mcp.list_tools())})

    t = threading.Thread(target=_init_mcp, daemon=True)
    t.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_json("INFO", "mcp_swarm_shutdown")


if __name__ == "__main__":
    main()
