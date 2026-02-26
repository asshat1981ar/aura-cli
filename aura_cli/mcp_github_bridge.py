"""
HTTP bridge: exposes GitHub MCP server (stdio/JSON-RPC) as HTTP at localhost:8001.

AURA's model_adapter expects:
  GET  /tools          → list available tools
  POST /call           → {"tool_name": ..., "args": {...}}
  GET  /health         → {"status": "ok"}

This bridge spawns `npx @modelcontextprotocol/server-github` as a subprocess,
speaks JSON-RPC 2.0 over its stdin/stdout, and wraps responses as HTTP JSON.

Usage:
  GITHUB_PERSONAL_ACCESS_TOKEN=<token> python -m aura_cli.mcp_github_bridge
  or via: ./scripts/start_mcp_github.sh
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")

logging.basicConfig(level=logging.INFO, format="[mcp-bridge] %(levelname)s %(message)s")
log = logging.getLogger("mcp_bridge")


class MCPProcess:
    """Manages the npx GitHub MCP server subprocess and JSON-RPC communication."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._req_id = 0
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, Any] = {}
        self._tools_cache: list | None = None

    def start(self):
        env = {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN}
        self._proc = subprocess.Popen(
            ["npx", "-y", "@modelcontextprotocol/server-github"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        # Start reader thread
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        # Initialize MCP session
        self._initialize()
        log.info("GitHub MCP process started (pid=%s)", self._proc.pid)

    def _initialize(self):
        """Send MCP initialize handshake."""
        resp = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "aura-mcp-bridge", "version": "1.0"},
        })
        if resp:
            self._rpc_notify("notifications/initialized", {})
            log.info("MCP initialized: %s", resp.get("result", {}).get("serverInfo", {}))

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
            log.error("MCP process pipe broken")
            return None

        if event.wait(timeout):
            result = self._results.pop(req_id, None)
            self._pending.pop(req_id, None)
            return result
        else:
            self._pending.pop(req_id, None)
            log.warning("RPC timeout for method=%s id=%s", method, req_id)
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
        content = resp.get("result", {}).get("content", [])
        # Flatten text content blocks
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return {"result": "\n".join(texts) if texts else resp.get("result", {})}

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


_mcp = MCPProcess()


class BridgeHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that bridges AURA's MCP HTTP calls to the stdio process."""

    def log_message(self, fmt, *args):
        log.info(fmt, *args)

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
        elif self.path == "/metrics":
            self._send_json(200, {"uptime": "ok"})
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
            self._send_json(200, result)
        else:
            self._send_json(404, {"error": "not found"})


def main():
    if not GITHUB_TOKEN:
        log.error("GITHUB_PERSONAL_ACCESS_TOKEN not set. Export it before starting.")
        sys.exit(1)

    # Start HTTP server first (so health checks work immediately)
    server = HTTPServer(("127.0.0.1", PORT), BridgeHandler)
    log.info("HTTP bridge listening on http://localhost:%s", PORT)

    # Initialize MCP process in background thread
    def _init_mcp():
        log.info("Starting GitHub MCP process...")
        _mcp.start()
        log.info("GitHub MCP ready — %d tools available", len(_mcp.list_tools()))

    t = threading.Thread(target=_init_mcp, daemon=True)
    t.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
