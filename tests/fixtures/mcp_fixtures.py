import json
import pytest
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict


class MockMCPServer:
    def __init__(self, port: int):
        self.port = port
        self.thread = None
        self.httpd = None

    def _make_handler(self):
        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload: Dict[str, Any], status: int = 200):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):  # noqa: A003
                return

            def do_GET(self):  # noqa: N802
                if self.path == "/health":
                    self._send_json({"status": "ok", "version": "1.0.0"})
                    return
                if self.path == "/tools":
                    self._send_json({"tools": [{"name": "echo", "description": "echo back"}]})
                    return
                self._send_json({"error": "not found"}, status=404)

            def do_POST(self):  # noqa: N802
                if self.path != "/call":
                    self._send_json({"error": "not found"}, status=404)
                    return
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length) if content_length else b"{}"
                req = json.loads(raw.decode("utf-8"))
                args = req.get("arguments", {})
                self._send_json({"status": "success", "result": f"Echo: {args.get('text', '')}"})

        return Handler

    def start(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._make_handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self._wait_until_ready()

    def _wait_until_ready(self, timeout_s: float = 5.0):
        """Block until the TCP port is accepting connections."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError(f"MockMCPServer on port {self.port} did not become ready")

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=1)


@pytest.fixture
def mock_mcp_server():
    server = MockMCPServer(port=9001)
    server.start()
    try:
        yield server
    finally:
        server.stop()
