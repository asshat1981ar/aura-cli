import contextlib
import importlib.util
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_android_local_models.py"
SPEC = importlib.util.spec_from_file_location("check_android_local_models", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in {"/health", "/v1/models", "/"}:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_server(port: int):
    server = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_check_model_endpoint_succeeds_for_live_server():
    port = _free_port()
    server = _start_server(port)
    try:
        ok, message = MODULE.check_model_endpoint("127.0.0.1", port, 1.0)
    finally:
        server.shutdown()
        server.server_close()

    assert ok is True
    assert "/health" in message


def test_check_model_endpoint_fails_for_closed_port():
    port = _free_port()
    ok, message = MODULE.check_model_endpoint("127.0.0.1", port, 0.2)

    assert ok is False
    assert "unreachable" in message


def test_wait_for_optional_embedding_endpoint_succeeds():
    coder_port = _free_port()
    planner_port = _free_port()
    embedding_port = _free_port()
    coder = _start_server(coder_port)
    planner = _start_server(planner_port)
    embedding = _start_server(embedding_port)
    try:
        for name, port in [("coder", coder_port), ("planner", planner_port), ("embedding", embedding_port)]:
            ok, message = MODULE.wait_for_model_endpoint(name, "127.0.0.1", port, 1.0)
            assert ok is True
            assert name in message
    finally:
        coder.shutdown()
        coder.server_close()
        planner.shutdown()
        planner.server_close()
        embedding.shutdown()
        embedding.server_close()
