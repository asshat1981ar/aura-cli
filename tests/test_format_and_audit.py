import asyncio

from tools import mcp_server as server
from tools.mcp_server import CallRequest


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _DirectClient:
    def post(self, path, *, headers=None, json=None):
        if path != "/call":
            raise AssertionError(f"Unhandled path in test harness: {path}")
        try:
            payload = asyncio.run(
                server.call_tool(
                    CallRequest(tool_name=json["tool_name"], args=json.get("args", {})),
                    headers.get("Authorization", "").split(" ", 1)[1] if headers else "anon",
                )
            )
            return _Response(200, payload)
        except Exception as exc:
            status = getattr(exc, "status_code", 500)
            detail = getattr(exc, "detail", str(exc))
            return _Response(status, {"detail": detail})


def setup_client(monkeypatch, enable_run=True):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    if enable_run:
        monkeypatch.setenv("MCP_ENABLE_RUN", "1")
    else:
        monkeypatch.delenv("MCP_ENABLE_RUN", raising=False)
    return _DirectClient(), {"Authorization": "Bearer t"}


def test_format_echo(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_MAX_READ_BYTES", "20000")
    server.ENABLE_RUN = True
    server.RUN_ALLOW = {"echo", "python", "python3", "ruff", "pip", "npm", "npx", "pip-audit"}
    # use echo as formatter to avoid tool deps
    c, hdrs = setup_client(monkeypatch)
    resp = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "format", "args": {"cmd": "echo ok", "mode": "check"}},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["returncode"] == 0
    assert "ok" in data["stdout"]


def test_dependency_audit_pip(monkeypatch, tmp_path):
    # Offline-safe audit using python -c
    monkeypatch.setattr(server, "ENABLE_RUN", True)
    monkeypatch.setattr(server, "RUN_ALLOW", {"pip-audit", "python", "python3", "pip"})
    c, hdrs = setup_client(monkeypatch)
    resp = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "dependency_audit", "args": {"cmd": ["python3", "-c", "print('audit-ok')"]}},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["returncode"] == 0
    assert "audit-ok" in str(data["report"])
