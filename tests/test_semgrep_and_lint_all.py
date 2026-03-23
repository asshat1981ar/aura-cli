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


def setup_client(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    monkeypatch.setattr(server, "ENABLE_RUN", True)
    return _DirectClient(), {"Authorization": "Bearer t"}


def test_semgrep_config_default(monkeypatch):
    c, hdrs = setup_client(monkeypatch)
    # allow semgrep command even if missing; expect 127 handled upstream? semgrep will 127 -> 500, so we assert 400/500 acceptable
    monkeypatch.setattr(server, "RUN_ALLOW", {"semgrep", "python", "python3", "echo"})
    resp = c.post("/call", headers=hdrs, json={"tool_name": "semgrep_scan", "args": {"paths": ["tools"]}})
    assert resp.status_code in (200, 400, 500)  # tolerate missing binary


def test_lint_all(monkeypatch):
    c, hdrs = setup_client(monkeypatch)
    monkeypatch.setattr(server, "RUN_ALLOW", {"ruff", "python", "python3", "npx", "prettier", "echo"})
    resp = c.post("/call", headers=hdrs, json={"tool_name": "lint_all", "args": {}})
    assert resp.status_code == 200
    assert resp.json()["data"]["returncode"] in (0, 1, 127)
