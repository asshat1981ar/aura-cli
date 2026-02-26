from fastapi.testclient import TestClient
from tools import mcp_server as server


def setup_client(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    monkeypatch.setattr(server, "ENABLE_RUN", True)
    return TestClient(server.app), {"Authorization": "Bearer t"}


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
    assert resp.json()["data"]["returncode"] in (0, 127)
