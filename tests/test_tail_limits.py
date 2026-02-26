from fastapi.testclient import TestClient
from tools import mcp_server as server


def setup_auth(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    c = TestClient(server.app)
    return c, {"Authorization": "Bearer t"}


def test_limits_tool(monkeypatch):
    c, hdrs = setup_auth(monkeypatch)
    r = c.post("/call", headers=hdrs, json={"tool_name": "limits", "args": {}})
    assert r.status_code == 200
    data = r.json()["data"]
    assert "limits" in data
    assert "flags" in data


def test_tail_logs_empty(monkeypatch, tmp_path):
    c, hdrs = setup_auth(monkeypatch)
    log = server.PROJECT_ROOT / "tmp_tail.log"
    log.write_text("line1\nline2\nline3")
    r = c.post("/call", headers=hdrs, json={"tool_name": "tail_logs", "args": {"path": str(log), "lines": 2}})
    assert r.status_code == 200
    lines = r.json()["data"]["lines"]
    assert lines == ["line2", "line3"]
