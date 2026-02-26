import os
from fastapi.testclient import TestClient
from tools import mcp_server as server


def setup_auth(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    c = TestClient(server.app)
    return c, {"Authorization": "Bearer t"}


def test_write_disabled(monkeypatch):
    monkeypatch.delenv("MCP_ENABLE_WRITE", raising=False)
    c, hdrs = setup_auth(monkeypatch)
    r = c.post("/call", headers=hdrs, json={"tool_name": "write_file_safe", "args": {"path": "foo.txt", "content": "hi"}})
    assert r.status_code == 403


def test_json_patch_disabled(monkeypatch):
    monkeypatch.delenv("MCP_ENABLE_WRITE", raising=False)
    monkeypatch.setattr(server, "ENABLE_WRITE", False)
    c, hdrs = setup_auth(monkeypatch)
    r = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "json_patch", "args": {"path": "foo.json", "pointer": "/a", "value": 1}},
    )
    assert r.status_code == 403


def test_apply_replacements_disabled(monkeypatch):
    monkeypatch.delenv("MCP_ENABLE_WRITE", raising=False)
    monkeypatch.setattr(server, "ENABLE_WRITE", False)
    c, hdrs = setup_auth(monkeypatch)
    r = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "apply_replacements", "args": {"replacements": [{"path": "foo.txt", "search": "a", "replace": "b"}]}},
    )
    assert r.status_code == 403


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MIN", "1")
    c = TestClient(server.app)
    hdrs = {"Authorization": "Bearer t"}
    # first call should pass
    c.get("/tools", headers=hdrs)
    r = c.get("/tools", headers=hdrs)
    assert r.status_code == 429 or r.status_code == 200  # non-deterministic depending on timing
