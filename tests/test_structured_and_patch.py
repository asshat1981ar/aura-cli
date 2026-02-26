import json
from pathlib import Path

from fastapi.testclient import TestClient

from tools import mcp_server as server


def setup_client(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    return TestClient(server.app), {"Authorization": "Bearer t"}


def test_structured_search(monkeypatch):
    c, hdrs = setup_client(monkeypatch)
    tmp = server.PROJECT_ROOT / "tmp_struct.txt"
    tmp.write_text("hello TODO world\n")
    r = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "structured_search", "args": {"query": "TODO", "paths": [str(tmp.relative_to(server.PROJECT_ROOT))]}},
    )
    tmp.unlink(missing_ok=True)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["count"] >= 1
    assert data["results"][0]["path"].endswith("tmp_struct.txt")


def test_json_patch(monkeypatch):
    monkeypatch.setattr(server, "ENABLE_WRITE", True)
    c, hdrs = setup_client(monkeypatch)
    tmp = server.PROJECT_ROOT / "tmp_json_patch.json"
    tmp.write_text(json.dumps({"foo": {"bar": 1}}))
    r = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "json_patch", "args": {"path": str(tmp.relative_to(server.PROJECT_ROOT)), "pointer": "/foo/bar", "value": 2}},
    )
    assert r.status_code == 200
    loaded = json.loads(tmp.read_text())
    tmp.unlink(missing_ok=True)
    assert loaded["foo"]["bar"] == 2


def test_secrets_scan(monkeypatch):
    c, hdrs = setup_client(monkeypatch)
    tmp = server.PROJECT_ROOT / "tmp_secret.txt"
    tmp.write_text("sk-12345678901234567890123456789012")
    r = c.post(
        "/call",
        headers=hdrs,
        json={"tool_name": "secrets_scan", "args": {"paths": [str(tmp.relative_to(server.PROJECT_ROOT))]}},
    )
    tmp.unlink(missing_ok=True)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["count"] >= 1
