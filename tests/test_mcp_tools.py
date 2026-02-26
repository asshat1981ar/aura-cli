import os
from fastapi.testclient import TestClient
from tools import mcp_server as server


def client(token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return TestClient(server.app), headers


def test_rate_limit_and_health(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    c, hdrs = client("t")
    r = c.get("/health", headers=hdrs)
    assert r.status_code == 200
    data = r.json()
    assert "limits" in data and "metrics" in data


def test_list_tools(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    c, hdrs = client("t")
    r = c.get("/tools", headers=hdrs)
    assert r.status_code == 200
    tools = r.json()["data"]["tools"]
    names = {t["name"] for t in tools}
    expected = {
        "read_file",
        "run_sandboxed",
        "dependency_audit",
        "format",
        "docstring_fill",
        "code_intel_xref",
        "recent_files",
        "env_snapshot",
        "tail_logs",
        "limits",
        "structured_search",
        "secrets_scan",
        "package_scripts",
        "apply_replacements",
        "json_patch",
        "yaml_patch",
        "compress",
        "decompress",
        "git_blame_snippet",
        "git_file_history",
        "semgrep_scan",
        "quick_fix",
        "linter_capabilities",
    }
    for expected in expected:
        assert expected in names


def test_read_file_jail(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    # override project root temporarily
    orig_root = server.PROJECT_ROOT
    server.PROJECT_ROOT = tmp_path
    p = tmp_path / "hello.txt"
    p.write_text("hi")
    c, hdrs = client("t")
    r = c.post("/call", headers=hdrs, json={"tool_name": "read_file", "args": {"path": "hello.txt"}})
    assert r.status_code == 200
    assert "hi" in r.json()["data"]["content"]
    server.PROJECT_ROOT = orig_root
