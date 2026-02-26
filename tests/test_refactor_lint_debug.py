from pathlib import Path

from fastapi.testclient import TestClient

from tools import mcp_server as server


def setup_client(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    return TestClient(server.app), {"Authorization": "Bearer t"}


def test_refactor_plan(monkeypatch):
    c, hdrs = setup_client(monkeypatch)
    p = server.PROJECT_ROOT / "tmp_refactor.py"
    p.write_text("def foo():\n    return 1\nclass Bar:\n    pass\n")
    r = c.post("/call", headers=hdrs, json={"tool_name": "refactor_plan", "args": {"path": str(p.relative_to(server.PROJECT_ROOT))}})
    p.unlink(missing_ok=True)
    assert r.status_code == 200
    items = r.json()["data"]["items"]
    assert any("foo" in i for i in items)


def test_lint_files(monkeypatch):
    c, hdrs = setup_client(monkeypatch)
    p = server.PROJECT_ROOT / "tmp_lint.py"
    p.write_text("x=1\n")
    # allow ruff call
    monkeypatch.setattr(server, "RUN_ALLOW", {"ruff", "python", "python3", "echo", "npx", "prettier"})
    monkeypatch.setattr(server, "ENABLE_RUN", True)
    r = c.post("/call", headers=hdrs, json={"tool_name": "lint_files", "args": {"paths": [str(p.relative_to(server.PROJECT_ROOT))]}})
    p.unlink(missing_ok=True)
    assert r.status_code == 200
    assert r.json()["data"]["returncode"] in (0, 127)


def test_debug_trace(monkeypatch):
    c, hdrs = setup_client(monkeypatch)
    mod = "site"  # safe stdlib module
    monkeypatch.setattr(server, "RUN_ALLOW", {"python", "python3"})
    monkeypatch.setattr(server, "ENABLE_RUN", True)
    r = c.post("/call", headers=hdrs, json={"tool_name": "debug_trace", "args": {"module": mod}})
    assert r.status_code == 200
