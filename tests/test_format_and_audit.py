
from fastapi.testclient import TestClient

from tools import mcp_server as server


def setup_client(monkeypatch, enable_run=True):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    if enable_run:
        monkeypatch.setenv("MCP_ENABLE_RUN", "1")
    else:
        monkeypatch.delenv("MCP_ENABLE_RUN", raising=False)
    c = TestClient(server.app)
    return c, {"Authorization": "Bearer t"}


def test_format_uses_ruff(monkeypatch, tmp_path):
    # format now delegates to `ruff format` on jailed paths; cmd arg is ignored
    monkeypatch.setattr(server, "ENABLE_RUN", True)
    monkeypatch.setattr(server, "RUN_ALLOW", {"ruff"})
    c, hdrs = setup_client(monkeypatch)
    # Create a real file inside PROJECT_ROOT so _jail succeeds
    p = server.PROJECT_ROOT / "tmp_fmt_test.py"
    p.write_text("x=1\n")
    try:
        resp = c.post(
            "/call",
            headers=hdrs,
            json={"tool_name": "format", "args": {"paths": [str(p.relative_to(server.PROJECT_ROOT))]}},
        )
    finally:
        p.unlink(missing_ok=True)
    assert resp.status_code == 200
    data = resp.json()["data"]
    # ruff format exits 0 on success (or 127 if not installed)
    assert data["returncode"] in (0, 127)


def test_dependency_audit_fixed_cmd(monkeypatch):
    # dependency_audit no longer accepts a cmd argument; it always runs pip-audit
    monkeypatch.setattr(server, "ENABLE_RUN", True)
    monkeypatch.setattr(server, "RUN_ALLOW", {"pip-audit"})
    c, hdrs = setup_client(monkeypatch)
    resp = c.post(
        "/call",
        headers=hdrs,
        # cmd arg is now ignored
        json={"tool_name": "dependency_audit", "args": {}},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    # pip-audit may not be installed in CI; 127 is acceptable
    assert data["returncode"] in (0, 1, 127)
    assert "report" in data
