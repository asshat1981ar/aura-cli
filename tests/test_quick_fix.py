from fastapi.testclient import TestClient
from tools import mcp_server as server


def setup_client(monkeypatch):
    monkeypatch.setenv("MCP_API_TOKEN", "t")
    monkeypatch.setattr(server, "ENABLE_RUN", True)
    return TestClient(server.app), {"Authorization": "Bearer t"}


def test_quick_fix_noop(monkeypatch, tmp_path):
    c, hdrs = setup_client(monkeypatch)
    p = server.PROJECT_ROOT / "tmp_fix.py"
    p.write_text("x=1\n")
    monkeypatch.setattr(server, "RUN_ALLOW", {"ruff", "python", "python3", "npx", "prettier", "echo"})
    r = c.post("/call", headers=hdrs, json={"tool_name": "quick_fix", "args": {"paths": [str(p.relative_to(server.PROJECT_ROOT))]}})
    p.unlink(missing_ok=True)
    assert r.status_code == 200
    # Allow ruff missing (returncode 127) or success (0)
    assert r.json()["data"]["returncode"] in (0, 127)
