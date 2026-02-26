import json
import os
import tempfile
from pathlib import Path

import pytest
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
        json={"tool_name": "dependency_audit", "args": {"cmd": ["python", "-c", "print('audit-ok')"]}},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["returncode"] == 0
    assert "audit-ok" in str(data["report"])
