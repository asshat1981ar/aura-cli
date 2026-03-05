"""Comprehensive tests for AURA Agent API (aura_cli/server.py)."""
import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from aura_cli.server import app


@pytest.fixture(autouse=True)
def set_api_env(monkeypatch):
    """Ensure AGENT_API_TOKEN and AGENT_API_ENABLE_RUN are set for every test
    in this module and restored afterwards."""
    monkeypatch.setenv("AGENT_API_TOKEN", "test-token")
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")


@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint_auth_failure(client):
    # No auth header
    response = client.get("/health")
    assert response.status_code == 401
    
    # Invalid token
    response = client.get("/health", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 403

def test_health_endpoint_success(client):
    response = client.get("/health", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "providers" in data
    assert "run_enabled" in data

def test_metrics_endpoint_success(client):
    response = client.get("/metrics", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "skill_metrics" in data

def test_tools_endpoint_success(client):
    response = client.get("/tools", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["tools"], list)

def test_execute_ask_success(client):
    with patch("aura_cli.server.model_adapter.respond", return_value="Test answer"):
        response = client.post(
            "/execute",
            headers={"Authorization": "Bearer test-token"},
            json={"tool_name": "ask", "args": ["What is 2+2?"]}
        )
        assert response.status_code == 200
        assert response.json()["data"] == "Test answer"

def test_execute_run_streaming(client):
    # Mocking create_subprocess_exec is complex, but we can verify the response type
    # and that it returns an SSE stream.
    response = client.post(
        "/execute",
        headers={"Authorization": "Bearer test-token"},
        json={"tool_name": "run", "args": ["ls"]}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    # Check some content
    lines = [line for line in response.iter_lines()]
    assert any("type" in line for line in lines)
    assert any("exit" in line for line in lines)

def test_execute_goal_streaming(client):
    # Mock orchestrator.run_cycle
    mock_entry = {
        "cycle_id": "c1",
        "stop_reason": "test_stop",
        "phase_outputs": {"verification": {"status": "pass"}}
    }
    with patch("aura_cli.server.orchestrator.run_cycle", return_value=mock_entry):
        response = client.post(
            "/execute",
            headers={"Authorization": "Bearer test-token"},
            json={"tool_name": "goal", "args": ["Fix the bug"]}
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        lines = [line for line in response.iter_lines()]
        assert any("start" in line for line in lines)
        assert any("cycle" in line for line in lines)
        assert any("complete" in line for line in lines)

def test_execute_unknown_tool(client):
    response = client.post(
        "/execute",
        headers={"Authorization": "Bearer test-token"},
        json={"tool_name": "invalid", "args": []}
    )
    assert response.status_code == 404

def test_execute_run_disabled(client, monkeypatch):
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "0")
    response = client.post(
        "/execute",
        headers={"Authorization": "Bearer test-token"},
        json={"tool_name": "run", "args": ["ls"]}
    )
    assert response.status_code == 403
