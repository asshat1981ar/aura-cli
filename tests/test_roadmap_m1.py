import os
import pytest
from core.types import AgentSpec, MCPServerConfig, TaskRequest, TaskResult, ExecutionContext
from core.config_manager import ConfigManager, DEFAULT_CONFIG


def test_mcp_server_config_instantiation():
    config = MCPServerConfig(name="test_server", command="node", args=["server.js"], port=8001)
    assert config.name == "test_server"
    assert config.command == "node"
    assert config.args == ["server.js"]
    assert config.port == 8001
    assert config.enabled is True


def test_agent_spec_instantiation():
    spec = AgentSpec(name="test_agent", description="A test agent", capabilities=["test"], source="mcp", mcp_server="test_server")
    assert spec.name == "test_agent"
    assert spec.source == "mcp"
    assert spec.mcp_server == "test_server"


def test_config_manager_m1_flags():
    # Test that new flags are in effective config
    # Note: Updated for M6 defaults (flipped to True)
    cm = ConfigManager()
    assert cm.get("enable_mcp_registry") is True
    assert cm.get("enable_new_orchestrator") is True
    assert cm.get("force_legacy_orchestrator") is False
    assert cm.get("new_orchestrator_shadow_mode") is False


def test_config_manager_env_overrides(monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_MCP_REGISTRY", "true")
    monkeypatch.setenv("AURA_ENABLE_NEW_ORCHESTRATOR", "1")
    monkeypatch.setenv("AURA_FORCE_LEGACY_ORCHESTRATOR", "yes")
    monkeypatch.setenv("AURA_NEW_ORCHESTRATOR_SHADOW_MODE", "true")

    cm = ConfigManager()
    assert cm.get("enable_mcp_registry") is True
    assert cm.get("enable_new_orchestrator") is True
    assert cm.get("force_legacy_orchestrator") is True
    assert cm.get("new_orchestrator_shadow_mode") is True


def test_task_request_result_contracts():
    ctx = ExecutionContext(project_root="/tmp")
    req = TaskRequest(task_id="t1", agent_name="tester", input_data={"foo": "bar"}, context=ctx)
    assert req.task_id == "t1"
    assert req.context.project_root == "/tmp"

    res = TaskResult(task_id="t1", status="success", output={"result": "ok"})
    assert res.task_id == "t1"
    assert res.status == "success"
