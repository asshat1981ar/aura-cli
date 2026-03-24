import pytest
from core.orchestrator import LoopOrchestrator
from core.config_manager import ConfigManager
from agents.registry import default_agents

def test_orchestrator_canary_routing_mcp_discovery(monkeypatch):
    # Enable canary
    from core.config_manager import config
    monkeypatch.setattr(config, "get", lambda k, default=None: True if k == "enable_new_orchestrator" else config.effective_config.get(k, default))
    
    # We need a brain and model mock
    from unittest.mock import MagicMock
    brain = MagicMock()
    model = MagicMock()
    
    agents = default_agents(brain, model)
    orch = LoopOrchestrator(agents=agents)
    
    # Run mcp_discovery phase
    # This should trigger the canary path in _run_phase
    result = orch._run_phase("mcp_discovery", {"project_root": "."})
    
    assert "status" in result
    assert result["status"] == "success"

def test_orchestrator_canary_routing_mcp_health(monkeypatch):
    # Enable canary
    from core.config_manager import config
    monkeypatch.setattr(config, "get", lambda k, default=None: True if k == "enable_new_orchestrator" else config.effective_config.get(k, default))
    
    from unittest.mock import MagicMock
    brain = MagicMock()
    model = MagicMock()
    
    agents = default_agents(brain, model)
    orch = LoopOrchestrator(agents=agents)
    
    # Run mcp_health phase
    result = orch._run_phase("mcp_health", {})
    
    assert "status" in result
    assert "summary" in result

def test_orchestrator_canary_routing_wave2(monkeypatch):
    # Enable canary
    from core.config_manager import config
    monkeypatch.setattr(config, "get", lambda k, default=None: True if k == "enable_new_orchestrator" else config.effective_config.get(k, default))
    
    from unittest.mock import MagicMock
    brain = MagicMock()
    model = MagicMock()
    
    agents = default_agents(brain, model)
    orch = LoopOrchestrator(agents=agents)
    
    # Run code_search phase
    result = orch._run_phase("code_search", {"query": "test"})
    # Result depends on implementation, but should not raise exception
    assert isinstance(result, dict)
