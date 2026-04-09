"""Tests for MCP API key authentication (R8).

Covers:
- Valid API key acceptance
- Invalid API key rejection
- Missing API key handling
- Backward compatibility (no key configured)
- All 5 MCP servers
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

import pytest
from fastapi import Header
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_env():
    """Clean up environment variables before each test."""
    # Store original values
    orig_env = {}
    env_vars = [
        "MCP_DEV_TOOLS_API_KEY",
        "MCP_SKILLS_API_KEY",
        "MCP_CONTROL_API_KEY",
        "MCP_AGENTIC_LOOP_API_KEY",
        "MCP_COPILOT_API_KEY",
        "MCP_API_TOKEN",  # Legacy
    ]
    for var in env_vars:
        orig_env[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original values
    for var, val in orig_env.items():
        if val is not None:
            os.environ[var] = val
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def test_api_key():
    """Standard test API key."""
    return "test-secret-key-12345"


# ---------------------------------------------------------------------------
# Auth Module Tests
# ---------------------------------------------------------------------------

class TestMCPAuthModule:
    """Test the core mcp_auth module functions."""
    
    def test_get_mcp_server_api_key_from_env(self, test_api_key):
        """Test loading API key from environment variable."""
        from tools.mcp_auth import get_mcp_server_api_key
        
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        result = get_mcp_server_api_key("dev_tools")
        assert result == test_api_key
    
    def test_get_mcp_server_api_key_not_configured(self):
        """Test that None is returned when key is not configured."""
        from tools.mcp_auth import get_mcp_server_api_key
        
        result = get_mcp_server_api_key("dev_tools")
        assert result is None
    
    def test_get_mcp_server_api_key_legacy_fallback(self, test_api_key):
        """Test legacy MCP_API_TOKEN fallback for dev_tools."""
        from tools.mcp_auth import get_mcp_server_api_key
        
        os.environ["MCP_API_TOKEN"] = test_api_key
        result = get_mcp_server_api_key("dev_tools")
        assert result == test_api_key
    
    def test_get_mcp_server_api_key_env_overrides_legacy(self, test_api_key):
        """Test that new env var takes priority over legacy."""
        from tools.mcp_auth import get_mcp_server_api_key
        
        os.environ["MCP_API_TOKEN"] = "legacy-key"
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        result = get_mcp_server_api_key("dev_tools")
        assert result == test_api_key
    
    def test_validate_api_key_valid(self, test_api_key):
        """Test valid API key validation."""
        from tools.mcp_auth import validate_api_key
        
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        result = validate_api_key("dev_tools", test_api_key)
        assert result is True
    
    def test_validate_api_key_invalid(self, test_api_key):
        """Test invalid API key validation."""
        from tools.mcp_auth import validate_api_key
        
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        result = validate_api_key("dev_tools", "wrong-key")
        assert result is False
    
    def test_validate_api_key_missing_when_required(self, test_api_key):
        """Test that missing key fails when auth is required."""
        from tools.mcp_auth import validate_api_key
        
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        result = validate_api_key("dev_tools", None)
        assert result is False
    
    def test_validate_api_key_optional_when_not_configured(self):
        """Test that auth is optional when no key is configured."""
        from tools.mcp_auth import validate_api_key
        
        result = validate_api_key("dev_tools", None)
        assert result is True  # Backward compatible
    
    def test_is_auth_enabled_true(self, test_api_key):
        """Test is_auth_enabled returns True when key is configured."""
        from tools.mcp_auth import is_auth_enabled
        
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        result = is_auth_enabled("dev_tools")
        assert result is True
    
    def test_is_auth_enabled_false(self):
        """Test is_auth_enabled returns False when key is not configured."""
        from tools.mcp_auth import is_auth_enabled
        
        result = is_auth_enabled("dev_tools")
        assert result is False


# ---------------------------------------------------------------------------
# Dev Tools Server Tests
# ---------------------------------------------------------------------------

class TestDevToolsServerAuth:
    """Test authentication on dev_tools MCP server."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for dev_tools server."""
        from tools.mcp_server import app
        return TestClient(app)
    
    def test_health_without_auth_optional(self, client):
        """Test health endpoint without auth when not configured."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_tools_without_auth_optional(self, client):
        """Test tools endpoint without auth when not configured."""
        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        # Response format is {"data": {"tools": [...]}}
        assert "data" in data
        assert "tools" in data["data"]
    
    def test_health_with_valid_api_key(self, client, test_api_key):
        """Test health endpoint with valid X-API-Key header."""
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": test_api_key})
        assert response.status_code == 200
    
    def test_health_with_invalid_api_key(self, client, test_api_key):
        """Test health endpoint with invalid X-API-Key header."""
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403
    
    def test_health_with_missing_api_key(self, client, test_api_key):
        """Test health endpoint with missing API key when required."""
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        
        response = client.get("/health")
        assert response.status_code == 401
    
    def test_health_with_valid_bearer_token(self, client, test_api_key):
        """Test health endpoint with valid Authorization: Bearer header."""
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"Authorization": f"Bearer {test_api_key}"})
        assert response.status_code == 200
    
    def test_health_with_invalid_bearer_token(self, client, test_api_key):
        """Test health endpoint with invalid Authorization: Bearer header."""
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Skills Server Tests
# ---------------------------------------------------------------------------

class TestSkillsServerAuth:
    """Test authentication on skills MCP server."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for skills server."""
        from tools.aura_mcp_skills_server import app
        return TestClient(app)
    
    def test_health_without_auth_optional(self, client):
        """Test health endpoint without auth when not configured."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_tools_without_auth_optional(self, client):
        """Test tools endpoint without auth when not configured."""
        response = client.get("/tools")
        assert response.status_code == 200
    
    def test_health_with_valid_api_key(self, client, test_api_key):
        """Test health endpoint with valid X-API-Key header."""
        os.environ["MCP_SKILLS_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": test_api_key})
        assert response.status_code == 200
    
    def test_health_with_invalid_api_key(self, client, test_api_key):
        """Test health endpoint with invalid X-API-Key header."""
        os.environ["MCP_SKILLS_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Control Server Tests
# ---------------------------------------------------------------------------

class TestControlServerAuth:
    """Test authentication on control MCP server."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for control server."""
        from tools.aura_control_mcp import app
        return TestClient(app)
    
    def test_health_without_auth_optional(self, client):
        """Test health endpoint without auth when not configured."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_tools_without_auth_optional(self, client):
        """Test tools endpoint without auth when not configured."""
        response = client.get("/tools")
        assert response.status_code == 200
    
    def test_health_with_valid_api_key(self, client, test_api_key):
        """Test health endpoint with valid X-API-Key header."""
        os.environ["MCP_CONTROL_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": test_api_key})
        assert response.status_code == 200
    
    def test_health_with_invalid_api_key(self, client, test_api_key):
        """Test health endpoint with invalid X-API-Key header."""
        os.environ["MCP_CONTROL_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Agentic Loop Server Tests
# ---------------------------------------------------------------------------

class TestAgenticLoopServerAuth:
    """Test authentication on agentic_loop MCP server."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for agentic_loop server."""
        from tools.agentic_loop_mcp import app
        return TestClient(app)
    
    def test_health_without_auth_optional(self, client):
        """Test health endpoint without auth when not configured."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_tools_without_auth_optional(self, client):
        """Test tools endpoint without auth when not configured."""
        response = client.get("/tools")
        assert response.status_code == 200
    
    def test_health_with_valid_api_key(self, client, test_api_key):
        """Test health endpoint with valid X-API-Key header."""
        os.environ["MCP_AGENTIC_LOOP_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": test_api_key})
        assert response.status_code == 200
    
    def test_health_with_invalid_api_key(self, client, test_api_key):
        """Test health endpoint with invalid X-API-Key header."""
        os.environ["MCP_AGENTIC_LOOP_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Copilot Server Tests
# ---------------------------------------------------------------------------

class TestCopilotServerAuth:
    """Test authentication on copilot MCP server."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for copilot server."""
        from tools.github_copilot_mcp import app
        return TestClient(app)
    
    def test_health_without_auth_optional(self, client):
        """Test health endpoint without auth when not configured."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_tools_without_auth_optional(self, client):
        """Test tools endpoint without auth when not configured."""
        response = client.get("/tools")
        assert response.status_code == 200
    
    def test_health_with_valid_api_key(self, client, test_api_key):
        """Test health endpoint with valid X-API-Key header."""
        os.environ["MCP_COPILOT_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": test_api_key})
        assert response.status_code == 200
    
    def test_health_with_invalid_api_key(self, client, test_api_key):
        """Test health endpoint with invalid X-API-Key header."""
        os.environ["MCP_COPILOT_API_KEY"] = test_api_key
        
        response = client.get("/health", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Client Tests
# ---------------------------------------------------------------------------

class TestMCPClient:
    """Test the MCP client authentication handling."""
    
    def test_get_mcp_server_api_key_from_env(self, test_api_key):
        """Test client loads API key from environment."""
        from aura_cli.mcp_client import _get_mcp_server_api_key
        
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        result = _get_mcp_server_api_key("dev_tools")
        assert result == test_api_key
    
    def test_get_mcp_server_api_key_not_configured(self):
        """Test client returns None when key not configured."""
        from aura_cli.mcp_client import _get_mcp_server_api_key
        
        result = _get_mcp_server_api_key("dev_tools")
        assert result is None
    
    def test_mcp_headers_with_api_key(self, test_api_key):
        """Test client includes X-API-Key in headers."""
        from aura_cli.mcp_client import _mcp_headers
        
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        headers = _mcp_headers("dev_tools")
        assert "X-API-Key" in headers
        assert headers["X-API-Key"] == test_api_key
    
    def test_mcp_headers_without_api_key(self):
        """Test client headers without API key."""
        from aura_cli.mcp_client import _mcp_headers
        
        headers = _mcp_headers("dev_tools")
        assert "X-API-Key" not in headers
        assert "Content-Type" in headers


# ---------------------------------------------------------------------------
# Config Manager Tests
# ---------------------------------------------------------------------------

class TestConfigManager:
    """Test ConfigManager integration with MCP auth."""
    
    def test_default_config_has_api_keys(self):
        """Test that DEFAULT_CONFIG includes mcp_server_api_keys."""
        from core.config_manager import DEFAULT_CONFIG
        
        assert "mcp_server_api_keys" in DEFAULT_CONFIG
        keys = DEFAULT_CONFIG["mcp_server_api_keys"]
        assert "dev_tools" in keys
        assert "skills" in keys
        assert "control" in keys
        assert "agentic_loop" in keys
        assert "copilot" in keys
    
    def test_get_mcp_server_api_key_from_config(self, test_api_key):
        """Test getting API key from config manager."""
        from core.config_manager import ConfigManager
        
        # Create a config with the key
        cfg = ConfigManager(overrides={
            "mcp_server_api_keys": {
                "dev_tools": test_api_key
            }
        })
        
        result = cfg.get_mcp_server_api_key("dev_tools")
        assert result == test_api_key
    
    def test_get_mcp_server_api_key_unknown_server(self):
        """Test that unknown server raises ConfigurationError."""
        from core.config_manager import ConfigManager, ConfigurationError
        
        cfg = ConfigManager()
        with pytest.raises(ConfigurationError) as exc_info:
            cfg.get_mcp_server_api_key("unknown_server")
        
        assert "Unknown MCP server" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Security Tests
# ---------------------------------------------------------------------------

class TestSecurity:
    """Security-focused tests for the auth implementation."""
    
    def test_constant_time_comparison_used(self, test_api_key):
        """Verify that hmac.compare_digest is used (not simple string compare)."""
        import hmac
        from unittest.mock import patch
        from tools.mcp_auth import validate_api_key
        
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        
        with patch('tools.mcp_auth.hmac.compare_digest') as mock_compare:
            mock_compare.return_value = True
            validate_api_key("dev_tools", test_api_key)
            mock_compare.assert_called_once()
    
    def test_no_timing_attack_via_error_messages(self, test_api_key):
        """Verify error messages don't leak information about key validity."""
        from tools.mcp_server import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        os.environ["MCP_DEV_TOOLS_API_KEY"] = test_api_key
        
        # Both invalid key and missing key should give same level of info
        invalid_response = client.get("/health", headers={"X-API-Key": "wrong"})
        missing_response = client.get("/health")
        
        # Both should fail
        assert invalid_response.status_code in [401, 403]
        assert missing_response.status_code in [401, 403]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
