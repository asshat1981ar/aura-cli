"""Tests for core/config_manager.py."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from core.config_manager import (
    ConfigManager,
    DEFAULT_CONFIG,
    _validate_positive_int,
    _validate_bool,
    _validate_string,
    _validate_float_range,
    _KEY_VALIDATORS,
)
from core.exceptions import ConfigurationError


class TestValidators:
    """Test configuration validators."""
    
    def test_validate_positive_int_valid(self):
        """Test positive integer validation with valid values."""
        is_valid, value, reason = _validate_positive_int("test", 5)
        assert is_valid is True
        assert value == 5
        assert reason == ""
    
    def test_validate_positive_int_invalid_zero(self):
        """Test positive integer validation with zero."""
        is_valid, value, reason = _validate_positive_int("test", 0)
        assert is_valid is False
        assert "positive integer" in reason
    
    def test_validate_positive_int_invalid_negative(self):
        """Test positive integer validation with negative."""
        is_valid, value, reason = _validate_positive_int("test", -5)
        assert is_valid is False
        assert "positive integer" in reason
    
    def test_validate_positive_int_invalid_string(self):
        """Test positive integer validation with string."""
        is_valid, value, reason = _validate_positive_int("test", "abc")
        assert is_valid is False
        assert "integer" in reason
    
    def test_validate_bool_true(self):
        """Test boolean validation with True."""
        is_valid, value, reason = _validate_bool("test", True)
        assert is_valid is True
        assert value is True
    
    def test_validate_bool_false(self):
        """Test boolean validation with False."""
        is_valid, value, reason = _validate_bool("test", False)
        assert is_valid is True
        assert value is False
    
    def test_validate_bool_string_true(self):
        """Test boolean validation with string 'true'."""
        is_valid, value, reason = _validate_bool("test", "true")
        assert is_valid is True
        assert value is True
    
    def test_validate_bool_string_false(self):
        """Test boolean validation with string 'false'."""
        is_valid, value, reason = _validate_bool("test", "FALSE")
        assert is_valid is True
        assert value is False
    
    def test_validate_bool_string_yes(self):
        """Test boolean validation with string 'yes'."""
        is_valid, value, reason = _validate_bool("test", "yes")
        assert is_valid is True
        assert value is True
    
    def test_validate_bool_string_no(self):
        """Test boolean validation with string 'no'."""
        is_valid, value, reason = _validate_bool("test", "no")
        assert is_valid is True
        assert value is False
    
    def test_validate_bool_invalid(self):
        """Test boolean validation with invalid value."""
        is_valid, value, reason = _validate_bool("test", "maybe")
        assert is_valid is False
        assert "boolean" in reason
    
    def test_validate_string_valid(self):
        """Test string validation with valid string."""
        is_valid, value, reason = _validate_string("test", "hello")
        assert is_valid is True
        assert value == "hello"
    
    def test_validate_string_none(self):
        """Test string validation with None."""
        is_valid, value, reason = _validate_string("test", None)
        assert is_valid is True
        assert value is None
    
    def test_validate_string_invalid(self):
        """Test string validation with invalid value."""
        is_valid, value, reason = _validate_string("test", 123)
        assert is_valid is False
        assert "string" in reason
    
    def test_validate_float_range_valid(self):
        """Test float range validation with valid value."""
        is_valid, value, reason = _validate_float_range("test", 50.0, 0.0, 100.0)
        assert is_valid is True
        assert value == 50.0
    
    def test_validate_float_range_boundary(self):
        """Test float range validation at boundaries."""
        is_valid, value, reason = _validate_float_range("test", 0.0, 0.0, 100.0)
        assert is_valid is True
        
        is_valid, value, reason = _validate_float_range("test", 100.0, 0.0, 100.0)
        assert is_valid is True
    
    def test_validate_float_range_invalid_low(self):
        """Test float range validation below range."""
        is_valid, value, reason = _validate_float_range("test", -10.0, 0.0, 100.0)
        assert is_valid is False
        assert "[0.0, 100.0]" in reason
    
    def test_validate_float_range_invalid_high(self):
        """Test float range validation above range."""
        is_valid, value, reason = _validate_float_range("test", 150.0, 0.0, 100.0)
        assert is_valid is False
        assert "[0.0, 100.0]" in reason
    
    def test_validate_float_range_invalid_string(self):
        """Test float range validation with string."""
        is_valid, value, reason = _validate_float_range("test", "abc", 0.0, 100.0)
        assert is_valid is False
        assert "number" in reason


class TestConfigManagerInit:
    """Test ConfigManager initialization."""
    
    def test_init_finds_config_file(self):
        """Test initialization finds config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"max_iterations": 20}, f)
            f.flush()
            
            cm = ConfigManager(config_file=f.name)
            assert cm.config_file == Path(f.name)
            os.unlink(f.name)
    
    def test_init_with_overrides(self):
        """Test initialization with runtime overrides."""
        cm = ConfigManager(overrides={"dry_run": True})
        assert cm.runtime_overrides == {"dry_run": True}


class TestConfigManagerGet:
    """Test ConfigManager.get method."""
    
    def test_get_existing_key(self):
        """Test getting existing config key."""
        cm = ConfigManager()
        result = cm.get("max_iterations")
        assert result is not None
    
    def test_get_nonexistent_key(self):
        """Test getting non-existent key returns None."""
        cm = ConfigManager()
        assert cm.get("nonexistent") is None
    
    def test_get_with_default(self):
        """Test getting with default value."""
        cm = ConfigManager()
        assert cm.get("nonexistent", "default") == "default"
    
    def test_get_validates_value(self):
        """Test get validates and coerces values."""
        cm = ConfigManager()
        # max_iterations should be validated as positive int
        val = cm.get("max_iterations")
        assert isinstance(val, int)
        assert val > 0


class TestConfigManagerShowConfig:
    """Test ConfigManager.show_config method."""
    
    def test_show_config_returns_dict(self):
        """Test show_config returns effective config."""
        cm = ConfigManager()
        config = cm.show_config()
        assert isinstance(config, dict)
        assert "max_iterations" in config


class TestConfigManagerSetRuntimeOverride:
    """Test ConfigManager.set_runtime_override method."""
    
    def test_set_runtime_override(self):
        """Test setting runtime override."""
        cm = ConfigManager()
        cm.set_runtime_override("test_key", "test_value")
        assert cm.get("test_key") == "test_value"


class TestConfigManagerRefresh:
    """Test ConfigManager.refresh method."""
    
    def test_refresh_reloads_config(self):
        """Test refresh reloads configuration."""
        cm = ConfigManager()
        initial = cm.get("max_iterations")
        cm.set_runtime_override("max_iterations", initial + 10)
        assert cm.get("max_iterations") == initial + 10
        
        # Remove override and refresh
        cm.runtime_overrides = {}
        cm.refresh()
        assert cm.get("max_iterations") == initial


class TestConfigManagerValidateValue:
    """Test ConfigManager._validate_value method."""
    
    def test_validate_value_returns_coerced(self):
        """Test _validate_value returns coerced value."""
        cm = ConfigManager()
        # String "5" should be coerced to int 5
        result = cm._validate_value("max_iterations", "5")
        assert result == 5
        assert isinstance(result, int)
    
    def test_validate_value_returns_default_on_error(self):
        """Test _validate_value returns default on invalid input."""
        cm = ConfigManager()
        default = DEFAULT_CONFIG["max_iterations"]
        # Invalid value should fall back to default
        result = cm._validate_value("max_iterations", -5)
        assert result == default
    
    def test_validate_value_no_validator(self):
        """Test _validate_value returns value when no validator exists."""
        cm = ConfigManager()
        result = cm._validate_value("unknown_key", "value")
        assert result == "value"


class TestConfigManagerLoadFromFile:
    """Test ConfigManager._load_from_file method."""
    
    def test_load_from_valid_file(self):
        """Test loading from valid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"max_iterations": 25}, f)
            f.flush()
            
            cm = ConfigManager(config_file=f.name)
            assert cm.get("max_iterations") == 25
            os.unlink(f.name)
    
    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file returns empty dict."""
        cm = ConfigManager(config_file="/nonexistent/path.json")
        result = cm._load_from_file()
        assert result == {}
    
    def test_load_from_file_invalid_json_raises(self):
        """Test loading from invalid JSON raises exception during init."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json: }")
            f.flush()
            
            # Exception is raised during ConfigManager init (in refresh)
            with pytest.raises(Exception):  # Any exception is fine here
                ConfigManager(config_file=f.name)
            
            os.unlink(f.name)
    
    def test_load_nested_aura_config(self):
        """Test loading config from nested 'aura' key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"aura": {"max_iterations": 30}, "other": "data"}, f)
            f.flush()
            
            cm = ConfigManager(config_file=f.name)
            assert cm.get("max_iterations") == 30
            os.unlink(f.name)


class TestConfigManagerEnvironmentOverrides:
    """Test environment variable loading."""
    
    def test_env_override_api_key(self):
        """Test AURA_API_KEY env override."""
        os.environ["AURA_API_KEY"] = "test-key-123"
        try:
            cm = ConfigManager()
            # Reload to pick up env
            cm.refresh()
            assert cm.get("api_key") == "test-key-123"
        finally:
            del os.environ["AURA_API_KEY"]
    
    def test_env_override_max_iterations(self):
        """Test AURA_MAX_ITERATIONS env override."""
        os.environ["AURA_MAX_ITERATIONS"] = "25"
        try:
            cm = ConfigManager()
            cm.refresh()
            assert cm.get("max_iterations") == 25
        finally:
            del os.environ["AURA_MAX_ITERATIONS"]
    
    def test_env_override_dry_run(self):
        """Test AURA_DRY_RUN env override."""
        os.environ["AURA_DRY_RUN"] = "true"
        try:
            cm = ConfigManager()
            cm.refresh()
            assert cm.get("dry_run") is True
        finally:
            del os.environ["AURA_DRY_RUN"]
    
    def test_env_override_openai_key(self):
        """Test OPENAI_API_KEY env override."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        try:
            cm = ConfigManager()
            cm.refresh()
            assert cm.get("openai_api_key") == "sk-test-key"
        finally:
            del os.environ["OPENAI_API_KEY"]


class TestDefaultConfig:
    """Test DEFAULT_CONFIG constant."""
    
    def test_default_config_has_required_keys(self):
        """Test that DEFAULT_CONFIG has all required keys."""
        required = ["model_name", "max_iterations", "max_cycles", "dry_run"]
        for key in required:
            assert key in DEFAULT_CONFIG, f"Missing required key: {key}"
    
    def test_default_config_values_are_valid(self):
        """Test that all default values pass validation."""
        for key, value in DEFAULT_CONFIG.items():
            if key in _KEY_VALIDATORS:
                validator = _KEY_VALIDATORS[key]
                is_valid, _, reason = validator(key, value)
                assert is_valid, f"Default for {key} is invalid: {reason}"
    
    def test_default_config_has_nested_dicts(self):
        """Test that DEFAULT_CONFIG has expected nested dicts."""
        nested_keys = ["beads", "model_routing", "semantic_memory", "local_model_profiles", "local_model_routing"]
        for key in nested_keys:
            assert key in DEFAULT_CONFIG, f"Missing nested config: {key}"
            assert isinstance(DEFAULT_CONFIG[key], dict), f"{key} should be a dict"

    def test_default_config_has_mcp_servers(self):
        """Test that DEFAULT_CONFIG has mcp_servers registry."""
        assert "mcp_servers" in DEFAULT_CONFIG, "Missing mcp_servers config"
        assert isinstance(DEFAULT_CONFIG["mcp_servers"], dict)
        expected_servers = ["dev_tools", "skills", "control", "agentic_loop", "copilot"]
        for server in expected_servers:
            assert server in DEFAULT_CONFIG["mcp_servers"], f"Missing MCP server: {server}"
            assert isinstance(DEFAULT_CONFIG["mcp_servers"][server], int)


class TestMCPServerPortRegistry:
    """Test MCP server port registry (Roadmap R5)."""
    
    def test_get_mcp_server_port_returns_default(self):
        """Test getting default port for known MCP servers."""
        cm = ConfigManager()
        assert cm.get_mcp_server_port("dev_tools") == 8001
        assert cm.get_mcp_server_port("skills") == 8002
        assert cm.get_mcp_server_port("control") == 8003
        assert cm.get_mcp_server_port("agentic_loop") == 8006
        assert cm.get_mcp_server_port("copilot") == 8007
    
    def test_get_mcp_server_port_unknown_raises(self):
        """Test that unknown server name raises ConfigurationError."""
        cm = ConfigManager()
        with pytest.raises(ConfigurationError) as exc_info:
            cm.get_mcp_server_port("unknown_server")
        assert "Unknown MCP server name" in str(exc_info.value)
    
    def test_get_mcp_server_port_from_config_file(self, tmp_path):
        """Test loading MCP port from config file."""
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps({
            "mcp_servers": {
                "dev_tools": 9001,
                "skills": 9002
            }
        }))
        cm = ConfigManager(config_file=str(config_file))
        assert cm.get_mcp_server_port("dev_tools") == 9001
        assert cm.get_mcp_server_port("skills") == 9002
        # Other servers should use defaults
        assert cm.get_mcp_server_port("control") == 8003
    
    def test_get_mcp_server_port_env_override(self, tmp_path):
        """Test MCP port override via environment variable."""
        os.environ["AURA_MCP_SERVERS_DEV_TOOLS_PORT"] = "9101"
        try:
            cm = ConfigManager()
            cm.refresh()  # Reload to pick up env
            assert cm.get_mcp_server_port("dev_tools") == 9101
        finally:
            del os.environ["AURA_MCP_SERVERS_DEV_TOOLS_PORT"]


class TestPortAvailabilityChecking:
    """Test port availability checking utilities (Roadmap R5)."""
    
    def test_check_port_available_with_available_port(self):
        """Test checking an available port."""
        cm = ConfigManager()
        # Find an available port
        available_port = cm.find_available_port(start_port=18000, end_port=18100)
        assert available_port is not None
        # Should be available
        assert cm.check_port_available(available_port) is True
    
    def test_check_port_available_with_unavailable_port(self):
        """Test checking an unavailable port by temporarily binding it."""
        import socket
        cm = ConfigManager()
        # Find an available port and bind to it (use 0.0.0.0 to block all interfaces)
        port = cm.find_available_port(start_port=18000, end_port=18100)
        assert port is not None
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            # Port should now be unavailable on 0.0.0.0
            assert cm.check_port_available(port, host="0.0.0.0") is False
    
    def test_find_available_port_finds_first_available(self):
        """Test finding available port in range."""
        cm = ConfigManager()
        # Find a port in a high range that's likely to be free
        port = cm.find_available_port(start_port=18200, end_port=18300)
        assert port is not None
        assert 18200 <= port <= 18300
    
    def test_find_available_port_respects_exclusions(self):
        """Test that find_available_port respects exclude_ports."""
        cm = ConfigManager()
        # Exclude the first few ports in range
        exclude = {18200, 18201, 18202}
        port = cm.find_available_port(start_port=18200, end_port=18210, exclude_ports=exclude)
        assert port is not None
        assert port not in exclude


class TestMCPServerPortConflicts:
    """Test MCP server port conflict detection (Roadmap R5)."""
    
    def test_check_mcp_server_port_conflicts_all_available(self):
        """Test conflict detection when all ports are available."""
        cm = ConfigManager()
        # Use a config with high ports that should be available
        cm.effective_config["mcp_servers"] = {
            "dev_tools": 18301,
            "skills": 18302,
        }
        
        results = cm.check_mcp_server_port_conflicts()
        
        assert "dev_tools" in results
        assert "skills" in results
        assert results["dev_tools"]["available"] is True
        assert results["skills"]["available"] is True
        assert results["dev_tools"]["error"] is None
        assert results["skills"]["error"] is None
    
    def test_check_mcp_server_port_conflicts_unknown_server(self):
        """Test conflict detection with unknown server."""
        cm = ConfigManager()
        results = cm.check_mcp_server_port_conflicts(servers=["unknown_server"])
        
        assert "unknown_server" in results
        assert results["unknown_server"]["available"] is False
        assert "Unknown MCP server" in results["unknown_server"]["error"]
    
    def test_check_mcp_server_port_conflicts_with_raise(self):
        """Test that raise_on_conflict raises ConfigurationError."""
        import socket
        cm = ConfigManager()
        
        # Find and bind a port to create a conflict (use 0.0.0.0)
        port = cm.find_available_port(start_port=18400, end_port=18500)
        assert port is not None
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            
            cm.effective_config["mcp_servers"] = {"test_server": port}
            
            with pytest.raises(ConfigurationError) as exc_info:
                cm.check_mcp_server_port_conflicts(servers=["test_server"], raise_on_conflict=True)
            
            assert "port conflicts" in str(exc_info.value).lower()
    
    def test_suggest_mcp_port_reassignments_empty_when_no_conflicts(self):
        """Test that suggestions are empty when no conflicts."""
        cm = ConfigManager()
        # Use high ports that should be available
        cm.effective_config["mcp_servers"] = {
            "dev_tools": 18501,
            "skills": 18502,
        }
        
        suggestions = cm.suggest_mcp_port_reassignments()
        
        # Should be empty since no conflicts
        assert suggestions == {}


class TestMCPPortEnvOverrides:
    """Test environment variable overrides for MCP ports."""
    
    def test_env_override_mcp_servers_port(self):
        """Test AURA_MCP_SERVERS_*_PORT env override."""
        os.environ["AURA_MCP_SERVERS_SKILLS_PORT"] = "9202"
        os.environ["AURA_MCP_SERVERS_CONTROL_PORT"] = "9203"
        try:
            cm = ConfigManager()
            cm.refresh()
            assert cm.get_mcp_server_port("skills") == 9202
            assert cm.get_mcp_server_port("control") == 9203
        finally:
            del os.environ["AURA_MCP_SERVERS_SKILLS_PORT"]
            del os.environ["AURA_MCP_SERVERS_CONTROL_PORT"]
    
    def test_env_override_mcp_port_invalid_ignored(self):
        """Test that invalid MCP port env values are ignored."""
        os.environ["AURA_MCP_SERVERS_DEV_TOOLS_PORT"] = "not_a_number"
        try:
            cm = ConfigManager()
            cm.refresh()
            # Should fall back to default
            assert cm.get_mcp_server_port("dev_tools") == 8001
        finally:
            del os.environ["AURA_MCP_SERVERS_DEV_TOOLS_PORT"]
