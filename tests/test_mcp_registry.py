"""Unit tests for core/mcp_registry.py."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from core.mcp_registry import (
    MCPServiceSpec,
    describe_service,
    get_registered_service,
    iter_service_specs,
    list_registered_services,
)


class TestMCPServiceSpec:
    def test_frozen_dataclass(self):
        spec = MCPServiceSpec(
            config_name="test",
            server_name="aura-test",
            title="Test",
            kind="tooling",
            auth_env="TEST_TOKEN",
            port_envs=("TEST_PORT",),
            endpoints=("/health",),
            capabilities=("test-cap",),
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.config_name = "other"  # type: ignore[misc]

    def test_fields_accessible(self):
        spec = MCPServiceSpec(
            config_name="skills",
            server_name="aura-skills",
            title="Skills",
            kind="skills",
            auth_env="MCP_API_TOKEN",
            port_envs=("MCP_SKILLS_PORT",),
            endpoints=("/health", "/tools"),
            capabilities=("skill-execution",),
        )
        assert spec.config_name == "skills"
        assert spec.kind == "skills"


class TestDescribeService:
    def _dev_tools_spec(self) -> MCPServiceSpec:
        from core.mcp_registry import _SERVICE_SPECS
        return next(s for s in _SERVICE_SPECS if s.config_name == "dev_tools")

    @patch("core.mcp_registry._config")
    def test_describe_returns_dict_with_expected_keys(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 8001
        spec = self._dev_tools_spec()
        result = describe_service(spec, host="127.0.0.1")
        for key in ("name", "title", "kind", "port", "url", "endpoints", "capabilities"):
            assert key in result

    @patch("core.mcp_registry._config")
    def test_url_contains_host_and_port(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 9999
        spec = self._dev_tools_spec()
        result = describe_service(spec, host="10.0.0.1")
        assert "10.0.0.1" in result["url"]
        assert "9999" in result["url"]

    @patch("core.mcp_registry._config")
    def test_auth_configured_false_when_env_unset(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 8001
        spec = self._dev_tools_spec()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(spec.auth_env, None)
            result = describe_service(spec, host="127.0.0.1")
        assert result["auth_configured"] is False

    @patch("core.mcp_registry._config")
    def test_auth_configured_true_when_env_set(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 8001
        spec = self._dev_tools_spec()
        with patch.dict(os.environ, {spec.auth_env: "secret"}):
            result = describe_service(spec, host="127.0.0.1")
        assert result["auth_configured"] is True

    @patch("core.mcp_registry._config")
    def test_port_env_overrides_config(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 8001
        spec = self._dev_tools_spec()
        with patch.dict(os.environ, {"PORT": "9876"}):
            result = describe_service(spec, host="127.0.0.1")
        assert result["port"] == 9876


class TestListRegisteredServices:
    @patch("core.mcp_registry._config")
    def test_returns_list(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 8001
        services = list_registered_services(host="127.0.0.1")
        assert isinstance(services, list)
        assert len(services) >= 1

    @patch("core.mcp_registry._config")
    def test_all_services_have_name(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 8001
        for svc in list_registered_services(host="127.0.0.1"):
            assert "name" in svc


class TestGetRegisteredService:
    @patch("core.mcp_registry._config")
    def test_get_known_service(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 8002
        svc = get_registered_service("skills", host="127.0.0.1")
        assert svc["config_name"] == "skills"

    @patch("core.mcp_registry._config")
    def test_get_dev_tools(self, mock_config):
        mock_config.get_mcp_server_port.return_value = 8001
        svc = get_registered_service("dev_tools", host="127.0.0.1")
        assert svc["name"] == "aura-dev-tools"

    def test_unknown_config_name_raises_key_error(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_registered_service("nonexistent")


class TestIterServiceSpecs:
    def test_iter_returns_specs(self):
        specs = list(iter_service_specs())
        assert len(specs) >= 1
        assert all(isinstance(s, MCPServiceSpec) for s in specs)

    def test_known_config_names_present(self):
        names = {s.config_name for s in iter_service_specs()}
        assert "dev_tools" in names
        assert "skills" in names
        assert "control" in names
