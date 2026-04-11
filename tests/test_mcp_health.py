"""Unit tests for core/mcp_health.py — check_mcp_health, get_health_summary."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.mcp_health import check_mcp_health, get_health_summary


class TestGetHealthSummary:
    def test_all_healthy(self):
        results = [
            {"name": "github", "status": "healthy"},
            {"name": "slack", "status": "healthy"},
        ]
        summary = get_health_summary(results)
        assert summary["total_servers"] == 2
        assert summary["healthy_count"] == 2
        assert summary["unhealthy_count"] == 0
        assert summary["all_healthy"] is True

    def test_some_unhealthy(self):
        results = [
            {"name": "github", "status": "healthy"},
            {"name": "slack", "status": "unhealthy", "error": "refused"},
        ]
        summary = get_health_summary(results)
        assert summary["total_servers"] == 2
        assert summary["healthy_count"] == 1
        assert summary["unhealthy_count"] == 1
        assert summary["all_healthy"] is False

    def test_empty_results(self):
        summary = get_health_summary([])
        assert summary["total_servers"] == 0
        assert summary["healthy_count"] == 0
        assert summary["unhealthy_count"] == 0
        assert summary["all_healthy"] is True

    def test_all_unhealthy(self):
        results = [{"name": "svc", "status": "unhealthy"}]
        summary = get_health_summary(results)
        assert summary["all_healthy"] is False
        assert summary["unhealthy_count"] == 1


class TestCheckMcpHealth:
    def test_returns_healthy_dict_on_success(self):
        mock_client = MagicMock()
        mock_client.get_health = AsyncMock(return_value={"status": "ok"})

        with patch("core.mcp_health.config") as mock_config, patch("core.mcp_health.MCPAsyncClient", return_value=mock_client):
            mock_config.get_mcp_server_port.return_value = 8001
            result = asyncio.run(check_mcp_health("github"))

        assert result["name"] == "github"
        assert result["status"] == "healthy"
        assert result["port"] == 8001
        assert "health_data" in result

    def test_returns_unhealthy_dict_on_exception(self):
        with patch("core.mcp_health.config") as mock_config, patch("core.mcp_health.MCPAsyncClient", side_effect=ConnectionError("refused")), patch("core.mcp_health.log_json"):
            mock_config.get_mcp_server_port.return_value = 8002
            result = asyncio.run(check_mcp_health("slack"))

        assert result["name"] == "slack"
        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_unhealthy_result_contains_error_string(self):
        with patch("core.mcp_health.config") as mock_config, patch("core.mcp_health.MCPAsyncClient", side_effect=TimeoutError("timed out")), patch("core.mcp_health.log_json"):
            mock_config.get_mcp_server_port.return_value = 8003
            result = asyncio.run(check_mcp_health("sentry"))

        assert "timed out" in result["error"]
