"""Tests for health checks."""

import pytest

from aura.health.checks import HealthChecks
from aura.health.models import CheckType, HealthStatus


class TestSystemCheck:
    @pytest.mark.asyncio
    async def test_system_check(self):
        result = await HealthChecks.system_check()
        
        assert result.name == "system"
        assert result.type == CheckType.SYSTEM
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.UNHEALTHY)
        assert "platform" in result.details or result.message is not None


class TestDiskCheck:
    @pytest.mark.asyncio
    async def test_disk_check(self):
        result = await HealthChecks.disk_check()
        
        assert result.name == "disk"
        assert result.type == CheckType.DISK
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)
        
        if result.status == HealthStatus.HEALTHY:
            assert "total_bytes" in result.details
            assert "used_percent" in result.details
    
    @pytest.mark.asyncio
    async def test_disk_check_with_custom_path(self, tmp_path):
        result = await HealthChecks.disk_check(path=str(tmp_path))
        
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)


class TestMemoryCheck:
    @pytest.mark.asyncio
    async def test_memory_check(self):
        result = await HealthChecks.memory_check()
        
        assert result.name == "memory"
        assert result.type == CheckType.MEMORY
        # May be UNKNOWN if psutil not installed
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN)
    
    @pytest.mark.asyncio
    async def test_memory_check_thresholds(self):
        result = await HealthChecks.memory_check(warning_percent=0.0, critical_percent=0.0)
        # With 0 thresholds, any usage should be unhealthy
        if result.status != HealthStatus.UNKNOWN:
            assert result.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)


class TestCpuCheck:
    @pytest.mark.asyncio
    async def test_cpu_check(self):
        result = await HealthChecks.cpu_check()
        
        assert result.name == "cpu"
        assert result.type == CheckType.CPU
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN)


class TestDatabaseCheck:
    @pytest.mark.asyncio
    async def test_database_check_memory(self):
        result = await HealthChecks.database_check()
        
        assert result.name == "database"
        assert result.type == CheckType.DATABASE
        # In-memory SQLite should always work
        assert result.status == HealthStatus.HEALTHY
        assert result.details.get("connection") == "successful"
    
    @pytest.mark.asyncio
    async def test_database_check_with_path(self, tmp_path):
        db_path = tmp_path / "test.db"
        result = await HealthChecks.database_check(connection_string=str(db_path))
        
        assert result.status == HealthStatus.HEALTHY


class TestApiCheck:
    @pytest.mark.asyncio
    async def test_api_check_without_aiohttp(self):
        # Test that check returns UNKNOWN when aiohttp not installed
        # or returns appropriate status if it is installed
        result = await HealthChecks.api_check("http://localhost:8080/health")
        
        assert result.type == CheckType.API
        # May be UNKNOWN (no aiohttp) or UNHEALTHY (connection failed)
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN)
    
    @pytest.mark.asyncio
    async def test_api_check_connection_error(self):
        result = await HealthChecks.api_check("http://localhost:59999")
        
        # Should be either UNHEALTHY (if aiohttp installed) or UNKNOWN (if not)
        assert result.status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN)
        assert result.message is not None
