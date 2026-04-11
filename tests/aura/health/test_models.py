"""Tests for health models."""

import pytest
from datetime import datetime

from aura.health.models import (
    CheckResult,
    CheckType,
    HealthReport,
    HealthStatus,
    ThresholdConfig,
)


class TestHealthStatus:
    def test_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestCheckType:
    def test_type_values(self):
        assert CheckType.SYSTEM.value == "system"
        assert CheckType.DATABASE.value == "database"
        assert CheckType.API.value == "api"
        assert CheckType.DISK.value == "disk"
        assert CheckType.MEMORY.value == "memory"
        assert CheckType.CPU.value == "cpu"
        assert CheckType.CUSTOM.value == "custom"


class TestThresholdConfig:
    def test_evaluate_healthy(self):
        config = ThresholdConfig(warning_threshold=70.0, critical_threshold=90.0)
        assert config.evaluate(50.0) == HealthStatus.HEALTHY
        assert config.evaluate(69.9) == HealthStatus.HEALTHY

    def test_evaluate_degraded(self):
        config = ThresholdConfig(warning_threshold=70.0, critical_threshold=90.0)
        assert config.evaluate(70.0) == HealthStatus.DEGRADED
        assert config.evaluate(89.9) == HealthStatus.DEGRADED

    def test_evaluate_unhealthy(self):
        config = ThresholdConfig(warning_threshold=70.0, critical_threshold=90.0)
        assert config.evaluate(90.0) == HealthStatus.UNHEALTHY
        assert config.evaluate(100.0) == HealthStatus.UNHEALTHY


class TestCheckResult:
    def test_healthy_result(self):
        result = CheckResult(
            name="test",
            type=CheckType.SYSTEM,
            status=HealthStatus.HEALTHY,
            response_time_ms=10.5,
        )

        assert result.is_healthy is True
        assert result.name == "test"
        assert result.response_time_ms == 10.5

    def test_unhealthy_result(self):
        result = CheckResult(
            name="test",
            type=CheckType.DATABASE,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=0,
            message="Connection failed",
        )

        assert result.is_healthy is False
        assert result.message == "Connection failed"

    def test_default_timestamp(self):
        before = datetime.utcnow()
        result = CheckResult(
            name="test",
            type=CheckType.CUSTOM,
            status=HealthStatus.HEALTHY,
            response_time_ms=1.0,
        )
        after = datetime.utcnow()

        assert before <= result.timestamp <= after


class TestHealthReport:
    @pytest.fixture
    def sample_checks(self):
        return [
            CheckResult("check1", CheckType.SYSTEM, HealthStatus.HEALTHY, 10.0),
            CheckResult("check2", CheckType.DISK, HealthStatus.DEGRADED, 20.0),
            CheckResult("check3", CheckType.MEMORY, HealthStatus.UNHEALTHY, 5.0),
        ]

    def test_report_creation(self, sample_checks):
        report = HealthReport(
            status=HealthStatus.DEGRADED,
            checks=sample_checks,
            duration_ms=100.0,
        )

        assert report.status == HealthStatus.DEGRADED
        assert report.total_count == 3
        assert report.healthy_count == 1
        assert len(report.failed_checks) == 2

    def test_all_healthy(self):
        checks = [
            CheckResult("c1", CheckType.SYSTEM, HealthStatus.HEALTHY, 1.0),
            CheckResult("c2", CheckType.DISK, HealthStatus.HEALTHY, 2.0),
        ]
        report = HealthReport(status=HealthStatus.HEALTHY, checks=checks)

        assert report.healthy_count == 2
        assert report.failed_checks == []

    def test_to_dict(self, sample_checks):
        report = HealthReport(
            status=HealthStatus.DEGRADED,
            checks=sample_checks,
            duration_ms=50.0,
        )

        data = report.to_dict()

        assert data["status"] == "degraded"
        assert data["duration_ms"] == 50.0
        assert data["summary"]["total"] == 3
        assert data["summary"]["healthy"] == 1
        assert data["summary"]["failed"] == 2
        assert len(data["checks"]) == 3
