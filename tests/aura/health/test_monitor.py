"""Tests for health monitor."""

import asyncio
import pytest

from aura.health.models import CheckResult, CheckType, HealthStatus
from aura.health.monitor import BackgroundHealthMonitor, HealthMonitor


class TestHealthMonitor:
    @pytest.fixture
    def monitor(self):
        return HealthMonitor()

    @pytest.fixture
    def healthy_check(self):
        async def check():
            return CheckResult("test", CheckType.CUSTOM, HealthStatus.HEALTHY, 1.0)

        return check

    @pytest.fixture
    def unhealthy_check(self):
        async def check():
            return CheckResult("test", CheckType.CUSTOM, HealthStatus.UNHEALTHY, 1.0, "Error")

        return check

    def test_register_check(self, monitor, healthy_check):
        monitor.register_check("my_check", healthy_check)
        assert "my_check" in monitor._checks

    def test_unregister_check(self, monitor, healthy_check):
        monitor.register_check("my_check", healthy_check)
        monitor.unregister_check("my_check")
        assert "my_check" not in monitor._checks

    @pytest.mark.asyncio
    async def test_run_check(self, monitor, healthy_check):
        monitor.register_check("healthy", healthy_check)
        result = await monitor.run_check("healthy")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_run_unknown_check(self, monitor):
        result = await monitor.run_check("unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_run_all_checks(self, monitor, healthy_check, unhealthy_check):
        monitor.register_check("good", healthy_check)
        monitor.register_check("bad", unhealthy_check)

        report = await monitor.run_all_checks()

        assert report.status == HealthStatus.UNHEALTHY
        assert report.total_count == 2
        assert report.healthy_count == 1

    @pytest.mark.asyncio
    async def test_run_selected_checks(self, monitor, healthy_check, unhealthy_check):
        monitor.register_check("good", healthy_check)
        monitor.register_check("bad", unhealthy_check)

        report = await monitor.run_all_checks(names=["good"])

        assert report.total_count == 1
        assert report.healthy_count == 1

    @pytest.mark.asyncio
    async def test_all_healthy(self, monitor, healthy_check):
        monitor.register_check("c1", healthy_check)
        monitor.register_check("c2", healthy_check)

        report = await monitor.run_all_checks()

        assert report.status == HealthStatus.HEALTHY
        assert monitor.is_healthy() is True

    @pytest.mark.asyncio
    async def test_one_degraded(self, monitor, healthy_check):
        async def degraded_check():
            return CheckResult("d", CheckType.CUSTOM, HealthStatus.DEGRADED, 1.0)

        monitor.register_check("good", healthy_check)
        monitor.register_check("warn", degraded_check)

        report = await monitor.run_all_checks()

        assert report.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_exception_handled(self, monitor):
        async def failing_check():
            raise ValueError("Check failed")

        monitor.register_check("failing", failing_check)
        result = await monitor.run_check("failing")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message

    @pytest.mark.asyncio
    async def test_sync_check_function(self, monitor):
        def sync_check():
            return CheckResult("sync", CheckType.CUSTOM, HealthStatus.HEALTHY, 1.0)

        monitor.register_check("sync", sync_check)
        result = await monitor.run_check("sync")

        assert result.status == HealthStatus.HEALTHY

    def test_get_last_report_empty(self, monitor):
        assert monitor.get_last_report() is None
        assert monitor.is_healthy() is False

    @pytest.mark.asyncio
    async def test_get_last_report_after_run(self, monitor, healthy_check):
        monitor.register_check("test", healthy_check)
        await monitor.run_all_checks()

        report = monitor.get_last_report()
        assert report is not None
        assert report.total_count == 1


class TestBackgroundHealthMonitor:
    @pytest.fixture
    def bg_monitor(self):
        return BackgroundHealthMonitor(check_interval=0.1)

    @pytest.fixture
    def healthy_check(self):
        async def check():
            return CheckResult("test", CheckType.CUSTOM, HealthStatus.HEALTHY, 0.1)

        return check

    @pytest.mark.asyncio
    async def test_start_stop(self, bg_monitor, healthy_check):
        bg_monitor.register_check("test", healthy_check)

        await bg_monitor.start()
        assert bg_monitor._running is True
        assert bg_monitor._task is not None

        await bg_monitor.stop()
        assert bg_monitor._running is False
        assert bg_monitor._task is None

    @pytest.mark.asyncio
    async def test_background_checks_run(self, bg_monitor, healthy_check):
        bg_monitor.register_check("test", healthy_check)

        await bg_monitor.start()
        await asyncio.sleep(0.15)  # Wait for at least one check
        await bg_monitor.stop()

        assert bg_monitor.get_last_report() is not None

    @pytest.mark.asyncio
    async def test_status_change_callback(self, bg_monitor):
        calls = []

        async def toggle_check():
            # Toggle between healthy and unhealthy
            toggle_check.state = not getattr(toggle_check, "state", False)
            status = HealthStatus.UNHEALTHY if toggle_check.state else HealthStatus.HEALTHY
            return CheckResult("toggle", CheckType.CUSTOM, status, 0.1)

        def on_change(report):
            calls.append(report.status)

        bg_monitor.register_check("toggle", toggle_check)
        bg_monitor.on_status_change(on_change)

        await bg_monitor.start()
        await asyncio.sleep(0.25)  # Allow multiple checks
        await bg_monitor.stop()

        # Should have received at least one callback
        assert len(calls) >= 1
