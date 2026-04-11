"""Health monitoring system."""

import asyncio
import time
from typing import Callable, Dict, List, Optional

from .models import CheckResult, HealthReport, HealthStatus


class HealthMonitor:
    """Monitor system health with configurable checks."""

    def __init__(self):
        self._checks: Dict[str, Callable[[], CheckResult]] = {}
        self._check_configs: Dict[str, dict] = {}
        self._last_report: Optional[HealthReport] = None

    def register_check(
        self,
        name: str,
        check_func: Callable[[], CheckResult],
        config: Optional[dict] = None,
    ):
        """Register a health check."""
        self._checks[name] = check_func
        self._check_configs[name] = config or {}

    def unregister_check(self, name: str):
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._check_configs.pop(name, None)

    async def run_check(self, name: str) -> Optional[CheckResult]:
        """Run a single check by name."""
        check_func = self._checks.get(name)
        if not check_func:
            return None

        try:
            result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
            return result
        except Exception as e:
            return CheckResult(
                name=name,
                type=self._check_configs.get(name, {}).get("type", "custom"),
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                message=f"Check failed: {str(e)}",
            )

    async def run_all_checks(self, names: Optional[List[str]] = None) -> HealthReport:
        """Run all or selected checks."""
        start = time.time()

        check_names = names or list(self._checks.keys())

        # Run all checks concurrently
        tasks = [self.run_check(name) for name in check_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        checks: List[CheckResult] = []
        for result in results:
            if isinstance(result, Exception):
                checks.append(
                    CheckResult(
                        name="unknown",
                        type="custom",
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=0,
                        message=str(result),
                    )
                )
            elif result:
                checks.append(result)

        # Determine overall status
        if not checks:
            overall_status = HealthStatus.UNKNOWN
        elif any(c.status == HealthStatus.UNHEALTHY for c in checks):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in checks):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        duration_ms = (time.time() - start) * 1000

        self._last_report = HealthReport(
            status=overall_status,
            checks=checks,
            duration_ms=duration_ms,
        )

        return self._last_report

    def get_last_report(self) -> Optional[HealthReport]:
        """Get the last health report."""
        return self._last_report

    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        if not self._last_report:
            return False
        return self._last_report.status == HealthStatus.HEALTHY


class BackgroundHealthMonitor(HealthMonitor):
    """Health monitor that runs checks continuously in background."""

    def __init__(self, check_interval: int = 60):
        super().__init__()
        self.check_interval = check_interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._callbacks: List[Callable[[HealthReport], None]] = []

    def on_status_change(self, callback: Callable[[HealthReport], None]):
        """Register callback for status changes."""
        self._callbacks.append(callback)

    async def start(self):
        """Start background monitoring."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self):
        """Background monitoring loop."""
        previous_status: Optional[HealthStatus] = None

        while self._running:
            try:
                report = await self.run_all_checks()

                # Notify if status changed
                if report.status != previous_status:
                    for callback in self._callbacks:
                        try:
                            callback(report)
                        except Exception:
                            pass
                    previous_status = report.status

                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
