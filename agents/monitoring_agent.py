"""Monitoring agent — collects and analyzes system metrics, health, and alerts.

Extends core/health_monitor.py by exposing it as a routable agent and
adding Prometheus metric push capabilities.
"""

from __future__ import annotations

import socket
import time
from typing import Dict, List, Optional


class MonitoringAgentAdapter:
    """Pipeline adapter for monitoring and observability operations.

    Provides health scanning, metric collection, alert evaluation,
    and system diagnostics.
    """

    name = "monitoring"

    def __init__(self, health_monitor=None):
        self.monitor = health_monitor
        self._metric_history: List[Dict] = []
        self._max_history = 1000
        self._alert_thresholds: Dict[str, float] = {
            "error_rate": 0.05,
            "p99_latency_ms": 10000,
            "memory_usage_pct": 80.0,
            "goal_failure_rate": 0.50,
        }

    def run(self, input_data: dict) -> dict:
        """Execute a monitoring action.

        Args:
            input_data: Dict with keys:
                - action (str): "scan", "query", "alert", "status", "diagnose".
                - servers (dict, optional): {name: port} map for health scan.
                - metric_name (str, optional): For query action.
                - time_range_minutes (int, optional): For query action.

        Returns:
            Dict with action-specific results.
        """
        action = input_data.get("action", "status")

        if action == "scan":
            return self._health_scan(input_data.get("servers", {}))
        elif action == "query":
            return self._query_metrics(
                input_data.get("metric_name", ""),
                input_data.get("time_range_minutes", 60),
            )
        elif action == "alert":
            return self._check_alerts()
        elif action == "status":
            return self._system_status()
        elif action == "diagnose":
            return self._diagnose(input_data)
        else:
            return {"error": f"Unknown action: {action}"}

    def _health_scan(self, servers: Dict[str, int]) -> dict:
        """Scan MCP servers for health status."""
        if not servers:
            servers = {
                "dev_tools": 8001,
                "skills": 8002,
                "control": 8003,
                "thinking": 8004,
                "agentic_loop": 8006,
                "copilot": 8007,
                "hub": 8010,
            }

        results = {}
        healthy_count = 0
        for name, port in servers.items():
            is_up = self._check_port(port)
            status = "healthy" if is_up else "unreachable"
            if is_up:
                healthy_count += 1
            results[name] = {"port": port, "status": status}

        return {
            "action": "scan",
            "total": len(servers),
            "healthy": healthy_count,
            "unhealthy": len(servers) - healthy_count,
            "servers": results,
        }

    def _check_port(self, port: int, host: str = "127.0.0.1") -> bool:
        """Check if a TCP port is listening."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0

    def _query_metrics(self, metric_name: str, time_range_minutes: int) -> dict:
        """Query stored metric history."""
        cutoff = time.time() - (time_range_minutes * 60)
        matching = [m for m in self._metric_history if m.get("name") == metric_name and m.get("timestamp", 0) >= cutoff] if metric_name else [m for m in self._metric_history if m.get("timestamp", 0) >= cutoff]
        return {
            "action": "query",
            "metric_name": metric_name,
            "time_range_minutes": time_range_minutes,
            "count": len(matching),
            "metrics": matching[-100:],
        }

    def _check_alerts(self) -> dict:
        """Check current metrics against alert thresholds."""
        alerts = []

        # Check from health monitor if available
        if self.monitor and hasattr(self.monitor, "get_metrics"):
            try:
                metrics = self.monitor.get_metrics()
                for name, threshold in self._alert_thresholds.items():
                    value = metrics.get(name)
                    if value is not None and value > threshold:
                        alerts.append(
                            {
                                "metric": name,
                                "value": value,
                                "threshold": threshold,
                                "severity": "warning" if value < threshold * 1.5 else "critical",
                            }
                        )
            except (OSError, IOError, ValueError):
                pass

        return {
            "action": "alert",
            "alert_count": len(alerts),
            "alerts": alerts,
            "thresholds": self._alert_thresholds,
        }

    def _system_status(self) -> dict:
        """Return overall system status summary."""
        scan = self._health_scan({})
        alerts = self._check_alerts()

        overall = "healthy"
        if scan["unhealthy"] > 0:
            overall = "degraded"
        if alerts["alert_count"] > 0:
            overall = "warning"
        if scan["unhealthy"] > scan["total"] // 2:
            overall = "critical"

        return {
            "action": "status",
            "overall": overall,
            "servers": scan,
            "alerts": alerts,
            "metric_history_size": len(self._metric_history),
            "timestamp": time.time(),
        }

    def _diagnose(self, input_data: dict) -> dict:
        """Run diagnostics using the health monitor."""
        if self.monitor and hasattr(self.monitor, "diagnose"):
            try:
                return {"action": "diagnose", "results": self.monitor.diagnose()}
            except Exception as exc:
                return {"action": "diagnose", "error": str(exc)}
        return {"action": "diagnose", "status": "health_monitor_not_available"}

    def record_metric(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """Record a metric for history tracking."""
        self._metric_history.append(
            {
                "name": name,
                "value": value,
                "labels": labels or {},
                "timestamp": time.time(),
            }
        )
        if len(self._metric_history) > self._max_history:
            self._metric_history = self._metric_history[-self._max_history :]
