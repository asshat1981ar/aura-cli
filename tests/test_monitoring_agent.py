"""Unit tests for agents/monitoring_agent.py."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agents.monitoring_agent import MonitoringAgentAdapter


class TestMonitoringAgentAdapterInit:
    def test_default_init(self):
        agent = MonitoringAgentAdapter()
        assert agent.name == "monitoring"
        assert agent.monitor is None
        assert agent._metric_history == []
        assert agent._max_history == 1000

    def test_init_with_monitor(self):
        mock_monitor = MagicMock()
        agent = MonitoringAgentAdapter(health_monitor=mock_monitor)
        assert agent.monitor is mock_monitor


class TestRunDispatch:
    def setup_method(self):
        self.agent = MonitoringAgentAdapter()

    def test_unknown_action_returns_error(self):
        result = self.agent.run({"action": "unknown_action"})
        assert "error" in result
        assert "unknown_action" in result["error"]

    @patch.object(MonitoringAgentAdapter, "_health_scan")
    def test_scan_action_dispatches(self, mock_scan):
        mock_scan.return_value = {"action": "scan"}
        result = self.agent.run({"action": "scan", "servers": {}})
        mock_scan.assert_called_once_with({})

    @patch.object(MonitoringAgentAdapter, "_check_alerts")
    def test_alert_action_dispatches(self, mock_alerts):
        mock_alerts.return_value = {"action": "alert"}
        self.agent.run({"action": "alert"})
        mock_alerts.assert_called_once()

    @patch.object(MonitoringAgentAdapter, "_diagnose")
    def test_diagnose_action_dispatches(self, mock_diagnose):
        mock_diagnose.return_value = {"action": "diagnose"}
        self.agent.run({"action": "diagnose"})
        mock_diagnose.assert_called_once()


class TestHealthScan:
    @patch.object(MonitoringAgentAdapter, "_check_port", return_value=False)
    def test_scan_with_all_unreachable(self, _):
        agent = MonitoringAgentAdapter()
        result = agent._health_scan({"svc_a": 9001, "svc_b": 9002})
        assert result["total"] == 2
        assert result["healthy"] == 0
        assert result["unhealthy"] == 2
        assert result["servers"]["svc_a"]["status"] == "unreachable"

    @patch.object(MonitoringAgentAdapter, "_check_port", return_value=True)
    def test_scan_with_all_healthy(self, _):
        agent = MonitoringAgentAdapter()
        result = agent._health_scan({"svc_a": 9001})
        assert result["healthy"] == 1
        assert result["servers"]["svc_a"]["status"] == "healthy"

    @patch.object(MonitoringAgentAdapter, "_check_port", return_value=False)
    def test_scan_uses_default_servers_when_empty(self, mock_check):
        agent = MonitoringAgentAdapter()
        result = agent._health_scan({})
        assert result["total"] == 7  # default server count
        assert mock_check.call_count == 7


class TestQueryMetrics:
    def test_query_returns_empty_when_no_history(self):
        agent = MonitoringAgentAdapter()
        result = agent._query_metrics("error_rate", 60)
        assert result["count"] == 0
        assert result["metrics"] == []

    def test_query_filters_by_name(self):
        agent = MonitoringAgentAdapter()
        agent.record_metric("error_rate", 0.01)
        agent.record_metric("latency", 200.0)
        result = agent._query_metrics("error_rate", 60)
        assert result["count"] == 1
        assert result["metrics"][0]["name"] == "error_rate"

    def test_query_no_name_returns_all(self):
        agent = MonitoringAgentAdapter()
        agent.record_metric("error_rate", 0.01)
        agent.record_metric("latency", 200.0)
        result = agent._query_metrics("", 60)
        assert result["count"] == 2


class TestAlerts:
    def test_check_alerts_no_monitor(self):
        agent = MonitoringAgentAdapter()
        result = agent._check_alerts()
        assert result["action"] == "alert"
        assert result["alert_count"] == 0

    def test_check_alerts_with_exceeded_threshold(self):
        mock_monitor = MagicMock()
        mock_monitor.get_metrics.return_value = {"error_rate": 0.9}  # > 0.05 threshold
        agent = MonitoringAgentAdapter(health_monitor=mock_monitor)
        result = agent._check_alerts()
        assert result["alert_count"] >= 1
        assert any(a["metric"] == "error_rate" for a in result["alerts"])

    def test_check_alerts_critical_severity(self):
        mock_monitor = MagicMock()
        mock_monitor.get_metrics.return_value = {"error_rate": 0.15}  # > 0.05 * 1.5 = 0.075
        agent = MonitoringAgentAdapter(health_monitor=mock_monitor)
        result = agent._check_alerts()
        alert = next(a for a in result["alerts"] if a["metric"] == "error_rate")
        assert alert["severity"] == "critical"


class TestRecordMetric:
    def test_record_stores_metric(self):
        agent = MonitoringAgentAdapter()
        agent.record_metric("cpu", 55.0, labels={"host": "node1"})
        assert len(agent._metric_history) == 1
        entry = agent._metric_history[0]
        assert entry["name"] == "cpu"
        assert entry["value"] == 55.0
        assert entry["labels"]["host"] == "node1"

    def test_record_trims_at_max_history(self):
        agent = MonitoringAgentAdapter()
        agent._max_history = 5
        for i in range(10):
            agent.record_metric("m", float(i))
        assert len(agent._metric_history) == 5
        # Last value should be the most recent
        assert agent._metric_history[-1]["value"] == 9.0


class TestDiagnose:
    def test_diagnose_without_monitor(self):
        agent = MonitoringAgentAdapter()
        result = agent._diagnose({})
        assert result["status"] == "health_monitor_not_available"

    def test_diagnose_with_monitor(self):
        mock_monitor = MagicMock()
        mock_monitor.diagnose.return_value = {"ok": True}
        agent = MonitoringAgentAdapter(health_monitor=mock_monitor)
        result = agent._diagnose({})
        assert result["results"] == {"ok": True}

    def test_diagnose_monitor_exception(self):
        mock_monitor = MagicMock()
        mock_monitor.diagnose.side_effect = RuntimeError("fail")
        agent = MonitoringAgentAdapter(health_monitor=mock_monitor)
        result = agent._diagnose({})
        assert "error" in result
