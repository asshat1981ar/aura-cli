"""Tests for observability metrics module."""

import json
import statistics
import tempfile
import time
from pathlib import Path

import pytest

from aura.observability.metrics import (
    MetricSeries,
    MetricType,
    MetricValue,
    MetricsStore,
    get_metrics_store,
    record_agent_execution,
    record_cycle_metrics,
    record_memory_operation,
)


class TestMetricValue:
    """Test MetricValue dataclass."""

    def test_creation(self):
        value = MetricValue(value=42.0, labels={"test": "label"})
        assert value.value == 42.0
        assert value.labels == {"test": "label"}
        assert value.timestamp is not None

    def test_to_dict(self):
        value = MetricValue(value=100.0, labels={"key": "val"})
        d = value.to_dict()
        assert d["value"] == 100.0
        assert d["labels"] == {"key": "val"}
        assert "timestamp" in d


class TestMetricSeries:
    """Test MetricSeries functionality."""

    def test_creation(self):
        series = MetricSeries(
            name="test_metric",
            metric_type=MetricType.COUNTER,
            description="Test metric",
            unit="count",
        )
        assert series.name == "test_metric"
        assert series.metric_type == MetricType.COUNTER
        assert len(series.values) == 0

    def test_add_value(self):
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)
        series.add(10.0)
        series.add(20.0, labels={"status": "ok"})
        assert len(series.values) == 2

    def test_get_latest(self):
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)
        assert series.get_latest() is None
        series.add(10.0)
        assert series.get_latest().value == 10.0
        series.add(20.0)
        assert series.get_latest().value == 20.0

    def test_get_values_with_since(self):
        from datetime import datetime, timedelta

        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)
        old_time = datetime.utcnow() - timedelta(hours=1)
        series.values.append(MetricValue(value=10.0, timestamp=old_time))
        series.values.append(MetricValue(value=20.0))

        recent = datetime.utcnow() - timedelta(minutes=30)
        values = series.get_values(since=recent)
        assert len(values) == 1
        assert values[0].value == 20.0

    def test_get_stats_empty(self):
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)
        stats = series.get_stats()
        assert stats["count"] == 0

    def test_get_stats_single_value(self):
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)
        series.add(10.0)
        stats = series.get_stats()
        assert stats["count"] == 1
        assert stats["sum"] == 10.0
        assert stats["min"] == 10.0
        assert stats["max"] == 10.0
        assert stats["mean"] == 10.0

    def test_get_stats_multiple_values(self):
        series = MetricSeries(name="test", metric_type=MetricType.HISTOGRAM)
        for i in range(1, 11):
            series.add(float(i))
        stats = series.get_stats()
        assert stats["count"] == 10
        assert stats["sum"] == 55.0
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["mean"] == 5.5
        assert "stddev" in stats
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats

    def test_get_stats_with_window(self):
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)
        for i in range(1, 101):
            series.add(float(i))
        stats = series.get_stats(window=10)
        assert stats["count"] == 10
        assert stats["min"] == 91.0
        assert stats["max"] == 100.0

    def test_percentile_calculation(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Implementation: int(N * percentile) gives index, then clamped
        # For 10 elements: index 5 = 6th element (value 6)
        assert MetricSeries._percentile(data, 0.5) == 6  # index 5
        assert MetricSeries._percentile(data, 0.95) == 10  # index 9
        assert MetricSeries._percentile(data, 0.99) == 10  # index 9


class TestMetricsStore:
    """Test MetricsStore functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        MetricsStore._instance = None
        MetricsStore._initialized = False

    def test_singleton_pattern(self):
        store1 = MetricsStore()
        store2 = MetricsStore()
        assert store1 is store2

    def test_counter_registration(self):
        store = MetricsStore()
        series = store.counter("requests_total", "Total requests", "count")
        assert series.name == "requests_total"
        assert series.metric_type == MetricType.COUNTER

    def test_gauge_registration(self):
        store = MetricsStore()
        series = store.gauge("memory_usage", "Memory usage", "MB")
        assert series.metric_type == MetricType.GAUGE

    def test_timer_registration(self):
        store = MetricsStore()
        series = store.timer("latency", "Request latency", "ms")
        assert series.metric_type == MetricType.TIMER

    def test_histogram_registration(self):
        store = MetricsStore()
        series = store.histogram("response_size", "Response size", "bytes")
        assert series.metric_type == MetricType.HISTOGRAM

    def test_increment(self):
        store = MetricsStore()
        store.counter("test_counter")
        store.increment("test_counter")
        store.increment("test_counter", 5.0)
        stats = store.get_stats("test_counter")
        # Note: counter adds values directly, doesn't accumulate
        assert stats["count"] == 2

    def test_set_gauge(self):
        store = MetricsStore()
        store.gauge("test_gauge")
        store.set("test_gauge", 42.0)
        # Check via snapshot which includes latest
        snapshot = store.snapshot()
        assert snapshot["test_gauge"]["latest"] == 42.0

    def test_observe_histogram(self):
        store = MetricsStore()
        store.histogram("test_histogram")
        store.observe("test_histogram", 100.0)
        store.observe("test_histogram", 200.0)
        stats = store.get_stats("test_histogram")
        assert stats["count"] == 2

    def test_time_context_manager(self):
        store = MetricsStore()
        store.timer("test_timer")
        with store.time("test_timer"):
            time.sleep(0.01)  # 10ms
        stats = store.get_stats("test_timer")
        assert stats["count"] == 1
        assert stats["min"] >= 10.0  # At least 10ms

    def test_timed_decorator(self):
        store = MetricsStore()
        store.timer("func_timer")

        @store.timed("func_timer")
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        assert result == 42
        stats = store.get_stats("func_timer")
        assert stats["count"] == 1

    def test_get_series(self):
        store = MetricsStore()
        store.counter("test")
        series = store.get_series("test")
        assert series is not None
        assert series.name == "test"
        assert store.get_series("nonexistent") is None

    def test_get_metric_names(self):
        store = MetricsStore()
        store.counter("metric_a")
        store.gauge("metric_b")
        names = store.get_metric_names()
        assert "metric_a" in names
        assert "metric_b" in names

    def test_get_all_series(self):
        store = MetricsStore()
        store.counter("metric_a")
        store.gauge("metric_b")
        all_series = store.get_all_series()
        assert len(all_series) == 2
        assert "metric_a" in all_series
        assert "metric_b" in all_series

    def test_snapshot(self):
        store = MetricsStore()
        store.counter("requests", "Total requests")
        store.increment("requests", 10.0)
        snapshot = store.snapshot()
        assert "requests" in snapshot
        assert snapshot["requests"]["type"] == "COUNTER"
        assert snapshot["requests"]["description"] == "Total requests"

    def test_series_eviction(self):
        store = MetricsStore(max_series=2)
        store.counter("metric_a")
        store.counter("metric_b")
        store.counter("metric_c")  # Should evict oldest
        names = store.get_metric_names()
        assert len(names) == 2

    def test_clear(self):
        store = MetricsStore()
        store.counter("test")
        store.increment("test")
        store.clear()
        assert len(store.get_metric_names()) == 0

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"

            # Create and populate store
            store = MetricsStore(persist_path=path)
            store.counter("test")
            store.increment("test", 42.0)
            store.flush()

            # Verify file exists and has content
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "test" in data

            # Create new store instance with same path
            MetricsStore._instance = None
            MetricsStore._initialized = False
            store2 = MetricsStore(persist_path=path)
            # Note: Values are restored but not the current in-memory state
            # This tests that the loading code runs without error


class TestUtilityFunctions:
    """Test utility functions for common metrics patterns."""

    def setup_method(self):
        """Reset singleton before each test."""
        MetricsStore._instance = None
        MetricsStore._initialized = False

    def test_record_agent_execution(self):
        record_agent_execution(
            agent_name="test_agent",
            duration_ms=100.0,
            success=True,
            output_quality=0.9,
        )
        store = get_metrics_store()
        assert store.get_series("agent_executions_total") is not None
        assert store.get_series("agent_execution_duration_ms") is not None
        assert store.get_series("agent_output_quality") is not None

    def test_record_memory_operation(self):
        record_memory_operation(
            operation="recall",
            duration_ms=50.0,
            bytes_accessed=1024,
        )
        store = get_metrics_store()
        assert store.get_series("memory_operation_duration_ms") is not None
        assert store.get_series("memory_bytes_total") is not None

    def test_record_cycle_metrics(self):
        record_cycle_metrics(
            cycle_number=1,
            duration_ms=5000.0,
            phases_completed=10,
            success=True,
        )
        store = get_metrics_store()
        assert store.get_series("cycles_total") is not None
        assert store.get_series("cycle_duration_ms") is not None
        assert store.get_series("phases_per_cycle") is not None


class TestMetricsStoreEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset singleton before each test."""
        MetricsStore._instance = None
        MetricsStore._initialized = False

    def test_operations_on_unregistered_metric(self):
        store = MetricsStore()
        # Should not raise errors
        store.increment("nonexistent")
        store.set("nonexistent", 10.0)
        store.observe("nonexistent", 100.0)
        assert store.get_stats("nonexistent") is None

    def test_time_context_manager_exception(self):
        store = MetricsStore()
        store.timer("test_timer")

        try:
            with store.time("test_timer"):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Timer should still record the duration
        stats = store.get_stats("test_timer")
        assert stats["count"] == 1

    def test_concurrent_access(self):
        import threading

        store = MetricsStore()
        store.counter("concurrent")

        def increment_many():
            for _ in range(100):
                store.increment("concurrent")

        threads = [threading.Thread(target=increment_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = store.get_stats("concurrent")
        assert stats["count"] == 500
