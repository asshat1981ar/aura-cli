"""Metrics collection and storage for AURA observability.

Provides lightweight, in-memory metrics collection with optional
persistent storage. Supports counters, gauges, histograms, and timers.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional


class MetricType(Enum):
    """Types of metrics supported."""

    COUNTER = auto()  # Monotonically increasing (e.g., total requests)
    GAUGE = auto()  # Point-in-time value (e.g., memory usage)
    HISTOGRAM = auto()  # Distribution of values (e.g., response times)
    TIMER = auto()  # Specialized histogram for durations


@dataclass
class MetricValue:
    """A single metric sample."""

    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


@dataclass
class MetricSeries:
    """A time series of metric values."""

    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    values: Deque[MetricValue] = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new value to the series."""
        self.values.append(MetricValue(value=value, labels=labels or {}))

    def get_latest(self) -> Optional[MetricValue]:
        """Get the most recent value."""
        return self.values[-1] if self.values else None

    def get_values(self, since: Optional[datetime] = None) -> List[MetricValue]:
        """Get values, optionally filtered by time."""
        if since is None:
            return list(self.values)
        return [v for v in self.values if v.timestamp >= since]

    def get_stats(self, window: Optional[int] = None) -> Dict[str, float]:
        """Calculate statistics for this metric.

        Args:
            window: Number of recent values to consider (None = all)
        """
        values = list(self.values)[-window:] if window else list(self.values)
        if not values:
            return {"count": 0}

        nums = [v.value for v in values]
        stats = {
            "count": len(nums),
            "sum": sum(nums),
            "min": min(nums),
            "max": max(nums),
            "mean": statistics.mean(nums),
        }

        if len(nums) >= 2:
            stats["stddev"] = statistics.stdev(nums)
            stats["p50"] = self._percentile(nums, 0.5)
            stats["p95"] = self._percentile(nums, 0.95)
            stats["p99"] = self._percentile(nums, 0.99)

        return stats

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile using nearest-rank method."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[max(0, min(index, len(sorted_data) - 1))]


class MetricsStore:
    """Central store for all metrics.

    Thread-safe singleton for collecting and retrieving metrics.
    Supports in-memory storage with optional persistence.

    Example:
        >>> store = MetricsStore()
        >>> store.counter("requests_total", "Total requests")
        >>> store.increment("requests_total")
        >>> store.gauge("memory_mb", "Memory usage")
        >>> store.set("memory_mb", 512.0)
        >>> store.timer("request_duration_ms", "Request latency")
        >>> with store.time("request_duration_ms"):
        ...     do_work()
    """

    _instance: Optional[MetricsStore] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> MetricsStore:
        """Singleton pattern for global metrics access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_series: int = 100,
        persist_path: Optional[Path] = None,
        auto_flush_interval: int = 60,
    ):
        if self._initialized:
            return

        self._initialized = True
        self._max_series = max_series
        self._persist_path = persist_path
        self._auto_flush_interval = auto_flush_interval

        self._series: Dict[str, MetricSeries] = {}
        self._series_lock = threading.RLock()
        self._flush_lock = threading.Lock()

        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if persist_path:
            self._load_from_disk()
            self._start_flush_thread()

    # ------------------------------------------------------------------
    # Metric Registration
    # ------------------------------------------------------------------

    def counter(
        self,
        name: str,
        description: str = "",
        unit: str = "count",
    ) -> MetricSeries:
        """Register or get a counter metric."""
        return self._get_or_create_series(name, MetricType.COUNTER, description, unit)

    def gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "",
    ) -> MetricSeries:
        """Register or get a gauge metric."""
        return self._get_or_create_series(name, MetricType.GAUGE, description, unit)

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "",
    ) -> MetricSeries:
        """Register or get a histogram metric."""
        return self._get_or_create_series(name, MetricType.HISTOGRAM, description, unit)

    def timer(
        self,
        name: str,
        description: str = "",
        unit: str = "ms",
    ) -> MetricSeries:
        """Register or get a timer metric."""
        return self._get_or_create_series(name, MetricType.TIMER, description, unit)

    def _get_or_create_series(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str,
    ) -> MetricSeries:
        """Get existing series or create new one."""
        with self._series_lock:
            if name not in self._series:
                if len(self._series) >= self._max_series:
                    # Evict oldest series
                    oldest = min(self._series.items(), key=lambda x: x[1].get_latest().timestamp if x[1].get_latest() else datetime.min)
                    del self._series[oldest[0]]

                self._series[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                    description=description,
                    unit=unit,
                )
            return self._series[name]

    # ------------------------------------------------------------------
    # Metric Operations
    # ------------------------------------------------------------------

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        series = self._series.get(name)
        if series:
            series.add(value, labels)

    def set(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value."""
        series = self._series.get(name)
        if series:
            series.add(value, labels)

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value for histogram/timer metrics."""
        series = self._series.get(name)
        if series:
            series.add(value, labels)

    @contextmanager
    def time(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.

        Example:
            >>> with store.time("operation_duration"):
            ...     do_expensive_work()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.observe(name, elapsed_ms, labels)

    def timed(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator for timing functions.

        Example:
            >>> @store.timed("function_duration")
            ... def my_function():
            ...     pass
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                with self.time(name, labels):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Query Operations
    # ------------------------------------------------------------------

    def get_series(self, name: str) -> Optional[MetricSeries]:
        """Get a metric series by name."""
        return self._series.get(name)

    def get_all_series(self) -> Dict[str, MetricSeries]:
        """Get all registered series."""
        with self._series_lock:
            return dict(self._series)

    def get_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        return list(self._series.keys())

    def get_stats(self, name: str, window: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Get statistics for a specific metric."""
        series = self._series.get(name)
        return series.get_stats(window) if series else None

    def snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of all metrics."""
        with self._series_lock:
            return {
                name: {
                    "type": series.metric_type.name,
                    "description": series.description,
                    "unit": series.unit,
                    "stats": series.get_stats(),
                    "latest": series.get_latest().value if series.get_latest() else None,
                }
                for name, series in self._series.items()
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Load metrics from disk if available."""
        if not self._persist_path or not self._persist_path.exists():
            return

        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)

            for name, series_data in data.items():
                metric_type = MetricType[series_data.get("type", "GAUGE")]
                series = self._get_or_create_series(
                    name,
                    metric_type,
                    series_data.get("description", ""),
                    series_data.get("unit", ""),
                )

                for val_data in series_data.get("values", []):
                    series.add(
                        value=val_data["value"],
                        labels=val_data.get("labels", {}),
                    )
        except (json.JSONDecodeError, KeyError, IOError):
            # Ignore corrupted/missing files
            pass

    def flush(self) -> None:
        """Persist metrics to disk."""
        if not self._persist_path:
            return

        with self._flush_lock:
            try:
                self._persist_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot = self.snapshot()

                # Include raw values for recovery
                with self._series_lock:
                    for name, series in self._series.items():
                        snapshot[name]["values"] = [
                            v.to_dict()
                            for v in list(series.values)[-100:]  # Last 100 values
                        ]

                with open(self._persist_path, "w") as f:
                    json.dump(snapshot, f, indent=2)
            except IOError:
                pass

    def _start_flush_thread(self) -> None:
        """Start background flush thread."""

        def flush_loop():
            while not self._stop_event.wait(self._auto_flush_interval):
                self.flush()

        self._flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._flush_thread.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shutdown metrics store and flush remaining data."""
        self._stop_event.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5)
        self.flush()

    def clear(self) -> None:
        """Clear all metrics."""
        with self._series_lock:
            self._series.clear()


# Global convenience functions for common metrics patterns
def get_metrics_store(
    persist_path: Optional[Path] = None,
    max_series: int = 100,
) -> MetricsStore:
    """Get or create the global metrics store instance."""
    return MetricsStore(persist_path=persist_path, max_series=max_series)


# Pre-defined AURA-specific metrics
def record_agent_execution(
    agent_name: str,
    duration_ms: float,
    success: bool,
    output_quality: Optional[float] = None,
) -> None:
    """Record metrics for an agent execution."""
    store = get_metrics_store()

    # Ensure metrics are registered
    store.counter("agent_executions_total", "Total agent executions")
    store.timer("agent_execution_duration_ms", "Agent execution duration")
    store.gauge("agent_output_quality", "Agent output quality (0-1)")

    labels = {"agent": agent_name, "status": "success" if success else "failure"}
    store.increment("agent_executions_total", 1.0, labels)
    store.observe("agent_execution_duration_ms", duration_ms, labels)

    if output_quality is not None:
        store.set("agent_output_quality", output_quality, labels)


def record_memory_operation(operation: str, duration_ms: float, bytes_accessed: int) -> None:
    """Record memory operation metrics."""
    store = get_metrics_store()

    store.timer("memory_operation_duration_ms", "Memory operation duration")
    store.counter("memory_bytes_total", "Total memory bytes accessed")

    labels = {"operation": operation}
    store.observe("memory_operation_duration_ms", duration_ms, labels)
    store.increment("memory_bytes_total", bytes_accessed, labels)


def record_cycle_metrics(
    cycle_number: int,
    duration_ms: float,
    phases_completed: int,
    success: bool,
) -> None:
    """Record orchestrator cycle metrics."""
    store = get_metrics_store()

    store.counter("cycles_total", "Total orchestrator cycles")
    store.timer("cycle_duration_ms", "Cycle duration")
    store.gauge("phases_per_cycle", "Phases completed per cycle")

    labels = {"status": "success" if success else "failure"}
    store.increment("cycles_total", 1.0, labels)
    store.observe("cycle_duration_ms", duration_ms, labels)
    store.set("phases_per_cycle", float(phases_completed), labels)
