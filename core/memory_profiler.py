"""Memory profiling utilities for performance optimization."""

from __future__ import annotations

import gc
import tracemalloc
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import time

from core.logging_utils import log_json


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    size: int  # bytes
    count: int  # number of objects
    peak: int  # peak memory usage
    timestamp: float


class MemoryProfiler:
    """Profile memory usage of operations."""

    def __init__(self):
        self._snapshots: List[MemorySnapshot] = []
        self._tracing = False

    def start_tracing(self) -> None:
        """Start memory tracing."""
        if not self._tracing:
            tracemalloc.start()
            self._tracing = True
            log_json("INFO", "memory_tracing_started")

    def stop_tracing(self) -> None:
        """Stop memory tracing."""
        if self._tracing:
            tracemalloc.stop()
            self._tracing = False
            log_json("INFO", "memory_tracing_stopped")

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        if self._tracing:
            current, peak = tracemalloc.get_traced_memory()
            snapshot = MemorySnapshot(
                size=current,
                count=len(gc.get_objects()),
                peak=peak,
                timestamp=time.time(),
            )
        else:
            # Fallback without tracing
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            snapshot = MemorySnapshot(
                size=memory_info.rss,
                count=len(gc.get_objects()),
                peak=memory_info.rss,
                timestamp=time.time(),
            )

        self._snapshots.append(snapshot)
        return snapshot

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self._snapshots:
            return {}

        latest = self._snapshots[-1]

        return {
            "current_mb": round(latest.size / (1024 * 1024), 2),
            "peak_mb": round(latest.peak / (1024 * 1024), 2),
            "object_count": latest.count,
            "snapshot_count": len(self._snapshots),
        }

    def get_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory allocations.

        Args:
            limit: Number of top allocations to return

        Returns:
            List of allocation statistics
        """
        if not self._tracing:
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")[:limit]

        return [
            {
                "file": stat.traceback.format()[-1] if stat.traceback else "unknown",
                "size_mb": round(stat.size / (1024 * 1024), 2),
                "count": stat.count,
            }
            for stat in top_stats
        ]

    def compare_snapshots(
        self,
        snapshot1_idx: int = -2,
        snapshot2_idx: int = -1,
    ) -> Dict[str, Any]:
        """Compare two snapshots.

        Args:
            snapshot1_idx: Index of first snapshot
            snapshot2_idx: Index of second snapshot

        Returns:
            Comparison statistics
        """
        if len(self._snapshots) < 2:
            return {}

        s1 = self._snapshots[snapshot1_idx]
        s2 = self._snapshots[snapshot2_idx]

        size_diff = s2.size - s1.size
        count_diff = s2.count - s1.count

        return {
            "size_diff_mb": round(size_diff / (1024 * 1024), 2),
            "size_diff_percent": round((size_diff / s1.size) * 100, 2) if s1.size > 0 else 0,
            "object_diff": count_diff,
            "time_elapsed": round(s2.timestamp - s1.timestamp, 2),
        }

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()


# Global profiler instance
_profiler: Optional[MemoryProfiler] = None


def get_profiler() -> MemoryProfiler:
    """Get global memory profiler."""
    global _profiler
    if _profiler is None:
        _profiler = MemoryProfiler()
    return _profiler


@contextmanager
def profile_memory(operation_name: str):
    """Context manager to profile memory usage.

    Args:
        operation_name: Name of the operation being profiled

    Example:
        with profile_memory("agent_execution"):
            result = await agent.run()
    """
    profiler = get_profiler()
    gc.collect()  # Clean up before measurement

    profiler.take_snapshot()  # registers baseline for compare_snapshots
    start_time = time.time()

    try:
        yield
    finally:
        gc.collect()
        profiler.take_snapshot()  # registers post-op snapshot for comparison
        elapsed = time.time() - start_time

        comparison = profiler.compare_snapshots(-2, -1)

        log_json(
            "INFO",
            "memory_profiled_operation",
            {
                "operation": operation_name,
                "elapsed_ms": round(elapsed * 1000, 2),
                **comparison,
            },
        )


def memory_traced(func: Callable) -> Callable:
    """Decorator to trace memory usage of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with profile_memory(func.__name__):
            return func(*args, **kwargs)

    return wrapper


class ObjectPool:
    """Pool for reusing expensive objects."""

    def __init__(self, factory: Callable, max_size: int = 10):
        self._factory = factory
        self._max_size = max_size
        self._available: List[Any] = []
        self._in_use: set = set()

    def acquire(self) -> Any:
        """Acquire an object from the pool."""
        if self._available:
            obj = self._available.pop()
        else:
            obj = self._factory()

        self._in_use.add(id(obj))
        return obj

    def release(self, obj: Any) -> None:
        """Release an object back to the pool."""
        obj_id = id(obj)
        if obj_id in self._in_use:
            self._in_use.discard(obj_id)
            if len(self._available) < self._max_size:
                self._available.append(obj)

    @contextmanager
    def borrow(self):
        """Context manager to borrow an object."""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory metrics in MB
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
            "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
            "percent": process.memory_percent(),
        }
    except ImportError:
        # Fallback without psutil
        return {
            "rss_mb": 0,
            "vms_mb": 0,
            "percent": 0,
        }


def force_garbage_collection() -> Dict[str, Any]:
    """Force garbage collection and return stats.

    Returns:
        GC statistics
    """
    gc.collect()

    counts = gc.get_count()
    thresholds = gc.get_threshold()

    return {
        "generation_counts": list(counts),
        "thresholds": list(thresholds),
        "objects_tracked": len(gc.get_objects()),
    }
