import threading
from typing import Dict

class SkillMetrics:
    """Thread-safe per-skill metrics tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, float]] = {}

    def _ensure(self, skill: str) -> None:
        if skill not in self._data:
            self._data[skill] = {"call_count": 0, "total_latency_ms": 0.0, "error_count": 0}

    def record(self, skill: str, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self._ensure(skill)
            self._data[skill]["call_count"] += 1
            self._data[skill]["total_latency_ms"] += latency_ms
            if error:
                self._data[skill]["error_count"] += 1

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out = {}
            for k, v in self._data.items():
                entry = dict(v)
                # 'count' alias for backward compatibility with older consumers
                entry["count"] = entry["call_count"]
                out[k] = entry
            return out


SKILL_METRICS = SkillMetrics()
