"""Karpathy-style experiment tracking with measure-keep/discard discipline.

Tracks self-improvement experiments with real metrics, enforcing the pattern:
capture baseline → mutate → measure → keep if improved, discard if regressed.

Inspired by Karpathy's "autoresearch" (March 2026): edit code, run experiment,
measure performance, keep or discard, repeat.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from core.logging_utils import log_json


@dataclass
class ExperimentResult:
    """Result of a single self-improvement experiment."""

    experiment_id: str
    hypothesis: str
    change_description: str
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    improvement: dict[str, float] = field(default_factory=dict)
    kept: bool = False
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    cycle_number: int = 0

    @property
    def net_improvement(self) -> float:
        if not self.improvement:
            return 0.0
        return sum(self.improvement.values()) / len(self.improvement)


class MetricsCollector:
    """Collects real metrics from AURA's runtime for experiment evaluation."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir

    def collect(self) -> dict[str, float]:
        metrics = {}
        metrics["test_pass_rate"] = self._get_test_pass_rate()
        metrics["avg_cycle_seconds"] = self._get_avg_cycle_time()
        metrics["goal_completion_rate"] = self._get_goal_completion_rate()
        metrics["verify_success_rate"] = self._get_verify_success_rate()
        metrics["avg_retries"] = self._get_avg_retries()
        return metrics

    def _get_test_pass_rate(self) -> float:
        log_path = self.memory_dir / "decision_log.jsonl"
        if not log_path.exists():
            return 0.5
        passed = total = 0
        try:
            lines = log_path.read_text().strip().split("\n")
            for line in lines[-100:]:
                entry = json.loads(line)
                if entry.get("event") == "verify_result":
                    total += 1
                    if entry.get("details", {}).get("success"):
                        passed += 1
        except (json.JSONDecodeError, KeyError, OSError):
            pass
        return passed / max(total, 1)

    def _get_avg_cycle_time(self) -> float:
        log_path = self.memory_dir / "decision_log.jsonl"
        if not log_path.exists():
            return 60.0
        times: list[float] = []
        try:
            lines = log_path.read_text().strip().split("\n")
            for line in lines[-50:]:
                entry = json.loads(line)
                if "cycle_duration" in entry.get("details", {}):
                    times.append(float(entry["details"]["cycle_duration"]))
        except (json.JSONDecodeError, KeyError, OSError):
            pass
        return sum(times) / max(len(times), 1) if times else 60.0

    def _get_goal_completion_rate(self) -> float:
        archive_path = self.memory_dir / "goal_archive.json"
        queue_path = self.memory_dir / "goal_queue.json"
        completed = queued = 0
        try:
            if archive_path.exists():
                data = json.loads(archive_path.read_text())
                completed = len(data) if isinstance(data, list) else 0
            if queue_path.exists():
                data = json.loads(queue_path.read_text())
                queued = len(data) if isinstance(data, list) else 0
        except (json.JSONDecodeError, OSError):
            pass
        total = completed + queued
        return completed / max(total, 1)

    def _get_verify_success_rate(self) -> float:
        return self._get_test_pass_rate()

    def _get_avg_retries(self) -> float:
        log_path = self.memory_dir / "decision_log.jsonl"
        if not log_path.exists():
            return 1.0
        retry_counts: list[float] = []
        try:
            lines = log_path.read_text().strip().split("\n")
            for line in lines[-50:]:
                entry = json.loads(line)
                if "retry_count" in entry.get("details", {}):
                    retry_counts.append(float(entry["details"]["retry_count"]))
        except (json.JSONDecodeError, KeyError, OSError):
            pass
        return sum(retry_counts) / max(len(retry_counts), 1) if retry_counts else 1.0


class ExperimentTracker:
    """Tracks experiments with keep/discard discipline, persists to JSONL."""

    def __init__(self, experiments_path: Path, metrics_collector: MetricsCollector):
        self.experiments_path = experiments_path
        self.metrics = metrics_collector
        self.experiments: list[ExperimentResult] = []
        self._load()

    def _load(self):
        if not self.experiments_path.exists():
            return
        try:
            for line in self.experiments_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                data = json.loads(line)
                self.experiments.append(ExperimentResult(**{k: v for k, v in data.items() if k in ExperimentResult.__dataclass_fields__}))
        except (json.JSONDecodeError, TypeError, OSError):
            pass

    def start_experiment(self, experiment_id: str, hypothesis: str) -> dict[str, float]:
        """Capture baseline metrics before an experiment."""
        log_json("INFO", "experiment_started", details={"id": experiment_id, "hypothesis": hypothesis})
        return self.metrics.collect()

    def finish_experiment(self, experiment_id: str, hypothesis: str, change_description: str, metrics_before: dict[str, float], cycle_number: int = 0, duration: float = 0.0) -> ExperimentResult:
        """Finish experiment, measure improvement, decide keep/discard."""
        metrics_after = self.metrics.collect()
        improvement = {k: metrics_after.get(k, 0) - metrics_before.get(k, 0) for k in set(metrics_before) | set(metrics_after)}

        net = sum(improvement.values()) / max(len(improvement), 1)
        kept = net > 0
        reason = f"Net improvement: {net:+.4f}" if kept else f"Net regression: {net:+.4f} — discarding"

        result = ExperimentResult(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            change_description=change_description,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement=improvement,
            kept=kept,
            reason=reason,
            duration_seconds=duration,
            cycle_number=cycle_number,
        )
        self.experiments.append(result)
        self._persist(result)

        log_json("INFO", "experiment_finished", details={"id": experiment_id, "kept": kept, "net_improvement": net, "reason": reason})
        return result

    def _persist(self, result: ExperimentResult):
        with open(self.experiments_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def get_summary(self) -> dict[str, Any]:
        """Get experiment summary for CLI display."""
        if not self.experiments:
            return {"total": 0, "kept": 0, "discarded": 0, "net_improvement": 0.0}
        kept = [e for e in self.experiments if e.kept]
        discarded = [e for e in self.experiments if not e.kept]
        return {
            "total": len(self.experiments),
            "kept": len(kept),
            "discarded": len(discarded),
            "keep_rate": len(kept) / len(self.experiments),
            "net_improvement": sum(e.net_improvement for e in kept),
            "top_improvements": [{"id": e.experiment_id, "improvement": e.net_improvement} for e in sorted(kept, key=lambda x: x.net_improvement, reverse=True)[:5]],
        }
