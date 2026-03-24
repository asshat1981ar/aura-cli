"""Quality trend analyzer: detect regressions and auto-enqueue fixes.

Tracks quality metrics (test count, syntax errors, import errors, coverage)
across cycles, detects degradation trends, and automatically enqueues
remediation goals when quality drops below thresholds.
"""
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

from core.investigate_test_drop import investigate_test_count_drop
from core.logging_utils import log_json

@dataclass
class QualitySnapshot:
    """Quality metrics from a single cycle."""
    cycle_id: str = ""
    goal: str = ""
    timestamp: float = field(default_factory=time.time)
    test_count: int = 0
    syntax_errors: int = 0
    import_errors: int = 0
    verify_status: str = ""  # pass/fail/skip
    changes_applied: int = 0
    cycle_duration_s: float = 0.0

    @property
    def health_score(self) -> float:
        """Overall health 0.0-1.0."""
        score = 0.5
        if self.verify_status == "pass":
            score += 0.3
        elif self.verify_status == "fail":
            score -= 0.3
        if self.syntax_errors == 0:
            score += 0.1
        else:
            score -= min(self.syntax_errors * 0.05, 0.3)
        if self.import_errors == 0:
            score += 0.1
        else:
            score -= min(self.import_errors * 0.05, 0.2)
        return max(0.0, min(1.0, score))

@dataclass
class TrendAlert:
    """An alert triggered by quality regression."""
    alert_type: str  # "regression", "degradation", "threshold_breach"
    metric: str
    current_value: float
    previous_value: float
    threshold: float
    severity: str  # "low", "medium", "high", "critical"
    suggested_goal: str = ""
    timestamp: float = field(default_factory=time.time)

class QualityTrendAnalyzer:
    """Tracks quality metrics across cycles and detects regressions."""

    def __init__(self, store_path: Path | None = None,
                 window_size: int = 20,
                 thresholds: dict | None = None):
        self.store_path = store_path or Path(__file__).parent.parent / "memory" / "quality_trends.json"
        self.window_size = window_size
        self.thresholds = thresholds or {
            "min_health_score": 0.4,
            "max_syntax_errors": 3,
            "max_import_errors": 2,
            "min_test_count_drop": -5,  # Alert if tests drop by this many
            "regression_window": 3,     # Alert if N consecutive failures
        }
        self.snapshots: deque[QualitySnapshot] = deque(maxlen=window_size * 5)
        self.alerts: list[TrendAlert] = []
        self._load()

    def record(self, snapshot: QualitySnapshot):
        """Record a quality snapshot and check for regressions."""
        self.snapshots.append(snapshot)
        new_alerts = self._analyze(snapshot)
        self.alerts.extend(new_alerts)
        self._save()

        if new_alerts:
            log_json("WARN", "quality_alerts_triggered",
                     details={"count": len(new_alerts),
                              "types": [a.alert_type for a in new_alerts]})
        return new_alerts

    def record_from_cycle(self, cycle_entry: dict) -> list[TrendAlert]:
        """Create snapshot from orchestrator cycle entry and record it."""
        phase_outputs = cycle_entry.get("phase_outputs", {})
        quality = phase_outputs.get("quality", {})
        verification = phase_outputs.get("verification", {})

        snapshot = QualitySnapshot(
            cycle_id=cycle_entry.get("cycle_id", ""),
            goal=cycle_entry.get("goal", ""),
            timestamp=cycle_entry.get("completed_at", time.time()),
            test_count=quality.get("test_count", 0),
            syntax_errors=len(quality.get("syntax_errors", [])),
            import_errors=len(quality.get("import_errors", [])),
            verify_status=verification.get("status", "skip"),
            changes_applied=len(phase_outputs.get("apply_result", {}).get("applied", [])),
            cycle_duration_s=cycle_entry.get("duration_s", 0),
        )
        return self.record(snapshot)

    def get_trend(self, metric: str = "health_score", window: int = 0) -> list[float]:
        """Get trend data for a metric."""
        n = window or self.window_size
        recent = list(self.snapshots)[-n:]
        return [getattr(s, metric, s.health_score) for s in recent]

    def get_summary(self) -> dict:
        """Get quality trend summary for CLI display."""
        if not self.snapshots:
            return {"total_cycles": 0, "health": "unknown"}

        recent = list(self.snapshots)[-self.window_size:]
        health_scores = [s.health_score for s in recent]
        avg_health = sum(health_scores) / len(health_scores)

        return {
            "total_cycles": len(self.snapshots),
            "window": len(recent),
            "avg_health": round(avg_health, 3),
            "current_health": round(recent[-1].health_score, 3),
            "trend": "improving" if len(health_scores) > 1 and health_scores[-1] > health_scores[0]
                     else "declining" if len(health_scores) > 1 and health_scores[-1] < health_scores[0]
                     else "stable",
            "total_alerts": len(self.alerts),
            "recent_alerts": [{"type": a.alert_type, "metric": a.metric,
                             "severity": a.severity} for a in self.alerts[-5:]],
            "test_count_current": recent[-1].test_count,
            "syntax_errors_current": recent[-1].syntax_errors,
        }

    def get_remediation_goals(self) -> list[str]:
        """Generate remediation goals for active alerts."""
        goals = []
        for alert in self.alerts[-10:]:
            if alert.suggested_goal:
                goals.append(alert.suggested_goal)
        return list(set(goals))

    def _analyze(self, snapshot: QualitySnapshot) -> list[TrendAlert]:
        """Analyze snapshot for regressions and threshold breaches."""
        alerts = []

        # Health score threshold
        if snapshot.health_score < self.thresholds["min_health_score"]:
            alerts.append(TrendAlert(
                alert_type="threshold_breach", metric="health_score",
                current_value=snapshot.health_score, previous_value=0,
                threshold=self.thresholds["min_health_score"],
                severity="high",
                suggested_goal=f"Fix quality regression: health score {snapshot.health_score:.2f} below threshold",
            ))

        # Syntax errors threshold
        if snapshot.syntax_errors > self.thresholds["max_syntax_errors"]:
            alerts.append(TrendAlert(
                alert_type="threshold_breach", metric="syntax_errors",
                current_value=snapshot.syntax_errors, previous_value=0,
                threshold=self.thresholds["max_syntax_errors"],
                severity="critical",
                suggested_goal=f"Fix {snapshot.syntax_errors} syntax errors introduced in recent changes",
            ))

        # Test count regression
        if len(self.snapshots) >= 2:
            prev = list(self.snapshots)[-2]
            test_delta = snapshot.test_count - prev.test_count
            if test_delta < self.thresholds["min_test_count_drop"]:
                investigation = investigate_test_count_drop(
                    prev.test_count,
                    snapshot.test_count,
                    goal=snapshot.goal or None,
                    verification={"status": snapshot.verify_status},
                )
                alerts.append(TrendAlert(
                    alert_type="regression", metric="test_count",
                    current_value=snapshot.test_count, previous_value=prev.test_count,
                    threshold=self.thresholds["min_test_count_drop"],
                    severity=investigation["severity"],
                    suggested_goal=investigation["suggested_goal"],
                ))

        # Consecutive failure detection
        recent = list(self.snapshots)[-self.thresholds["regression_window"]:]
        if len(recent) >= self.thresholds["regression_window"]:
            if all(s.verify_status == "fail" for s in recent):
                alerts.append(TrendAlert(
                    alert_type="degradation", metric="verify_status",
                    current_value=0, previous_value=1,
                    threshold=self.thresholds["regression_window"],
                    severity="critical",
                    suggested_goal=f"Critical: {len(recent)} consecutive verification failures — investigate root cause",
                ))

        return alerts

    def _load(self):
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text())
            for s in data.get("snapshots", []):
                self.snapshots.append(QualitySnapshot(**{
                    k: v for k, v in s.items() if k in QualitySnapshot.__dataclass_fields__
                }))
            for a in data.get("alerts", []):
                self.alerts.append(TrendAlert(**{
                    k: v for k, v in a.items() if k in TrendAlert.__dataclass_fields__
                }))
        except (json.JSONDecodeError, TypeError, OSError):
            pass

    def _save(self):
        data = {
            "snapshots": [asdict(s) for s in self.snapshots],
            "alerts": [asdict(a) for a in self.alerts[-50:]],
            "updated_at": time.time(),
        }
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass
