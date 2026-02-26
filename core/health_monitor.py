"""
Codebase Health Monitor.

Periodically runs a battery of analysis skills against the project root,
stores time-series health snapshots, detects quality drift vs the previous
snapshot, and auto-generates remediation goals when metrics breach thresholds.

Usage::

    from core.health_monitor import HealthMonitor
    monitor = HealthMonitor(skills, goal_queue, memory_store, project_root=".")
    monitor.on_cycle_complete(cycle_entry)   # auto-trigger every N cycles
    report = monitor.run_scan()              # or trigger manually
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json

# Skills to run for the health scan
HEALTH_SKILLS: List[str] = [
    "complexity_scorer",
    "tech_debt_quantifier",
    "test_coverage_analyzer",
    "security_scanner",
    "architecture_validator",
]

# Per-skill thresholds and auto-goal templates.
# Each entry has either ``max`` (value must not exceed) or ``min`` (must not go below).
# ``drift`` is the relative change fraction that triggers a drift alert.
THRESHOLDS: Dict[str, Dict] = {
    "complexity_scorer": {
        "field": "high_risk_count",
        "max": 10,
        "drift": 0.20,
        "goal_template": "Reduce cyclomatic complexity — {value} high-risk functions detected",
    },
    "tech_debt_quantifier": {
        "field": "debt_score",
        "max": 60,
        "drift": 0.15,
        "goal_template": "Address accumulated tech debt (score: {value})",
    },
    "test_coverage_analyzer": {
        "field": "coverage_pct",
        "min": 55.0,
        "drift": -0.10,
        "goal_template": "Improve test coverage from {value}% toward 75% target",
    },
    "security_scanner": {
        "field": "critical_count",
        "max": 0,
        "drift": 0.0,
        "goal_template": "Fix {value} critical security finding(s) detected by scanner",
    },
    "architecture_validator": {
        "field": "coupling_score",
        "max": 0.75,
        "drift": 0.15,
        "goal_template": "Reduce architectural coupling (score: {value:.2f})",
    },
}


class HealthMonitor:
    """Run health skill battery, detect drift, and auto-queue remediation goals.

    Attributes:
        TRIGGER_EVERY_N: How many completed cycles between automatic scans.
    """

    TRIGGER_EVERY_N: int = 10

    def __init__(
        self,
        skills: Dict[str, Any],
        goal_queue,
        memory_store,
        project_root: str = ".",
    ):
        self.skills = skills
        self.queue = goal_queue
        self.memory = memory_store
        self.root = str(project_root)
        self._cycle_count: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def on_cycle_complete(self, cycle_entry: Dict[str, Any]) -> None:
        """Call after every completed cycle.  Triggers scan every N cycles."""
        self._cycle_count += 1
        if self._cycle_count % self.TRIGGER_EVERY_N == 0:
            self.run_scan()

    def run_scan(self) -> Dict[str, Any]:
        """Run the full health battery.  Never raises."""
        try:
            return self._run_scan()
        except Exception as exc:
            log_json("ERROR", "health_monitor_failed", details={"error": str(exc)})
            return {"error": str(exc)}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_scan(self) -> Dict[str, Any]:
        from core.skill_dispatcher import dispatch_skills

        available = {k: self.skills[k] for k in HEALTH_SKILLS if k in self.skills}
        if not available:
            log_json("WARN", "health_monitor_no_skills")
            return {"skipped": True, "reason": "no_health_skills_available"}

        log_json("INFO", "health_monitor_scan_start",
                 details={"skills": list(available.keys()), "root": self.root})

        results = dispatch_skills("default", available, self.root, timeout=30.0)

        # Load previous snapshot for drift comparison
        prev_snapshots = self.memory.query("health_snapshots", limit=1)
        prev: Dict = prev_snapshots[-1] if prev_snapshots else {}
        prev_metrics: Dict = prev.get("metrics", {})

        auto_goals: List[str] = []
        metrics: Dict[str, Any] = {}
        breaches: List[Dict] = []

        for skill_name, cfg in THRESHOLDS.items():
            result = results.get(skill_name, {})
            if "error" in result or not result:
                continue
            value = result.get(cfg["field"])
            if value is None:
                continue

            metrics[skill_name] = value
            breach_reason = self._check_breach(skill_name, cfg, value, prev_metrics)
            if breach_reason:
                breaches.append({"skill": skill_name, "value": value, "reason": breach_reason})
                try:
                    goal_text = cfg["goal_template"].format(value=value)
                except Exception:
                    goal_text = f"Health issue in {skill_name}: {value}"
                self.queue.add(goal_text)
                auto_goals.append(goal_text)
                log_json("INFO", "health_monitor_goal_generated",
                         details={"goal": goal_text, "reason": breach_reason})

        snapshot = {
            "timestamp": time.time(),
            "metrics": metrics,
            "auto_goals_count": len(auto_goals),
        }
        self.memory.put("health_snapshots", snapshot)

        report = {
            "snapshot": snapshot,
            "breaches": breaches,
            "auto_goals": auto_goals,
            "skills_ran": list(results.keys()),
        }
        log_json("INFO", "health_monitor_scan_complete",
                 details={"metrics": metrics, "breaches": len(breaches),
                          "auto_goals": len(auto_goals)})
        return report

    def _check_breach(
        self,
        skill_name: str,
        cfg: Dict,
        value: Any,
        prev_metrics: Dict,
    ) -> Optional[str]:
        """Return a breach reason string, or None if within bounds."""
        # Absolute threshold check
        if "max" in cfg and isinstance(value, (int, float)) and value > cfg["max"]:
            return f"absolute_max_exceeded (value={value}, max={cfg['max']})"
        if "min" in cfg and isinstance(value, (int, float)) and value < cfg["min"]:
            return f"absolute_min_breached (value={value}, min={cfg['min']})"

        # Drift check vs previous snapshot
        prev_value = prev_metrics.get(skill_name)
        if prev_value is not None and isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
            denom = abs(prev_value) or 1.0
            drift = (value - prev_value) / denom
            drift_threshold = cfg.get("drift", 0.15)
            if drift_threshold >= 0 and drift > drift_threshold:
                return f"drift_up (prev={prev_value}, now={value}, drift={drift:.2%})"
            if drift_threshold < 0 and drift < drift_threshold:
                return f"drift_down (prev={prev_value}, now={value}, drift={drift:.2%})"

        return None
