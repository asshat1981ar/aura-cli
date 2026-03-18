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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.logging_utils import log_json


# ---------------------------------------------------------------------------
# Subsystem probe — fast connectivity / availability checks
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    ok: bool
    latency_ms: float
    detail: str = ""
    error: str = ""


@dataclass
class HealthReport:
    timestamp: float
    all_ok: bool
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "all_ok": self.all_ok,
            "checks": [c.__dict__ for c in self.checks],
            "failed_count": len(self.failed),
        }


class SystemHealthProbe:
    """Lightweight connectivity and availability probe for AURA subsystems.

    Run at the start of every cycle and after failures.  All checks are
    independent; a failure in one never prevents the others from running.

    Args:
        brain: Brain instance (SQLite-backed memory).
        model: ModelAdapter instance (LLM endpoint wrapper).
        goal_queue: GoalQueue instance.
        memory_controller: MemoryController (global singleton).
        vector_store: VectorStore instance (optional).
    """

    def __init__(
        self,
        brain: Any = None,
        model: Any = None,
        goal_queue: Any = None,
        memory_controller: Any = None,
        vector_store: Any = None,
    ):
        self._brain = brain
        self._model = model
        self._goal_queue = goal_queue
        self._mc = memory_controller
        self._vs = vector_store

        self._checks: List[Callable[[], CheckResult]] = []
        if brain is not None:
            self._checks.append(self.check_brain_db)
        if memory_controller is not None:
            self._checks.append(self.check_memory_controller)
        if model is not None:
            self._checks.append(self.check_model_adapter)

        # Skill registry importability is meaningful even without a fully
        # constructed runtime, so always include it in the lightweight probe.
        self._checks.append(self.check_skill_registry)

        if goal_queue is not None:
            self._checks.append(self.check_goal_queue)
        if vector_store is not None:
            self._checks.append(self.check_vector_store)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _timed(self, name: str, fn: Callable[[], Optional[str]]) -> CheckResult:
        t0 = time.monotonic()
        try:
            detail = fn() or ""
            return CheckResult(
                name=name,
                ok=True,
                latency_ms=(time.monotonic() - t0) * 1000,
                detail=detail,
            )
        except Exception as exc:
            return CheckResult(
                name=name,
                ok=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_brain_db(self) -> CheckResult:
        """Verify SQLite connectivity and schema version."""
        def _probe():
            if self._brain is None:
                raise RuntimeError("Brain not configured")
            self._brain.db.execute("SELECT 1").fetchone()
            return f"schema_v{getattr(self._brain, 'SCHEMA_VERSION', '?')}"
        return self._timed("brain_db", _probe)

    def check_memory_controller(self) -> CheckResult:
        """Verify the memory controller has a persistent store attached."""
        def _probe():
            if self._mc is None:
                raise RuntimeError("MemoryController not configured")
            _ = self._mc.persistent_store
            return "ok"
        return self._timed("memory_controller", _probe)

    def check_model_adapter(self) -> CheckResult:
        """Verify the model adapter has a callable respond method."""
        def _probe():
            if self._model is None:
                raise RuntimeError("ModelAdapter not configured")
            if not callable(getattr(self._model, "respond", None)):
                raise RuntimeError("ModelAdapter missing .respond()")
            provider = getattr(self._model, "provider", "unknown")
            return f"provider={provider}"
        return self._timed("model_adapter", _probe)

    def check_skill_registry(self) -> CheckResult:
        """Verify all skills load without import errors."""
        def _probe():
            from agents.skills.registry import all_skills
            skills = all_skills()
            return f"{len(skills)} skills loaded"
        return self._timed("skill_registry", _probe)

    def check_goal_queue(self) -> CheckResult:
        """Verify the goal queue is accessible and readable."""
        def _probe():
            count = len(getattr(self._goal_queue, "queue", []))
            return f"{count} pending goals"
        return self._timed("goal_queue", _probe)

    def check_vector_store(self) -> CheckResult:
        """Verify the vector store index attribute is accessible."""
        def _probe():
            if self._vs is None:
                raise RuntimeError("VectorStore not configured")
            _ = getattr(self._vs, "index", None)
            return "index accessible"
        return self._timed("vector_store", _probe)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> HealthReport:
        """Run every registered check and return a :class:`HealthReport`."""
        results = [check() for check in self._checks]
        report = HealthReport(
            timestamp=time.time(),
            all_ok=all(r.ok for r in results) if results else True,
            checks=results,
        )
        level = "INFO" if report.all_ok else "WARN"
        log_json(level, "system_health_probe_complete", details=report.as_dict())
        return report

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

    TRIGGER_EVERY_N: int = 10  # overridden from config in __init__

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

        results = dispatch_skills("health_monitor", available, self.root, timeout=30.0)

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
