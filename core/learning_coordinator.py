"""Autonomous Learning Loop coordinator (PRD-003).

LearningCoordinator is the single integration point that converts all
learning signals from a completed cycle into structured LearningArtifact
records and actionable goal-queue entries.

Signal sources consumed each cycle:
  1. ReflectorAgent.learnings  — per-cycle strings (reflection["learnings"])
  2. QualityTrendAnalyzer alerts — List[TrendAlert] from quality regressions
  3. DeepReflectionLoop reports  — every 5 cycles, persisted to "reflection_reports"
     memory tier; coordinator polls that tier so DeepReflectionLoop needs no changes.

Usage::

    coord = LearningCoordinator(memory_store)
    goals = coord.on_cycle_complete(entry, reflection, quality_alerts)
    # goals: List[str] — high-severity remediation goals to enqueue immediately

    # During Phase 10 (discover):
    backlog = coord.generate_backlog(limit=3)
    # backlog: List[str] — pending goals for the goal queue
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from core.learning_types import LearningArtifact
from core.logging_utils import log_json


class LearningCoordinator:
    """Converts cycle learning signals into LearningArtifacts and backlog goals.

    Attributes:
        ENQUEUE_SEVERITIES: Severity levels that trigger immediate goal enqueueing.
        MAX_GOALS_PER_CYCLE: Upper bound on goals returned per on_cycle_complete call.
    """

    ENQUEUE_SEVERITIES: frozenset = frozenset({"high", "critical"})
    MAX_GOALS_PER_CYCLE: int = 3

    def __init__(self, memory_store) -> None:
        """
        Args:
            memory_store: MemoryStore instance for persisting artifacts and
                          reading reflection_reports.
        """
        self.memory = memory_store
        self._last_reflection_ts: float = 0.0
        self._pending_goals: List[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def on_cycle_complete(
        self,
        cycle_entry: Dict[str, Any],
        reflection: Dict[str, Any],
        quality_alerts: List[Any],  # List[TrendAlert]
    ) -> List[str]:
        """Process a completed cycle.

        Converts all learning signals into LearningArtifacts, persists them,
        and returns high-severity remediation goals to enqueue immediately.

        Args:
            cycle_entry:    Orchestrator cycle entry dict (cycle_id, goal, goal_type, …).
            reflection:     Reflection phase output dict (may contain "learnings" list).
            quality_alerts: Quality alerts from QualityTrendAnalyzer.record_from_cycle().

        Returns:
            List of goal strings (at most MAX_GOALS_PER_CYCLE) ready to enqueue.
        """
        try:
            return self._process_cycle(cycle_entry, reflection, quality_alerts)
        except Exception as exc:
            log_json("WARN", "learning_coordinator_cycle_error", details={"error": str(exc)})
            return []

    def generate_backlog(self, limit: int = 3) -> List[str]:
        """Return pending high-severity goals accumulated between cycles.

        Called by Phase 10 (discover) to drain the backlog into the goal queue.
        Each call advances the cursor — goals are returned at most once.

        Args:
            limit: Maximum number of goals to return.

        Returns:
            List of goal strings (may be shorter than limit if queue is small).
        """
        result = self._pending_goals[:limit]
        self._pending_goals = self._pending_goals[limit:]
        return result

    def get_recent_artifacts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Read recent learning artifacts from the memory store.

        Args:
            limit: Maximum number of artifacts to return (most recent first).

        Returns:
            List of artifact dicts as stored by dataclasses.asdict().
        """
        return self.memory.query("learning_artifacts", limit=limit)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _process_cycle(
        self,
        cycle_entry: Dict[str, Any],
        reflection: Dict[str, Any],
        quality_alerts: List[Any],
    ) -> List[str]:
        cycle_id = cycle_entry.get("cycle_id", "")
        goal = cycle_entry.get("goal", "")
        goal_type = cycle_entry.get("goal_type", "unknown")

        artifacts: List[LearningArtifact] = []

        # 1. Per-cycle learnings from ReflectorAgent (low-severity strings)
        for learning in reflection.get("learnings", []):
            if isinstance(learning, str) and learning.strip():
                artifacts.append(
                    LearningArtifact(
                        cycle_id=cycle_id,
                        goal=goal,
                        goal_type=goal_type,
                        artifact_type="cycle_learning",
                        insight=learning.strip(),
                        evidence={"source": "reflector"},
                        severity="low",
                    )
                )

        # 2. Quality alerts → quality_regression artifacts
        for alert in quality_alerts:
            try:
                art = LearningArtifact(
                    cycle_id=cycle_id,
                    goal=goal,
                    goal_type=goal_type,
                    artifact_type="quality_regression",
                    insight=f"{alert.alert_type}: {alert.metric} = {alert.current_value:.2f}",
                    evidence={
                        "alert_type": alert.alert_type,
                        "metric": alert.metric,
                        "current_value": alert.current_value,
                        "previous_value": alert.previous_value,
                        "threshold": alert.threshold,
                    },
                    suggested_goal=alert.suggested_goal or None,
                    severity=alert.severity,
                )
                artifacts.append(art)
            except Exception as exc:
                log_json("WARN", "learning_coordinator_alert_conversion_failed", details={"error": str(exc)})

        # 3. Check for a new DeepReflectionLoop report (runs every 5 cycles)
        reflection_artifacts, reflection_goals = self._sync_reflection_reports(cycle_entry)
        artifacts.extend(reflection_artifacts)

        # 4. Collect immediate high-severity goals and mark acted_on BEFORE persisting
        # so stored records accurately reflect which goals were enqueued.
        immediate_goals: List[str] = []
        scheduled_goals = set()

        # Reflection reports may already have marked their artifacts as acted_on
        # upstream. Explicitly merge the returned reflection_goals here so they
        # are not lost just because those artifacts are no longer actionable.
        for reflection_goal in reflection_goals:
            if not reflection_goal or reflection_goal in scheduled_goals:
                continue

            matching_reflection_artifacts = [
                art
                for art in reflection_artifacts
                if art.suggested_goal == reflection_goal and art.severity in self.ENQUEUE_SEVERITIES
            ]
            if not matching_reflection_artifacts:
                continue

            if len(immediate_goals) < self.MAX_GOALS_PER_CYCLE:
                immediate_goals.append(reflection_goal)
                for art in matching_reflection_artifacts:
                    art.acted_on = True
            else:
                # Overflow: defer to backlog so Phase 10 can drain them later.
                # Ensure persisted artifacts reflect that they were deferred.
                self._pending_goals.append(reflection_goal)
                for art in matching_reflection_artifacts:
                    art.acted_on = False

            scheduled_goals.add(reflection_goal)

        for art in artifacts:
            if (
                art.is_actionable()
                and art.severity in self.ENQUEUE_SEVERITIES
                and art.suggested_goal
                and art.suggested_goal not in scheduled_goals
            ):
                if len(immediate_goals) < self.MAX_GOALS_PER_CYCLE:
                    immediate_goals.append(art.suggested_goal)
                    art.mark_acted_on()
                else:
                    # Overflow: defer to backlog so Phase 10 can drain them later
                    self._pending_goals.append(art.suggested_goal)
                scheduled_goals.add(art.suggested_goal)

        # 4b. Add reflection goals up to the cap, deduped against immediate_goals.
        # Reflection artifacts are already marked acted_on in _sync_reflection_reports,
        # so they are skipped by is_actionable() above; enqueue their goals here instead.
        _seen: set = set(immediate_goals)
        for g in reflection_goals:
            if g in _seen:
                continue
            _seen.add(g)
            if len(immediate_goals) < self.MAX_GOALS_PER_CYCLE:
                immediate_goals.append(g)
            else:
                self._pending_goals.append(g)

        # 5. Persist all artifacts (acted_on is now set correctly)
        for art in artifacts:
            try:
                self.memory.put("learning_artifacts", dataclasses.asdict(art))
            except Exception as exc:
                log_json("WARN", "learning_artifact_persist_failed", details={"error": str(exc)})

        log_json(
            "INFO",
            "learning_coordinator_cycle_complete",
            details={
                "cycle_id": cycle_id,
                "artifacts": len(artifacts),
                "immediate_goals": len(immediate_goals),
                "pending_backlog": len(self._pending_goals),
            },
        )
        return immediate_goals

    def _sync_reflection_reports(
        self,
        cycle_entry: Dict[str, Any],
    ) -> Tuple[List[LearningArtifact], List[str]]:
        """Poll memory for a new DeepReflectionLoop report.

        DeepReflectionLoop already persists its report to
        MemoryStore.put("reflection_reports", report) after every 5 cycles.
        This method reads the latest one and converts its insights to
        LearningArtifacts — no changes needed to DeepReflectionLoop itself.
        """
        artifacts: List[LearningArtifact] = []
        goals: List[str] = []

        try:
            reports = self.memory.query("reflection_reports", limit=1)
            if not reports:
                return artifacts, goals

            latest = reports[-1]
            ts = latest.get("timestamp", 0.0)
            if ts <= self._last_reflection_ts:
                return artifacts, goals  # already processed this report

            self._last_reflection_ts = ts
            cycle_id = cycle_entry.get("cycle_id", "")
            goal = cycle_entry.get("goal", "")
            goal_type = cycle_entry.get("goal_type", "unknown")

            severity_map = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}
            type_map = {
                "phase_failure": "phase_failure",
                "low_value_skill": "skill_weakness",
                "goal_type_struggling": "phase_failure",
            }

            for insight in latest.get("insights", []):
                raw_severity = str(insight.get("severity", "LOW")).upper()
                severity = severity_map.get(raw_severity, "low")
                artifact_type = type_map.get(insight.get("type", ""), "cycle_learning")
                suggested_goal: Optional[str] = None
                if severity in self.ENQUEUE_SEVERITIES:
                    suggested_goal = self._insight_to_goal(insight)

                art = LearningArtifact(
                    cycle_id=cycle_id,
                    goal=goal,
                    goal_type=goal_type,
                    artifact_type=artifact_type,
                    insight=insight.get("message", str(insight)),
                    evidence=insight,
                    suggested_goal=suggested_goal,
                    severity=severity,
                )
                artifacts.append(art)
                if suggested_goal and len(goals) < self.MAX_GOALS_PER_CYCLE:
                    goals.append(suggested_goal)
                    art.mark_acted_on()

        except Exception as exc:
            log_json(
                "WARN",
                "learning_coordinator_reflection_sync_failed",
                details={"error": str(exc)},
            )

        return artifacts, goals

    def _insight_to_goal(self, insight: Dict[str, Any]) -> Optional[str]:
        """Convert a DeepReflectionLoop insight dict to a remediation goal string."""
        itype = insight.get("type", "")
        if itype == "phase_failure":
            phase = insight.get("phase", "unknown")
            rate = insight.get("failure_rate", 0)
            return f"Investigate and fix high failure rate ({rate:.0%}) in {phase} phase"
        if itype == "goal_type_struggling":
            gt = insight.get("goal_type", "unknown")
            rate = insight.get("success_rate", 0)
            return f"Improve handling of '{gt}' goals (only {rate:.0%} success rate)"
        return insight.get("message") or None
