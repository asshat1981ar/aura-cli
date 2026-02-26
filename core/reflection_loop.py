"""
Deep Self-Reflection Loop.

After every N completed cycles (default 5), reads the full decision log,
computes phase failure rates and skill signal rates, extracts actionable
insights, and writes them to Brain.weaknesses.  Also emits a structured
``reflection_report`` to MemoryStore for downstream consumers.

Usage::

    from core.reflection_loop import DeepReflectionLoop
    loop = DeepReflectionLoop(memory_store, brain)
    loop.on_cycle_complete(cycle_entry)   # call after every run_cycle()
    report = loop.run()                   # or trigger manually
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from core.logging_utils import log_json


class DeepReflectionLoop:
    """Analyse historical cycle data and surface actionable weaknesses.

    Attributes:
        TRIGGER_EVERY_N: How many completed cycles between automatic runs.
        MIN_CYCLES:      Minimum history length before analysis is meaningful.
    """

    TRIGGER_EVERY_N: int = 5
    MIN_CYCLES: int = 3

    # Thresholds that classify a pattern as a weakness
    PHASE_FAIL_RATE_WARN: float = 0.40   # 40%+ failure rate → MEDIUM
    PHASE_FAIL_RATE_HIGH: float = 0.65   # 65%+ failure rate → HIGH
    SKILL_LOW_SIGNAL_RATE: float = 0.25  # <25% actionable hits → LOW VALUE
    SKILL_MIN_RUNS: int = 3              # require at least 3 runs before judging

    def __init__(self, memory_store, brain):
        self.memory = memory_store
        self.brain = brain
        self._cycle_count: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def on_cycle_complete(self, cycle_entry: Dict[str, Any]) -> None:
        """Call this after every ``LoopOrchestrator.run_cycle()`` returns."""
        self._cycle_count += 1
        if self._cycle_count % self.TRIGGER_EVERY_N == 0:
            self.run()

    def run(self) -> Dict[str, Any]:
        """Analyse history and write insights.  Never raises."""
        try:
            return self._run()
        except Exception as exc:
            log_json("ERROR", "reflection_loop_failed", details={"error": str(exc)})
            return {"error": str(exc)}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self) -> Dict[str, Any]:
        history = self.memory.read_log(limit=100)
        if len(history) < self.MIN_CYCLES:
            log_json("INFO", "reflection_loop_skipped_insufficient_history",
                     details={"entries": len(history)})
            return {"skipped": True, "reason": "insufficient_history"}

        phase_stats: Dict[str, Dict] = {}   # phase → {total, failed}
        skill_stats: Dict[str, Dict] = {}   # skill → {ran, actionable}
        goal_type_outcomes: Dict[str, Dict] = {}  # goal_type → {total, pass}

        for entry in history:
            po = entry.get("phase_outputs", {})
            goal_type = entry.get("goal_type", "unknown")

            # Goal-type outcome tracking
            gt = goal_type_outcomes.setdefault(goal_type, {"total": 0, "pass": 0})
            gt["total"] += 1
            verif = po.get("verification", {})
            if isinstance(verif, dict) and verif.get("status") in ("pass", "skip"):
                gt["pass"] += 1

            # Phase failure rates
            for phase, output in po.items():
                if phase.startswith("_") or phase in ("skill_context", "apply_result"):
                    continue
                ps = phase_stats.setdefault(phase, {"total": 0, "failed": 0})
                ps["total"] += 1
                if isinstance(output, dict) and output.get("status") == "fail":
                    ps["failed"] += 1

            # Skill hit rates
            for skill_name, result in po.get("skill_context", {}).items():
                ss = skill_stats.setdefault(skill_name, {"ran": 0, "actionable": 0})
                ss["ran"] += 1
                if isinstance(result, dict) and "error" not in result:
                    has_content = any(
                        v for k, v in result.items()
                        if v not in (None, [], {}, "", 0)
                    )
                    if has_content:
                        ss["actionable"] += 1

        insights = self._extract_insights(phase_stats, skill_stats, goal_type_outcomes)
        self._write_to_brain(insights)

        report = {
            "timestamp": time.time(),
            "cycles_analyzed": len(history),
            "cycle_count": self._cycle_count,
            "phase_stats": phase_stats,
            "skill_stats": skill_stats,
            "goal_type_outcomes": goal_type_outcomes,
            "insights": insights,
        }
        self.memory.put("reflection_reports", report)
        log_json("INFO", "reflection_loop_complete",
                 details={"insights": len(insights), "cycles_analyzed": len(history)})
        return report

    def _extract_insights(
        self,
        phase_stats: Dict,
        skill_stats: Dict,
        goal_type_outcomes: Dict,
    ) -> List[Dict]:
        insights: List[Dict] = []

        # Phase failure rate insights
        for phase, s in phase_stats.items():
            if s["total"] == 0:
                continue
            rate = s["failed"] / s["total"]
            if rate >= self.PHASE_FAIL_RATE_HIGH:
                insights.append({
                    "type": "phase_failure",
                    "phase": phase,
                    "failure_rate": round(rate, 3),
                    "severity": "HIGH",
                    "message": f"Phase '{phase}' failing {rate:.0%} of the time",
                })
            elif rate >= self.PHASE_FAIL_RATE_WARN:
                insights.append({
                    "type": "phase_failure",
                    "phase": phase,
                    "failure_rate": round(rate, 3),
                    "severity": "MEDIUM",
                    "message": f"Phase '{phase}' failing {rate:.0%} of the time",
                })

        # Skill low-signal insights
        for skill, s in skill_stats.items():
            if s["ran"] < self.SKILL_MIN_RUNS:
                continue
            rate = s["actionable"] / s["ran"]
            if rate < self.SKILL_LOW_SIGNAL_RATE:
                insights.append({
                    "type": "low_value_skill",
                    "skill": skill,
                    "actionable_rate": round(rate, 3),
                    "runs": s["ran"],
                    "severity": "LOW",
                    "message": f"Skill '{skill}' produces actionable output only {rate:.0%} of the time",
                })

        # Goal-type low success rate
        for gt, s in goal_type_outcomes.items():
            if s["total"] < 3:
                continue
            success_rate = s["pass"] / s["total"]
            if success_rate < 0.40:
                insights.append({
                    "type": "goal_type_struggling",
                    "goal_type": gt,
                    "success_rate": round(success_rate, 3),
                    "severity": "HIGH",
                    "message": f"Goal type '{gt}' succeeds only {success_rate:.0%} of the time",
                })

        return insights

    def _write_to_brain(self, insights: List[Dict]) -> None:
        import json
        for insight in insights:
            self.brain.add_weakness(json.dumps(insight))
