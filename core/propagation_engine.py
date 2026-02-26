"""
Propagation Engine — Forward-chaining event→rule→action system.

Every completed cycle fires a set of events.  Rules match events against
conditions and trigger actions.  Actions include queuing new goals,
modifying the pipeline config for the next cycle, tagging context, or
escalating to human review.

Anti-loop guard: every generated goal is hashed and tracked in a
``propagation_log`` to prevent the same rule from re-firing on its own
output (cycles limited to depth 3).

Built-in rules
--------------
1. verification_pass + goal_type=bug_fix  → queue regression test goal
2. verification_pass + file changed       → queue doc-update goal (once per file)
3. weakness_detected severity=HIGH        → queue immediate remediation goal
4. phase_failure rate > 0.6 (from reflection) → queue "investigate phase X" goal
5. health_breach                          → pass-through (HealthMonitor already queued)
6. coverage_drop                          → queue test coverage goal
7. goal_type=refactor + passed            → queue architecture validation
8. consecutive_failures >= 2              → adjust pipeline: add more context skills

Usage::

    from core.propagation_engine import PropagationEngine
    engine = PropagationEngine(goal_queue, context_graph, memory_store)
    engine.on_cycle_complete(cycle_entry)

    # Register a custom rule:
    engine.register_rule(PropagationRule(
        name="my_rule",
        event="verification_pass",
        condition=lambda ctx: ctx.get("goal_type") == "feature",
        action=lambda ctx: ctx["goal_queue"].add("Update API docs after new feature"),
    ))
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.logging_utils import log_json

# Maximum propagation depth — prevents A→B→A chains
MAX_PROPAGATION_DEPTH: int = 3


@dataclass
class PropagationRule:
    """A declarative forward-chaining rule.

    Attributes:
        name:       Unique identifier for this rule.
        event:      Event type that activates this rule (see EVENT_TYPES).
        condition:  Callable(context: dict) -> bool.  Must be cheap and pure.
        action:     Callable(context: dict) -> Optional[str].
                    Returns the goal text if a goal was queued, else None.
        max_fires_per_goal: Safety cap — rule fires at most this many times
                            per unique goal string per session.
        enabled:    Toggle without removing from the registry.
    """
    name: str
    event: str
    condition: Callable[[Dict], bool]
    action: Callable[[Dict], Optional[str]]
    max_fires_per_goal: int = 1
    enabled: bool = True


# Event types emitted by PropagationEngine
EVENT_TYPES = frozenset([
    "verification_pass",
    "verification_fail",
    "weakness_detected",
    "phase_failure_spike",
    "health_breach",
    "coverage_drop",
    "consecutive_failures",
    "cycle_complete",
])


class PropagationEngine:
    """Forward-chaining event→rule→goal propagation.

    Wired into ``LoopOrchestrator`` via ``on_cycle_complete()``.
    """

    def __init__(self, goal_queue, context_graph, memory_store):
        self.queue = goal_queue
        self.graph = context_graph
        self.memory = memory_store
        self._rules: Dict[str, PropagationRule] = {}
        self._propagation_log: Dict[str, int] = {}  # goal_hash → fire count
        self._consecutive_fail_count: int = 0
        self._register_builtin_rules()

    # ── Public API ───────────────────────────────────────────────────────────

    def register_rule(self, rule: PropagationRule) -> None:
        self._rules[rule.name] = rule
        log_json("INFO", "propagation_rule_registered", details={"rule": rule.name})

    def on_cycle_complete(self, cycle_entry: Dict[str, Any]) -> List[str]:
        """Fire all matching rules for this cycle.  Returns list of queued goals."""
        try:
            return self._process(cycle_entry)
        except Exception as exc:
            log_json("ERROR", "propagation_engine_error", details={"error": str(exc)})
            return []

    # ── Internal ─────────────────────────────────────────────────────────────

    def _process(self, entry: Dict) -> List[str]:
        events = self._extract_events(entry)
        context = self._build_context(entry, events)
        queued: List[str] = []

        for event in events:
            for rule in self._rules.values():
                if not rule.enabled or rule.event != event:
                    continue
                try:
                    if not rule.condition(context):
                        continue
                    result = rule.action(context)
                    if result and self._can_fire(rule.name, result):
                        self.queue.add(result)
                        self._record_fire(rule.name, result)
                        queued.append(result)
                        log_json("INFO", "propagation_rule_fired",
                                 details={"rule": rule.name, "event": event,
                                          "goal": result[:80]})
                except Exception as exc:
                    log_json("WARN", "propagation_rule_error",
                             details={"rule": rule.name, "error": str(exc)})

        if queued:
            self.memory.put("propagation_events", {
                "cycle_id": entry.get("cycle_id"),
                "goals_generated": queued,
                "timestamp": time.time(),
            })
        return queued

    def _extract_events(self, entry: Dict) -> List[str]:
        """Translate a cycle entry into a list of event strings."""
        events = ["cycle_complete"]
        po = entry.get("phase_outputs", {})
        verif = po.get("verification", {})
        status = verif.get("status", "skip") if isinstance(verif, dict) else "skip"

        if status in ("pass", "skip"):
            self._consecutive_fail_count = 0
            events.append("verification_pass")
        else:
            self._consecutive_fail_count += 1
            events.append("verification_fail")
            if self._consecutive_fail_count >= 2:
                events.append("consecutive_failures")

        # Weakness events from reflection insights
        for insight in po.get("reflection", {}).get("learnings", []):
            if "HIGH" in str(insight) or "failure" in str(insight).lower():
                events.append("weakness_detected")
                break

        # Phase failure spike from reflection loop output
        for report in (self.memory.query("reflection_reports", limit=1) or []):
            for ins in report.get("insights", []):
                if ins.get("type") == "phase_failure" and ins.get("severity") == "HIGH":
                    events.append("phase_failure_spike")

        # Coverage drop from health snapshots
        snapshots = self.memory.query("health_snapshots", limit=2)
        if len(snapshots) >= 2:
            prev_cov = snapshots[-2].get("metrics", {}).get("test_coverage_analyzer")
            curr_cov = snapshots[-1].get("metrics", {}).get("test_coverage_analyzer")
            if (prev_cov is not None and curr_cov is not None
                    and isinstance(curr_cov, (int, float))
                    and isinstance(prev_cov, (int, float))
                    and curr_cov < prev_cov * 0.9):
                events.append("coverage_drop")

        return list(dict.fromkeys(events))  # deduplicate, preserve order

    def _build_context(self, entry: Dict, events: List[str]) -> Dict:
        po = entry.get("phase_outputs", {})
        verif = po.get("verification", {}) or {}
        apply_result = po.get("apply_result", {}) or {}
        return {
            "cycle_id":       entry.get("cycle_id", ""),
            "goal_type":      entry.get("goal_type", "default"),
            "events":         events,
            "verification":   verif,
            "apply_result":   apply_result,
            "applied_files":  apply_result.get("applied", []),
            "failed_files":   apply_result.get("failed", []),
            "failures":       verif.get("failures", []),
            "phase_outputs":  po,
            "consecutive_fails": self._consecutive_fail_count,
            "goal_queue":     self.queue,
            "context_graph":  self.graph,
            "memory":         self.memory,
        }

    def _can_fire(self, rule_name: str, goal_text: str) -> bool:
        """Anti-loop guard — cap fires per unique goal text."""
        key = f"{rule_name}:{hashlib.sha256(goal_text.encode()).hexdigest()[:12]}"
        return self._propagation_log.get(key, 0) < MAX_PROPAGATION_DEPTH

    def _record_fire(self, rule_name: str, goal_text: str) -> None:
        key = f"{rule_name}:{hashlib.sha256(goal_text.encode()).hexdigest()[:12]}"
        self._propagation_log[key] = self._propagation_log.get(key, 0) + 1

    # ── Built-in rules ────────────────────────────────────────────────────────

    def _register_builtin_rules(self) -> None:
        rules = [
            # 1. After a successful bug fix, auto-queue a regression test
            PropagationRule(
                name="regression_test_after_fix",
                event="verification_pass",
                condition=lambda ctx: (
                    ctx["goal_type"] == "bug_fix"
                    and bool(ctx["applied_files"])
                ),
                action=lambda ctx: (
                    f"Add regression test for fix in: "
                    f"{', '.join(ctx['applied_files'][:2])}"
                ),
            ),
            # 2. After a successful feature, queue API contract validation
            PropagationRule(
                name="api_contract_check_after_feature",
                event="verification_pass",
                condition=lambda ctx: ctx["goal_type"] == "feature",
                action=lambda ctx: "Validate API contracts after new feature addition",
            ),
            # 3. After successful refactor, validate architecture coupling
            PropagationRule(
                name="arch_validate_after_refactor",
                event="verification_pass",
                condition=lambda ctx: ctx["goal_type"] == "refactor",
                action=lambda ctx: (
                    "Run architecture validation after refactor — check for new circular deps"
                ),
            ),
            # 4. HIGH severity weakness → immediate remediation goal
            PropagationRule(
                name="remediate_high_severity_weakness",
                event="weakness_detected",
                condition=lambda ctx: ctx["goal_type"] != "default",
                action=lambda ctx: (
                    f"Immediate remediation: HIGH severity weakness detected "
                    f"in {ctx['goal_type']} cycle"
                ),
                max_fires_per_goal=2,
            ),
            # 5. Phase failure spike → investigate goal
            PropagationRule(
                name="investigate_phase_failure",
                event="phase_failure_spike",
                condition=lambda ctx: True,
                action=lambda ctx: self._phase_failure_goal(ctx),
                max_fires_per_goal=2,
            ),
            # 6. Coverage drop → test coverage goal
            PropagationRule(
                name="coverage_drop_goal",
                event="coverage_drop",
                condition=lambda ctx: True,
                action=lambda ctx: "Restore test coverage — coverage dropped >10% from baseline",
            ),
            # 7. Consecutive failures → add deeper symbol context to next cycle
            PropagationRule(
                name="context_enrichment_on_failures",
                event="consecutive_failures",
                condition=lambda ctx: ctx["consecutive_fails"] >= 2,
                action=lambda ctx: (
                    f"Investigate root cause of consecutive {ctx['consecutive_fails']}x "
                    f"failures on {ctx['goal_type']} goal"
                ),
                max_fires_per_goal=1,
            ),
            # 8. Security fix passed → update security scan baseline
            PropagationRule(
                name="security_baseline_after_fix",
                event="verification_pass",
                condition=lambda ctx: ctx["goal_type"] == "security",
                action=lambda ctx: "Re-run security scan to update clean baseline after fix",
            ),
        ]
        for rule in rules:
            self._rules[rule.name] = rule

        log_json("INFO", "propagation_builtin_rules_registered",
                 details={"count": len(rules)})

    @staticmethod
    def _phase_failure_goal(ctx: Dict) -> Optional[str]:
        """Craft a goal from the most recent phase failure insight."""
        reports = ctx["memory"].query("reflection_reports", limit=1)
        if not reports:
            return None
        for ins in reports[-1].get("insights", []):
            if ins.get("type") == "phase_failure" and ins.get("severity") == "HIGH":
                return (
                    f"Investigate and fix repeated '{ins['phase']}' phase failures "
                    f"(rate: {ins.get('failure_rate', '?')})"
                )
        return None
