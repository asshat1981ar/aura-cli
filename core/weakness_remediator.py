"""
Weakness Remediator.

Reads Brain.weaknesses, deduplicates via SHA-256 hash, scores by severity
and frequency, crafts targeted goal strings for the top-K unaddressed
weaknesses, and enqueues them in GoalQueue with ``source=weakness_remediator``.

Prevents re-queuing the same weakness by calling Brain.mark_weakness_queued().

Usage::

    from core.weakness_remediator import WeaknessRemediator
    remediator = WeaknessRemediator()
    result = remediator.run(brain, goal_queue, limit=3)
    # {"goals_generated": 2, "goals": [...]}
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from core.logging_utils import log_json

# Maps structured insight types to human-readable goal templates
_GOAL_TEMPLATES: Dict[str, str] = {
    "phase_failure": (
        "Investigate and fix repeated '{phase}' phase failures "
        "(failure rate: {failure_rate:.0%}, severity: {severity})"
    ),
    "low_value_skill": (
        "Evaluate or reconfigure '{skill}' skill — "
        "only {actionable_rate:.0%} actionable output rate after {runs} runs"
    ),
    "goal_type_struggling": (
        "Improve handling of '{goal_type}' goals — "
        "only {success_rate:.0%} success rate"
    ),
    "negative_sentiment": "Address recurring issue: {description}",
    "keyword": "Investigate flagged issue: {description}",
}

# Severity → numeric priority weight (higher = more urgent)
_SEVERITY_WEIGHT: Dict[str, float] = {
    "HIGH":   3.0,
    "MEDIUM": 2.0,
    "LOW":    1.0,
}


class WeaknessRemediator:
    """Turn Brain weaknesses into actionable GoalQueue entries."""

    def run(self, brain, goal_queue, limit: int = 3) -> Dict[str, Any]:
        """Process unqueued weaknesses and generate remediation goals.

        Args:
            brain:      Brain instance (provides recall_weaknesses, mark_weakness_queued).
            goal_queue: GoalQueue instance with an ``add()`` method.
            limit:      Maximum number of new goals to create per invocation.

        Returns:
            ``{"goals_generated": int, "goals": [str]}``
        """
        try:
            return self._run(brain, goal_queue, limit)
        except Exception as exc:
            log_json("ERROR", "weakness_remediator_failed", details={"error": str(exc)})
            return {"error": str(exc), "goals_generated": 0, "goals": []}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self, brain, goal_queue, limit: int) -> Dict[str, Any]:
        raw_weaknesses = brain.recall_weaknesses()
        if not raw_weaknesses:
            return {"goals_generated": 0, "goals": [], "skipped": True}

        queued_hashes = set(brain.recall_queued_weakness_hashes())

        # Parse, hash, score each weakness
        candidates: List[Tuple[float, str, Dict, str]] = []  # (score, hash, parsed, goal_text)

        for raw in raw_weaknesses:
            w_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
            if w_hash in queued_hashes:
                continue

            parsed = self._parse_weakness(raw)
            score = self._score_weakness(parsed)
            goal_text = self._craft_goal(parsed)
            if goal_text:
                candidates.append((score, w_hash, parsed, goal_text))

        # Sort by score descending, deduplicate goal text
        candidates.sort(key=lambda x: x[0], reverse=True)
        seen_goals: set = set()
        new_goals: List[str] = []

        for score, w_hash, parsed, goal_text in candidates:
            if len(new_goals) >= limit:
                break
            # Simple dedup: skip if very similar goal already queued this run
            goal_key = goal_text[:60].lower()
            if goal_key in seen_goals:
                continue
            seen_goals.add(goal_key)

            goal_queue.add(goal_text)
            brain.mark_weakness_queued(w_hash)
            new_goals.append(goal_text)
            log_json("INFO", "weakness_remediator_goal_queued",
                     details={"goal": goal_text, "score": round(score, 2),
                               "type": parsed.get("type", "unknown")})

        return {"goals_generated": len(new_goals), "goals": new_goals}

    def _parse_weakness(self, raw: str) -> Dict:
        """Try to parse a structured JSON weakness; fall back to raw string."""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        # Legacy text-based weakness from Brain.analyze_critique_for_weaknesses
        lower = raw.lower()
        if any(k in lower for k in ["fail", "error", "crash", "exception"]):
            return {"type": "keyword", "description": raw[:200]}
        return {"type": "negative_sentiment", "description": raw[:200]}

    def _score_weakness(self, w: Dict) -> float:
        """Higher score = more urgent to remediate."""
        base = _SEVERITY_WEIGHT.get(w.get("severity", "LOW"), 1.0)
        # Boost by failure rate / actionable rate where available
        if "failure_rate" in w:
            base += float(w["failure_rate"]) * 2.0
        if "actionable_rate" in w:
            base -= float(w["actionable_rate"])  # lower actionable → more urgent
        if "success_rate" in w:
            base += (1.0 - float(w["success_rate"])) * 2.0
        return base

    def _craft_goal(self, w: Dict) -> str:
        """Build a goal string from a parsed weakness dict."""
        w_type = w.get("type", "negative_sentiment")
        template = _GOAL_TEMPLATES.get(w_type, "Fix weakness: {description}")
        try:
            return template.format(**{**w, "description": w.get("description", str(w))[:120]})
        except (KeyError, ValueError):
            return f"Fix weakness ({w_type}): {str(w)[:120]}"
