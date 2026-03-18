"""Flow-of-Thought arbitration for bounded self-propagating development."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Iterable, List

_PRIORITY_ORDER = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


def _coerce_float(value: Any, default: float = 0.5) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _stable_id(*parts: str) -> str:
    payload = "|".join(str(part or "").strip() for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _priority_for(raw: Dict[str, Any], risk_level: str | None) -> str:
    priority = str(raw.get("priority") or "").strip().lower()
    if priority in _PRIORITY_ORDER:
        return priority

    severity = str(raw.get("severity") or "").strip().upper()
    if severity in {"CRITICAL", "HIGH"}:
        return "high"
    if severity == "MEDIUM":
        return "medium"

    if risk_level in {"critical", "high"}:
        return "high"
    if risk_level == "medium":
        return "medium"
    return "low"


def _listify(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


class FoTArbiter:
    """Deduplicate, rank, and optionally queue bounded follow-up candidates."""

    def __init__(
        self,
        *,
        goal_queue: Any = None,
        max_selected: int = 5,
        max_auto_queue: int = 2,
        goal_validator: Callable[[str], tuple[bool, str | None]] | None = None,
    ) -> None:
        self.goal_queue = goal_queue
        self.max_selected = max(1, int(max_selected or 1))
        self.max_auto_queue = max(0, int(max_auto_queue or 0))
        self.goal_validator = goal_validator

    def normalize_candidate(self, raw: Any, *, source: str | None = None) -> Dict[str, Any] | None:
        if isinstance(raw, str):
            raw = {"recommended_goal": raw}
        if not isinstance(raw, dict):
            return None

        development_context = raw.get("development_context", {})
        if not isinstance(development_context, dict):
            development_context = {}

        source_name = str(source or raw.get("source") or "unknown").strip() or "unknown"
        recommended_goal = str(
            raw.get("recommended_goal")
            or raw.get("goal")
            or raw.get("suggested_goal")
            or ""
        ).strip()
        summary = str(
            raw.get("summary")
            or raw.get("message")
            or raw.get("description")
            or recommended_goal
            or ""
        ).strip()
        if not summary and not recommended_goal:
            return None

        risk_level = str(raw.get("risk_level") or raw.get("risk") or "").strip().lower() or None
        fitness_snapshot = raw.get("fitness_snapshot", {})
        if not isinstance(fitness_snapshot, dict):
            fitness_snapshot = {}
        confidence = _coerce_float(
            raw.get("confidence", fitness_snapshot.get("score")),
            default=0.5,
        )
        target_subsystem = str(
            raw.get("target_subsystem")
            or development_context.get("target_subsystem")
            or ""
        ).strip() or None

        requires_human_review = bool(
            raw.get("requires_human_review")
            or raw.get("requires_operator_review")
        )
        beads_recheck_required = bool(raw.get("beads_recheck_required"))
        if target_subsystem == "recursive_self_improvement":
            beads_recheck_required = True
            if risk_level in {"high", "critical"}:
                requires_human_review = True

        evidence = [str(item) for item in _listify(raw.get("evidence")) if str(item).strip()]
        evidence.extend(
            f"cycle:{item}"
            for item in _listify(raw.get("source_cycles"))
            if str(item).strip()
        )
        if raw.get("rule_name"):
            evidence.append(f"rule:{raw['rule_name']}")
        if raw.get("event"):
            evidence.append(f"event:{raw['event']}")

        queueable = bool(raw.get("queueable", bool(recommended_goal)))
        queue_block_reason = raw.get("queue_block_reason")

        return {
            "candidate_id": str(
                raw.get("candidate_id")
                or raw.get("proposal_id")
                or raw.get("hash")
                or f"fot-{_stable_id(source_name, recommended_goal or summary)}"
            ),
            "source": source_name,
            "summary": summary,
            "recommended_goal": recommended_goal,
            "target_subsystem": target_subsystem,
            "evidence": evidence,
            "priority": _priority_for(raw, risk_level),
            "confidence": confidence,
            "queueable": queueable,
            "queue_block_reason": queue_block_reason,
            "requires_human_review": requires_human_review,
            "beads_recheck_required": beads_recheck_required,
            "recommended_actions": [
                str(item)
                for item in _listify(raw.get("recommended_actions") or raw.get("actions"))
                if str(item).strip()
            ],
            "proposal_id": raw.get("proposal_id"),
            "source_cycles": [str(item) for item in _listify(raw.get("source_cycles")) if str(item).strip()],
            "fitness_snapshot": dict(fitness_snapshot),
            "risk_level": risk_level,
            "development_context": development_context,
        }

    def _ranking_key(self, candidate: Dict[str, Any]) -> tuple[Any, ...]:
        fitness_snapshot = candidate.get("fitness_snapshot", {})
        if not isinstance(fitness_snapshot, dict):
            fitness_snapshot = {}
        score = _coerce_float(fitness_snapshot.get("score"), default=candidate.get("confidence", 0.5))
        return (
            _PRIORITY_ORDER.get(candidate.get("priority"), 3),
            -score,
            -_coerce_float(candidate.get("confidence"), default=0.5),
            candidate.get("source") != "beads",
            candidate.get("recommended_goal") or candidate.get("summary") or "",
        )

    def _block(self, candidate: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "candidate_id": candidate.get("candidate_id"),
            "source": candidate.get("source"),
            "goal": candidate.get("recommended_goal"),
            "summary": candidate.get("summary"),
            "reason": reason,
            "proposal_id": candidate.get("proposal_id"),
        }

    def arbitrate(
        self,
        raw_candidates: Iterable[Any],
        *,
        auto_queue: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        normalized: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        for raw in raw_candidates or []:
            candidate = self.normalize_candidate(raw)
            if candidate is None:
                continue
            dedupe_key = (
                str(candidate.get("recommended_goal") or "").strip().lower()
                or str(candidate.get("summary") or "").strip().lower()
            )
            if dedupe_key in seen_keys:
                blocked.append(self._block(candidate, "duplicate_candidate"))
                continue
            seen_keys.add(dedupe_key)
            normalized.append(candidate)

        ranked = sorted(normalized, key=self._ranking_key)
        selected: List[Dict[str, Any]] = []
        for candidate in ranked:
            if len(selected) >= self.max_selected:
                blocked.append(self._block(candidate, "candidate_budget_exceeded"))
                continue
            selected.append(candidate)

        existing_goals = set()
        if self.goal_queue is not None:
            try:
                existing_goals = {str(item) for item in list(getattr(self.goal_queue, "queue", []))}
            except Exception:
                existing_goals = set()

        auto_queued_goals: List[str] = []
        for candidate in selected:
            goal = str(candidate.get("recommended_goal") or "").strip()
            intrinsic_reason = candidate.get("queue_block_reason")
            if intrinsic_reason:
                blocked.append(self._block(candidate, str(intrinsic_reason)))
                continue
            if not auto_queue or dry_run:
                continue
            if not candidate.get("queueable"):
                blocked.append(self._block(candidate, "candidate_not_queueable"))
                continue
            if not goal:
                blocked.append(self._block(candidate, "missing_recommended_goal"))
                continue
            if candidate.get("requires_human_review"):
                blocked.append(self._block(candidate, "requires_human_review"))
                continue
            if candidate.get("beads_recheck_required"):
                blocked.append(self._block(candidate, "beads_recheck_required"))
                continue
            if len(auto_queued_goals) >= self.max_auto_queue:
                blocked.append(self._block(candidate, "auto_queue_limit_reached"))
                continue
            if goal in existing_goals or goal in auto_queued_goals:
                blocked.append(self._block(candidate, "duplicate_goal"))
                continue
            if callable(self.goal_validator):
                ok, reason = self.goal_validator(goal)
                if not ok:
                    blocked.append(self._block(candidate, reason or "invalid_goal"))
                    continue
            if self.goal_queue is not None:
                self.goal_queue.add(goal)
            auto_queued_goals.append(goal)
            existing_goals.add(goal)

        selected_sources = sorted({str(candidate.get("source") or "unknown") for candidate in selected})
        return {
            "status": "ok",
            "candidates_considered": len(normalized),
            "candidates_selected": len(selected),
            "selected_candidates": selected,
            "queue_delta": len(auto_queued_goals),
            "auto_queued_goals": auto_queued_goals,
            "blocked_candidates": blocked,
            "sources": selected_sources,
            "requires_human_review": sum(1 for candidate in selected if candidate.get("requires_human_review")),
            "beads_rechecks": sum(1 for candidate in selected if candidate.get("beads_recheck_required")),
        }
