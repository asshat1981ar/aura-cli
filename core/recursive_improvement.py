import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.development_weakness import build_development_context
from core.fitness import FitnessFunction

logger = logging.getLogger(__name__)

class RecursiveImprovementService:
    """
    Runtime service that evaluates improvement opportunities from cycle history.
    Emits structured proposals for operator review.
    """
    VALID_MODES = {"propose", "auto_queue"}

    def __init__(
        self,
        fitness_weights: Optional[Dict[str, float]] = None,
        *,
        project_root: str | Path | None = None,
        goal_queue: Any = None,
        enabled: bool = True,
        mode: str = "propose",
        max_proposals: int = 3,
        max_auto_queue: int = 2,
    ):
        self.fitness = FitnessFunction(weights=fitness_weights)
        self._recent_cycles: List[Dict] = []
        self.project_root = Path(project_root) if project_root is not None else None
        self.goal_queue = goal_queue
        self.enabled = bool(enabled)
        self.mode = mode if mode in self.VALID_MODES else "propose"
        self.max_proposals = max(1, int(max_proposals))
        self.max_auto_queue = max(0, int(max_auto_queue))

    def _resolve_mode(self, mode: str | None = None) -> str:
        candidate = str(mode or self.mode or "propose").strip().lower()
        if candidate not in self.VALID_MODES:
            return "propose"
        return candidate

    def _build_development_context(self, goal: str, skill_context: Dict[str, Any], beads: Dict[str, Any]) -> Dict[str, Any]:
        if self.project_root is None:
            return {}

        active_context = {
            "skill_context": skill_context,
            "beads_gate": beads,
        }
        try:
            return build_development_context(self.project_root, goal=goal, active_context=active_context)
        except Exception:
            return {}

    def normalize_cycle_entry(self, cycle_entry: Dict) -> Dict:
        """Normalize a cycle entry into the fields used for proposal scoring."""
        phase_outputs = cycle_entry.get("phase_outputs", {}) if isinstance(cycle_entry, dict) else {}
        verification = phase_outputs.get("verification", {}) if isinstance(phase_outputs, dict) else {}
        verification_status = cycle_entry.get("verification_status")
        if verification_status is None and isinstance(verification, dict):
            verification_status = verification.get("status")

        retry_count = cycle_entry.get("retries")
        if retry_count is None and isinstance(phase_outputs, dict):
            retry_count = phase_outputs.get("retry_count", 0)

        skill_context = phase_outputs.get("skill_context", {}) if isinstance(phase_outputs, dict) else {}
        if not isinstance(skill_context, dict):
            skill_context = {}
        hotspot = skill_context.get("structural_hotspot", {})
        hotspot_files = []
        if isinstance(hotspot, dict):
            hotspot_files = [str(item) for item in hotspot.get("files", []) if str(item)]

        architecture = skill_context.get("architecture_validator", {})
        complexity = skill_context.get("complexity_scorer", {})
        coverage = skill_context.get("test_coverage_analyzer", {})
        beads = cycle_entry.get("beads") or phase_outputs.get("beads_gate", {})
        if not isinstance(beads, dict):
            beads = {}
        development_context = self._build_development_context(
            str(cycle_entry.get("goal", "")),
            skill_context,
            beads,
        )

        return {
            "cycle_id": cycle_entry.get("cycle_id", "unknown"),
            "goal": cycle_entry.get("goal", ""),
            "verification_status": verification_status or "unknown",
            "retries": int(retry_count or 0),
            "hotspot_files": hotspot_files,
            "metrics": {
                "coupling_score": _numeric_or_none(architecture.get("coupling_score")),
                "high_risk_count": _numeric_or_none(complexity.get("high_risk_count")),
                "coverage_pct": _numeric_or_none(coverage.get("coverage_pct")),
            },
            "development_context": development_context,
            "beads_follow_up_goals": [
                str(item) for item in beads.get("follow_up_goals", []) if str(item)
            ],
        }

    def observe_cycle(self, cycle_entry: Dict) -> List[Dict]:
        """Append a normalized cycle entry and return recent history."""
        normalized = self.normalize_cycle_entry(cycle_entry)
        self._recent_cycles.append(normalized)
        if len(self._recent_cycles) > 25:
            self._recent_cycles = self._recent_cycles[-25:]
        return list(self._recent_cycles)

    def _coerce_history(self, cycle_history: List[Dict]) -> List[Dict]:
        normalized_history: List[Dict] = []
        for entry in cycle_history or []:
            if isinstance(entry, dict) and {"cycle_id", "verification_status", "retries"}.issubset(entry.keys()):
                normalized_history.append(entry)
            else:
                normalized_history.append(self.normalize_cycle_entry(entry))
        return normalized_history

    def assess_queueability(self, recommended_goal: str, development_context: Dict[str, Any] | None = None) -> tuple[bool, str | None]:
        goal = str(recommended_goal or "").strip()
        if not goal:
            return False, "missing_recommended_goal"

        lowered = goal.lower()
        if len(goal) > 240:
            return False, "goal_too_broad"

        for needle in (
            "rewrite the system",
            "refactor everything",
            "entire system",
            "entire codebase",
        ):
            if needle in lowered:
                return False, "goal_too_broad"

        development_context = development_context or {}
        prototype_inventory = development_context.get("prototype_inventory", {}) if isinstance(development_context, dict) else {}
        deprecated_paths = prototype_inventory.get("deprecated_paths", []) if isinstance(prototype_inventory, dict) else []
        for path in deprecated_paths:
            if str(path).lower() in lowered:
                return False, "deprecated_recursive_self_improvement_path"

        return True, None

    def evaluate_candidates(self, cycle_history: List[Dict]) -> List[Dict]:
        """
        Analyze recent cycle outcomes to identify improvement opportunities.
        """
        proposals = []
        normalized_history = self._coerce_history(cycle_history)
        if not normalized_history:
            return proposals

        total_cycles = len(normalized_history)
        failed_cycles = [c for c in normalized_history if c.get("verification_status") == "fail"]
        avg_retry_rate = sum(c.get("retries", 0) for c in normalized_history) / max(total_cycles, 1)
        success_rate = 1.0 - (len(failed_cycles) / max(total_cycles, 1))
        latest = normalized_history[-1]
        latest_metrics = latest.get("metrics", {}) if isinstance(latest.get("metrics"), dict) else {}
        coverage_pct = latest_metrics.get("coverage_pct")
        high_risk_count = latest_metrics.get("high_risk_count")
        coupling_score = latest_metrics.get("coupling_score")
        development_context = latest.get("development_context", {}) if isinstance(latest, dict) else {}

        if len(failed_cycles) > 2:
            proposal = self.create_proposal(
                proposal_id=f"ri_{int(time.time())}_001",
                summary="High failure rate detected in recent cycles.",
                source_cycles=[c.get("cycle_id", "unknown") for c in failed_cycles],
                metrics={
                    "success_rate": success_rate,
                    "retry_rate": avg_retry_rate,
                    "complexity_delta": 0.05,
                },
                hypotheses=["System instability or capability gap in specific goals"],
                actions=["Review failure logs", "Decompose complex goals", "Add targeted tests"],
                recommended_goal=self.build_goal_recommendation(
                    "Stabilize recurring verification failures with targeted regression coverage and narrower follow-up goals."
                ),
                development_context=development_context,
                risk="low",
            )
            proposals.append(proposal)

        if avg_retry_rate >= 1.5:
            proposal = self.create_proposal(
                proposal_id=f"ri_{int(time.time())}_002",
                summary="Retry churn is elevated across recent cycles.",
                source_cycles=[c.get("cycle_id", "unknown") for c in normalized_history[-5:]],
                metrics={
                    "success_rate": success_rate,
                    "retry_rate": avg_retry_rate,
                    "complexity_delta": 0.03,
                },
                hypotheses=["Act/verify handoff is unstable for the current goal mix"],
                actions=["Tighten synthesized task scopes", "Add a regression fixture for retry-heavy failures"],
                recommended_goal=self.build_goal_recommendation(
                    "Reduce retry churn in the act/verify handoff with a targeted regression fix."
                ),
                development_context=development_context,
                risk="medium",
            )
            proposals.append(proposal)

        if latest.get("hotspot_files"):
            hotspot_file = latest["hotspot_files"][0]
            proposal = self.create_proposal(
                proposal_id=f"ri_{int(time.time())}_003",
                summary=f"Structural hotspot detected in {hotspot_file}.",
                source_cycles=[latest.get("cycle_id", "unknown")],
                metrics={
                    "success_rate": success_rate,
                    "retry_rate": avg_retry_rate,
                    "complexity_delta": _bounded_complexity_delta(high_risk_count),
                },
                hypotheses=["The hotspot is accumulating complexity faster than tests are containing it"],
                actions=["Refactor the hotspot incrementally", "Backfill regression tests before widening the scope"],
                recommended_goal=f"Refactor and add regression coverage for structural hotspot: {hotspot_file}",
                development_context=development_context,
                risk="medium",
            )
            proposals.append(proposal)

        if coverage_pct is not None and coverage_pct < 60.0:
            proposal = self.create_proposal(
                proposal_id=f"ri_{int(time.time())}_004",
                summary="Coverage is trending low on the latest architectural scan.",
                source_cycles=[latest.get("cycle_id", "unknown")],
                metrics={
                    "success_rate": success_rate,
                    "retry_rate": avg_retry_rate,
                    "complexity_delta": 0.02,
                },
                hypotheses=["Missing regression coverage is increasing verification churn"],
                actions=["Backfill focused tests for central modules", "Prefer test-first follow-up goals"],
                recommended_goal=self.build_goal_recommendation(
                    "Backfill focused regression coverage for the highest-risk central module."
                ),
                development_context=development_context,
                risk="low",
            )
            proposals.append(proposal)

        if coupling_score is not None and coupling_score >= 0.75:
            proposal = self.create_proposal(
                proposal_id=f"ri_{int(time.time())}_005",
                summary="Architectural coupling remains high in the latest scan.",
                source_cycles=[latest.get("cycle_id", "unknown")],
                metrics={
                    "success_rate": success_rate,
                    "retry_rate": avg_retry_rate,
                    "complexity_delta": 0.06,
                },
                hypotheses=["High coupling is slowing safe iteration and creating broad change surfaces"],
                actions=["Break refactors into single-subsystem goals", "Prefer narrow interfaces over cross-module edits"],
                recommended_goal=self.build_goal_recommendation(
                    "Reduce architectural coupling with a scoped single-subsystem refactor."
                ),
                development_context=development_context,
                risk="medium",
            )
            proposals.append(proposal)

        if development_context.get("target_subsystem") == "recursive_self_improvement":
            weaknesses = development_context.get("weaknesses", []) or []
            if weaknesses:
                canonical_path = development_context.get("canonical_path") or "core/recursive_improvement.py"
                proposal = self.create_proposal(
                    proposal_id=f"ri_{int(time.time())}_006",
                    summary="Recursive self-improvement guardrails remain a live focus area.",
                    source_cycles=[latest.get("cycle_id", "unknown")],
                    metrics={
                        "success_rate": success_rate,
                        "retry_rate": avg_retry_rate,
                        "complexity_delta": 0.02,
                    },
                    hypotheses=[weakness.get("summary", "Canonical RSI rollout still has open guardrails.") for weakness in weaknesses[:2]],
                    actions=[
                        "Keep execution routed through the canonical orchestrator path",
                        "Preserve the canonical RSI runtime path",
                        "Add tests that block deprecated prototype overlap",
                    ],
                    recommended_goal=f"Harden canonical recursive self-improvement routing in {canonical_path} and keep deprecated prototype paths retired.",
                    development_context=development_context,
                    risk="medium",
                )
                proposals.append(proposal)

        return proposals[: self.max_proposals]

    def build_goal_recommendation(self, goal: str) -> str:
        return str(goal).strip()

    def create_goal_proposal(
        self,
        goal: str,
        *,
        summary: str | None = None,
        source_cycles: List[str] | None = None,
        risk: str = "low",
        queueable: bool = True,
    ) -> Dict:
        recommended_goal = self.build_goal_recommendation(goal)
        return self.create_proposal(
            proposal_id=f"ri_{int(time.time())}_manual",
            summary=summary or "Contextual self-development goal identified.",
            source_cycles=source_cycles or [],
            metrics={
                "success_rate": 0.5,
                "retry_rate": 0.0,
                "complexity_delta": 0.02,
            },
            hypotheses=["The current repo signals point to a bounded self-development opportunity"],
            actions=["Queue the goal through the canonical orchestrator path"],
            recommended_goal=recommended_goal,
            queueable=queueable,
            risk=risk,
        )

    def create_proposal(
        self,
        proposal_id: str,
        summary: str,
        source_cycles: List[str],
        metrics: Dict,
        hypotheses: List[str],
        actions: List[str],
        recommended_goal: str | None = None,
        queueable: bool = False,
        queue_block_reason: str | None = None,
        development_context: Dict[str, Any] | None = None,
        risk: str = "medium",
    ) -> Dict:
        """
        Creates a structured proposal following the canonical data contract.
        """
        fitness_score = self.fitness.calculate(metrics)

        if queueable and recommended_goal:
            queueable, queue_block_reason = self.assess_queueability(
                recommended_goal,
                development_context=development_context,
            )

        proposal = {
            "proposal_id": proposal_id,
            "source": "recursive_improvement",
            "source_cycles": source_cycles,
            "summary": summary,
            "fitness_snapshot": {
                "score": fitness_score,
                "success_rate": metrics.get("success_rate", 0.0),
                "retry_rate": metrics.get("retry_rate", 0.0),
                "complexity_delta": metrics.get("complexity_delta", 0.0),
            },
            "hypotheses": hypotheses,
            "recommended_actions": actions,
            "recommended_goal": recommended_goal,
            "queueable": bool(queueable and recommended_goal),
            "queue_block_reason": queue_block_reason,
            "risk_level": risk,
            "requires_operator_review": True,
            "development_context": development_context or {},
        }
        return proposal

    def queue_proposals(
        self,
        proposals: List[Dict],
        *,
        goal_queue: Any = None,
        mode: str | None = None,
    ) -> List[str]:
        if self._resolve_mode(mode) != "auto_queue":
            return []

        queue = goal_queue or self.goal_queue
        if queue is None:
            return []

        pending = list(getattr(queue, "queue", [])) if hasattr(queue, "queue") else []
        queued_goals: List[str] = []
        for proposal in proposals:
            goal = str(proposal.get("recommended_goal") or "").strip()
            if not goal:
                continue
            if not proposal.get("queueable", False):
                if not proposal.get("queue_block_reason"):
                    proposal["queue_block_reason"] = "proposal_not_queueable"
                continue
            if len(queued_goals) >= self.max_auto_queue:
                proposal["queueable"] = False
                proposal["queue_block_reason"] = "auto_queue_limit_reached"
                continue
            if goal in pending or goal in queued_goals:
                proposal["queueable"] = False
                proposal["queue_block_reason"] = "already_queued"
                continue
            queue.add(goal)
            queued_goals.append(goal)
        return queued_goals

    def build_cycle_payload(
        self,
        cycle_entry: Dict,
        *,
        mode: str | None = None,
        allow_queue: bool = True,
    ) -> Dict[str, Any]:
        recent_history = self.observe_cycle(cycle_entry)
        proposals = self.evaluate_candidates(recent_history)
        queued_goals = self.queue_proposals(proposals, mode=mode) if allow_queue else []
        normalized = self._recent_cycles[-1] if self._recent_cycles else self.normalize_cycle_entry(cycle_entry)
        queue_block_reasons = [
            {
                "goal": str(proposal.get("recommended_goal") or ""),
                "reason": proposal.get("queue_block_reason") or "proposal_not_queueable",
                "proposal_id": proposal.get("proposal_id"),
            }
            for proposal in proposals
            if proposal.get("queue_block_reason")
        ]

        payload = {
            "status": "ok" if self.enabled else "disabled",
            "self_dev_mode": self._resolve_mode(mode),
            "proposal_count": len(proposals),
            "proposals": proposals,
            "follow_up_goals": [
                proposal["recommended_goal"]
                for proposal in proposals
                if proposal.get("recommended_goal")
            ],
            "auto_queued_goals": queued_goals,
            "queue_block_reasons": queue_block_reasons,
            "development_context": normalized.get("development_context", {}),
        }
        cycle_entry.setdefault("phase_outputs", {})["ralph"] = payload
        return payload

    def _seed_history(self, goal: str) -> List[Dict]:
        return [
            {
                "cycle_id": "manual-self-dev",
                "goal": goal,
                "verification_status": "unknown",
                "retries": 0,
                "hotspot_files": [],
                "metrics": {},
                "development_context": self._build_development_context(goal, {}, {}),
                "beads_follow_up_goals": [],
            }
        ]

    def run_manual(
        self,
        *,
        goal: str,
        cycle_history: List[Dict] | None = None,
        mode: str | None = None,
        goal_queue: Any = None,
        allow_queue: bool = True,
    ) -> Dict[str, Any]:
        normalized_history = self._coerce_history(cycle_history or self._seed_history(goal))
        proposals = self.evaluate_candidates(normalized_history)
        if not proposals and goal.strip():
            proposals = [
                self.create_goal_proposal(
                    goal,
                    summary="Manual self-development request routed through the canonical runtime path.",
                    source_cycles=[entry.get("cycle_id", "unknown") for entry in normalized_history[-3:]],
                    risk="low",
                    queueable=True,
                )
            ]
        effective_mode = self._resolve_mode(mode)
        queued_goals = (
            self.queue_proposals(proposals, goal_queue=goal_queue, mode=effective_mode)
            if allow_queue
            else []
        )
        return {
            "status": "ok" if self.enabled else "disabled",
            "source": "runtime_attached",
            "goal": goal,
            "self_dev_mode": effective_mode,
            "analyzed_cycles": len(normalized_history),
            "proposal_count": len(proposals),
            "proposals": proposals,
            "follow_up_goals": [
                proposal["recommended_goal"]
                for proposal in proposals
                if proposal.get("recommended_goal")
            ],
            "auto_queued_goals": queued_goals,
            "queue_block_reasons": [
                {
                    "goal": str(proposal.get("recommended_goal") or ""),
                    "reason": proposal.get("queue_block_reason") or "proposal_not_queueable",
                    "proposal_id": proposal.get("proposal_id"),
                }
                for proposal in proposals
                if proposal.get("queue_block_reason")
            ],
            "queue_delta": len(queued_goals),
            "recent_cycle_ids": [entry.get("cycle_id", "unknown") for entry in normalized_history],
        }

    def log_proposal(self, proposal: Dict):
        logger.info(f"Generated Recursive Improvement Proposal: {proposal['proposal_id']}")
        logger.info(json.dumps(proposal, indent=2))


def _numeric_or_none(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bounded_complexity_delta(high_risk_count) -> float:
    if not isinstance(high_risk_count, (int, float)):
        return 0.03
    return max(0.0, min(float(high_risk_count) / 100.0, 1.0))
