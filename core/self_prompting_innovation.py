from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class InnovationProposal:
    proposal_id: str
    title: str
    category: str
    goal: str
    rationale: str
    evidence: list[str]
    smallest_surface: str
    expected_value: str
    risk_level: str
    verification_cost: str
    recommended_action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "category": self.category,
            "goal": self.goal,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
            "smallest_surface": self.smallest_surface,
            "expected_value": self.expected_value,
            "risk_level": self.risk_level,
            "verification_cost": self.verification_cost,
            "recommended_action": self.recommended_action,
        }


class SelfPromptingInnovationLoop:
    def __init__(
        self,
        *,
        architecture_explorer: Callable[[], dict[str, Any]],
        capability_researcher: Callable[[str], dict[str, Any]],
        verification_reviewer: Callable[[list[InnovationProposal], dict[str, Any]], dict[str, Any]],
        summarize_subagents: Callable[..., list[dict[str, Any]]],
        proposal_builder: Callable[[str, dict[str, Any], dict[str, Any]], list[InnovationProposal]],
        goal_queue=None,
        orchestrator=None,
        auto_execute_queued: bool = True,
        innovation_goal_limit: int = 2,
        trigger_every_n: int = 20,
        surface_cooldown_cycles: int = 20,
    ):
        self._architecture_explorer = architecture_explorer
        self._capability_researcher = capability_researcher
        self._verification_reviewer = verification_reviewer
        self._summarize_subagents = summarize_subagents
        self._proposal_builder = proposal_builder
        self.goal_queue = goal_queue
        self.orchestrator = orchestrator
        self.auto_execute_queued = auto_execute_queued
        self.innovation_goal_limit = max(1, int(innovation_goal_limit))
        self.TRIGGER_EVERY_N = max(1, int(trigger_every_n))
        self.surface_cooldown_cycles = max(1, int(surface_cooldown_cycles))
        self._cycle_count = 0
        self._surface_last_queued: dict[str, int] = {}
        self._proposal_last_queued: dict[str, int] = {}

    def _select_proposals(
        self,
        proposals: list[InnovationProposal],
        *,
        focus: str,
        proposal_limit: int,
    ) -> list[InnovationProposal]:
        focus_priority = {
            "capability": {
                "skill": 0,
                "mcp": 1,
                "capability": 2,
                "verification": 3,
                "orchestration": 4,
                "developer-surface": 5,
            },
            "quality": {
                "verification": 0,
                "orchestration": 1,
                "capability": 2,
                "skill": 3,
                "mcp": 4,
                "developer-surface": 5,
            },
            "throughput": {
                "developer-surface": 0,
                "capability": 1,
                "skill": 2,
                "mcp": 3,
                "verification": 4,
                "orchestration": 5,
            },
            "research": {
                "capability": 0,
                "verification": 1,
                "mcp": 2,
                "skill": 3,
                "orchestration": 4,
                "developer-surface": 5,
            },
        }
        category_priority = focus_priority.get(focus, focus_priority["capability"])
        risk_priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        ranked = sorted(
            proposals,
            key=lambda item: (
                category_priority.get(item.category, 99),
                risk_priority.get(item.risk_level, 99),
                item.title,
            ),
        )
        return ranked[: max(1, int(proposal_limit))]

    def _proposal_surface_key(self, proposal: InnovationProposal) -> str:
        return str(proposal.smallest_surface or proposal.goal).strip().lower()

    def _suppression_reason(self, proposal: InnovationProposal) -> str | None:
        surface_key = self._proposal_surface_key(proposal)
        proposal_cycle = self._proposal_last_queued.get(proposal.proposal_id)
        if proposal_cycle is not None and self._cycle_count - proposal_cycle < self.surface_cooldown_cycles:
            return "recent_proposal"

        surface_cycle = self._surface_last_queued.get(surface_key)
        if surface_cycle is not None and self._cycle_count - surface_cycle < self.surface_cooldown_cycles:
            return "surface_cooldown"
        return None

    def _queue_selected_goals(self, selected: list[InnovationProposal], *, dry_run: bool) -> dict[str, Any]:
        if not selected:
            return {
                "attempted": False,
                "queued": [],
                "skipped": [],
                "suppressed": [],
                "queue_strategy": None,
            }

        queued_proposals: list[InnovationProposal] = []
        skipped: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        existing = set(getattr(self.goal_queue, "queue", []) or []) if self.goal_queue is not None else set()

        for proposal in selected[: self.innovation_goal_limit]:
            suppression_reason = self._suppression_reason(proposal)
            if suppression_reason:
                suppressed.append({"goal": proposal.goal, "proposal_id": proposal.proposal_id, "reason": suppression_reason})
                continue
            if proposal.goal in existing:
                skipped.append({"goal": proposal.goal, "proposal_id": proposal.proposal_id, "reason": "already_queued"})
                continue
            queued_proposals.append(proposal)

        if dry_run:
            return {
                "attempted": False,
                "queued": [],
                "skipped": skipped + [{"goal": proposal.goal, "proposal_id": proposal.proposal_id, "reason": "dry_run"} for proposal in queued_proposals],
                "suppressed": suppressed,
                "queue_strategy": None,
            }

        if self.goal_queue is None:
            return {
                "attempted": False,
                "queued": [],
                "skipped": skipped + [{"goal": proposal.goal, "proposal_id": proposal.proposal_id, "reason": "goal_queue_unavailable"} for proposal in queued_proposals],
                "suppressed": suppressed,
                "queue_strategy": None,
            }

        new_goals = [proposal.goal for proposal in queued_proposals]
        if not new_goals:
            return {
                "attempted": True,
                "queued": [],
                "skipped": skipped,
                "suppressed": suppressed,
                "queue_strategy": None,
            }

        if hasattr(self.goal_queue, "prepend_batch"):
            self.goal_queue.prepend_batch(new_goals)
            strategy = "prepend"
        elif hasattr(self.goal_queue, "batch_add"):
            self.goal_queue.batch_add(new_goals)
            strategy = "append"
        else:
            for goal in new_goals:
                self.goal_queue.add(goal)
            strategy = "append"

        for proposal in queued_proposals:
            self._proposal_last_queued[proposal.proposal_id] = self._cycle_count
            self._surface_last_queued[self._proposal_surface_key(proposal)] = self._cycle_count

        return {
            "attempted": True,
            "queued": new_goals,
            "skipped": skipped,
            "suppressed": suppressed,
            "queue_strategy": strategy,
        }

    def _execute_selected_goals(self, goals: list[str], *, dry_run: bool, execution_limit: int) -> dict[str, Any]:
        if not goals:
            return {"attempted": False, "executed": []}
        if dry_run:
            return {"attempted": False, "executed": [{"goal": goal, "status": "planned"} for goal in goals]}
        if self.orchestrator is None:
            return {
                "attempted": False,
                "executed": [{"goal": goal, "status": "orchestrator_unavailable"} for goal in goals],
            }
        executed = []
        for goal in goals[: max(1, int(execution_limit))]:
            result = self.orchestrator.run_loop(goal, max_cycles=1, dry_run=False)
            executed.append(
                {
                    "goal": goal,
                    "status": result.get("stop_reason", "unknown"),
                    "history_length": len(result.get("history", [])),
                }
            )
        return {"attempted": True, "executed": executed}

    def run(
        self,
        goal: str,
        *,
        execute_queued: bool | None = None,
        dry_run: bool = False,
        proposal_limit: int | None = None,
        focus: str = "capability",
    ) -> dict[str, Any]:
        execute_queued = self.auto_execute_queued if execute_queued is None else execute_queued
        proposal_limit = max(1, int(proposal_limit or self.innovation_goal_limit))

        with ThreadPoolExecutor(max_workers=2) as pool:
            architecture_future = pool.submit(self._architecture_explorer)
            capability_future = pool.submit(self._capability_researcher, goal)
            architecture = architecture_future.result()
            capability = capability_future.result()

        proposals = self._proposal_builder(goal, architecture, capability)
        selected = self._select_proposals(proposals, focus=focus, proposal_limit=proposal_limit)
        verification = self._verification_reviewer(selected, architecture)
        queue_result = self._queue_selected_goals(selected, dry_run=dry_run)
        execution_result = (
            self._execute_selected_goals(queue_result.get("queued", []), dry_run=dry_run, execution_limit=proposal_limit)
            if execute_queued
            else {"attempted": False, "executed": []}
        )

        return {
            "workflow": "innovation",
            "focus": focus,
            "subagents": self._summarize_subagents(architecture, capability, verification),
            "analysis": {
                "architecture": architecture,
                "capability": capability,
                "verification": verification,
            },
            "proposals": [proposal.as_dict() for proposal in proposals],
            "selected_proposals": [proposal.as_dict() for proposal in selected],
            "queue": queue_result,
            "implementation": execution_result,
        }

    def on_cycle_complete(self, entry: dict) -> dict[str, Any] | None:
        self._cycle_count += 1
        goal = entry.get("goal", "evolve and improve the AURA system")
        skill_context = str(entry.get("phase_outputs", {}).get("skill_context", {}))
        is_hotspot = "structural_hotspot" in skill_context or "refactor hotspot" in goal.lower()
        if self._cycle_count % self.TRIGGER_EVERY_N == 0 or is_hotspot:
            return self.run(goal, execute_queued=False, dry_run=False, proposal_limit=self.innovation_goal_limit)
        return None
