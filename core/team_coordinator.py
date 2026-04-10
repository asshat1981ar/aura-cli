"""Agent Team Coordinator: orchestrate multiple AURA agents on decomposed sub-goals.

Implements the "team lead" pattern where one AURA instance:
1. Decomposes a complex goal into independent sub-goals
2. Assigns sub-goals to worker agents (local threads or remote A2A peers)
3. Monitors progress and handles failures
4. Synthesizes results from all workers

Inspired by Anthropic's Agent Teams (Feb 2026).
"""

import asyncio
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from core.logging_utils import log_json


class WorkerStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class SubGoal:
    """A decomposed sub-goal assigned to a worker."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    parent_goal: str = ""
    assigned_to: str = ""  # worker ID or peer URL
    status: WorkerStatus = WorkerStatus.PENDING
    result: dict = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0
    dependencies: list[str] = field(default_factory=list)  # sub-goal IDs this depends on
    priority: int = 0  # higher = more important


@dataclass
class TeamResult:
    """Result from a team execution."""

    goal: str
    sub_goals: list[SubGoal] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    total_duration: float = 0.0
    synthesis: str = ""


class TeamCoordinator:
    """Coordinates a team of AURA agents working on decomposed sub-goals."""

    def __init__(self, model=None, orchestrator_factory: Callable | None = None, max_workers: int = 3, a2a_client=None):
        self.model = model
        self.orchestrator_factory = orchestrator_factory  # Creates new orchestrator instances
        self.max_workers = max_workers
        self.a2a_client = a2a_client
        self.active_teams: dict[str, TeamResult] = {}

    def decompose_goal(self, goal: str, context: dict | None = None) -> list[SubGoal]:
        """Decompose a complex goal into independent sub-goals."""
        if not self.model:
            return [SubGoal(description=goal, parent_goal=goal)]

        prompt = f"""Decompose this goal into independent, parallelizable sub-goals.
Each sub-goal should be self-contained and testable independently.

Goal: {goal}

Context: {json.dumps(context or {}, default=str)[:1000]}

Respond with JSON array:
[{{"description": "sub-goal description", "priority": 1, "dependencies": []}}]

Rules:
- Maximum 5 sub-goals
- Mark dependencies by index (e.g., [0] means depends on first sub-goal)
- Higher priority number = more important
- Each sub-goal should be completable in 1-3 cycles"""

        try:
            respond_fn = getattr(self.model, "respond_for_role", None)
            if callable(respond_fn):
                response = respond_fn("planning", prompt)
            else:
                response = self.model.respond(prompt)
            return self._parse_sub_goals(response, goal)
        except Exception as exc:
            log_json("WARN", "team_decompose_failed", details={"error": str(exc)})
            return [SubGoal(description=goal, parent_goal=goal)]

    def execute_team(self, goal: str, sub_goals: list[SubGoal], dry_run: bool = False) -> TeamResult:
        """Execute sub-goals across worker agents."""
        team_id = uuid.uuid4().hex[:8]
        t0 = time.time()
        result = TeamResult(goal=goal, sub_goals=sub_goals)
        self.active_teams[team_id] = result

        log_json("INFO", "team_execution_started", details={"team_id": team_id, "sub_goals": len(sub_goals)})

        # Sort by dependencies (topological-ish: no-dep first, then dependent)
        independent = [sg for sg in sub_goals if not sg.dependencies]
        dependent = [sg for sg in sub_goals if sg.dependencies]

        # Phase 1: Run independent sub-goals in parallel
        self._run_parallel(independent, dry_run)

        # Phase 2: Run dependent sub-goals (sequentially for now)
        for sg in dependent:
            deps_met = all(any(done.id == dep_id and done.status == WorkerStatus.COMPLETED for done in sub_goals) for dep_id in sg.dependencies)
            if deps_met:
                self._run_single(sg, dry_run)
            else:
                sg.status = WorkerStatus.FAILED
                sg.result = {"error": "dependencies not met"}

        result.success_count = sum(1 for sg in sub_goals if sg.status == WorkerStatus.COMPLETED)
        result.failure_count = sum(1 for sg in sub_goals if sg.status == WorkerStatus.FAILED)
        result.total_duration = time.time() - t0

        log_json("INFO", "team_execution_complete", details={"team_id": team_id, "success": result.success_count, "failed": result.failure_count, "duration_s": round(result.total_duration, 1)})
        return result

    def _run_parallel(self, sub_goals: list[SubGoal], dry_run: bool):
        """Run sub-goals in parallel using thread pool."""
        if not sub_goals:
            return

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sub_goals))) as pool:
            futures = {}
            for sg in sub_goals:
                sg.status = WorkerStatus.RUNNING
                sg.started_at = time.time()
                future = pool.submit(self._execute_sub_goal, sg, dry_run)
                futures[future] = sg

            for future in as_completed(futures):
                sg = futures[future]
                try:
                    result = future.result(timeout=300)
                    sg.result = result
                    sg.status = WorkerStatus.COMPLETED
                except Exception as exc:
                    sg.result = {"error": str(exc)}
                    sg.status = WorkerStatus.FAILED
                sg.completed_at = time.time()

    def _run_single(self, sub_goal: SubGoal, dry_run: bool):
        """Run a single sub-goal."""
        sub_goal.status = WorkerStatus.RUNNING
        sub_goal.started_at = time.time()
        try:
            result = self._execute_sub_goal(sub_goal, dry_run)
            sub_goal.result = result
            sub_goal.status = WorkerStatus.COMPLETED
        except Exception as exc:
            sub_goal.result = {"error": str(exc)}
            sub_goal.status = WorkerStatus.FAILED
        sub_goal.completed_at = time.time()

    def _execute_sub_goal(self, sub_goal: SubGoal, dry_run: bool) -> dict:
        """Execute a sub-goal using local orchestrator or A2A peer."""
        # Try A2A delegation first
        if self.a2a_client and sub_goal.assigned_to:
            peer = self.a2a_client.find_capable_peer("autonomous_goal")
            if peer:
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(self.a2a_client.delegate(peer.url, "autonomous_goal", sub_goal.description))
                    return result or {"status": "delegated"}
                finally:
                    loop.close()

        # Local execution via orchestrator factory
        if self.orchestrator_factory:
            orchestrator = self.orchestrator_factory()
            return orchestrator.run_cycle(sub_goal.description, dry_run=dry_run)

        # Fallback: return description as a goal for manual processing
        return {"status": "pending_manual", "goal": sub_goal.description}

    def _parse_sub_goals(self, response: str, parent_goal: str) -> list[SubGoal]:
        """Parse sub-goals from model response."""
        try:
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                sub_goals = []
                for _i, item in enumerate(data[:5]):
                    if isinstance(item, dict):
                        sg = SubGoal(
                            description=item.get("description", str(item)),
                            parent_goal=parent_goal,
                            priority=int(item.get("priority", 0)),
                            dependencies=[sub_goals[d].id for d in item.get("dependencies", []) if d < len(sub_goals)],
                        )
                        sub_goals.append(sg)
                return sub_goals if sub_goals else [SubGoal(description=parent_goal, parent_goal=parent_goal)]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return [SubGoal(description=parent_goal, parent_goal=parent_goal)]

    def get_team_status(self, team_id: str) -> dict | None:
        """Get status of an active team."""
        result = self.active_teams.get(team_id)
        if not result:
            return None
        return {
            "goal": result.goal,
            "sub_goals": [{"id": sg.id, "desc": sg.description[:50], "status": sg.status.value, "priority": sg.priority} for sg in result.sub_goals],
            "success": result.success_count,
            "failed": result.failure_count,
            "duration_s": result.total_duration,
        }
