from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

from core.self_prompting_innovation import InnovationProposal, SelfPromptingInnovationLoop


class _QueueStub:
    def __init__(self):
        self.queue = deque()

    def prepend_batch(self, goals):
        self.queue.extendleft(reversed(goals))


def _proposal(proposal_id: str, title: str, *, goal: str, surface: str, category: str = "capability") -> InnovationProposal:
    return InnovationProposal(
        proposal_id=proposal_id,
        title=title,
        category=category,
        goal=goal,
        rationale="because",
        evidence=["evidence"],
        smallest_surface=surface,
        expected_value="high",
        risk_level="low",
        verification_cost="targeted tests",
        recommended_action="queue",
    )


def _make_loop(*, proposals=None, goal_queue=None, orchestrator=None, trigger_every_n=3, cooldown=5):
    proposals = proposals or []
    return SelfPromptingInnovationLoop(
        architecture_explorer=lambda: {"role": "architecture_explorer"},
        capability_researcher=lambda goal: {"role": "capability_researcher", "goal": goal},
        verification_reviewer=lambda selected, architecture: {"role": "verification_reviewer", "count": len(selected)},
        summarize_subagents=lambda *reports: [{"role": report.get("role"), "status": "ok"} for report in reports],
        proposal_builder=lambda goal, architecture, capability: list(proposals),
        goal_queue=goal_queue,
        orchestrator=orchestrator,
        auto_execute_queued=False,
        innovation_goal_limit=2,
        trigger_every_n=trigger_every_n,
        surface_cooldown_cycles=cooldown,
    )


def test_on_cycle_complete_triggers_on_cadence():
    loop = _make_loop()
    loop.run = MagicMock(return_value={"selected_proposals": []})

    loop.on_cycle_complete({"goal": "one"})
    loop.on_cycle_complete({"goal": "two"})
    assert loop.run.call_count == 0

    loop.on_cycle_complete({"goal": "three"})
    loop.run.assert_called_once()


def test_on_cycle_complete_triggers_immediately_for_hotspot():
    loop = _make_loop(trigger_every_n=50)
    loop.run = MagicMock(return_value={"selected_proposals": []})

    loop.on_cycle_complete({"goal": "refactor hotspot in orchestrator"})
    loop.run.assert_called_once()


def test_run_suppresses_recent_surface_requeue():
    queue = _QueueStub()
    proposals = [_proposal("p1", "Improve orchestration", goal="improve orchestration", surface="core/orchestrator.py")]
    loop = _make_loop(proposals=proposals, goal_queue=queue, cooldown=10)

    first = loop.run("improve orchestration")
    second = loop.run("improve orchestration again")

    assert first["queue"]["queued"] == ["improve orchestration"]
    assert second["queue"]["queued"] == []
    assert second["queue"]["suppressed"][0]["reason"] in {"recent_proposal", "surface_cooldown"}


def test_run_respects_queue_limit_and_execution_limit():
    queue = _QueueStub()
    orchestrator = MagicMock()
    orchestrator.run_loop.return_value = {"stop_reason": "PASS", "history": [{"cycle": 1}]}
    proposals = [
        _proposal("p1", "A", goal="goal-a", surface="a.py"),
        _proposal("p2", "B", goal="goal-b", surface="b.py"),
        _proposal("p3", "C", goal="goal-c", surface="c.py"),
    ]
    loop = SelfPromptingInnovationLoop(
        architecture_explorer=lambda: {"role": "architecture_explorer"},
        capability_researcher=lambda goal: {"role": "capability_researcher", "goal": goal},
        verification_reviewer=lambda selected, architecture: {"role": "verification_reviewer", "count": len(selected)},
        summarize_subagents=lambda *reports: [{"role": report.get("role"), "status": "ok"} for report in reports],
        proposal_builder=lambda goal, architecture, capability: list(proposals),
        goal_queue=queue,
        orchestrator=orchestrator,
        auto_execute_queued=True,
        innovation_goal_limit=2,
        trigger_every_n=3,
        surface_cooldown_cycles=5,
    )

    result = loop.run("expand capabilities", execute_queued=True, proposal_limit=3)

    assert len(result["selected_proposals"]) == 3
    assert result["queue"]["queued"] == ["goal-a", "goal-b"]
    assert len(result["implementation"]["executed"]) == 2
