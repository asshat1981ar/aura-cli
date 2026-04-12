"""Comprehensive tests for core.evolution_loop.EvolutionLoop.

Focuses on:
- InnovationProposal generation (_build_innovation_proposals)
- Capability gap detection (_capability_researcher / analyze_capability_needs)
- Proposal filtering and scoring (_select_proposals)
- Queue management (_queue_selected_goals)
- Dry-run behaviour
- on_cycle_complete trigger logic
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from core.evolution_loop import EvolutionLoop, InnovationProposal


# ── Factory helpers ───────────────────────────────────────────────────────────


def _make_loop(
    *,
    goal_queue=None,
    orchestrator=None,
    skills: dict | None = None,
    project_root: str = "/tmp",
    auto_execute_queued: bool = False,
    innovation_goal_limit: int = 3,
) -> EvolutionLoop:
    """Build an EvolutionLoop with all heavy dependencies mocked out."""
    planner = MagicMock()
    planner.plan.return_value = ["hypothesis step"]
    coder = MagicMock()
    coder.implement.return_value = "# code"
    critic = MagicMock()
    critic.critique_code.return_value = '{"score": 8}'
    critic.validate_mutation.return_value = '{"decision": "REJECTED", "confidence_score": 0.5, "impact_assessment": "low", "reasoning": "test"}'
    brain = MagicMock()
    brain.recall_with_budget.return_value = []
    brain.recall_weaknesses.return_value = []
    vector_store = MagicMock()
    vector_store.search.return_value = []
    git_tools = MagicMock()
    git_tools.repo_path = project_root
    mutator = MagicMock()

    with patch("core.evolution_loop.ExperimentTracker"), patch("core.evolution_loop.MetricsCollector"):
        loop = EvolutionLoop(
            planner=planner,
            coder=coder,
            critic=critic,
            brain=brain,
            vector_store=vector_store,
            git_tools=git_tools,
            mutator=mutator,
            goal_queue=goal_queue,
            orchestrator=orchestrator,
            project_root=project_root,
            skills=skills or {},
            auto_execute_queued=auto_execute_queued,
            innovation_goal_limit=innovation_goal_limit,
        )
    return loop


def _make_architecture(
    *,
    hotspots: list | None = None,
    debt_score: int = 90,
    clone_count: int = 0,
) -> dict:
    return {
        "role": "architecture_explorer",
        "structural": {"hotspots": hotspots or []},
        "tech_debt": {
            "debt_score": debt_score,
            "summary": "Some technical debt found.",
        },
        "clones": {"clone_count": clone_count, "consolidation_suggestions": []},
    }


def _make_capability(
    *,
    missing_skills: list | None = None,
    recommended_skills: list | None = None,
    provisioning_actions: list | None = None,
) -> dict:
    return {
        "role": "capability_researcher",
        "capability_plan": {
            "missing_skills": missing_skills or [],
            "recommended_skills": recommended_skills or [],
            "provisioning_actions": provisioning_actions or [],
        },
        "mcp": {"server_count": 2, "servers": ["github", "filesystem"]},
        "skill_count": 5,
    }


# ── InnovationProposal dataclass ──────────────────────────────────────────────


class TestInnovationProposalDataclass(unittest.TestCase):
    def _make_proposal(self, **overrides) -> InnovationProposal:
        defaults = dict(
            proposal_id="test:id",
            title="Test Proposal",
            category="skill",
            goal="Do something useful",
            rationale="Because it is needed",
            evidence=["evidence1"],
            smallest_surface="agents/skills",
            expected_value="high",
            risk_level="low",
            verification_cost="unit tests",
            recommended_action="queue",
        )
        defaults.update(overrides)
        return InnovationProposal(**defaults)

    def test_as_dict_returns_all_fields(self):
        proposal = self._make_proposal()
        d = proposal.as_dict()
        for field in (
            "proposal_id",
            "title",
            "category",
            "goal",
            "rationale",
            "evidence",
            "smallest_surface",
            "expected_value",
            "risk_level",
            "verification_cost",
            "recommended_action",
        ):
            self.assertIn(field, d)

    def test_evidence_is_a_list_copy(self):
        proposal = self._make_proposal(evidence=["a", "b"])
        d = proposal.as_dict()
        self.assertEqual(d["evidence"], ["a", "b"])
        # Ensure it is a copy, not the same object
        self.assertIsNot(d["evidence"], proposal.evidence)

    def test_frozen_dataclass_raises_on_mutation(self):
        proposal = self._make_proposal()
        with self.assertRaises((AttributeError, TypeError)):
            proposal.title = "New Title"  # type: ignore[misc]


# ── _build_innovation_proposals ───────────────────────────────────────────────


class TestBuildInnovationProposals(unittest.TestCase):
    def setUp(self):
        self.loop = _make_loop()

    def test_missing_skill_produces_skill_proposal(self):
        architecture = _make_architecture()
        capability = _make_capability(missing_skills=["dockerfile_analyzer"])
        proposals = self.loop._build_innovation_proposals("build goal", architecture, capability)
        ids = [p.proposal_id for p in proposals]
        self.assertIn("skill:dockerfile_analyzer", ids)

    def test_missing_skill_proposal_has_correct_category(self):
        architecture = _make_architecture()
        capability = _make_capability(missing_skills=["dockerfile_analyzer"])
        proposals = self.loop._build_innovation_proposals("build goal", architecture, capability)
        skill_proposals = [p for p in proposals if p.proposal_id == "skill:dockerfile_analyzer"]
        self.assertEqual(skill_proposals[0].category, "skill")

    def test_recommended_skill_produces_capability_proposal(self):
        architecture = _make_architecture()
        capability = _make_capability(recommended_skills=["git_history_analyzer"])
        proposals = self.loop._build_innovation_proposals("git goal", architecture, capability)
        ids = [p.proposal_id for p in proposals]
        self.assertIn("enable:git_history_analyzer", ids)

    def test_provisioning_action_produces_mcp_proposal(self):
        architecture = _make_architecture()
        capability = _make_capability(provisioning_actions=[{"action": "ensure_mcp_servers", "reason": "needed"}])
        proposals = self.loop._build_innovation_proposals("mcp goal", architecture, capability)
        ids = [p.proposal_id for p in proposals]
        self.assertIn("mcp:ensure_mcp_servers", ids)

    def test_hotspot_produces_orchestration_proposal(self):
        architecture = _make_architecture(
            hotspots=[
                {"file": "core/orchestrator.py", "risk_level": "high"},
            ]
        )
        capability = _make_capability()
        proposals = self.loop._build_innovation_proposals("improve goal", architecture, capability)
        hotspot_proposals = [p for p in proposals if p.proposal_id.startswith("hotspot:")]
        self.assertEqual(len(hotspot_proposals), 1)
        self.assertEqual(hotspot_proposals[0].category, "orchestration")

    def test_only_first_two_hotspots_included(self):
        hotspots = [{"file": f"core/file{i}.py", "risk_level": "medium"} for i in range(5)]
        architecture = _make_architecture(hotspots=hotspots)
        capability = _make_capability()
        proposals = self.loop._build_innovation_proposals("goal", architecture, capability)
        hotspot_proposals = [p for p in proposals if p.proposal_id.startswith("hotspot:")]
        self.assertLessEqual(len(hotspot_proposals), 2)

    def test_low_debt_score_produces_verification_proposal(self):
        architecture = _make_architecture(debt_score=50)  # < 80 threshold
        capability = _make_capability()
        proposals = self.loop._build_innovation_proposals("refactor goal", architecture, capability)
        ids = [p.proposal_id for p in proposals]
        self.assertIn("verification:debt", ids)

    def test_high_debt_score_does_not_produce_verification_proposal(self):
        architecture = _make_architecture(debt_score=95)  # >= 80 threshold
        capability = _make_capability()
        proposals = self.loop._build_innovation_proposals("refactor goal", architecture, capability)
        ids = [p.proposal_id for p in proposals]
        self.assertNotIn("verification:debt", ids)

    def test_clones_produce_refactor_proposal(self):
        architecture = _make_architecture(clone_count=3)
        capability = _make_capability()
        proposals = self.loop._build_innovation_proposals("clean up goal", architecture, capability)
        ids = [p.proposal_id for p in proposals]
        self.assertIn("refactor:clones", ids)

    def test_zero_clones_no_refactor_proposal(self):
        architecture = _make_architecture(clone_count=0)
        capability = _make_capability()
        proposals = self.loop._build_innovation_proposals("clean up goal", architecture, capability)
        ids = [p.proposal_id for p in proposals]
        self.assertNotIn("refactor:clones", ids)

    def test_empty_capability_and_architecture_returns_empty_list(self):
        architecture = _make_architecture()
        capability = _make_capability()
        proposals = self.loop._build_innovation_proposals("goal", architecture, capability)
        self.assertEqual(proposals, [])


# ── _select_proposals (filtering / scoring) ───────────────────────────────────


class TestSelectProposals(unittest.TestCase):
    def _make_proposals(self, categories_risks: list[tuple[str, str]]) -> list[InnovationProposal]:
        return [
            InnovationProposal(
                proposal_id=f"p{i}",
                title=f"Proposal {i}",
                category=cat,
                goal=f"goal {i}",
                rationale="rationale",
                evidence=["ev"],
                smallest_surface="file.py",
                expected_value="medium",
                risk_level=risk,
                verification_cost="unit tests",
                recommended_action="queue",
            )
            for i, (cat, risk) in enumerate(categories_risks)
        ]

    def setUp(self):
        self.loop = _make_loop()

    def test_limits_output_to_proposal_limit(self):
        proposals = self._make_proposals([("skill", "low")] * 10)
        selected = self.loop._select_proposals(proposals, focus="capability", proposal_limit=3)
        self.assertEqual(len(selected), 3)

    def test_returns_at_least_one_proposal(self):
        proposals = self._make_proposals([("skill", "low")])
        selected = self.loop._select_proposals(proposals, focus="capability", proposal_limit=0)
        self.assertGreaterEqual(len(selected), 1)

    def test_capability_focus_prioritises_skill_category(self):
        proposals = self._make_proposals(
            [
                ("orchestration", "low"),
                ("skill", "low"),
                ("verification", "low"),
            ]
        )
        selected = self.loop._select_proposals(proposals, focus="capability", proposal_limit=1)
        self.assertEqual(selected[0].category, "skill")

    def test_quality_focus_prioritises_verification_category(self):
        proposals = self._make_proposals(
            [
                ("skill", "low"),
                ("orchestration", "low"),
                ("verification", "low"),
            ]
        )
        selected = self.loop._select_proposals(proposals, focus="quality", proposal_limit=1)
        self.assertEqual(selected[0].category, "verification")

    def test_throughput_focus_prioritises_developer_surface(self):
        proposals = self._make_proposals(
            [
                ("skill", "low"),
                ("developer-surface", "low"),
                ("verification", "low"),
            ]
        )
        selected = self.loop._select_proposals(proposals, focus="throughput", proposal_limit=1)
        self.assertEqual(selected[0].category, "developer-surface")

    def test_unknown_focus_falls_back_to_capability_ordering(self):
        proposals = self._make_proposals(
            [
                ("orchestration", "low"),
                ("skill", "low"),
            ]
        )
        selected = self.loop._select_proposals(proposals, focus="unknown_focus", proposal_limit=1)
        self.assertEqual(selected[0].category, "skill")

    def test_lower_risk_ranked_higher_within_same_category(self):
        proposals = self._make_proposals(
            [
                ("skill", "high"),
                ("skill", "low"),
            ]
        )
        selected = self.loop._select_proposals(proposals, focus="capability", proposal_limit=2)
        self.assertEqual(selected[0].risk_level, "low")


# ── Capability gap detection ──────────────────────────────────────────────────


class TestCapabilityGapDetection(unittest.TestCase):
    def test_capability_researcher_returns_role(self):
        loop = _make_loop()
        with patch("core.evolution_loop.analyze_capability_needs") as mock_analyze:
            mock_analyze.return_value = {
                "missing_skills": ["new_skill"],
                "recommended_skills": [],
                "provisioning_actions": [],
            }
            result = loop._capability_researcher("build docker image")
        self.assertEqual(result["role"], "capability_researcher")

    def test_capability_researcher_includes_skill_count(self):
        loop = _make_loop(skills={"skill_a": MagicMock(), "skill_b": MagicMock()})
        with patch("core.evolution_loop.analyze_capability_needs") as mock_analyze:
            mock_analyze.return_value = {"missing_skills": [], "recommended_skills": [], "provisioning_actions": []}
            result = loop._capability_researcher("some goal")
        self.assertEqual(result["skill_count"], 2)

    def test_capability_researcher_passes_goal_to_analyze(self):
        loop = _make_loop()
        with patch("core.evolution_loop.analyze_capability_needs") as mock_analyze:
            mock_analyze.return_value = {"missing_skills": [], "recommended_skills": [], "provisioning_actions": []}
            loop._capability_researcher("add github integration")
        call_args = mock_analyze.call_args
        self.assertEqual(call_args[0][0], "add github integration")


# ── _queue_selected_goals ─────────────────────────────────────────────────────


class TestQueueSelectedGoals(unittest.TestCase):
    def _make_proposal(self, goal: str) -> InnovationProposal:
        return InnovationProposal(
            proposal_id="p",
            title="T",
            category="skill",
            goal=goal,
            rationale="r",
            evidence=[],
            smallest_surface="f",
            expected_value="high",
            risk_level="low",
            verification_cost="tests",
            recommended_action="queue",
        )

    def test_dry_run_does_not_queue(self):
        queue = MagicMock()
        loop = _make_loop(goal_queue=queue)
        proposals = [self._make_proposal("goal A")]
        result = loop._queue_selected_goals(proposals, dry_run=True)
        self.assertFalse(result["attempted"])
        queue.add.assert_not_called()
        queue.prepend_batch.assert_not_called() if hasattr(queue, "prepend_batch") else None

    def test_no_goal_queue_reports_skipped(self):
        loop = _make_loop(goal_queue=None)
        proposals = [self._make_proposal("goal B")]
        result = loop._queue_selected_goals(proposals, dry_run=False)
        self.assertFalse(result["attempted"])
        skipped_reasons = [s["reason"] for s in result["skipped"]]
        self.assertIn("goal_queue_unavailable", skipped_reasons)

    def test_uses_prepend_batch_when_available(self):
        queue = MagicMock()
        queue.queue = []
        loop = _make_loop(goal_queue=queue)
        proposals = [self._make_proposal("goal X")]
        loop._queue_selected_goals(proposals, dry_run=False)
        queue.prepend_batch.assert_called_once_with(["goal X"])

    def test_falls_back_to_add_when_no_prepend_batch(self):
        queue = MagicMock(spec=["add", "queue"])
        queue.queue = []
        loop = _make_loop(goal_queue=queue)
        proposals = [self._make_proposal("goal Y")]
        loop._queue_selected_goals(proposals, dry_run=False)
        queue.add.assert_called_once_with("goal Y")

    def test_already_queued_goals_are_skipped(self):
        queue = MagicMock()
        queue.queue = ["goal already there"]
        loop = _make_loop(goal_queue=queue)
        proposals = [self._make_proposal("goal already there")]
        result = loop._queue_selected_goals(proposals, dry_run=False)
        self.assertEqual(result["queued"], [])
        self.assertEqual(result["skipped"][0]["reason"], "already_queued")

    def test_empty_proposals_returns_not_attempted(self):
        loop = _make_loop(goal_queue=MagicMock())
        result = loop._queue_selected_goals([], dry_run=False)
        self.assertFalse(result["attempted"])


# ── on_cycle_complete trigger ─────────────────────────────────────────────────


class TestOnCycleCompleteTrigger(unittest.TestCase):
    def test_does_not_trigger_before_n_cycles(self):
        loop = _make_loop()
        with patch.object(loop, "run") as mock_run:
            for _ in range(loop.TRIGGER_EVERY_N - 1):
                loop.on_cycle_complete({"goal": "test"})
            mock_run.assert_not_called()

    def test_triggers_at_nth_cycle(self):
        loop = _make_loop()
        with patch.object(loop, "run") as mock_run:
            for _ in range(loop.TRIGGER_EVERY_N):
                loop.on_cycle_complete({"goal": "test"})
            mock_run.assert_called_once()

    def test_hotspot_signal_triggers_early(self):
        loop = _make_loop()
        entry = {
            "goal": "something",
            "phase_outputs": {"skill_context": {"structural_hotspot": {"file": "x.py"}}},
        }
        with patch.object(loop, "run") as mock_run:
            loop.on_cycle_complete(entry)
            mock_run.assert_called_once()

    def test_refactor_hotspot_goal_triggers_early(self):
        loop = _make_loop()
        entry = {"goal": "refactor hotspot in core/orchestrator.py", "phase_outputs": {}}
        with patch.object(loop, "run") as mock_run:
            loop.on_cycle_complete(entry)
            mock_run.assert_called_once()


# ── _summarize_subagents ──────────────────────────────────────────────────────


class TestSummarizeSubagents(unittest.TestCase):
    def setUp(self):
        self.loop = _make_loop()

    def test_ok_status_when_no_error(self):
        reports = [{"role": "architecture_explorer", "data": "ok"}]
        summary = self.loop._summarize_subagents(*reports)
        self.assertEqual(summary[0]["status"], "ok")

    def test_error_status_when_error_key_present(self):
        reports = [{"role": "capability_researcher", "error": "something went wrong"}]
        summary = self.loop._summarize_subagents(*reports)
        self.assertEqual(summary[0]["status"], "error")

    def test_role_preserved_in_summary(self):
        reports = [{"role": "verification_reviewer"}]
        summary = self.loop._summarize_subagents(*reports)
        self.assertEqual(summary[0]["role"], "verification_reviewer")


# ── _mutation_plan_to_dsl ─────────────────────────────────────────────────────


class TestMutationPlanToDsl(unittest.TestCase):
    def setUp(self):
        self.loop = _make_loop()

    def test_add_file_mutation_without_old_content(self):
        plan = {"mutations": [{"file_path": "foo.py", "new_content": "print('hello')"}]}
        dsl = self.loop._mutation_plan_to_dsl(plan)
        self.assertIn("ADD_FILE foo.py", dsl)
        self.assertIn("print('hello')", dsl)

    def test_replace_in_file_mutation_with_old_content(self):
        plan = {"mutations": [{"file_path": "bar.py", "old_content": "old", "new_content": "new"}]}
        dsl = self.loop._mutation_plan_to_dsl(plan)
        self.assertIn("REPLACE_IN_FILE bar.py", dsl)
        self.assertIn("old", dsl)
        self.assertIn("new", dsl)

    def test_skips_mutation_without_file_path(self):
        plan = {"mutations": [{"new_content": "orphan"}]}
        dsl = self.loop._mutation_plan_to_dsl(plan)
        self.assertEqual(dsl.strip(), "")

    def test_empty_mutations_list_returns_empty_string(self):
        plan = {"mutations": []}
        dsl = self.loop._mutation_plan_to_dsl(plan)
        self.assertEqual(dsl.strip(), "")


if __name__ == "__main__":
    unittest.main()
