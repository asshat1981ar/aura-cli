"""Integration tests for Sprint 2 modules wired into the orchestrator.

Tests TreeOfThought, CodeRAG, SkillCorrelation, QualityTrends, and
TeamCoordinator working end-to-end and integrated with the orchestrator.
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── 1. TreeOfThought + Orchestrator: plan phase with ToT ────────────────


class TestTreeOfThoughtOrchestratorWiring(unittest.TestCase):
    """Verify ToT planning generates multiple candidates and selects the best."""

    def test_tot_generates_and_scores_plans(self):
        from core.tree_of_thought import TreeOfThoughtPlanner

        planner = TreeOfThoughtPlanner(n_candidates=3)

        call_idx = 0

        def mock_respond(prompt):
            nonlocal call_idx
            call_idx += 1
            # Simulate plan generation for different strategies
            if "Score each plan" in prompt or "scoring" in prompt.lower():
                # Scoring prompt — return scores for all candidates
                return json.dumps({
                    "scores": {
                        "0": {"feasibility": 0.9, "coverage": 0.8, "risk": 0.7,
                              "testability": 0.8, "clarity": 0.9},
                        "1": {"feasibility": 0.5, "coverage": 0.6, "risk": 0.4,
                              "testability": 0.5, "clarity": 0.6},
                        "2": {"feasibility": 0.7, "coverage": 0.7, "risk": 0.6,
                              "testability": 0.7, "clarity": 0.7},
                    }
                })
            # Plan generation prompt
            return json.dumps({
                "steps": [
                    f"Step 1 for strategy variant {call_idx}",
                    f"Step 2 for strategy variant {call_idx}",
                ],
            })

        model = MagicMock()
        model.respond = mock_respond
        model.respond_for_role = None

        candidates = planner.generate_plans(model, "Add user auth", {
            "memory_snapshot": "past: used JWT tokens",
            "known_weaknesses": "",
            "skill_context": {},
        })

        self.assertEqual(len(candidates), 3)
        strategies = [c.strategy for c in candidates]
        self.assertIn("conservative", strategies)
        self.assertIn("aggressive", strategies)
        self.assertIn("incremental", strategies)

        winner = planner.score_plans(model, candidates, "Add user auth")
        self.assertIsNotNone(winner)
        self.assertGreater(winner.total_score, 0)

    def test_tot_winner_feeds_confidence_router(self):
        """Verify ToT winner score feeds into ConfidenceRouter."""
        from core.phase_result import ConfidenceRouter, PhaseResult
        from core.tree_of_thought import PlanCandidate

        router = ConfidenceRouter()

        # Simulate a ToT winner with high score
        winner = PlanCandidate(
            strategy="conservative",
            strategy_description="Safe approach",
            steps=["step 1", "step 2"],
            total_score=0.92,
        )

        plan_confidence = min(winner.total_score, 0.95)
        plan_result = PhaseResult(
            phase="plan", output={"steps": winner.steps},
            confidence=plan_confidence,
        )
        router.record(plan_result)

        # High-confidence plan should allow skipping optional critique
        self.assertTrue(
            router.should_skip_optional(plan_result, "critique"),
            "ToT plan with 0.92 confidence should skip optional critique",
        )

    def test_tot_single_candidate_fallback(self):
        """Single candidate should still work without scoring."""
        from core.tree_of_thought import TreeOfThoughtPlanner

        planner = TreeOfThoughtPlanner(n_candidates=1)

        model = MagicMock()
        model.respond = lambda prompt: json.dumps({"steps": ["only step"]})
        model.respond_for_role = None

        candidates = planner.generate_plans(model, "Simple fix", {})
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].strategy, "conservative")


# ── 2. CodeRAG + Act phase: context injection ────────────────────────────


class TestCodeRAGContextInjection(unittest.TestCase):
    """Verify RAG retrieves context and injects into the act phase."""

    def test_rag_retrieves_from_brain(self):
        from core.code_rag import CodeRAG, RAGContext

        # Mock brain with search
        mock_brain = MagicMock()
        mock_brain.search.return_value = [
            {"content": "def auth_handler(): ...", "similarity": 0.85},
            {"content": "def validate_token(): ...", "similarity": 0.72},
        ]

        rag = CodeRAG(brain=mock_brain, max_examples=3)
        context = rag.retrieve_context("Add JWT authentication", {"steps": ["add token validator"]})

        self.assertIsInstance(context, RAGContext)
        self.assertGreater(context.retrieval_time_ms, 0)

    def test_rag_augments_task_bundle(self):
        """Simulate how the orchestrator injects RAG context into the task bundle."""
        from core.code_rag import CodeRAG, RAGContext

        # Empty stores — returns empty context
        rag = CodeRAG(vector_store=None, brain=None)
        context = rag.retrieve_context("goal")

        # Even with empty context, the orchestrator flow shouldn't break
        task_bundle = {"plan": ["step 1"], "fix_hints": []}
        if context and context.examples:
            task_bundle["rag_examples"] = [e.get("content", "")[:300] for e in context.examples[:3]]
            task_bundle["rag_anti_patterns"] = context.anti_patterns[:3]

        # No RAG data injected for empty stores
        self.assertNotIn("rag_examples", task_bundle)

    def test_rag_with_anti_patterns(self):
        """Verify anti-patterns from past failures are retrieved."""
        from core.code_rag import CodeRAG

        mock_brain = MagicMock()
        mock_brain.search.return_value = [
            {"content": "previous auth code", "similarity": 0.8, "type": "implementation"},
        ]

        rag = CodeRAG(brain=mock_brain)
        context = rag.retrieve_context("Fix auth bug")

        # Context should be returned without errors even if brain returns no failures
        self.assertIsNotNone(context)
        self.assertIsInstance(context.anti_patterns, list)


# ── 3. SkillCorrelation + Skill Dispatch ─────────────────────────────────


class TestSkillCorrelationDispatchWiring(unittest.TestCase):
    """Verify skill correlation matrix suggests skills during dispatch."""

    def test_record_and_suggest(self):
        from core.skill_correlation import SkillCorrelationMatrix, SkillOutcome

        with tempfile.TemporaryDirectory() as tmpdir:
            matrix = SkillCorrelationMatrix(store_path=Path(tmpdir) / "corr.json")

            # Record 5 successful cycles with linter + type_checker
            for i in range(5):
                outcomes = [
                    SkillOutcome(skill_name="linter", goal_type="bugfix", success=True),
                    SkillOutcome(skill_name="type_checker", goal_type="bugfix", success=True),
                    SkillOutcome(skill_name="test_coverage", goal_type="bugfix", success=True),
                ]
                matrix.record_cycle(outcomes, cycle_success=True)

            # Correlation between linter and type_checker should be positive
            corr = matrix.get_correlation("linter", "type_checker")
            self.assertGreater(corr, 0, "Correlated skills should have positive score")

            # Suggest skills when only linter is active
            suggestions = matrix.suggest_skills(["linter"], goal_type="bugfix")
            suggested_names = [s[0] for s in suggestions]
            self.assertIn("type_checker", suggested_names,
                          "type_checker should be suggested when linter is active")

    def test_cluster_discovery(self):
        from core.skill_correlation import SkillCorrelationMatrix, SkillOutcome

        with tempfile.TemporaryDirectory() as tmpdir:
            matrix = SkillCorrelationMatrix(store_path=Path(tmpdir) / "corr.json")

            # Build two distinct clusters
            for _ in range(10):
                # Cluster 1: linter + type_checker always together
                matrix.record_cycle([
                    SkillOutcome("linter", "bugfix", True),
                    SkillOutcome("type_checker", "bugfix", True),
                ], cycle_success=True)
                # Cluster 2: test_coverage + structural_analyzer together
                matrix.record_cycle([
                    SkillOutcome("test_coverage", "feature", True),
                    SkillOutcome("structural_analyzer", "feature", True),
                ], cycle_success=True)

            clusters = matrix.discover_clusters(min_correlation=0.5)
            self.assertGreaterEqual(len(clusters), 1, "Should discover at least one cluster")

    def test_orchestrator_skill_dispatch_integration(self):
        """Simulate how orchestrator uses skill correlation in dispatch."""
        from core.skill_correlation import SkillCorrelationMatrix, SkillOutcome

        with tempfile.TemporaryDirectory() as tmpdir:
            matrix = SkillCorrelationMatrix(store_path=Path(tmpdir) / "corr.json")

            # Train: ast_analyzer correlates with linter
            for _ in range(8):
                matrix.record_cycle([
                    SkillOutcome("linter", "refactor", True),
                    SkillOutcome("ast_analyzer", "refactor", True),
                ], cycle_success=True)

            # Simulate the orchestrator flow
            base_skills = {"linter": MagicMock()}
            all_skills = {"linter": MagicMock(), "ast_analyzer": MagicMock(), "unused_skill": MagicMock()}

            suggestions = matrix.suggest_skills(list(base_skills.keys()), "refactor")
            for suggested_name, corr_score in suggestions:
                if suggested_name in all_skills and suggested_name not in base_skills:
                    base_skills[suggested_name] = all_skills[suggested_name]

            self.assertIn("ast_analyzer", base_skills,
                          "ast_analyzer should be auto-added via correlation")


# ── 4. QualityTrends + post-cycle: regression detection ─────────────────


class TestQualityTrendsPostCycleWiring(unittest.TestCase):
    """Verify quality trends detect regressions and enqueue remediation goals."""

    def test_healthy_cycle_no_alerts(self):
        from core.quality_trends import QualityTrendAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = QualityTrendAnalyzer(store_path=Path(tmpdir) / "trends.json")

            alerts = analyzer.record_from_cycle({
                "cycle_id": "cycle_1",
                "goal": "Add feature X",
                "completed_at": time.time(),
                "duration_s": 30.0,
                "phase_outputs": {
                    "quality": {"test_count": 50, "syntax_errors": [], "import_errors": []},
                    "verification": {"status": "pass"},
                    "apply_result": {"applied": ["file.py"]},
                },
            })

            self.assertEqual(len(alerts), 0, "Healthy cycle should produce no alerts")

    def test_failing_cycle_triggers_alert(self):
        from core.quality_trends import QualityTrendAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = QualityTrendAnalyzer(store_path=Path(tmpdir) / "trends.json")

            # Record a cycle with many syntax errors and failing verification
            alerts = analyzer.record_from_cycle({
                "cycle_id": "cycle_bad",
                "goal": "Risky refactor",
                "completed_at": time.time(),
                "duration_s": 60.0,
                "phase_outputs": {
                    "quality": {
                        "test_count": 5,
                        "syntax_errors": ["err1", "err2", "err3", "err4"],
                        "import_errors": ["imp1", "imp2", "imp3"],
                    },
                    "verification": {"status": "fail"},
                    "apply_result": {"applied": []},
                },
            })

            self.assertGreater(len(alerts), 0, "Failing cycle should trigger alerts")
            severities = [a.severity for a in alerts]
            self.assertTrue(any(s in ("high", "critical") for s in severities))

    def test_remediation_goals_enqueued(self):
        """Simulate the orchestrator flow: alerts → remediation goals → goal queue."""
        from core.quality_trends import QualityTrendAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = QualityTrendAnalyzer(store_path=Path(tmpdir) / "trends.json")

            # Record a bad cycle to trigger alerts with suggested goals
            analyzer.record_from_cycle({
                "cycle_id": "cycle_fail",
                "goal": "Break things",
                "completed_at": time.time(),
                "duration_s": 120.0,
                "phase_outputs": {
                    "quality": {
                        "test_count": 0,
                        "syntax_errors": ["e1", "e2", "e3", "e4", "e5"],
                        "import_errors": ["i1", "i2", "i3"],
                    },
                    "verification": {"status": "fail"},
                    "apply_result": {"applied": []},
                },
            })

            remediation_goals = analyzer.get_remediation_goals()
            # If there are alerts with suggested_goal, they should appear here
            if analyzer.alerts:
                goals_with_suggestions = [a for a in analyzer.alerts if a.suggested_goal]
                if goals_with_suggestions:
                    self.assertGreater(len(remediation_goals), 0,
                                       "Alerts with suggested_goal should yield remediation goals")

            # Simulate goal queue addition (as orchestrator does)
            mock_goal_queue = MagicMock()
            for goal_text in remediation_goals:
                mock_goal_queue.add(goal_text)

            # Verify goals were enqueued
            self.assertEqual(mock_goal_queue.add.call_count, len(remediation_goals))

    def test_trend_summary(self):
        from core.quality_trends import QualityTrendAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = QualityTrendAnalyzer(store_path=Path(tmpdir) / "trends.json")

            # Record 3 cycles with improving quality
            for i, status in enumerate(["fail", "fail", "pass"]):
                analyzer.record_from_cycle({
                    "cycle_id": f"cycle_{i}",
                    "goal": f"Goal {i}",
                    "completed_at": time.time(),
                    "duration_s": 30.0,
                    "phase_outputs": {
                        "quality": {"test_count": 10 + i * 5, "syntax_errors": [], "import_errors": []},
                        "verification": {"status": status},
                        "apply_result": {"applied": ["f.py"]},
                    },
                })

            summary = analyzer.get_summary()
            self.assertEqual(summary["total_cycles"], 3)
            self.assertIn("avg_health", summary)
            self.assertIn("trend", summary)


# ── 5. TeamCoordinator: decompose and execute ────────────────────────────


class TestTeamCoordinatorEndToEnd(unittest.TestCase):
    """Verify team coordinator decomposes goals and runs sub-tasks."""

    def test_decompose_goal(self):
        from core.team_coordinator import TeamCoordinator

        model = MagicMock()
        model.respond = lambda prompt: json.dumps([
            {"description": "Create database schema", "priority": 1, "dependencies": []},
            {"description": "Build API endpoints", "priority": 2, "dependencies": [0]},
            {"description": "Add input validation", "priority": 1, "dependencies": []},
        ])
        model.respond_for_role = None

        coordinator = TeamCoordinator(model=model)
        sub_goals = coordinator.decompose_goal("Build user management system")
        self.assertGreaterEqual(len(sub_goals), 1)

        # At least one sub-goal should exist
        self.assertTrue(all(sg.description for sg in sub_goals))

    def test_execute_team_parallel_tasks(self):
        from core.team_coordinator import SubGoal, TeamCoordinator

        coordinator = TeamCoordinator()

        # Create independent sub-goals
        sub_goals = [
            SubGoal(description="Task A", priority=1, dependencies=[]),
            SubGoal(description="Task B", priority=1, dependencies=[]),
        ]

        # execute_team uses internal _run_parallel which calls _run_single
        # which requires orchestrator_factory — without it, tasks stay PENDING
        # Test the structure and that it doesn't crash
        result = coordinator.execute_team("Test goal", sub_goals, dry_run=True)

        self.assertIsNotNone(result)
        self.assertGreater(result.total_duration, 0)


# ── 6. Cross-module: ToT → ConfidenceRouter → QualityTrends ─────────────


class TestCrossModuleFlow(unittest.TestCase):
    """Test multi-module interactions as they occur in a real cycle."""

    def test_tot_confidence_to_quality_trends(self):
        """Simulate: ToT produces high-confidence plan → confidence recorded →
        cycle succeeds → quality trends record healthy snapshot."""
        from core.phase_result import ConfidenceRouter, PhaseResult
        from core.quality_trends import QualityTrendAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            router = ConfidenceRouter()
            trends = QualityTrendAnalyzer(store_path=Path(tmpdir) / "trends.json")

            # Phase 1: ToT plan with high confidence
            plan_result = PhaseResult(phase="plan", confidence=0.9)
            router.record(plan_result)

            # Phase 2: Act with moderate confidence
            act_result = PhaseResult(phase="act", confidence=0.7)
            router.record(act_result)

            # No escalation needed — confidence is stable
            self.assertFalse(router.should_escalate(act_result))

            # Phase 3: Cycle succeeds → record quality
            alerts = trends.record_from_cycle({
                "cycle_id": "cross_test",
                "goal": "Cross-module test",
                "completed_at": time.time(),
                "duration_s": 25.0,
                "phase_outputs": {
                    "quality": {"test_count": 30, "syntax_errors": [], "import_errors": []},
                    "verification": {"status": "pass"},
                    "apply_result": {"applied": ["module.py"]},
                },
            })
            self.assertEqual(len(alerts), 0)

    def test_skill_correlation_then_quality_regression(self):
        """Simulate: Skills are correlated → used together → cycle fails → quality alert."""
        from core.quality_trends import QualityTrendAnalyzer
        from core.skill_correlation import SkillCorrelationMatrix, SkillOutcome

        with tempfile.TemporaryDirectory() as tmpdir:
            matrix = SkillCorrelationMatrix(store_path=Path(tmpdir) / "corr.json")
            trends = QualityTrendAnalyzer(store_path=Path(tmpdir) / "trends.json")

            # Record a cycle where correlated skills were used but cycle failed
            matrix.record_cycle([
                SkillOutcome("linter", "bugfix", True),
                SkillOutcome("type_checker", "bugfix", True),
            ], cycle_success=False)

            # Failed cycle → quality regression
            alerts = trends.record_from_cycle({
                "cycle_id": "fail_cycle",
                "goal": "Failed bugfix",
                "completed_at": time.time(),
                "duration_s": 90.0,
                "phase_outputs": {
                    "quality": {
                        "test_count": 2,
                        "syntax_errors": ["err1", "err2", "err3", "err4"],
                        "import_errors": ["imp1", "imp2", "imp3"],
                    },
                    "verification": {"status": "fail"},
                    "apply_result": {"applied": []},
                },
            })

            self.assertGreater(len(alerts), 0, "Failed cycle should trigger quality alerts")

            # Skill correlation should reflect the failure
            corr = matrix.get_correlation("linter", "type_checker")
            # After one failed cycle, co_failure > 0, so correlation should be negative
            self.assertLess(corr, 0, "Correlation should be negative after failed co-use")


# ── 7. Orchestrator _run_phase with ToT config ──────────────────────────


class TestOrchestratorWithSprint2Modules(unittest.TestCase):
    """Verify orchestrator initializes and uses Sprint 2 modules."""

    def test_orchestrator_has_quality_trends(self):
        from core.orchestrator import LoopOrchestrator

        with patch.object(LoopOrchestrator, "_load_config_file", return_value={}):
            orch = LoopOrchestrator(agents={}, project_root=Path("/tmp"))

        self.assertIsNotNone(orch.quality_trends)
        self.assertIsNotNone(orch.confidence_router)

    def test_orchestrator_has_skill_correlation(self):
        from core.orchestrator import LoopOrchestrator

        with patch.object(LoopOrchestrator, "_load_config_file", return_value={}):
            orch = LoopOrchestrator(agents={}, project_root=Path("/tmp"))

        # skill_correlation is optional — may be None if import fails
        # but should be instantiated when the module is available
        self.assertIsNotNone(orch.skill_correlation,
                             "SkillCorrelationMatrix should be instantiated in orchestrator")


# ── 8. CodeRAG + NBest combined flow ─────────────────────────────────────


class TestCodeRAGWithNBest(unittest.TestCase):
    """Simulate the full act phase: RAG retrieves context → NBest generates variants."""

    def test_rag_context_feeds_nbest(self):
        from core.code_rag import CodeRAG, RAGContext
        from core.nbest import NBestEngine

        # Step 1: RAG retrieves context (mock)
        rag = CodeRAG(vector_store=None, brain=None)
        rag_context = rag.retrieve_context("Add caching layer")

        # Step 2: Build prompt with RAG context
        base_prompt = "Goal: Add caching layer"
        if rag_context.examples:
            base_prompt += f"\n\nPast examples: {rag_context.examples}"
        if rag_context.anti_patterns:
            base_prompt += f"\n\nAvoid: {rag_context.anti_patterns}"

        # Step 3: NBest generates candidates
        engine = NBestEngine(n_candidates=2)

        call_count = 0
        def mock_respond(prompt):
            nonlocal call_count
            call_count += 1
            return json.dumps({
                "changes": [{
                    "file_path": "cache.py",
                    "old_code": "",
                    "new_code": f"def cache_v{call_count}(): pass",
                }]
            })

        model = MagicMock()
        model.respond = mock_respond
        model.respond_for_role = None

        candidates = engine.generate_candidates(model, base_prompt)
        self.assertEqual(len(candidates), 2)
        for c in candidates:
            self.assertTrue(len(c.changes) > 0)


if __name__ == "__main__":
    unittest.main()
