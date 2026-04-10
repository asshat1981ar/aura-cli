"""Unit tests for agents/adversarial/ package."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.adversarial import (
    AdversarialAgent,
    AdversarialCritique,
    AdversarialLearner,
    AdversarialStrategy,
    CritiqueOutcomeTracker,
    Finding,
    StrategyResult,
    TargetType,
)
from agents.adversarial.strategies import (
    AssumptionChallengeStrategy,
    DevilsAdvocateStrategy,
    EdgeCaseHunterStrategy,
    ScalabilityFocusStrategy,
    SecurityMindsetStrategy,
    WorstCaseStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finding(severity="medium", category="test", confidence=0.8):
    return Finding(
        category=category,
        severity=severity,
        description="desc",
        evidence="evidence",
        recommendation="rec",
        confidence=confidence,
    )


def _strategy_result(strategy="test", findings=None, confidence=0.7):
    return StrategyResult(
        strategy=strategy,
        findings=findings or [],
        confidence=confidence,
        execution_time=0.1,
    )


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


class TestFinding:
    def test_default_confidence(self):
        f = Finding(
            category="c",
            severity="low",
            description="d",
            evidence="e",
            recommendation="r",
        )
        assert f.confidence == 0.8

    def test_custom_fields(self):
        f = _finding(severity="critical", category="security", confidence=0.95)
        assert f.severity == "critical"
        assert f.category == "security"
        assert f.confidence == 0.95


# ---------------------------------------------------------------------------
# StrategyResult dataclass
# ---------------------------------------------------------------------------


class TestStrategyResult:
    def test_creation_with_findings(self):
        r = _strategy_result(findings=[_finding()])
        assert r.strategy == "test"
        assert len(r.findings) == 1
        assert r.error is None

    def test_error_field(self):
        r = StrategyResult(strategy="s", findings=[], error="Timeout")
        assert r.error == "Timeout"
        assert r.confidence == 0.0


# ---------------------------------------------------------------------------
# AdversarialAgent — init
# ---------------------------------------------------------------------------


class TestAdversarialAgentInit:
    def test_default_init(self):
        agent = AdversarialAgent()
        assert agent.brain is None
        assert agent.model is None
        assert len(agent.strategies) == len(AdversarialStrategy)

    def test_capabilities_list(self):
        agent = AdversarialAgent()
        assert "adversarial_critique" in agent.capabilities
        assert "red_team" in agent.capabilities

    def test_active_critiques_starts_empty(self):
        agent = AdversarialAgent()
        assert agent.get_active_critiques() == []

    def test_init_with_brain_and_model(self):
        brain = MagicMock()
        model = MagicMock()
        agent = AdversarialAgent(brain=brain, model=model)
        assert agent.brain is brain
        assert agent.model is model


# ---------------------------------------------------------------------------
# AdversarialAgent — private helpers
# ---------------------------------------------------------------------------


class TestAdversarialAgentHelpers:
    def setup_method(self):
        self.agent = AdversarialAgent()

    def test_calculate_confidence_empty_results(self):
        conf = self.agent._calculate_confidence({})
        assert conf == 0.0

    def test_calculate_confidence_no_findings(self):
        results = {"s": _strategy_result(confidence=0.6)}
        conf = self.agent._calculate_confidence(results)
        assert 0 < conf <= 1.0

    def test_calculate_confidence_with_findings(self):
        results = {
            "a": _strategy_result(findings=[_finding()], confidence=0.8),
            "b": _strategy_result(findings=[_finding(), _finding()], confidence=0.6),
        }
        conf = self.agent._calculate_confidence(results)
        assert 0 < conf <= 1.0

    def test_calculate_risk_score_no_findings(self):
        assert self.agent._calculate_risk_score([]) == 0.0

    def test_calculate_risk_score_critical_finding(self):
        score = self.agent._calculate_risk_score([_finding(severity="critical", confidence=1.0)])
        assert score == 1.0

    def test_calculate_risk_score_low_finding(self):
        score = self.agent._calculate_risk_score([_finding(severity="low", confidence=1.0)])
        assert score == 0.1

    def test_calculate_risk_score_mixed_findings(self):
        score = self.agent._calculate_risk_score([_finding(severity="critical"), _finding(severity="low")])
        assert 0 < score <= 1.0

    def test_synthesize_no_findings_assessment(self):
        result = self.agent._synthesize_critiques({})
        assert "No significant" in result.assessment

    def test_synthesize_critical_finding_assessment(self):
        sr = _strategy_result(findings=[_finding(severity="critical")])
        result = self.agent._synthesize_critiques({"s": sr})
        assert "Critical" in result.assessment or "critical" in result.assessment.lower()

    def test_synthesize_high_finding_assessment(self):
        sr = _strategy_result(findings=[_finding(severity="high")])
        result = self.agent._synthesize_critiques({"s": sr})
        assert "high" in result.assessment.lower() or "Significant" in result.assessment

    def test_synthesize_collects_all_findings(self):
        sr = _strategy_result(findings=[_finding(), _finding()])
        result = self.agent._synthesize_critiques({"s": sr})
        assert len(result.findings) == 2


# ---------------------------------------------------------------------------
# AdversarialAgent — async critique
# ---------------------------------------------------------------------------


class TestAdversarialAgentCritique:
    def setup_method(self):
        self.agent = AdversarialAgent()

    def test_critique_returns_adversarial_critique(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                return await self.agent.critique(
                    target="def foo(): pass",
                    target_type=TargetType.CODE,
                )

        critique = asyncio.get_event_loop().run_until_complete(run())
        assert isinstance(critique, AdversarialCritique)
        assert critique.target_type == TargetType.CODE

    def test_critique_stores_in_active_critiques(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                return await self.agent.critique(
                    target="plan text",
                    target_type=TargetType.PLAN,
                )

        critique = asyncio.get_event_loop().run_until_complete(run())
        assert critique.critique_id in self.agent.get_active_critiques()

    def test_critique_code_uses_code_target_type(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                return await self.agent.critique_code("x = 1")

        critique = asyncio.get_event_loop().run_until_complete(run())
        assert critique.target_type == TargetType.CODE

    def test_critique_plan_uses_plan_target_type(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                return await self.agent.critique_plan("my plan", "my goal")

        critique = asyncio.get_event_loop().run_until_complete(run())
        assert critique.target_type == TargetType.PLAN

    def test_critique_has_risk_score(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                return await self.agent.critique("code", TargetType.CODE)

        critique = asyncio.get_event_loop().run_until_complete(run())
        assert 0.0 <= critique.risk_score <= 1.0

    def test_critique_has_timestamp(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                return await self.agent.critique("code", TargetType.CODE)

        critique = asyncio.get_event_loop().run_until_complete(run())
        assert critique.timestamp > 0

    def test_learn_from_outcome_validated(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                critique = await self.agent.critique("target", TargetType.CODE)
                await self.agent.learn_from_outcome(critique.critique_id, was_validated=True, actual_severity="high")
                return critique

        critique = asyncio.get_event_loop().run_until_complete(run())
        assert critique.validation_status == "validated"

    def test_learn_from_outcome_false_positive(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                critique = await self.agent.critique("target", TargetType.CODE)
                await self.agent.learn_from_outcome(critique.critique_id, was_validated=False)
                return critique

        critique = asyncio.get_event_loop().run_until_complete(run())
        assert critique.validation_status == "false_positive"

    def test_learn_from_outcome_unknown_id_is_noop(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                await self.agent.learn_from_outcome("nonexistent-id", was_validated=True)

        # Should not raise
        asyncio.get_event_loop().run_until_complete(run())

    def test_get_strategy_performance_returns_dict(self):
        agent = AdversarialAgent()
        stats = agent.get_strategy_performance()
        assert isinstance(stats, dict)

    def test_get_strategy_performance_filtered_by_type(self):
        agent = AdversarialAgent()
        stats = agent.get_strategy_performance(TargetType.CODE)
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# AdversarialLearner
# ---------------------------------------------------------------------------


class TestAdversarialLearner:
    def setup_method(self):
        self.learner = AdversarialLearner()

    def test_init_empty_state(self):
        assert self.learner.effectiveness == {}
        assert self.learner.pattern_memory == []
        assert self.learner.failure_patterns == []

    def test_record_feedback_validated_increments_validated(self):
        with patch("core.logging_utils.log_json"):
            self.learner.record_feedback(
                strategy="devils_advocate",
                target_type=TargetType.CODE,
                was_validated=True,
                severity="high",
                notes="good catch",
            )
        key = "devils_advocate:code"
        assert self.learner.effectiveness[key].validated_findings == 1
        assert self.learner.effectiveness[key].total_critiques == 1

    def test_record_feedback_false_positive_increments_fp(self):
        with patch("core.logging_utils.log_json"):
            self.learner.record_feedback(
                strategy="edge_case_hunter",
                target_type=TargetType.API,
                was_validated=False,
                severity="medium",
                notes="false alarm",
            )
        key = "edge_case_hunter:api"
        assert self.learner.effectiveness[key].false_positives == 1

    def test_success_rate_computed_correctly(self):
        with patch("core.logging_utils.log_json"):
            for _ in range(3):
                self.learner.record_feedback("s", TargetType.CODE, True, "low", "")
            for _ in range(1):
                self.learner.record_feedback("s", TargetType.CODE, False, "low", "")
        assert abs(self.learner.effectiveness["s:code"].success_rate - 0.75) < 0.01

    def test_get_relevant_notes_returns_recent(self):
        with patch("core.logging_utils.log_json"):
            self.learner.record_feedback("s", TargetType.CODE, True, "high", "note1")
            self.learner.record_feedback("s", TargetType.CODE, True, "high", "note2")
        notes = self.learner.get_relevant_notes(TargetType.CODE)
        assert "note1" in notes or "note2" in notes

    def test_get_relevant_notes_empty_when_no_match(self):
        notes = self.learner.get_relevant_notes(TargetType.DESIGN)
        assert notes == []

    def test_recommend_strategies_returns_list(self):
        with patch("core.logging_utils.log_json"):
            for _ in range(6):
                self.learner.record_feedback("good_strategy", TargetType.CODE, True, "high", "")
        recs = self.learner.recommend_strategies(TargetType.CODE)
        assert "good_strategy" in recs

    def test_recommend_strategies_skips_low_success(self):
        with patch("core.logging_utils.log_json"):
            for _ in range(10):
                self.learner.record_feedback("bad_strategy", TargetType.CODE, False, "low", "")
        recs = self.learner.recommend_strategies(TargetType.CODE, min_success_rate=0.5)
        assert "bad_strategy" not in recs

    def test_get_performance_stats_overall(self):
        with patch("core.logging_utils.log_json"):
            self.learner.record_feedback("s", TargetType.CODE, True, "high", "")
        stats = self.learner.get_performance_stats()
        assert "total_strategies" in stats
        assert stats["total_strategies"] >= 1

    def test_get_performance_stats_by_target_type(self):
        with patch("core.logging_utils.log_json"):
            self.learner.record_feedback("s", TargetType.CODE, True, "medium", "")
        stats = self.learner.get_performance_stats(TargetType.CODE)
        assert "average_success_rate" in stats

    def test_get_strategy_recommendations_empty(self):
        recs = self.learner.get_strategy_recommendations()
        assert recs == {"recommended": [], "avoid": [], "experimental": []}

    def test_get_strategy_recommendations_categorizes_correctly(self):
        with patch("core.logging_utils.log_json"):
            for _ in range(10):
                self.learner.record_feedback("great", TargetType.CODE, True, "high", "")
            for _ in range(10):
                self.learner.record_feedback("poor", TargetType.CODE, False, "low", "")
        recs = self.learner.get_strategy_recommendations()
        strats_rec = [r["strategy"] for r in recs["recommended"]]
        strats_avoid = [r["strategy"] for r in recs["avoid"]]
        assert "great" in strats_rec
        assert "poor" in strats_avoid


# ---------------------------------------------------------------------------
# CritiqueOutcomeTracker
# ---------------------------------------------------------------------------


class TestCritiqueOutcomeTracker:
    def setup_method(self):
        self.tracker = CritiqueOutcomeTracker()

    def _make_critique(self, critique_id="abc123"):
        c = MagicMock()
        c.target_type = TargetType.CODE
        c.findings = [_finding()]
        c.risk_score = 0.5
        c.strategy_results = {"devils_advocate": _strategy_result()}
        import time

        c.timestamp = time.time()
        return c

    def test_init_empty(self):
        assert len(self.tracker._tracked) == 0

    def test_start_tracking_creates_entry(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                return await self.tracker.start_tracking("id1", self._make_critique("id1"))

        tracked = asyncio.get_event_loop().run_until_complete(run())
        assert tracked.critique_id == "id1"
        assert self.tracker.get_tracked("id1") is not None

    def test_record_outcome_updates_tracked(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                await self.tracker.start_tracking("id2", self._make_critique("id2"))
                return await self.tracker.record_outcome("id2", was_validated=True, actual_severity="high")

        tracked = asyncio.get_event_loop().run_until_complete(run())
        assert tracked.outcome.was_validated is True
        assert tracked.outcome.actual_severity == "high"

    def test_record_outcome_unknown_id_returns_none(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                return await self.tracker.record_outcome("unknown", was_validated=True)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is None

    def test_get_pending_validation(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                await self.tracker.start_tracking("id3", self._make_critique("id3"))

        asyncio.get_event_loop().run_until_complete(run())
        pending = self.tracker.get_pending_validation()
        assert any(t.critique_id == "id3" for t in pending)

    def test_get_validation_stats_empty(self):
        stats = self.tracker.get_validation_stats()
        assert stats["total_tracked"] == 0
        assert stats["pending"] == 0

    def test_get_validation_stats_with_data(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                await self.tracker.start_tracking("x1", self._make_critique("x1"))
                await self.tracker.record_outcome("x1", was_validated=True)

        asyncio.get_event_loop().run_until_complete(run())
        stats = self.tracker.get_validation_stats()
        assert stats["total_tracked"] == 1
        assert stats["validated_true"] == 1
        assert stats["accuracy_rate"] == 1.0

    def test_get_stats_by_strategy(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                c = self._make_critique("y1")
                c.strategy_results = {"devils_advocate": _strategy_result("devils_advocate")}
                await self.tracker.start_tracking("y1", c)
                await self.tracker.record_outcome("y1", was_validated=True)

        asyncio.get_event_loop().run_until_complete(run())
        stats = self.tracker.get_stats_by_strategy()
        assert "devils_advocate" in stats
        assert stats["devils_advocate"]["validated_true"] == 1

    def test_trim_when_over_max(self):
        tracker = CritiqueOutcomeTracker(max_tracked=2)

        async def run():
            with patch("core.logging_utils.log_json"):
                for i in range(4):
                    c = self._make_critique(f"id{i}")
                    await tracker.start_tracking(f"id{i}", c)

        asyncio.get_event_loop().run_until_complete(run())
        assert len(tracker._tracked) <= 2

    def test_get_recent_outcomes_limit(self):
        async def run():
            with patch("core.logging_utils.log_json"):
                for i in range(5):
                    c = self._make_critique(f"r{i}")
                    await self.tracker.start_tracking(f"r{i}", c)

        asyncio.get_event_loop().run_until_complete(run())
        recent = self.tracker.get_recent_outcomes(limit=3)
        assert len(recent) <= 3


# ---------------------------------------------------------------------------
# Adversarial Strategies
# ---------------------------------------------------------------------------


class TestAdversarialStrategies:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_devils_advocate_heuristic_returns_findings(self):
        s = DevilsAdvocateStrategy(model=None)
        result = self._run(s.execute("some code", TargetType.CODE, {}, 0.8))
        assert isinstance(result, StrategyResult)
        assert result.strategy == "devils_advocate"
        assert len(result.findings) > 0

    def test_devils_advocate_plan_specific_finding(self):
        s = DevilsAdvocateStrategy(model=None)
        result = self._run(s.execute("a plan", TargetType.PLAN, {}, 0.8))
        severities = [f.severity for f in result.findings]
        assert any(sev in severities for sev in ["high", "medium", "critical"])

    def test_edge_case_hunter_code_findings(self):
        s = EdgeCaseHunterStrategy(model=None)
        result = self._run(s.execute("def foo(x): return x", TargetType.CODE, {}, 0.9))
        assert result.strategy == "edge_case_hunter"
        assert len(result.findings) > 0

    def test_edge_case_hunter_api_findings(self):
        s = EdgeCaseHunterStrategy(model=None)
        result = self._run(s.execute("POST /api/data", TargetType.API, {}, 0.8))
        assert any(f.category == "edge_case" for f in result.findings)

    def test_assumption_challenge_returns_findings(self):
        s = AssumptionChallengeStrategy(model=None)
        result = self._run(s.execute("code snippet", TargetType.CODE, {}, 0.7))
        assert result.strategy == "assumption_challenge"
        assert len(result.findings) > 0
        assert all(f.category == "assumption" for f in result.findings)

    def test_worst_case_strategy_returns_critical(self):
        s = WorstCaseStrategy(model=None)
        result = self._run(s.execute("system", TargetType.ARCHITECTURE, {}, 1.0))
        assert result.strategy == "worst_case"
        assert any(f.severity == "critical" for f in result.findings)

    def test_security_mindset_code_returns_findings(self):
        s = SecurityMindsetStrategy(model=None)
        result = self._run(s.execute("user_input + query", TargetType.CODE, {}, 0.9))
        assert result.strategy == "security_mindset"
        assert len(result.findings) > 0

    def test_security_mindset_non_code_no_findings(self):
        s = SecurityMindsetStrategy(model=None)
        result = self._run(s.execute("a plan", TargetType.PLAN, {}, 0.9))
        # No code/api target → no findings
        assert result.findings == []

    def test_scalability_focus_returns_findings(self):
        s = ScalabilityFocusStrategy(model=None)
        result = self._run(s.execute("for x in items:", TargetType.CODE, {}, 0.8))
        assert result.strategy == "scalability_focus"
        assert len(result.findings) >= 3

    def test_strategy_confidence_within_bounds(self):
        for StratCls in [
            DevilsAdvocateStrategy,
            EdgeCaseHunterStrategy,
            AssumptionChallengeStrategy,
            WorstCaseStrategy,
            SecurityMindsetStrategy,
            ScalabilityFocusStrategy,
        ]:
            s = StratCls(model=None)
            result = self._run(s.execute("target", TargetType.CODE, {}, 0.8))
            assert 0.0 <= result.confidence <= 1.0, f"{StratCls.__name__} confidence out of bounds"

    def test_strategy_with_model_calls_generate(self):
        model = MagicMock()
        model.generate.return_value = ""
        s = DevilsAdvocateStrategy(model=model)
        result = self._run(s.execute("target", TargetType.CODE, {}, 0.8))
        model.generate.assert_called_once()
        assert isinstance(result, StrategyResult)
