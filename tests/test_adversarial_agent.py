"""Tests for the Adversarial Agent."""

import pytest
import asyncio

from agents.adversarial import (
    AdversarialAgent,
    AdversarialCritique,
    Finding,
    StrategyResult,
    AdversarialLearner,
    StrategyEffectiveness,
    CritiqueOutcomeTracker,
)
from agents.adversarial.agent import TargetType, AdversarialStrategy
from agents.adversarial.strategies import (
    DevilsAdvocateStrategy,
    EdgeCaseHunterStrategy,
    AssumptionChallengeStrategy,
    WorstCaseStrategy,
    SecurityMindsetStrategy,
)


class TestFinding:
    """Tests for Finding dataclass."""
    
    def test_finding_creation(self):
        """Test finding creation."""
        finding = Finding(
            category="security",
            severity="high",
            description="Potential injection vulnerability",
            evidence="User input used directly in query",
            recommendation="Use parameterized queries",
            confidence=0.85
        )
        
        assert finding.category == "security"
        assert finding.severity == "high"
        assert finding.confidence == 0.85


class TestStrategyResult:
    """Tests for StrategyResult dataclass."""
    
    def test_result_creation(self):
        """Test strategy result creation."""
        finding = Finding(
            category="test",
            severity="medium",
            description="Test finding",
            evidence="Test evidence",
            recommendation="Test recommendation"
        )
        
        result = StrategyResult(
            strategy="test_strategy",
            findings=[finding],
            confidence=0.8,
            execution_time=1.5
        )
        
        assert result.strategy == "test_strategy"
        assert len(result.findings) == 1
        assert result.confidence == 0.8


class TestDevilsAdvocateStrategy:
    """Tests for DevilsAdvocateStrategy."""
    
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test strategy execution."""
        strategy = DevilsAdvocateStrategy(model=None)
        
        result = await strategy.execute(
            target="def add(a, b): return a + b",
            target_type=TargetType.CODE,
            context={},
            intensity=0.8
        )
        
        assert result.strategy == "devils_advocate"
        assert len(result.findings) > 0
        assert result.confidence > 0
    
    def test_estimate_confidence(self):
        """Test confidence estimation."""
        strategy = DevilsAdvocateStrategy()
        
        findings = [
            Finding("cat", "high", "desc", "ev", "rec"),
            Finding("cat", "medium", "desc", "ev", "rec"),
        ]
        
        confidence = strategy._estimate_confidence(findings, 0.8)
        
        assert confidence > 0.0
        assert confidence <= 1.0


class TestEdgeCaseHunterStrategy:
    """Tests for EdgeCaseHunterStrategy."""
    
    @pytest.mark.asyncio
    async def test_execute_code(self):
        """Test edge case hunting for code."""
        strategy = EdgeCaseHunterStrategy(model=None)
        
        result = await strategy.execute(
            target="def process(items): return [x * 2 for x in items]",
            target_type=TargetType.CODE,
            context={"language": "python"},
            intensity=0.8
        )
        
        assert result.strategy == "edge_case_hunter"
        assert len(result.findings) > 0
        
        # Should include edge cases for empty lists, None, etc.
        categories = [f.category for f in result.findings]
        assert "edge_case" in categories


class TestAssumptionChallengeStrategy:
    """Tests for AssumptionChallengeStrategy."""
    
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test assumption challenging."""
        strategy = AssumptionChallengeStrategy(model=None)
        
        result = await strategy.execute(
            target="The API will always return valid JSON",
            target_type=TargetType.DESIGN,
            context={},
            intensity=0.7
        )
        
        assert result.strategy == "assumption_challenge"
        assert len(result.findings) > 0


class TestAdversarialLearner:
    """Tests for AdversarialLearner."""
    
    def test_record_feedback_validated(self):
        """Test recording validated feedback."""
        learner = AdversarialLearner()
        
        learner.record_feedback(
            strategy="devils_advocate",
            target_type=TargetType.CODE,
            was_validated=True,
            severity="high",
            notes="Found real issue"
        )
        
        key = "devils_advocate:code"
        assert key in learner.effectiveness
        assert learner.effectiveness[key].validated_findings == 1
        assert learner.effectiveness[key].success_rate == 1.0
    
    def test_record_feedback_false_positive(self):
        """Test recording false positive feedback."""
        learner = AdversarialLearner()
        
        learner.record_feedback(
            strategy="edge_case_hunter",
            target_type=TargetType.API,
            was_validated=False,
            severity="low",
            notes="Issue already handled"
        )
        
        key = "edge_case_hunter:api"
        assert key in learner.effectiveness
        assert learner.effectiveness[key].false_positives == 1
        assert learner.effectiveness[key].success_rate == 0.0
    
    def test_recommend_strategies(self):
        """Test strategy recommendations."""
        learner = AdversarialLearner()
        
        # Add some feedback
        learner.record_feedback(
            strategy="devils_advocate",
            target_type=TargetType.CODE,
            was_validated=True,
            severity="high",
            notes=""
        )
        learner.record_feedback(
            strategy="edge_case_hunter",
            target_type=TargetType.CODE,
            was_validated=False,
            severity="low",
            notes=""
        )
        
        recommendations = learner.recommend_strategies(
            TargetType.CODE,
            min_success_rate=0.5
        )
        
        assert "devils_advocate" in recommendations
        assert "edge_case_hunter" not in recommendations
    
    def test_get_relevant_notes(self):
        """Test getting relevant notes."""
        learner = AdversarialLearner()
        
        learner.record_feedback(
            strategy="devils_advocate",
            target_type=TargetType.CODE,
            was_validated=True,
            severity="high",
            notes="Check for null inputs"
        )
        
        notes = learner.get_relevant_notes(TargetType.CODE)
        
        assert len(notes) > 0
        assert "Check for null inputs" in notes
    
    def test_get_performance_stats(self):
        """Test performance stats retrieval."""
        learner = AdversarialLearner()
        
        learner.record_feedback(
            strategy="test_strategy",
            target_type=TargetType.CODE,
            was_validated=True,
            severity="high",
            notes=""
        )
        
        stats = learner.get_performance_stats(TargetType.CODE)
        
        assert stats["total_strategies"] > 0
        assert "average_success_rate" in stats


class TestCritiqueOutcomeTracker:
    """Tests for CritiqueOutcomeTracker."""
    
    @pytest.mark.asyncio
    async def test_track_and_record(self):
        """Test tracking and recording outcomes."""
        tracker = CritiqueOutcomeTracker()
        
        # Create mock critique
        class MockCritique:
            def __init__(self):
                self.target_type = TargetType.CODE
                self.findings = [1, 2, 3]
                self.risk_score = 0.7
                self.strategy_results = {"s1": None, "s2": None}
                self.timestamp = 1234567890
        
        critique = MockCritique()
        
        # Start tracking
        tracked = await tracker.start_tracking("critique_1", critique)
        
        assert tracked.critique_id == "critique_1"
        assert tracked.outcome is None
        
        # Record outcome
        result = await tracker.record_outcome(
            "critique_1",
            was_validated=True,
            actual_severity="high",
            notes="Validated"
        )
        
        assert result is not None
        assert result.outcome is not None
        assert result.outcome.was_validated is True
    
    @pytest.mark.asyncio
    async def test_get_pending_validation(self):
        """Test getting pending validations."""
        tracker = CritiqueOutcomeTracker()
        
        class MockCritique:
            def __init__(self):
                self.target_type = TargetType.CODE
                self.findings = []
                self.risk_score = 0.5
                self.strategy_results = {}
                self.timestamp = 1234567890
        
        await tracker.start_tracking("critique_1", MockCritique())
        await tracker.start_tracking("critique_2", MockCritique())
        
        pending = tracker.get_pending_validation()
        
        assert len(pending) == 2
    
    def test_validation_stats(self):
        """Test validation statistics."""
        tracker = CritiqueOutcomeTracker()
        
        stats = tracker.get_validation_stats()
        
        assert stats["total_tracked"] == 0
        assert stats["validation_rate"] == 0


class TestAdversarialAgent:
    """Tests for AdversarialAgent."""
    
    @pytest.mark.asyncio
    async def test_critique_basic(self):
        """Test basic critique functionality."""
        agent = AdversarialAgent(brain=None, model=None)
        
        critique = await agent.critique(
            target="def divide(a, b): return a / b",
            target_type=TargetType.CODE,
            context={"language": "python"},
            strategies=[
                AdversarialStrategy.EDGE_CASE_HUNTER,
                AdversarialStrategy.ASSUMPTION_CHALLENGE,
            ],
            intensity=0.8
        )
        
        assert critique.critique_id is not None
        assert critique.target_type == TargetType.CODE
        assert len(critique.findings) > 0
        assert critique.risk_score >= 0.0
        assert critique.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_critique_code_convenience(self):
        """Test code critique convenience method."""
        agent = AdversarialAgent(brain=None, model=None)
        
        critique = await agent.critique_code(
            code="def process(data): return data[0]",
            language="python"
        )
        
        assert critique.target_type == TargetType.CODE
        assert len(critique.findings) > 0
    
    @pytest.mark.asyncio
    async def test_critique_plan_convenience(self):
        """Test plan critique convenience method."""
        agent = AdversarialAgent(brain=None, model=None)
        
        critique = await agent.critique_plan(
            plan="1. Step one\n2. Step two",
            goal="Implement feature"
        )
        
        assert critique.target_type == TargetType.PLAN
    
    @pytest.mark.asyncio
    async def test_learn_from_outcome(self):
        """Test learning from outcomes."""
        agent = AdversarialAgent(brain=None, model=None)
        
        # Create a critique first
        critique = await agent.critique(
            target="test",
            target_type=TargetType.CODE,
            strategies=[AdversarialStrategy.EDGE_CASE_HUNTER]
        )
        
        # Learn from outcome
        await agent.learn_from_outcome(
            critique.critique_id,
            was_validated=True,
            actual_severity="high",
            notes="Found real bug"
        )
        
        assert critique.validation_status == "validated"
    
    def test_get_strategy_performance(self):
        """Test strategy performance retrieval."""
        agent = AdversarialAgent(brain=None, model=None)
        
        # Add some learning data
        agent.learner.record_feedback(
            strategy="test_strategy",
            target_type=TargetType.CODE,
            was_validated=True,
            severity="medium",
            notes=""
        )
        
        performance = agent.get_strategy_performance(TargetType.CODE)
        
        assert performance["total_strategies"] > 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
