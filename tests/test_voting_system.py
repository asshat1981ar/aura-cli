"""Tests for the LLM Voting System."""

import pytest
import asyncio

from core.voting import (
    VotingEngine,
    VoteConfig,
    Vote,
    VotingStrategy,
    SimpleMajorityStrategy,
    WeightedConfidenceStrategy,
    BordaCountStrategy,
    ExpertPanelStrategy,
    ConsensusAnalyzer,
    ConsensusAnalysis,
)


class TestVote:
    """Tests for Vote dataclass."""
    
    def test_vote_creation(self):
        """Test vote creation."""
        vote = Vote(
            model_id="model_a",
            selection="option_1",
            confidence=0.9,
            reasoning="Best option"
        )
        
        assert vote.model_id == "model_a"
        assert vote.selection == "option_1"
        assert vote.confidence == 0.9


class TestSimpleMajorityStrategy:
    """Tests for SimpleMajorityStrategy."""
    
    def test_simple_majority(self):
        """Test basic majority voting."""
        strategy = SimpleMajorityStrategy()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.8),
            "model_b": Vote("model_b", "option_a", 0.7),
            "model_c": Vote("model_c", "option_b", 0.9),
        }
        
        result = strategy.aggregate(votes, ["option_a", "option_b"])
        
        assert result.winner == "option_a"
        assert result.confidence > 0.5
        assert result.tie_detected is False
    
    def test_tie_detection(self):
        """Test tie detection."""
        strategy = SimpleMajorityStrategy()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.8),
            "model_b": Vote("model_b", "option_b", 0.8),
        }
        
        result = strategy.aggregate(votes, ["option_a", "option_b"])
        
        assert result.tie_detected is True


class TestWeightedConfidenceStrategy:
    """Tests for WeightedConfidenceStrategy."""
    
    def test_weighted_voting(self):
        """Test weighted confidence voting."""
        strategy = WeightedConfidenceStrategy()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.9),
            "model_b": Vote("model_b", "option_a", 0.6),
            "model_c": Vote("model_c", "option_b", 0.8),
        }
        
        result = strategy.aggregate(votes, ["option_a", "option_b"])
        
        # option_a should win due to higher aggregate confidence
        assert result.winner == "option_a"
        assert result.confidence > 0.0
    
    def test_with_model_weights(self):
        """Test with custom model weights."""
        weights = {"model_a": 2.0, "model_b": 1.0, "model_c": 1.0}
        strategy = WeightedConfidenceStrategy(lambda: weights)
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.8),  # Weighted: 1.6
            "model_b": Vote("model_b", "option_b", 1.0),  # Weighted: 1.0
            "model_c": Vote("model_c", "option_b", 1.0),  # Weighted: 1.0
        }
        
        result = strategy.aggregate(votes, ["option_a", "option_b"])
        
        # With weights, option_a should win despite fewer votes
        assert result.winner == "option_a"


class TestBordaCountStrategy:
    """Tests for BordaCountStrategy."""
    
    def test_borda_count(self):
        """Test Borda count voting."""
        strategy = BordaCountStrategy()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.9),
            "model_b": Vote("model_b", "option_b", 0.8),
            "model_c": Vote("model_c", "option_c", 0.7),
        }
        
        result = strategy.aggregate(votes, ["option_a", "option_b", "option_c"])
        
        # option_a should win (first choice)
        assert result.winner == "option_a"


class TestExpertPanelStrategy:
    """Tests for ExpertPanelStrategy."""
    
    def test_expert_weighting(self):
        """Test expert domain weighting."""
        strategy = ExpertPanelStrategy()
        
        votes = {
            "claude-sonnet": Vote("claude-sonnet", "architecture_a", 0.8),
            "gpt-4": Vote("gpt-4", "architecture_b", 0.9),
        }
        
        # With architecture domain, Claude should have higher weight
        result = strategy.aggregate(
            votes, 
            ["architecture_a", "architecture_b"],
            domain="architecture"
        )
        
        # Claude's vote should be weighted higher for architecture
        assert result.winner == "architecture_a"
    
    def test_fallback_without_domain(self):
        """Test fallback without domain."""
        strategy = ExpertPanelStrategy()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.8),
            "model_b": Vote("model_b", "option_b", 0.9),
        }
        
        result = strategy.aggregate(votes, ["option_a", "option_b"], domain=None)
        
        # Should fall back to simple majority
        assert result.winner in ["option_a", "option_b"]


class TestConsensusAnalyzer:
    """Tests for ConsensusAnalyzer."""
    
    def test_analyze_unanimous(self):
        """Test unanimous consensus."""
        analyzer = ConsensusAnalyzer()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.9),
            "model_b": Vote("model_b", "option_a", 0.8),
            "model_c": Vote("model_c", "option_a", 0.85),
        }
        
        analysis = analyzer.analyze(votes, "option_a")
        
        assert analysis.consensus_level > 0.8
        assert analysis.is_unanimous is True
        assert analysis.requires_discussion is False
    
    def test_analyze_split(self):
        """Test split vote consensus."""
        analyzer = ConsensusAnalyzer()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.9),
            "model_b": Vote("model_b", "option_b", 0.9),
            "model_c": Vote("model_c", "option_c", 0.8),
        }
        
        analysis = analyzer.analyze(votes, "option_a")
        
        assert analysis.consensus_level < 0.7
        assert analysis.is_unanimous is False
        assert analysis.requires_discussion is True
    
    def test_calculate_strength(self):
        """Test strength calculation."""
        analyzer = ConsensusAnalyzer()
        
        analysis_high = ConsensusAnalysis(
            consensus_level=0.95,
            primary_cluster="option_a",
            secondary_clusters=[],
            entropy=0.1,
            is_unanimous=True,
            requires_discussion=False
        )
        
        analysis_low = ConsensusAnalysis(
            consensus_level=0.3,
            primary_cluster="option_a",
            secondary_clusters=["option_b"],
            entropy=0.8,
            is_unanimous=False,
            requires_discussion=True
        )
        
        assert analyzer.calculate_strength(analysis_high) == "strong"
        assert analyzer.calculate_strength(analysis_low) == "none"
    
    def test_identify_dissenters(self):
        """Test dissenter identification."""
        analyzer = ConsensusAnalyzer()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.9),
            "model_b": Vote("model_b", "option_b", 0.9),
            "model_c": Vote("model_c", "option_a", 0.8),
        }
        
        dissenters = analyzer.identify_dissenters(votes, "option_a")
        
        assert "model_b" in dissenters
        assert "model_a" not in dissenters
    
    def test_confidence_distribution(self):
        """Test confidence distribution analysis."""
        analyzer = ConsensusAnalyzer()
        
        votes = {
            "model_a": Vote("model_a", "option_a", 0.9),
            "model_b": Vote("model_b", "option_a", 0.7),
            "model_c": Vote("model_c", "option_a", 0.8),
        }
        
        dist = analyzer.analyze_confidence_distribution(votes)
        
        assert "mean" in dist
        assert "median" in dist
        assert dist["mean"] == pytest.approx(0.8, 0.01)


class TestVotingEngine:
    """Tests for VotingEngine."""
    
    @pytest.mark.asyncio
    async def test_vote_basic(self):
        """Test basic voting."""
        engine = VotingEngine()
        
        config = VoteConfig(
            models=["model_a", "model_b"],
            strategy=VotingStrategy.SIMPLE_MAJORITY,
            min_consensus=0.5
        )
        
        # Note: This uses simulated votes
        result = await engine.vote(
            prompt="Test decision",
            options=["option_a", "option_b"],
            config=config
        )
        
        assert result.vote_id is not None
        assert result.winner in ["option_a", "option_b"]
        assert result.consensus_level >= 0.0
    
    def test_record_outcome(self):
        """Test outcome recording."""
        engine = VotingEngine()
        
        # Simulate a vote result in history
        from core.voting.engine import VoteResult
        
        result = VoteResult(
            vote_id="test_vote",
            winner="option_a",
            winner_confidence=0.8,
            all_votes={
                "model_a": Vote("model_a", "option_a", 0.9),
                "model_b": Vote("model_b", "option_b", 0.7),
            },
            consensus_level=0.6,
            disagreement_analysis=None,
            confidence_breakdown={"option_a": 0.6, "option_b": 0.4},
            strategy_used=VotingStrategy.SIMPLE_MAJORITY,
            runtime_seconds=1.0
        )
        
        engine.history.append(result)
        
        # Record outcome
        engine.record_outcome("test_vote", "option_a", True)
        
        # Check accuracy history was updated
        assert "model_a" in engine._accuracy_history
        assert True in engine._accuracy_history["model_a"]
    
    def test_get_model_performance(self):
        """Test model performance retrieval."""
        engine = VotingEngine()
        
        # Simulate some history
        engine._accuracy_history = {
            "model_a": [True, True, False],
            "model_b": [False, False, True]
        }
        
        performance = engine.get_model_performance()
        
        assert "model_a" in performance
        assert performance["model_a"]["accuracy"] == pytest.approx(0.667, 0.01)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
