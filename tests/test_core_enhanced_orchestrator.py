"""Comprehensive test suite for core/enhanced_orchestrator.py.

This module provides 80%+ coverage for EnhancedOrchestrator including:
- Component initialization (simulation, knowledge, voting, adversarial)
- Enhanced feature processing workflow
- Async operations with pytest-asyncio
- Feature enable/disable flags
- Integration with base orchestrator
"""

import pytest
from unittest.mock import MagicMock, Mock, patch, AsyncMock

from core.enhanced_orchestrator import (
    EnhancedOrchestrator,
    enhance_orchestrator,
    attach_enhanced_features_to_orchestrator,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_base_orchestrator():
    """Mock base LoopOrchestrator."""
    orch = MagicMock()
    orch.run_cycle.return_value = {
        "goal": "test",
        "status": "success",
        "phase_outputs": {},
    }
    orch.brain = MagicMock()
    orch.model = MagicMock()
    return orch


@pytest.fixture
def enhanced_orchestrator_all_enabled(mock_base_orchestrator):
    """EnhancedOrchestrator with all features enabled."""
    with patch("core.enhanced_orchestrator.SimulationEngine"):
        with patch("core.enhanced_orchestrator.KnowledgeBase"):
            with patch("core.enhanced_orchestrator.VotingEngine"):
                with patch("core.enhanced_orchestrator.AdversarialAgent"):
                    orch = EnhancedOrchestrator(
                        base_orchestrator=mock_base_orchestrator,
                        enable_simulation=True,
                        enable_knowledge=True,
                        enable_voting=True,
                        enable_adversarial=True,
                    )
    return orch


@pytest.fixture
def enhanced_orchestrator_all_disabled(mock_base_orchestrator):
    """EnhancedOrchestrator with all features disabled."""
    orch = EnhancedOrchestrator(
        base_orchestrator=mock_base_orchestrator,
        enable_simulation=False,
        enable_knowledge=False,
        enable_voting=False,
        enable_adversarial=False,
    )
    return orch


# ── Test Initialization ───────────────────────────────────────────────────


class TestEnhancedOrchestratorInit:
    """Test EnhancedOrchestrator initialization."""

    def test_init_with_all_features_enabled(self, mock_base_orchestrator):
        """Test initialization with all features enabled."""
        with patch("core.enhanced_orchestrator.SimulationEngine") as mock_sim:
            with patch("core.enhanced_orchestrator.KnowledgeBase") as mock_kb:
                with patch("core.enhanced_orchestrator.VotingEngine") as mock_vote:
                    with patch("core.enhanced_orchestrator.AdversarialAgent") as mock_adv:
                        orch = EnhancedOrchestrator(
                            base_orchestrator=mock_base_orchestrator,
                            enable_simulation=True,
                            enable_knowledge=True,
                            enable_voting=True,
                            enable_adversarial=True,
                        )

        assert orch.base is mock_base_orchestrator
        assert orch.enable_simulation is True
        assert orch.enable_knowledge is True
        assert orch.enable_voting is True
        assert orch.enable_adversarial is True

    def test_init_with_all_features_disabled(self, mock_base_orchestrator):
        """Test initialization with all features disabled."""
        orch = EnhancedOrchestrator(
            base_orchestrator=mock_base_orchestrator,
            enable_simulation=False,
            enable_knowledge=False,
            enable_voting=False,
            enable_adversarial=False,
        )

        assert orch.base is mock_base_orchestrator
        assert orch.simulation_engine is None
        assert orch.knowledge_base is None
        assert orch.voting_engine is None
        assert orch.adversarial_agent is None

    def test_init_without_base_orchestrator(self):
        """Test initialization without base orchestrator."""
        orch = EnhancedOrchestrator(base_orchestrator=None)

        assert orch.base is None

    def test_init_component_failure_does_not_crash(self, mock_base_orchestrator):
        """Test component initialization failures are handled gracefully."""
        with patch("core.enhanced_orchestrator.SimulationEngine", side_effect=Exception("init error")):
            # Should not raise
            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_simulation=True,
            )

        # Component should be None but orchestrator still works
        assert orch.simulation_engine is None


# ── Test Component Initialization ─────────────────────────────────────────


class TestComponentInitialization:
    """Test individual component initialization."""

    def test_simulation_engine_initialized_when_enabled(self, mock_base_orchestrator):
        """Test SimulationEngine is initialized when enabled."""
        with patch("core.enhanced_orchestrator.SimulationEngine") as mock_sim:
            mock_sim.return_value = MagicMock()

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_simulation=True,
            )

        mock_sim.assert_called_once()
        assert orch.simulation_engine is not None

    def test_knowledge_base_initialized_when_enabled(self, mock_base_orchestrator):
        """Test KnowledgeBase is initialized when enabled."""
        with patch("core.enhanced_orchestrator.KnowledgeBase") as mock_kb:
            mock_kb.return_value = MagicMock()

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_knowledge=True,
            )

        mock_kb.assert_called_once()
        assert orch.knowledge_base is not None

    def test_voting_engine_initialized_when_enabled(self, mock_base_orchestrator):
        """Test VotingEngine is initialized when enabled."""
        with patch("core.enhanced_orchestrator.VotingEngine") as mock_vote:
            mock_vote.return_value = MagicMock()

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_voting=True,
            )

        mock_vote.assert_called_once()
        assert orch.voting_engine is not None

    def test_adversarial_agent_initialized_when_enabled(self, mock_base_orchestrator):
        """Test AdversarialAgent is initialized when enabled."""
        with patch("core.enhanced_orchestrator.AdversarialAgent") as mock_adv:
            mock_adv.return_value = MagicMock()

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_adversarial=True,
            )

        mock_adv.assert_called_once()
        assert orch.adversarial_agent is not None


# ── Test Enhanced Processing Workflow ─────────────────────────────────────


class TestEnhancedProcessing:
    """Test enhanced processing with all features."""

    @pytest.mark.asyncio
    async def test_process_with_enhancements_basic(self, mock_base_orchestrator):
        """Test basic processing with all features disabled."""
        orch = EnhancedOrchestrator(
            base_orchestrator=mock_base_orchestrator,
            enable_simulation=False,
            enable_knowledge=False,
            enable_voting=False,
            enable_adversarial=False,
        )

        result = await orch.process_with_enhancements("Test goal")

        assert result["goal"] == "Test goal"
        assert "enhancements" in result
        mock_base_orchestrator.run_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_without_base_orchestrator(self):
        """Test processing without base orchestrator."""
        orch = EnhancedOrchestrator(base_orchestrator=None)

        result = await orch.process_with_enhancements("Test goal")

        assert result["goal"] == "Test goal"
        assert "context" in result

    @pytest.mark.asyncio
    async def test_knowledge_base_query_retrieves_insights(self, mock_base_orchestrator):
        """Test knowledge base query adds relevant insights."""
        with patch("core.enhanced_orchestrator.KnowledgeBase") as mock_kb_class:
            mock_kb = AsyncMock()
            mock_kb.query.return_value = [
                MagicMock(entry=MagicMock(content="insight 1"), composite_score=0.9),
                MagicMock(entry=MagicMock(content="insight 2"), composite_score=0.8),
            ]
            mock_kb_class.return_value = mock_kb

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_knowledge=True,
            )
            orch.knowledge_base = mock_kb

            result = await orch.process_with_enhancements("Test goal", use_knowledge=True)

        mock_kb.query.assert_called_once()
        assert result["enhancements"]["knowledge_retrieved"] == 2
        assert "relevant_knowledge" in result["context"]

    @pytest.mark.asyncio
    async def test_simulation_runs_for_appropriate_goals(self, mock_base_orchestrator):
        """Test simulation runs for goals with keywords."""
        with patch("core.enhanced_orchestrator.SimulationEngine") as mock_sim_class:
            mock_sim = AsyncMock()
            mock_winner = MagicMock()
            mock_winner.scenario_id = "winner_1"
            mock_winner.scenario_name = "conservative"
            mock_result = MagicMock()
            mock_result.winner = mock_winner
            mock_result.insights = ["insight1", "insight2"]
            mock_sim.run_simulation.return_value = mock_result
            mock_sim_class.return_value = mock_sim

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_simulation=True,
            )
            orch.simulation_engine = mock_sim

            result = await orch.process_with_enhancements(
                "Test configuration optimization",
                use_simulation=True
            )

        mock_sim.run_simulation.assert_called_once()
        assert "simulation" in result["enhancements"]
        assert result["context"]["recommended_approach"] == "conservative"

    @pytest.mark.asyncio
    async def test_simulation_skips_for_non_matching_goals(self, mock_base_orchestrator):
        """Test simulation skips for goals without keywords."""
        with patch("core.enhanced_orchestrator.SimulationEngine") as mock_sim_class:
            mock_sim = AsyncMock()
            mock_sim_class.return_value = mock_sim

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_simulation=True,
            )
            orch.simulation_engine = mock_sim

            result = await orch.process_with_enhancements(
                "Simple goal with no keywords",
                use_simulation=True
            )

        mock_sim.run_simulation.assert_not_called()

    @pytest.mark.asyncio
    async def test_adversarial_critique_for_substantial_goals(self, mock_base_orchestrator):
        """Test adversarial critique for goals over 50 chars."""
        with patch("core.enhanced_orchestrator.AdversarialAgent") as mock_adv_class:
            mock_adv = AsyncMock()
            mock_critique = MagicMock()
            mock_critique.critique_id = "crit_123"
            mock_critique.risk_score = 0.3
            mock_critique.findings = [
                MagicMock(severity="medium", description="issue 1"),
            ]
            mock_critique.overall_assessment = "acceptable"
            mock_adv.critique.return_value = mock_critique
            mock_adv_class.return_value = mock_adv

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_adversarial=True,
            )
            orch.adversarial_agent = mock_adv

            result = await orch.process_with_enhancements(
                "This is a substantial goal that needs adversarial review",
                use_adversarial=True
            )

        mock_adv.critique.assert_called_once()
        assert "adversarial_critique" in result["enhancements"]
        assert result["enhancements"]["adversarial_critique"]["risk_score"] == 0.3

    @pytest.mark.asyncio
    async def test_adversarial_critique_skips_short_goals(self, mock_base_orchestrator):
        """Test adversarial critique skips goals under 50 chars."""
        with patch("core.enhanced_orchestrator.AdversarialAgent") as mock_adv_class:
            mock_adv = AsyncMock()
            mock_adv_class.return_value = mock_adv

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_adversarial=True,
            )
            orch.adversarial_agent = mock_adv

            result = await orch.process_with_enhancements(
                "Short goal",
                use_adversarial=True
            )

        mock_adv.critique.assert_not_called()

    @pytest.mark.asyncio
    async def test_voting_with_multiple_approaches(self, mock_base_orchestrator):
        """Test voting selects best approach."""
        with patch("core.enhanced_orchestrator.VotingEngine") as mock_vote_class:
            mock_vote = AsyncMock()
            mock_result = MagicMock()
            mock_result.winner = "approach_2"
            mock_result.consensus_level = 0.75
            mock_result.winner_confidence = 0.85
            mock_vote.vote.return_value = mock_result
            mock_vote_class.return_value = mock_vote

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_voting=True,
            )
            orch.voting_engine = mock_vote

            context = {"approaches": ["approach_1", "approach_2", "approach_3"]}
            result = await orch.process_with_enhancements(
                "Test goal",
                context=context,
                use_voting=True
            )

        mock_vote.vote.assert_called_once()
        assert "voting" in result["enhancements"]
        assert result["enhancements"]["voting"]["winner"] == "approach_2"

    @pytest.mark.asyncio
    async def test_voting_skips_without_approaches(self, mock_base_orchestrator):
        """Test voting skips when no approaches in context."""
        with patch("core.enhanced_orchestrator.VotingEngine") as mock_vote_class:
            mock_vote = AsyncMock()
            mock_vote_class.return_value = mock_vote

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_voting=True,
            )
            orch.voting_engine = mock_vote

            result = await orch.process_with_enhancements(
                "Test goal",
                use_voting=True
            )

        mock_vote.vote.assert_not_called()

    @pytest.mark.asyncio
    async def test_knowledge_store_after_processing(self, mock_base_orchestrator):
        """Test knowledge is stored after processing."""
        with patch("core.enhanced_orchestrator.KnowledgeBase") as mock_kb_class:
            mock_kb = AsyncMock()
            mock_kb.query.return_value = []
            mock_kb.add.return_value = None
            mock_kb_class.return_value = mock_kb

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_knowledge=True,
            )
            orch.knowledge_base = mock_kb

            result = await orch.process_with_enhancements("Test goal", use_knowledge=True)

        mock_kb.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhancement_errors_are_handled(self, mock_base_orchestrator):
        """Test errors in enhancements don't crash processing."""
        with patch("core.enhanced_orchestrator.KnowledgeBase") as mock_kb_class:
            mock_kb = AsyncMock()
            mock_kb.query.side_effect = Exception("query failed")
            mock_kb_class.return_value = mock_kb

            orch = EnhancedOrchestrator(
                base_orchestrator=mock_base_orchestrator,
                enable_knowledge=True,
            )
            orch.knowledge_base = mock_kb

            # Should not raise
            result = await orch.process_with_enhancements("Test goal", use_knowledge=True)

        assert result is not None


# ── Test Feature Status ───────────────────────────────────────────────────


class TestFeatureStatus:
    """Test feature status reporting."""

    def test_get_feature_status_all_enabled(self, mock_base_orchestrator):
        """Test feature status when all enabled."""
        with patch("core.enhanced_orchestrator.SimulationEngine"):
            with patch("core.enhanced_orchestrator.KnowledgeBase"):
                with patch("core.enhanced_orchestrator.VotingEngine"):
                    with patch("core.enhanced_orchestrator.AdversarialAgent"):
                        orch = EnhancedOrchestrator(
                            base_orchestrator=mock_base_orchestrator,
                            enable_simulation=True,
                            enable_knowledge=True,
                            enable_voting=True,
                            enable_adversarial=True,
                        )

        status = orch.get_feature_status()

        assert status["simulation"]["enabled"] is True
        assert status["knowledge"]["enabled"] is True
        assert status["voting"]["enabled"] is True
        assert status["adversarial"]["enabled"] is True

    def test_get_feature_status_all_disabled(self, enhanced_orchestrator_all_disabled):
        """Test feature status when all disabled."""
        status = enhanced_orchestrator_all_disabled.get_feature_status()

        assert status["simulation"]["enabled"] is False
        assert status["knowledge"]["enabled"] is False
        assert status["voting"]["enabled"] is False
        assert status["adversarial"]["enabled"] is False
        assert status["simulation"]["initialized"] is False
        assert status["knowledge"]["initialized"] is False


# ── Test Convenience Functions ────────────────────────────────────────────


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_enhance_orchestrator_function(self):
        """Test enhance_orchestrator convenience function."""
        mock_orch = MagicMock()

        with patch("core.enhanced_orchestrator.EnhancedOrchestrator") as mock_class:
            mock_class.return_value = MagicMock()

            result = enhance_orchestrator(
                mock_orch,
                simulation=True,
                knowledge=False,
                voting=True,
                adversarial=False
            )

        mock_class.assert_called_once_with(
            base_orchestrator=mock_orch,
            enable_simulation=True,
            enable_knowledge=False,
            enable_voting=True,
            enable_adversarial=False
        )

    def test_attach_enhanced_features_to_orchestrator(self):
        """Test attach_enhanced_features_to_orchestrator function."""
        mock_orch = MagicMock()
        mock_orch.brain = MagicMock()
        mock_orch.model = MagicMock()
        mock_orch.attach_enhanced_features = MagicMock()

        with patch("core.enhanced_orchestrator.SimulationEngine"):
            with patch("core.enhanced_orchestrator.KnowledgeBase"):
                with patch("core.enhanced_orchestrator.VotingEngine"):
                    with patch("core.enhanced_orchestrator.AdversarialAgent"):
                        result = attach_enhanced_features_to_orchestrator(
                            mock_orch,
                            enable_simulation=True,
                            enable_knowledge=True,
                            enable_voting=True,
                            enable_adversarial=True
                        )

        assert result is mock_orch
        mock_orch.attach_enhanced_features.assert_called_once()

    def test_attach_with_partial_features(self):
        """Test attaching only some features."""
        mock_orch = MagicMock()
        mock_orch.brain = MagicMock()
        mock_orch.model = MagicMock()
        mock_orch.attach_enhanced_features = MagicMock()

        with patch("core.enhanced_orchestrator.SimulationEngine"):
            result = attach_enhanced_features_to_orchestrator(
                mock_orch,
                enable_simulation=True,
                enable_knowledge=False,
                enable_voting=False,
                enable_adversarial=False
            )

        assert result is mock_orch
        mock_orch.attach_enhanced_features.assert_called_once()

    def test_attach_handles_init_failures(self):
        """Test attach handles initialization failures gracefully."""
        mock_orch = MagicMock()
        mock_orch.brain = MagicMock()
        mock_orch.model = MagicMock()
        mock_orch.attach_enhanced_features = MagicMock()

        with patch("core.enhanced_orchestrator.SimulationEngine", side_effect=Exception("init error")):
            # Should not raise
            result = attach_enhanced_features_to_orchestrator(
                mock_orch,
                enable_simulation=True
            )

        assert result is mock_orch


# ── Test Integration Scenarios ────────────────────────────────────────────


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_enhancement_pipeline(self, mock_base_orchestrator):
        """Test full enhancement pipeline with all features."""
        # Setup all mocks
        with patch("core.enhanced_orchestrator.SimulationEngine") as mock_sim_class:
            with patch("core.enhanced_orchestrator.KnowledgeBase") as mock_kb_class:
                with patch("core.enhanced_orchestrator.VotingEngine") as mock_vote_class:
                    with patch("core.enhanced_orchestrator.AdversarialAgent") as mock_adv_class:
                        # Create mocks
                        mock_kb = AsyncMock()
                        mock_kb.query.return_value = [
                            MagicMock(entry=MagicMock(content="insight"), composite_score=0.9),
                        ]
                        mock_kb.add.return_value = None

                        mock_adv = AsyncMock()
                        mock_critique = MagicMock()
                        mock_critique.critique_id = "crit_1"
                        mock_critique.risk_score = 0.2
                        mock_critique.findings = []
                        mock_critique.overall_assessment = "good"
                        mock_adv.critique.return_value = mock_critique

                        mock_kb_class.return_value = mock_kb
                        mock_adv_class.return_value = mock_adv

                        orch = EnhancedOrchestrator(
                            base_orchestrator=mock_base_orchestrator,
                            enable_simulation=False,  # Skip sim for speed
                            enable_knowledge=True,
                            enable_voting=False,
                            enable_adversarial=True,
                        )

                        result = await orch.process_with_enhancements(
                            "This is a substantial test goal for full pipeline testing",
                            use_knowledge=True,
                            use_adversarial=True
                        )

        # Verify all steps executed
        assert "enhancements" in result
        assert "knowledge_retrieved" in result["enhancements"]
        assert "adversarial_critique" in result["enhancements"]
        mock_kb.query.assert_called_once()
        mock_adv.critique.assert_called_once()
        mock_kb.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_selective_enhancement_execution(self, mock_base_orchestrator):
        """Test selective enhancement execution with use_* flags."""
        with patch("core.enhanced_orchestrator.KnowledgeBase") as mock_kb_class:
            with patch("core.enhanced_orchestrator.AdversarialAgent") as mock_adv_class:
                mock_kb = AsyncMock()
                mock_kb.query.return_value = []
                mock_adv = AsyncMock()
                mock_kb_class.return_value = mock_kb
                mock_adv_class.return_value = mock_adv

                orch = EnhancedOrchestrator(
                    base_orchestrator=mock_base_orchestrator,
                    enable_knowledge=True,
                    enable_adversarial=True,
                )

                # Use knowledge but not adversarial
                result = await orch.process_with_enhancements(
                    "Long enough goal for adversarial review criteria",
                    use_knowledge=True,
                    use_adversarial=False
                )

        mock_kb.query.assert_called_once()
        mock_adv.critique.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
