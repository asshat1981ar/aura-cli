"""Tests for the Simulation Engine."""

import pytest
import asyncio
from core.simulation import (
    SimulationEngine,
    SimulationConfig,
    SuccessCriterion,
    Scenario,
    ScenarioRegistry,
    ScenarioOutcome,
    IsolatedSimulationRunner,
    OutcomeAnalyzer,
)


class TestSimulationConfig:
    """Tests for SimulationConfig."""
    
    def test_success_criterion_evaluate_gt(self):
        """Test greater-than criterion evaluation."""
        criterion = SuccessCriterion(
            name="high_pass_rate",
            metric="test_pass_rate",
            operator="gt",
            threshold=0.8
        )
        
        assert criterion.evaluate({"test_pass_rate": 0.9}) is True
        assert criterion.evaluate({"test_pass_rate": 0.7}) is False
        assert criterion.evaluate({"test_pass_rate": 0.8}) is False
    
    def test_success_criterion_evaluate_gte(self):
        """Test greater-than-or-equal criterion evaluation."""
        criterion = SuccessCriterion(
            name="min_coverage",
            metric="coverage",
            operator="gte",
            threshold=0.8
        )
        
        assert criterion.evaluate({"coverage": 0.9}) is True
        assert criterion.evaluate({"coverage": 0.8}) is True
        assert criterion.evaluate({"coverage": 0.7}) is False


class TestScenarioRegistry:
    """Tests for ScenarioRegistry."""
    
    def test_register_and_get(self):
        """Test scenario registration and retrieval."""
        registry = ScenarioRegistry()
        
        def make_scenario():
            return Scenario(
                scenario_id="test_scenario",
                name="Test Scenario",
                description="A test scenario",
                goal="Test something",
                context={"key": "value"}
            )
        
        registry.register("test", make_scenario)
        
        scenario = registry.get("test")
        assert scenario.scenario_id == "test_scenario"
        assert scenario.name == "Test Scenario"
    
    def test_get_with_overrides(self):
        """Test getting scenario with overrides."""
        registry = ScenarioRegistry()
        
        def make_scenario():
            return Scenario(
                scenario_id="base",
                name="Base",
                description="Base scenario",
                goal="Base goal",
                context={"original": "value"}
            )
        
        registry.register("base", make_scenario)
        
        scenario = registry.get("base", original="overridden", new="added")
        assert scenario.context["original"] == "overridden"
        assert scenario.context["new"] == "added"


class TestSimulationEngine:
    """Tests for SimulationEngine."""
    
    @pytest.mark.asyncio
    async def test_run_simulation_basic(self):
        """Test basic simulation run."""
        engine = SimulationEngine()
        
        config = SimulationConfig(
            name="test_sim",
            base_scenario="prompt_strategy",
            variables={"prompt_style": ["concise", "detailed"]},
            max_parallel=2
        )
        
        result = await engine.run_simulation(config)
        
        assert result.run_id is not None
        assert len(result.outcomes) == 2
        assert result.config == config
    
    @pytest.mark.asyncio
    async def test_run_simulation_no_variables(self):
        """Test simulation with no variable variations."""
        engine = SimulationEngine()
        
        config = SimulationConfig(
            name="test_sim",
            base_scenario="prompt_strategy",
            variables={},
            max_parallel=1
        )
        
        result = await engine.run_simulation(config)
        
        assert len(result.outcomes) == 1
    
    def test_get_statistics_empty(self):
        """Test getting statistics with no history."""
        engine = SimulationEngine()
        
        stats = engine.get_statistics()
        
        assert stats["total_runs"] == 0


class TestIsolatedSimulationRunner:
    """Tests for IsolatedSimulationRunner."""
    
    @pytest.mark.asyncio
    async def test_run_scenario_success(self):
        """Test successful scenario execution."""
        runner = IsolatedSimulationRunner()
        
        scenario = Scenario(
            scenario_id="test",
            name="Test Scenario",
            description="Test",
            goal="Test goal",
            context={"prompt_style": "concise"}
        )
        
        outcome = await runner.run_scenario(scenario, timeout=10.0)
        
        assert outcome.scenario_id == "test"
        assert outcome.success is not None
        assert outcome.metrics is not None


class TestOutcomeAnalyzer:
    """Tests for OutcomeAnalyzer."""
    
    def test_analyze_empty_outcomes(self):
        """Test analysis with no outcomes."""
        analyzer = OutcomeAnalyzer()
        
        insights = analyzer.analyze([], {})
        
        assert len(insights) == 1
        assert insights[0].insight_type == "statistical"
    
    def test_analyze_with_outcomes(self):
        """Test analysis with sample outcomes."""
        analyzer = OutcomeAnalyzer()
        
        outcomes = [
            ScenarioOutcome(
                scenario_id="s1",
                scenario_name="Scenario 1",
                success=True,
                metrics={"score": 0.9, "time": 1.0},
                score=0.9
            ),
            ScenarioOutcome(
                scenario_id="s2",
                scenario_name="Scenario 2",
                success=True,
                metrics={"score": 0.8, "time": 1.2},
                score=0.8
            ),
        ]
        
        insights = analyzer.analyze(outcomes, {})
        
        assert len(insights) > 0
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        analyzer = OutcomeAnalyzer()
        
        outcomes = [
            ScenarioOutcome(
                scenario_id="s1",
                scenario_name="Scenario 1",
                success=True,
                metrics={"value": 0.9},
                score=0.9,
                parameter_overrides={"param": 0.5}
            ),
            ScenarioOutcome(
                scenario_id="s2",
                scenario_name="Scenario 2",
                success=True,
                metrics={"value": 0.7},
                score=0.7,
                parameter_overrides={"param": 0.3}
            ),
        ]
        
        sensitivity = analyzer.calculate_sensitivity(outcomes, "param")
        
        if sensitivity:  # May be None if insufficient data
            assert sensitivity.parameter_name == "param"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
