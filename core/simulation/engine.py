"""Core simulation orchestrator for running parallel "what-if" scenarios."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from core.logging_utils import log_json


@dataclass
class SuccessCriterion:
    """A criterion for evaluating simulation success."""
    name: str
    metric: str  # Metric to evaluate (e.g., "test_pass_rate", "execution_time")
    operator: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    weight: float = 1.0
    
    def evaluate(self, metrics: Dict[str, float]) -> bool:
        """Evaluate this criterion against metrics."""
        value = metrics.get(self.metric, 0.0)
        
        operators = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "eq": lambda v, t: abs(v - t) < 0.001,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
        }
        
        op_func = operators.get(self.operator)
        if not op_func:
            log_json("WARN", "unknown_operator", {"operator": self.operator})
            return False
            
        return op_func(value, self.threshold)


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    name: str
    base_scenario: str  # Reference to registered scenario
    variables: Dict[str, List[Any]]  # Parameter variations to test
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    max_parallel: int = 3
    timeout_seconds: float = 300.0
    description: str = ""


@dataclass
class SimulationResult:
    """Result from a single simulation run."""
    run_id: str
    config: SimulationConfig
    outcomes: List["ScenarioOutcome"]
    winner: Optional["ScenarioOutcome"]
    insights: List["Insight"]
    runtime_seconds: float
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SimulationEngine:
    """Orchestrates parallel simulations to test change outcomes."""
    
    def __init__(self, orchestrator=None, metrics_collector=None):
        """
        Initialize the simulation engine.
        
        Args:
            orchestrator: Optional LoopOrchestrator for running scenarios
            metrics_collector: Optional MetricsCollector for baseline metrics
        """
        self.orchestrator = orchestrator
        self.metrics = metrics_collector
        self._history: List[SimulationResult] = []
        
        # Lazy imports to avoid circular dependencies
        from core.simulation.scenario import ScenarioRegistry
        from core.simulation.runner import IsolatedSimulationRunner
        from core.simulation.outcome_analyzer import OutcomeAnalyzer
        
        self.registry = ScenarioRegistry()
        self.runner = IsolatedSimulationRunner()
        self.analyzer = OutcomeAnalyzer()
        
    async def run_simulation(self, config: SimulationConfig) -> SimulationResult:
        """
        Execute a simulation with multiple parameter variations.
        
        Args:
            config: Simulation configuration
            
        Returns:
            SimulationResult with outcomes, winner, and insights
        """
        start_time = time.time()
        run_id = str(uuid.uuid4())[:8]
        
        log_json("INFO", "simulation_started", {
            "run_id": run_id,
            "name": config.name,
            "variables": list(config.variables.keys()),
            "max_parallel": config.max_parallel
        })
        
        # 1. Capture baseline metrics if collector available
        baseline = {}
        if self.metrics:
            try:
                baseline = self.metrics.collect()
                log_json("INFO", "baseline_captured", baseline)
            except Exception as e:
                log_json("WARN", "baseline_capture_failed", {"error": str(e)})
        
        # 2. Generate scenario variations (cartesian product of variables)
        scenarios = self._generate_scenarios(config)
        log_json("INFO", "scenarios_generated", {
            "run_id": run_id,
            "count": len(scenarios)
        })
        
        # 3. Run scenarios with semaphore-controlled parallelism
        semaphore = asyncio.Semaphore(config.max_parallel)
        
        async def run_with_limit(scenario):
            async with semaphore:
                return await self.runner.run_scenario(
                    scenario, 
                    config.timeout_seconds
                )
        
        try:
            outcomes = await asyncio.gather(*[
                run_with_limit(scenario) for scenario in scenarios
            ])
        except Exception as e:
            log_json("ERROR", "simulation_execution_failed", {
                "run_id": run_id,
                "error": str(e)
            })
            outcomes = []
        
        # 4. Select winner based on success criteria
        winner = self._select_winner(outcomes, config.success_criteria)
        
        # 5. Extract actionable insights
        insights = self.analyzer.analyze(outcomes, baseline)
        
        runtime = time.time() - start_time
        
        result = SimulationResult(
            run_id=run_id,
            config=config,
            outcomes=outcomes,
            winner=winner,
            insights=insights,
            runtime_seconds=runtime,
            baseline_metrics=baseline
        )
        
        self._history.append(result)
        
        log_json("INFO", "simulation_completed", {
            "run_id": run_id,
            "outcomes_count": len(outcomes),
            "has_winner": winner is not None,
            "insights_count": len(insights),
            "runtime_seconds": runtime
        })
        
        return result
    
    def _generate_scenarios(self, config: SimulationConfig) -> List["Scenario"]:
        """Generate scenario variations from config variables."""
        from core.simulation.scenario import Scenario
        
        base_scenario = self.registry.get(config.base_scenario)
        
        if not config.variables:
            # No variations, return base scenario
            return [base_scenario]
        
        # Generate cartesian product of variable values
        import itertools
        
        variable_names = list(config.variables.keys())
        variable_values = [config.variables[name] for name in variable_names]
        
        scenarios = []
        for combination in itertools.product(*variable_values):
            overrides = dict(zip(variable_names, combination))
            
            # Create scenario with overrides
            scenario = Scenario(
                scenario_id=f"{base_scenario.scenario_id}_{len(scenarios)}",
                name=f"{base_scenario.name} ({self._format_overrides(overrides)})",
                description=base_scenario.description,
                goal=base_scenario.goal,
                context={**base_scenario.context, **overrides},
                parameter_overrides=overrides,
                expected_behavior=base_scenario.expected_behavior
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _format_overrides(self, overrides: Dict[str, Any]) -> str:
        """Format overrides for display."""
        parts = []
        for key, value in overrides.items():
            if isinstance(value, str) and len(value) > 20:
                value = value[:20] + "..."
            parts.append(f"{key}={value}")
        return ", ".join(parts)
    
    def _select_winner(
        self, 
        outcomes: List["ScenarioOutcome"], 
        criteria: List[SuccessCriterion]
    ) -> Optional["ScenarioOutcome"]:
        """Select the winning scenario based on success criteria."""
        if not outcomes or not criteria:
            # No criteria, select by best overall metrics
            valid_outcomes = [o for o in outcomes if o.success]
            if not valid_outcomes:
                return None
            return max(valid_outcomes, key=lambda o: o.score)
        
        # Score each outcome against criteria
        scored_outcomes = []
        for outcome in outcomes:
            if not outcome.metrics:
                continue
                
            total_score = 0.0
            total_weight = 0.0
            
            for criterion in criteria:
                passed = criterion.evaluate(outcome.metrics)
                weight = criterion.weight
                total_score += (1.0 if passed else 0.0) * weight
                total_weight += weight
            
            if total_weight > 0:
                normalized_score = total_score / total_weight
                scored_outcomes.append((outcome, normalized_score))
        
        if not scored_outcomes:
            return None
        
        # Return highest scoring outcome
        winner, score = max(scored_outcomes, key=lambda x: x[1])
        log_json("INFO", "winner_selected", {
            "scenario_id": winner.scenario_id,
            "score": score
        })
        
        return winner
    
    def get_history(self) -> List[SimulationResult]:
        """Get simulation run history."""
        return self._history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics from simulation history."""
        if not self._history:
            return {"total_runs": 0}
        
        total_runs = len(self._history)
        total_outcomes = sum(len(r.outcomes) for r in self._history)
        avg_runtime = sum(r.runtime_seconds for r in self._history) / total_runs
        wins = sum(1 for r in self._history if r.winner is not None)
        
        return {
            "total_runs": total_runs,
            "total_outcomes": total_outcomes,
            "avg_runtime_seconds": avg_runtime,
            "win_rate": wins / total_runs,
            "avg_outcomes_per_run": total_outcomes / total_runs
        }
