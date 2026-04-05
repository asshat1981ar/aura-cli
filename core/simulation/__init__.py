"""Simulation Engine for running "what-if" scenarios."""

from core.simulation.engine import SimulationEngine, SimulationConfig, SimulationResult, SuccessCriterion
from core.simulation.scenario import Scenario, ScenarioRegistry, SIMULATION_SCENARIOS
from core.simulation.runner import IsolatedSimulationRunner, ScenarioOutcome
from core.simulation.outcome_analyzer import OutcomeAnalyzer, Insight
from core.simulation.registry import TemplateRegistry

__all__ = [
    "SimulationEngine",
    "SimulationConfig",
    "SimulationResult",
    "SuccessCriterion",
    "Scenario",
    "ScenarioRegistry",
    "SIMULATION_SCENARIOS",
    "IsolatedSimulationRunner",
    "ScenarioOutcome",
    "OutcomeAnalyzer",
    "Insight",
    "TemplateRegistry",
]
