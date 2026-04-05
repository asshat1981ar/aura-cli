"""Scenario definition and registry for simulations."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class ScenarioType(Enum):
    """Types of simulation scenarios."""
    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    TEST_GENERATION = "test_generation"
    DEBUGGING = "debugging"
    PLANNING = "planning"
    CONFIGURATION = "configuration"
    PROMPT_ENGINEERING = "prompt_engineering"


@dataclass
class Scenario:
    """A single simulation scenario to test."""
    scenario_id: str
    name: str
    description: str
    goal: str
    context: Dict[str, Any] = field(default_factory=dict)
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: Optional[str] = None
    scenario_type: ScenarioType = ScenarioType.CODE_GENERATION
    tags: List[str] = field(default_factory=list)
    
    def with_overrides(self, **overrides) -> "Scenario":
        """Create a copy with additional overrides."""
        return Scenario(
            scenario_id=f"{self.scenario_id}_variant",
            name=f"{self.name} (variant)",
            description=self.description,
            goal=self.goal,
            context={**self.context, **overrides},
            parameter_overrides={**self.parameter_overrides, **overrides},
            expected_behavior=self.expected_behavior,
            scenario_type=self.scenario_type,
            tags=self.tags
        )


@dataclass
class ScenarioOutcome:
    """Outcome from running a single scenario."""
    scenario_id: str
    scenario_name: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)


class ScenarioRegistry:
    """Registry of reusable simulation scenarios."""
    
    def __init__(self):
        self._scenarios: Dict[str, Callable[[], Scenario]] = {}
        self._register_builtin_scenarios()
    
    def register(self, name: str, factory: Callable[[], Scenario]):
        """Register a scenario factory."""
        self._scenarios[name] = factory
    
    def get(self, name: str, **overrides) -> Scenario:
        """Get a scenario with optional parameter overrides."""
        factory = self._scenarios.get(name)
        if not factory:
            raise KeyError(f"Scenario '{name}' not registered")
        
        scenario = factory()
        
        if overrides:
            scenario = Scenario(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                description=scenario.description,
                goal=scenario.goal,
                context={**scenario.context, **overrides},
                parameter_overrides={**scenario.parameter_overrides, **overrides},
                expected_behavior=scenario.expected_behavior,
                scenario_type=scenario.scenario_type,
                tags=scenario.tags
            )
        
        return scenario
    
    def list_scenarios(self) -> List[str]:
        """List all registered scenario names."""
        return list(self._scenarios.keys())
    
    def _register_builtin_scenarios(self):
        """Register built-in simulation scenarios."""
        
        # Prompt strategy comparison
        self.register("prompt_strategy", lambda: Scenario(
            scenario_id="prompt_strategy",
            name="Prompt Strategy Comparison",
            description="Test different prompt strategies for code generation",
            goal="Generate unit tests for a given function",
            context={
                "function_code": "",
                "language": "python",
                "prompt_style": "detailed"
            },
            scenario_type=ScenarioType.PROMPT_ENGINEERING,
            tags=["prompting", "code_generation", "testing"]
        ))
        
        # Agent configuration optimization
        self.register("agent_configuration", lambda: Scenario(
            scenario_id="agent_configuration",
            name="Agent Configuration Optimization",
            description="Find optimal agent parameter configurations",
            goal="Complete a refactoring task efficiently",
            context={
                "refactoring_target": "",
                "max_iterations": 3,
                "temperature": 0.7,
                "model": "default"
            },
            scenario_type=ScenarioType.CONFIGURATION,
            tags=["configuration", "optimization", "agents"]
        ))
        
        # Planning approach comparison
        self.register("planning_approach", lambda: Scenario(
            scenario_id="planning_approach",
            name="Planning Approach Comparison",
            description="Compare Tree-of-Thought vs single-path planning",
            goal="Implement a new feature with proper planning",
            context={
                "feature_spec": "",
                "planning_method": "tree_of_thought",
                "n_candidates": 3
            },
            scenario_type=ScenarioType.PLANNING,
            tags=["planning", "tree_of_thought", "strategy"]
        ))
        
        # Code generation quality
        self.register("code_quality", lambda: Scenario(
            scenario_id="code_quality",
            name="Code Generation Quality",
            description="Test code generation quality under different constraints",
            goal="Generate high-quality code for a given specification",
            context={
                "specification": "",
                "language": "python",
                "quality_focus": "readability",
                "include_tests": True
            },
            scenario_type=ScenarioType.CODE_GENERATION,
            tags=["code_generation", "quality", "testing"]
        ))
        
        # Refactoring safety
        self.register("refactoring_safety", lambda: Scenario(
            scenario_id="refactoring_safety",
            name="Refactoring Safety",
            description="Test refactoring approaches for safety",
            goal="Refactor code while maintaining behavior",
            context={
                "code_to_refactor": "",
                "refactoring_type": "extract_method",
                "safety_checks": True
            },
            scenario_type=ScenarioType.REFACTORING,
            tags=["refactoring", "safety", "behavior_preservation"]
        ))
        
        # Test generation strategies
        self.register("test_generation", lambda: Scenario(
            scenario_id="test_generation",
            name="Test Generation Strategies",
            description="Compare different test generation approaches",
            goal="Generate comprehensive tests for a function",
            context={
                "function_code": "",
                "test_framework": "pytest",
                "coverage_target": 0.9,
                "include_edge_cases": True
            },
            scenario_type=ScenarioType.TEST_GENERATION,
            tags=["testing", "test_generation", "coverage"]
        ))
        
        # Debug strategy comparison
        self.register("debug_strategy", lambda: Scenario(
            scenario_id="debug_strategy",
            name="Debug Strategy Comparison",
            description="Compare different debugging approaches",
            goal="Fix a bug efficiently",
            context={
                "error_message": "",
                "stack_trace": "",
                "approach": "systematic"
            },
            scenario_type=ScenarioType.DEBUGGING,
            tags=["debugging", "error_handling", "troubleshooting"]
        ))


# Global scenario registry instance
SIMULATION_SCENARIOS = ScenarioRegistry()

# Built-in scenarios for quick access
BUILTIN_SCENARIOS = {
    "prompt_strategy": "Test different prompt strategies for code generation",
    "agent_configuration": "Find optimal agent parameter configurations",
    "planning_approach": "Compare Tree-of-Thought vs single-path planning",
    "code_quality": "Test code generation quality under different constraints",
    "refactoring_safety": "Test refactoring approaches for safety",
    "test_generation": "Compare different test generation approaches",
    "debug_strategy": "Compare different debugging approaches",
}
