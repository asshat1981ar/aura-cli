"""Isolated simulation runner with sandboxing capabilities."""

import asyncio
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.logging_utils import log_json
from core.simulation.scenario import Scenario, ScenarioOutcome


@dataclass
class ExecutionContext:
    """Context for isolated execution."""

    working_dir: Path
    environment_vars: Dict[str, str] = field(default_factory=dict)
    readonly_paths: List[Path] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)


class IsolatedSimulationRunner:
    """Runs scenarios in isolated environments to prevent side effects."""

    def __init__(self, temp_root: Optional[Path] = None):
        """
        Initialize the isolated runner.

        Args:
            temp_root: Root directory for temporary simulation workspaces
        """
        self.temp_root = temp_root or Path(tempfile.gettempdir()) / "aura_simulations"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._original_cwd = Path.cwd()
        self._active_runs: Dict[str, ExecutionContext] = {}

    async def run_scenario(self, scenario: Scenario, timeout: float) -> ScenarioOutcome:
        """
        Run a single scenario with full isolation.

        Args:
            scenario: The scenario to run
            timeout: Maximum execution time in seconds

        Returns:
            ScenarioOutcome with results
        """
        run_id = f"{scenario.scenario_id}_{int(time.time())}_{os.urandom(4).hex()}"
        run_dir = self.temp_root / run_id

        start_time = time.time()

        try:
            # Create isolated workspace
            run_dir.mkdir(parents=True)

            log_json("DEBUG", "simulation_run_starting", {"run_id": run_id, "scenario": scenario.scenario_id, "timeout": timeout})

            # Setup isolated context
            context = ExecutionContext(
                working_dir=run_dir,
                environment_vars={
                    "AURA_SIMULATION_MODE": "1",
                    "AURA_DISABLE_WRITES": "1",
                    "AURA_SIM_SCENARIO_ID": scenario.scenario_id,
                },
            )
            self._active_runs[run_id] = context

            # Execute with timeout and isolation
            with self._isolated_environment(run_dir, context.environment_vars):
                outcome = await self._execute_with_timeout(scenario, timeout)

            outcome.execution_time_seconds = time.time() - start_time

            log_json("DEBUG", "simulation_run_completed", {"run_id": run_id, "success": outcome.success, "duration": outcome.execution_time_seconds})

            return outcome

        except asyncio.TimeoutError:
            log_json("WARN", "simulation_timeout", {"run_id": run_id, "scenario": scenario.scenario_id, "timeout": timeout})
            return ScenarioOutcome(scenario_id=scenario.scenario_id, scenario_name=scenario.name, success=False, execution_time_seconds=time.time() - start_time, error_message=f"Timeout after {timeout}s")

        except Exception as e:
            log_json("ERROR", "simulation_run_failed", {"run_id": run_id, "scenario": scenario.scenario_id, "error": str(e)})
            return ScenarioOutcome(scenario_id=scenario.scenario_id, scenario_name=scenario.name, success=False, execution_time_seconds=time.time() - start_time, error_message=str(e))

        finally:
            # Cleanup
            self._active_runs.pop(run_id, None)
            self._cleanup_workspace(run_dir)

    async def _execute_with_timeout(self, scenario: Scenario, timeout: float) -> ScenarioOutcome:
        """Execute scenario with timeout."""
        try:
            return await asyncio.wait_for(self._execute_scenario(scenario), timeout=timeout)
        except asyncio.TimeoutError:
            raise

    async def _execute_scenario(self, scenario: Scenario) -> ScenarioOutcome:
        """
        Execute the actual scenario logic.

        This is a placeholder that can be extended with actual
        simulation execution logic based on scenario type.
        """
        metrics = {}
        logs = []
        artifacts = {}

        # Record scenario context
        logs.append(f"Executing scenario: {scenario.name}")
        logs.append(f"Goal: {scenario.goal}")
        logs.append(f"Parameters: {scenario.parameter_overrides}")

        # Simulate execution based on scenario type
        scenario_type = scenario.scenario_type

        if scenario_type.value == "prompt_engineering":
            # Simulate prompt strategy testing
            prompt_style = scenario.context.get("prompt_style", "default")

            # Mock metrics based on prompt style
            effectiveness = {
                "detailed": 0.85,
                "concise": 0.75,
                "chain_of_thought": 0.90,
                "few_shot": 0.88,
            }.get(prompt_style, 0.70)

            metrics = {
                "effectiveness": effectiveness,
                "response_quality": effectiveness * 0.95,
                "token_efficiency": 0.8 if prompt_style == "concise" else 0.6,
            }

        elif scenario_type.value == "configuration":
            # Simulate configuration testing
            temp = scenario.context.get("temperature", 0.7)
            max_iter = scenario.context.get("max_iterations", 3)

            metrics = {
                "completion_rate": min(0.95, 0.7 + temp * 0.2),
                "avg_iterations": max_iter,
                "quality_score": 0.8 - abs(temp - 0.5) * 0.3,
            }

        elif scenario_type.value == "planning":
            # Simulate planning approach testing
            method = scenario.context.get("planning_method", "single")
            n_candidates = scenario.context.get("n_candidates", 1)

            base_quality = 0.75
            if method == "tree_of_thought":
                base_quality += 0.1
            if n_candidates > 1:
                base_quality += min(0.1, n_candidates * 0.02)

            metrics = {
                "plan_quality": min(0.95, base_quality),
                "planning_time": 1.0 + n_candidates * 0.3,
                "success_prediction": base_quality * 0.9,
            }

        elif scenario_type.value == "code_generation":
            # Simulate code generation testing
            quality_focus = scenario.context.get("quality_focus", "balanced")

            quality_scores = {
                "readability": 0.90,
                "performance": 0.85,
                "maintainability": 0.88,
                "balanced": 0.82,
            }

            metrics = {
                "code_quality": quality_scores.get(quality_focus, 0.80),
                "test_coverage": 0.75 if scenario.context.get("include_tests") else 0.0,
                "compile_success": 0.95,
            }

        elif scenario_type.value == "refactoring":
            # Simulate refactoring testing
            safety_checks = scenario.context.get("safety_checks", True)

            base_safety = 0.85
            if safety_checks:
                base_safety += 0.1

            metrics = {
                "behavior_preservation": min(0.98, base_safety),
                "code_improvement": 0.15,  # 15% improvement
                "test_pass_rate": min(0.98, base_safety),
            }

        elif scenario_type.value == "test_generation":
            # Simulate test generation testing
            coverage_target = scenario.context.get("coverage_target", 0.8)

            metrics = {
                "coverage_achieved": min(coverage_target * 1.1, 0.95),
                "test_quality": 0.85,
                "execution_time": 0.5,
            }

        elif scenario_type.value == "debugging":
            # Simulate debugging testing
            approach = scenario.context.get("approach", "default")

            effectiveness = {
                "systematic": 0.90,
                "intuitive": 0.75,
                "bisection": 0.85,
                "default": 0.80,
            }.get(approach, 0.75)

            metrics = {
                "fix_success": effectiveness,
                "time_to_fix": 2.0 - effectiveness,
                "regression_risk": 1.0 - effectiveness,
            }

        else:
            # Generic simulation
            metrics = {
                "success_rate": 0.8,
                "quality": 0.75,
            }

        # Calculate overall score
        score = sum(metrics.values()) / len(metrics) if metrics else 0.0
        success = score >= 0.7

        return ScenarioOutcome(scenario_id=scenario.scenario_id, scenario_name=scenario.name, success=success, metrics=metrics, score=score, logs=logs, artifacts=artifacts)

    @contextmanager
    def _isolated_environment(self, working_dir: Path, env_vars: Dict[str, str]):
        """Context manager for isolated execution environment."""
        old_cwd = Path.cwd()
        old_env = dict(os.environ)

        try:
            # Change to isolated working directory
            os.chdir(working_dir)

            # Set simulation environment variables
            for key, value in env_vars.items():
                os.environ[key] = value

            yield

        finally:
            # Restore original state
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)

    def _cleanup_workspace(self, run_dir: Path):
        """Clean up temporary workspace."""
        try:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
        except Exception as e:
            log_json("WARN", "simulation_cleanup_failed", {"path": str(run_dir), "error": str(e)})

    def get_active_runs(self) -> List[str]:
        """Get list of currently active simulation runs."""
        return list(self._active_runs.keys())

    def force_cleanup_all(self):
        """Force cleanup of all simulation workspaces."""
        for run_id in list(self._active_runs.keys()):
            self._active_runs.pop(run_id, None)

        try:
            if self.temp_root.exists():
                shutil.rmtree(self.temp_root)
                self.temp_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log_json("ERROR", "force_cleanup_failed", {"error": str(e)})
