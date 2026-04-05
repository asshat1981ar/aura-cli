"""
Agentic Evaluation Framework

Patterns for self-improvement through iterative evaluation and refinement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from core.logging_utils import log_json


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating outputs."""
    name: str
    weight: float
    description: str
    threshold: float = 0.8


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    score: float
    passed: bool
    feedback: str
    dimensions: Dict[str, float] = field(default_factory=dict)
    iteration: int = 0


@dataclass
class RefinementHistory:
    """History of refinement iterations."""
    iterations: List[Tuple[str, EvaluationResult]] = field(default_factory=list)
    final_output: Optional[str] = None
    total_time_ms: float = 0.0


class Generator(Protocol):
    """Protocol for generation function."""
    def __call__(self, task: str) -> str: ...


class Evaluator(Protocol):
    """Protocol for evaluation function."""
    def __call__(self, output: str, task: str) -> EvaluationResult: ...


class Optimizer(Protocol):
    """Protocol for optimization function."""
    def __call__(self, output: str, feedback: EvaluationResult) -> str: ...


class BasicReflection:
    """
    Pattern 1: Basic Reflection
    Agent evaluates and improves its own output through self-critique.
    """
    
    def __init__(
        self,
        generator: Generator,
        criteria: List[str],
        max_iterations: int = 3,
        min_improvement: float = 0.05,
    ):
        self.generator = generator
        self.criteria = criteria
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.history: List[RefinementHistory] = []
    
    def evaluate(self, output: str, task: str) -> EvaluationResult:
        """Evaluate output against criteria."""
        # Use LLM to evaluate
        from core.model_adapter import ModelAdapter
        model = ModelAdapter()
        
        prompt = f"""
        Evaluate this output against criteria: {self.criteria}
        
        Task: {task}
        Output: {output}
        
        Return JSON with:
        {{
            "score": 0.0-1.0,
            "passed": true/false,
            "feedback": "detailed feedback",
            "dimensions": {{"criterion1": score, ...}}
        }}
        """
        
        try:
            response = model.generate_text(prompt)
            data = json.loads(response)
            return EvaluationResult(
                score=data.get("score", 0.0),
                passed=data.get("passed", False),
                feedback=data.get("feedback", ""),
                dimensions=data.get("dimensions", {}),
            )
        except Exception as e:
            log_json("WARN", "evaluation_parse_failed", {"error": str(e)})
            return EvaluationResult(score=0.0, passed=False, feedback=str(e))
    
    def optimize(self, output: str, feedback: EvaluationResult) -> str:
        """Refine output based on feedback."""
        from core.model_adapter import ModelAdapter
        model = ModelAdapter()
        
        prompt = f"""
        Improve this output based on feedback:
        
        Current output: {output}
        Feedback: {feedback.feedback}
        Score: {feedback.score}
        
        Provide improved version addressing all issues.
        """
        
        return model.generate_text(prompt)
    
    def run(self, task: str) -> Tuple[str, RefinementHistory]:
        """Run reflection loop."""
        start_time = time.time()
        history = RefinementHistory()
        
        output = self.generator(task)
        last_score = 0.0
        
        for iteration in range(self.max_iterations):
            result = self.evaluate(output, task)
            result.iteration = iteration
            
            history.iterations.append((output, result))
            
            if result.passed:
                log_json("INFO", "reflection_converged", {
                    "iteration": iteration,
                    "score": result.score,
                })
                break
            
            # Check for improvement
            improvement = result.score - last_score
            if iteration > 0 and improvement < self.min_improvement:
                log_json("INFO", "reflection_stalled", {
                    "iteration": iteration,
                    "improvement": improvement,
                })
                break
            
            last_score = result.score
            output = self.optimize(output, result)
        
        history.final_output = output
        history.total_time_ms = (time.time() - start_time) * 1000
        self.history.append(history)
        
        return output, history


class EvaluatorOptimizer:
    """
    Pattern 2: Evaluator-Optimizer
    Separate generation and evaluation into distinct components.
    """
    
    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        optimizer: Optimizer,
        score_threshold: float = 0.8,
        max_iterations: int = 3,
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.score_threshold = score_threshold
        self.max_iterations = max_iterations
        self.history: List[RefinementHistory] = []
    
    def run(self, task: str) -> Tuple[str, RefinementHistory]:
        """Run evaluator-optimizer loop."""
        start_time = time.time()
        history = RefinementHistory()
        
        output = self.generator(task)
        
        for iteration in range(self.max_iterations):
            evaluation = self.evaluator(output, task)
            evaluation.iteration = iteration
            
            history.iterations.append((output, evaluation))
            
            if evaluation.score >= self.score_threshold:
                log_json("INFO", "evaluator_optimizer_converged", {
                    "iteration": iteration,
                    "score": evaluation.score,
                })
                break
            
            output = self.optimizer(output, evaluation)
        
        history.final_output = output
        history.total_time_ms = (time.time() - start_time) * 1000
        self.history.append(history)
        
        return output, history


class CodeReflector:
    """
    Pattern 3: Code-Specific Reflection
    Test-driven refinement loop for code generation.
    """
    
    def __init__(
        self,
        max_iterations: int = 3,
        timeout_seconds: float = 30.0,
    ):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.history: List[RefinementHistory] = []
    
    def generate_code(self, spec: str) -> str:
        """Generate code from specification."""
        from core.model_adapter import ModelAdapter
        model = ModelAdapter()
        
        prompt = f"""
        Write Python code for this specification:
        {spec}
        
        Requirements:
        - Include type hints
        - Follow PEP 8
        - Add docstrings
        - Make it testable
        
        Return only the code, no explanations.
        """
        
        return model.generate_text(prompt)
    
    def generate_tests(self, spec: str, code: str) -> str:
        """Generate pytest tests for code."""
        from core.model_adapter import ModelAdapter
        model = ModelAdapter()
        
        prompt = f"""
        Generate pytest tests for this code:
        
        Specification: {spec}
        Code: {code}
        
        Include:
        - Unit tests for all functions
        - Edge case tests
        - Error handling tests
        - Use pytest fixtures if needed
        
        Return only the test code.
        """
        
        return model.generate_text(prompt)
    
    def run_tests(self, code: str, tests: str) -> Dict[str, Any]:
        """Run tests in sandbox."""
        import tempfile
        import subprocess
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code
            code_file = f"{tmpdir}/solution.py"
            with open(code_file, "w") as f:
                f.write(code)
            
            # Write tests
            test_file = f"{tmpdir}/test_solution.py"
            with open(test_file, "w") as f:
                f.write(f"from solution import *\n\n{tests}")
            
            # Run tests
            try:
                result = subprocess.run(
                    ["python3", "-m", "pytest", test_file, "-v"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=tmpdir,
                )
                
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Tests timed out after {self.timeout_seconds}s",
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    def fix_code(self, code: str, test_result: Dict[str, Any]) -> str:
        """Fix code based on test failures."""
        from core.model_adapter import ModelAdapter
        model = ModelAdapter()
        
        error = test_result.get("stderr", test_result.get("error", "Unknown error"))
        
        prompt = f"""
        Fix this code based on test failures:
        
        Current code:
        {code}
        
        Test output/error:
        {error}
        
        Provide fixed code only.
        """
        
        return model.generate_text(prompt)
    
    def reflect_and_fix(self, spec: str) -> Tuple[str, RefinementHistory]:
        """Run test-driven refinement loop."""
        start_time = time.time()
        history = RefinementHistory()
        
        code = self.generate_code(spec)
        tests = self.generate_tests(spec, code)
        
        for iteration in range(self.max_iterations):
            result = self.run_tests(code, tests)
            
            evaluation = EvaluationResult(
                score=1.0 if result["success"] else 0.0,
                passed=result["success"],
                feedback=result.get("stdout", result.get("error", "")),
                iteration=iteration,
            )
            
            history.iterations.append((code, evaluation))
            
            if result["success"]:
                log_json("INFO", "code_reflection_converged", {
                    "iteration": iteration,
                })
                break
            
            code = self.fix_code(code, result)
        
        history.final_output = code
        history.total_time_ms = (time.time() - start_time) * 1000
        self.history.append(history)
        
        return code, history


class RubricBasedEvaluator:
    """
    Pattern 4: Rubric-Based Evaluation
    Score outputs against weighted dimensions.
    """
    
    def __init__(self, rubric: Dict[str, Dict[str, float]]):
        """
        Initialize with rubric.
        
        Args:
            rubric: Dict of {dimension: {weight: float, threshold: float}}
        """
        self.rubric = rubric
        self.criteria: List[EvaluationCriteria] = [
            EvaluationCriteria(
                name=k,
                weight=v.get("weight", 1.0),
                description=v.get("description", ""),
                threshold=v.get("threshold", 0.8),
            )
            for k, v in rubric.items()
        ]
    
    def evaluate(self, output: str, context: str = "") -> EvaluationResult:
        """Evaluate against rubric."""
        from core.model_adapter import ModelAdapter
        model = ModelAdapter()
        
        dimensions_str = "\n".join([
            f"- {c.name}: {c.description} (weight: {c.weight})"
            for c in self.criteria
        ])
        
        prompt = f"""
        Evaluate this output against the rubric.
        
        Context: {context}
        
        Rubric dimensions:
        {dimensions_str}
        
        Output to evaluate:
        {output}
        
        Rate each dimension 1-5 and return JSON:
        {{
            "dimensions": {{"dim_name": score, ...}},
            "overall_score": 0.0-1.0,
            "feedback": "detailed feedback",
            "passed": true/false
        }}
        """
        
        try:
            response = model.generate_text(prompt)
            data = json.loads(response)
            
            dimensions = data.get("dimensions", {})
            weighted_score = sum(
                dimensions.get(c.name, 0) * c.weight
                for c in self.criteria
            ) / 5.0  # Normalize to 0-1
            
            return EvaluationResult(
                score=weighted_score,
                passed=data.get("passed", weighted_score >= 0.8),
                feedback=data.get("feedback", ""),
                dimensions=dimensions,
            )
        except Exception as e:
            log_json("WARN", "rubric_evaluation_failed", {"error": str(e)})
            return EvaluationResult(score=0.0, passed=False, feedback=str(e))


# Pre-defined rubrics for common tasks
RUBRICS = {
    "code_quality": {
        "correctness": {"weight": 0.4, "description": "Code works as intended"},
        "readability": {"weight": 0.3, "description": "Code is clear and well-structured"},
        "maintainability": {"weight": 0.2, "description": "Easy to modify and extend"},
        "efficiency": {"weight": 0.1, "description": "Good performance characteristics"},
    },
    "documentation": {
        "completeness": {"weight": 0.4, "description": "All aspects covered"},
        "clarity": {"weight": 0.3, "description": "Easy to understand"},
        "accuracy": {"weight": 0.3, "description": "Technically correct"},
    },
    "test_quality": {
        "coverage": {"weight": 0.4, "description": "Tests cover key paths"},
        "clarity": {"weight": 0.3, "description": "Tests are readable"},
        "maintainability": {"weight": 0.2, "description": "Easy to update"},
        "speed": {"weight": 0.1, "description": "Tests run quickly"},
    },
    "sadd_workstream": {
        "completeness": {"weight": 0.3, "description": "All tasks completed"},
        "quality": {"weight": 0.3, "description": "High quality output"},
        "alignment": {"weight": 0.2, "description": "Matches acceptance criteria"},
        "efficiency": {"weight": 0.2, "description": "Resource efficient"},
    },
}


def evaluate_sadd_workstream(
    workstream_title: str,
    output: str,
    acceptance_criteria: List[str],
) -> EvaluationResult:
    """
    Evaluate a SADD workstream output using rubric-based evaluation.
    """
    evaluator = RubricBasedEvaluator(RUBRICS["sadd_workstream"])
    
    context = f"""
    Workstream: {workstream_title}
    Acceptance Criteria:
    {chr(10).join(f"- {c}" for c in acceptance_criteria)}
    """
    
    return evaluator.evaluate(output, context)


def reflect_and_refine_workstream(
    workstream_title: str,
    task: str,
    acceptance_criteria: List[str],
) -> Tuple[str, RefinementHistory]:
    """
    Apply reflection loop to SADD workstream.
    """
    def generator(t: str) -> str:
        from core.model_adapter import ModelAdapter
        model = ModelAdapter()
        return model.generate_text(f"Complete this SADD workstream:\n{t}")
    
    criteria = [
        f"Meets: {c}" for c in acceptance_criteria
    ] + [
        "High code quality",
        "Proper error handling",
        "Good test coverage",
    ]
    
    reflector = BasicReflection(
        generator=generator,
        criteria=criteria,
        max_iterations=3,
    )
    
    return reflector.run(task)
