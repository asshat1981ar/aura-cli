"""Python-specialized agent for code analysis, generation, testing, and linting.

Uses AURA skills: linter_enforcer, type_checker, complexity_scorer,
test_coverage_analyzer for Python-specific workflows.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PythonAgentAdapter:
    """Pipeline adapter for Python-specific operations.

    Wraps available Python-oriented skills and the model adapter to provide
    Python code analysis, generation, testing, and backend operations.
    """

    name = "python_agent"

    def __init__(self, model_adapter=None, skills: Optional[Dict[str, Any]] = None):
        self.model = model_adapter
        self.skills = skills or {}

    def run(self, input_data: dict) -> dict:
        """Execute Python-specific analysis and code generation.

        Args:
            input_data: Dict with keys:
                - task (str): Natural-language task description.
                - code (str, optional): Existing code to analyze/modify.
                - file_path (str, optional): Path to the file.
                - project_root (str, optional): Project root path.
                - action (str, optional): Specific action — "analyze", "generate",
                  "lint", "test", "type_check". Defaults to "analyze".

        Returns:
            Dict with keys: action, analysis, suggestions, lint_results,
            type_check_results, generated_code (depending on action).
        """
        task = input_data.get("task", "")
        code = input_data.get("code", "")
        action = input_data.get("action", "analyze")

        result: Dict[str, Any] = {"action": action, "task": task}

        if action == "lint" or action == "analyze":
            result["lint_results"] = self._run_linter(code, input_data)

        if action == "type_check" or action == "analyze":
            result["type_check_results"] = self._run_type_checker(code, input_data)

        if action == "analyze":
            result["complexity"] = self._run_complexity(code, input_data)
            result["coverage"] = self._run_coverage(input_data)

        if action == "generate":
            result["generated_code"] = self._generate(task, code, input_data)

        if action == "test":
            result["test_results"] = self._run_tests(input_data)

        return result

    def _run_linter(self, code: str, input_data: dict) -> dict:
        """Run linter_enforcer skill if available."""
        skill = self.skills.get("linter_enforcer")
        if skill and hasattr(skill, "run"):
            try:
                return skill.run({"code": code, **input_data})
            except Exception as exc:
                return {"error": str(exc)}
        return {"status": "skill_not_available", "skill": "linter_enforcer"}

    def _run_type_checker(self, code: str, input_data: dict) -> dict:
        """Run type_checker skill if available."""
        skill = self.skills.get("type_checker")
        if skill and hasattr(skill, "run"):
            try:
                return skill.run({"code": code, **input_data})
            except Exception as exc:
                return {"error": str(exc)}
        return {"status": "skill_not_available", "skill": "type_checker"}

    def _run_complexity(self, code: str, input_data: dict) -> dict:
        """Run complexity_scorer skill if available."""
        skill = self.skills.get("complexity_scorer")
        if skill and hasattr(skill, "run"):
            try:
                return skill.run({"code": code, **input_data})
            except Exception as exc:
                return {"error": str(exc)}
        return {"status": "skill_not_available", "skill": "complexity_scorer"}

    def _run_coverage(self, input_data: dict) -> dict:
        """Run test_coverage_analyzer skill if available."""
        skill = self.skills.get("test_coverage_analyzer")
        if skill and hasattr(skill, "run"):
            try:
                return skill.run(input_data)
            except Exception as exc:
                return {"error": str(exc)}
        return {"status": "skill_not_available", "skill": "test_coverage_analyzer"}

    def _generate(self, task: str, code: str, input_data: dict) -> str:
        """Generate Python code using the model adapter."""
        if self.model is None:
            return ""
        prompt = f"Generate Python code for: {task}"
        if code:
            prompt += f"\n\nExisting code:\n{code}"
        try:
            if hasattr(self.model, "generate"):
                return self.model.generate(prompt)
            return ""
        except (OSError, IOError, ValueError):
            return ""

    def _run_tests(self, input_data: dict) -> dict:
        """Run test execution via tester skill or subprocess."""
        skill = self.skills.get("tester")
        if skill and hasattr(skill, "run"):
            try:
                return skill.run(input_data)
            except Exception as exc:
                return {"error": str(exc)}
        return {"status": "skill_not_available", "skill": "tester"}
