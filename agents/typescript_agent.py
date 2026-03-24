"""TypeScript-specialized agent for frontend/backend Node.js workflows.

Delegates to tree-sitter MCP for AST analysis, semgrep for pattern matching,
and uses api_contract_validator and schema_validator skills.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class TypeScriptAgentAdapter:
    """Pipeline adapter for TypeScript/Node.js-specific operations."""

    name = "typescript_agent"

    def __init__(self, model_adapter=None, skills: Optional[Dict[str, Any]] = None):
        self.model = model_adapter
        self.skills = skills or {}

    def run(self, input_data: dict) -> dict:
        """Execute TypeScript-specific operations.

        Args:
            input_data: Dict with keys:
                - task (str): Task description.
                - code (str, optional): Existing code.
                - file_path (str, optional): File path.
                - project_root (str, optional): Project root.
                - action (str, optional): "analyze", "generate", "lint",
                  "type_check", "build". Defaults to "analyze".

        Returns:
            Dict with action results.
        """
        task = input_data.get("task", "")
        code = input_data.get("code", "")
        action = input_data.get("action", "analyze")
        project_root = input_data.get("project_root", ".")

        result: Dict[str, Any] = {"action": action, "task": task}

        if action == "lint" or action == "analyze":
            result["lint_results"] = self._run_eslint(project_root)

        if action == "type_check" or action == "analyze":
            result["type_check_results"] = self._run_tsc(project_root)

        if action == "analyze":
            result["api_contract"] = self._run_skill("api_contract_validator", input_data)
            result["schema"] = self._run_skill("schema_validator", input_data)

        if action == "generate":
            result["generated_code"] = self._generate(task, code)

        if action == "build":
            result["build_results"] = self._run_build(project_root)

        return result

    def _run_eslint(self, project_root: str) -> dict:
        """Run ESLint on the project."""
        try:
            proc = subprocess.run(
                ["npx", "eslint", ".", "--format", "json", "--max-warnings", "50"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "exit_code": proc.returncode,
                "output": proc.stdout[:2000],
                "errors": proc.stderr[:500] if proc.returncode != 0 else "",
            }
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return {"error": str(exc)}

    def _run_tsc(self, project_root: str) -> dict:
        """Run TypeScript compiler check."""
        try:
            proc = subprocess.run(
                ["npx", "tsc", "--noEmit"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "exit_code": proc.returncode,
                "output": proc.stdout[:2000],
                "errors": proc.stderr[:500] if proc.returncode != 0 else "",
            }
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return {"error": str(exc)}

    def _run_build(self, project_root: str) -> dict:
        """Run npm build."""
        try:
            proc = subprocess.run(
                ["npm", "run", "build"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "exit_code": proc.returncode,
                "output": proc.stdout[:2000],
                "errors": proc.stderr[:500] if proc.returncode != 0 else "",
            }
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return {"error": str(exc)}

    def _run_skill(self, skill_name: str, input_data: dict) -> dict:
        """Run a named skill if available."""
        skill = self.skills.get(skill_name)
        if skill and hasattr(skill, "run"):
            try:
                return skill.run(input_data)
            except Exception as exc:
                return {"error": str(exc)}
        return {"status": "skill_not_available", "skill": skill_name}

    def _generate(self, task: str, code: str) -> str:
        """Generate TypeScript code using the model adapter."""
        if self.model is None:
            return ""
        prompt = f"Generate TypeScript code for: {task}"
        if code:
            prompt += f"\n\nExisting code:\n{code}"
        try:
            if hasattr(self.model, "generate"):
                return self.model.generate(prompt)
            return ""
        except Exception:
            return ""
