"""Skill: analyzes cycle summaries to identify frequently failing skills."""
from __future__ import annotations
import json
import os
from collections import Counter
from typing import Any, Dict

from agents.skills.base import SkillBase

_REMEDIATIONS = {
    "default": "Review skill input schema and ensure required fields are present.",
    "security_scanner": "Check that project_root is accessible and contains Python files.",
    "test_coverage_analyzer": "Ensure pytest and coverage are installed and tests exist.",
    "type_checker": "Verify mypy is installed and the project has valid Python syntax.",
    "dependency_analyzer": "Confirm requirements.txt or pyproject.toml is present.",
    "architecture_validator": "Ensure source files are importable and have no circular deps at top level.",
    "doc_generator": "Check that code files exist and are syntactically valid Python.",
    "performance_profiler": "Provide non-empty 'code' input with at least one function.",
    "web_fetcher": "Verify network connectivity and that the URL/query field is provided.",
    "multi_file_editor": "Supply a specific 'goal' and ensure project_root is writable.",
}

_CYCLE_SUMMARIES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "memory", "store", "cycle_summaries.json"
)


class SkillFailureAnalyzerSkill(SkillBase):
    name = "skill_failure_analyzer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        summaries_path = input_data.get("summaries_path", _CYCLE_SUMMARIES_PATH)

        try:
            with open(summaries_path, "r", encoding="utf-8") as fh:
                summaries = json.load(fh)
        except FileNotFoundError:
            return {
                "failing_skills": [],
                "total_cycles_analyzed": 0,
                "most_problematic": None,
                "error": f"Summaries file not found: {summaries_path}",
            }
        except json.JSONDecodeError as exc:
            return {
                "failing_skills": [],
                "total_cycles_analyzed": 0,
                "most_problematic": None,
                "error": f"Invalid JSON in summaries file: {exc}",
            }

        failure_counter: Counter = Counter()
        for cycle in summaries:
            for failure in cycle.get("failures", []):
                if isinstance(failure, str):
                    failure_counter[failure] += 1
                elif isinstance(failure, dict):
                    skill_name = failure.get("skill") or failure.get("name") or str(failure)
                    failure_counter[skill_name] += 1

        failing_skills = [
            {
                "skill": skill,
                "failure_count": count,
                "remediation": _REMEDIATIONS.get(skill, _REMEDIATIONS["default"]),
            }
            for skill, count in failure_counter.most_common()
        ]

        most_problematic = failure_counter.most_common(1)[0][0] if failure_counter else None

        return {
            "failing_skills": failing_skills,
            "total_cycles_analyzed": len(summaries),
            "most_problematic": most_problematic,
        }
