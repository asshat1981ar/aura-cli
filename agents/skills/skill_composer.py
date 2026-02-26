"""Skill: compose a multi-skill workflow plan from a natural-language goal."""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

# Goal category keyword mapping -> ordered list of recommended skills
_WORKFLOWS: Dict[str, List[Dict]] = {
    "refactor": [
        {"skill_name": "complexity_scorer", "reason": "Identify high-complexity functions to refactor"},
        {"skill_name": "refactoring_advisor", "reason": "Detect code smells and suggest improvements"},
        {"skill_name": "linter_enforcer", "reason": "Enforce style consistency after refactor"},
        {"skill_name": "generation_quality_checker", "reason": "Validate quality of generated refactored code"},
    ],
    "security": [
        {"skill_name": "security_scanner", "reason": "Detect secrets, injection risks, unsafe calls"},
        {"skill_name": "dependency_analyzer", "reason": "Check for vulnerable dependencies"},
        {"skill_name": "type_checker", "reason": "Static types reduce attack surface"},
    ],
    "test": [
        {"skill_name": "test_coverage_analyzer", "reason": "Find untested code paths"},
        {"skill_name": "generation_quality_checker", "reason": "Score generated test quality"},
        {"skill_name": "complexity_scorer", "reason": "High complexity = higher test priority"},
    ],
    "document": [
        {"skill_name": "doc_generator", "reason": "Generate docstring templates for undocumented items"},
        {"skill_name": "refactoring_advisor", "reason": "Clean up code before documenting"},
    ],
    "performance": [
        {"skill_name": "performance_profiler", "reason": "Find hotspots and anti-patterns"},
        {"skill_name": "complexity_scorer", "reason": "High complexity often correlates with slowness"},
        {"skill_name": "git_history_analyzer", "reason": "Hotspot files may need performance focus"},
    ],
    "architecture": [
        {"skill_name": "architecture_validator", "reason": "Detect circular imports and coupling"},
        {"skill_name": "tech_debt_quantifier", "reason": "Quantify structural debt"},
        {"skill_name": "incremental_differ", "reason": "Analyse impact of architectural changes"},
    ],
    "dependency": [
        {"skill_name": "dependency_analyzer", "reason": "Find conflicts and CVEs"},
        {"skill_name": "security_scanner", "reason": "Cross-check for hardcoded credentials"},
    ],
    "api": [
        {"skill_name": "api_contract_validator", "reason": "Extract and validate endpoint contracts"},
        {"skill_name": "schema_validator", "reason": "Validate request/response schemas"},
        {"skill_name": "type_checker", "reason": "Check API handler type safety"},
    ],
    "debt": [
        {"skill_name": "tech_debt_quantifier", "reason": "Measure overall debt score"},
        {"skill_name": "refactoring_advisor", "reason": "Prioritize code smells to fix"},
        {"skill_name": "linter_enforcer", "reason": "Enforce standards to prevent new debt"},
    ],
    "git": [
        {"skill_name": "git_history_analyzer", "reason": "Identify hotspot files"},
        {"skill_name": "incremental_differ", "reason": "Understand impact of recent changes"},
    ],
    "fix": [
        {"skill_name": "error_pattern_matcher", "reason": "Match error against known patterns"},
        {"skill_name": "type_checker", "reason": "Catch type-related bugs"},
        {"skill_name": "linter_enforcer", "reason": "Check for obvious style/logic errors"},
    ],
    "duplicate": [
        {"skill_name": "code_clone_detector", "reason": "Find exact and near-duplicate code blocks"},
        {"skill_name": "refactoring_advisor", "reason": "Suggest consolidation strategies"},
    ],
    "strategy": [
        {"skill_name": "adaptive_strategy_selector", "reason": "Choose best execution strategy for goal"},
    ],
}

_KEYWORD_CATEGORY = {
    "refactor": "refactor", "restructure": "refactor", "clean": "refactor", "simplify": "refactor",
    "security": "security", "secure": "security", "vulnerability": "security", "cve": "security", "secret": "security",
    "test": "test", "coverage": "test", "unit": "test", "pytest": "test",
    "doc": "document", "document": "document", "readme": "document", "docstring": "document",
    "perf": "performance", "performance": "performance", "speed": "performance", "optimize": "performance", "slow": "performance",
    "architect": "architecture", "design": "architecture", "import": "architecture", "coupling": "architecture",
    "depend": "dependency", "pip": "dependency", "package": "dependency", "requirements": "dependency",
    "api": "api", "endpoint": "api", "route": "api", "fastapi": "api", "flask": "api",
    "debt": "debt", "technical debt": "debt", "todo": "debt", "hack": "debt",
    "git": "git", "commit": "git", "history": "git", "hotspot": "git",
    "fix": "fix", "bug": "fix", "error": "fix", "debug": "fix",
    "duplicate": "duplicate", "clone": "duplicate", "repeat": "duplicate", "dry": "duplicate",
    "strategy": "strategy", "adapt": "strategy",
}


class SkillComposerSkill(SkillBase):
    name = "skill_composer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        goal: str = input_data.get("goal", "")
        available: Optional[List[str]] = input_data.get("available_skills")
        goal_lower = goal.lower()

        # Find matching category
        matched_category = "refactor"  # default
        best_match_count = 0
        for kw, cat in _KEYWORD_CATEGORY.items():
            count = goal_lower.count(kw)
            if count > best_match_count:
                best_match_count = count
                matched_category = cat

        workflow = _WORKFLOWS.get(matched_category, _WORKFLOWS["refactor"])

        # Filter to available skills if specified
        if available:
            workflow = [step for step in workflow if step["skill_name"] in available]

        log_json("INFO", "skill_composer_complete", details={"category": matched_category, "steps": len(workflow)})
        return {"workflow": workflow, "estimated_steps": len(workflow), "goal_category": matched_category, "goal": goal}
