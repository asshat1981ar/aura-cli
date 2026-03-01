"""
Skill Dispatcher — goal classification and parallel skill execution.

This module provides:

* ``SKILL_MAP`` — mapping from goal type to an ordered list of skill names.
* ``classify_goal(goal)`` — infers goal type from the goal string.
* ``dispatch_skills(goal_type, skills, project_root, timeout)`` — runs each
  skill callable against *project_root* and returns a dict of results.

Skills are plain callables with the signature ``skill(project_root: str) -> dict``.
All exceptions are caught per-skill so one failing skill never blocks others.
"""
from __future__ import annotations

import concurrent.futures
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Static skill map — goal_type -> ordered list of skill names
# ---------------------------------------------------------------------------

SKILL_MAP: Dict[str, List[str]] = {
    "bug_fix": [
        "security_scanner",
        "complexity_scorer",
        "test_coverage_analyzer",
        "tech_debt_quantifier",
        "architecture_validator",
    ],
    "feature": [
        "complexity_scorer",
        "architecture_validator",
        "tech_debt_quantifier",
        "test_coverage_analyzer",
        "security_scanner",
    ],
    "refactor": [
        "complexity_scorer",
        "architecture_validator",
        "tech_debt_quantifier",
        "security_scanner",
        "test_coverage_analyzer",
    ],
    "test": [
        "test_coverage_analyzer",
        "complexity_scorer",
        "tech_debt_quantifier",
        "security_scanner",
        "architecture_validator",
    ],
    "default": [
        "security_scanner",
        "complexity_scorer",
        "tech_debt_quantifier",
        "test_coverage_analyzer",
        "architecture_validator",
    ],
}

# ---------------------------------------------------------------------------
# Keyword patterns used by classify_goal
# ---------------------------------------------------------------------------

_GOAL_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("bug_fix",  re.compile(r"\b(fix|bug|error|crash|defect|patch|repair|broken|fail)\b", re.I)),
    ("test",     re.compile(r"\b(test|spec|coverage|unittest|pytest|assert)\b", re.I)),
    ("refactor", re.compile(r"\b(refactor|clean|restructure|reorganize|simplify|rename)\b", re.I)),
    ("feature",  re.compile(r"\b(add|implement|create|build|new|feature|extend|introduce)\b", re.I)),
]


def classify_goal(goal: str) -> str:
    """Return the goal type string most appropriate for *goal*.

    Matches are evaluated in priority order; the first match wins.
    Falls back to ``"default"`` when no pattern matches.
    """
    for goal_type, pattern in _GOAL_PATTERNS:
        if pattern.search(goal):
            return goal_type
    return "default"


# ---------------------------------------------------------------------------
# Skill dispatch
# ---------------------------------------------------------------------------

def dispatch_skills(
    goal_type: str,
    skills: Dict[str, Callable[..., Any]],
    project_root: str,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Run each skill in *skills* against *project_root* and return results.

    Args:
        goal_type:    Informational — logged but not used for dispatch here.
        skills:       Dict mapping skill name to callable
                      ``(project_root: str) -> dict``.
        project_root: Path passed as the first positional argument to each skill.
        timeout:      Per-skill wall-clock timeout in seconds (``None`` = no limit).

    Returns:
        Dict mapping skill name to its result dict, or
        ``{"error": "<msg>"}`` when the skill raised an exception or timed out.
    """
    if not skills:
        return {}

    results: Dict[str, Any] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(skills), 8)) as pool:
        future_to_name = {
            pool.submit(_run_skill, name, fn, project_root): name
            for name, fn in skills.items()
        }
        try:
            for future in concurrent.futures.as_completed(future_to_name, timeout=timeout):
                name = future_to_name[future]
                results[name] = future.result()
        except concurrent.futures.TimeoutError:
            # Mark any skills that didn't finish within the batch timeout
            for future, name in future_to_name.items():
                if name not in results:
                    log_json("WARN", "skill_dispatch_timeout",
                             details={"skill": name, "goal_type": goal_type})
                    results[name] = {"error": "timeout"}

    log_json("INFO", "skill_dispatch_complete",
             details={"goal_type": goal_type, "skills": list(results.keys())})
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_skill(name: str, fn: Callable[..., Any], project_root: str) -> Any:
    """Invoke a single skill callable, catching all exceptions."""
    try:
        return fn(project_root)
    except Exception as exc:
        log_json("WARN", "skill_execution_error",
                 details={"skill": name, "error": str(exc)})
        return {"error": str(exc)}
