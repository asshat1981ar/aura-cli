"""
Parallel skill dispatcher for goal-type-aware context enrichment.

Classifies the incoming goal into one of five types, selects the most
relevant skills for that type, then runs them concurrently in a thread
pool.  Results feed directly into the Planner as enriched context.

Usage::

    from core.skill_dispatcher import classify_goal, dispatch_skills
    from agents.skills.registry import all_skills

    skills = all_skills()
    goal_type = classify_goal("Fix the login crash when token is None")
    skill_context = dispatch_skills(goal_type, skills, project_root=".")
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Goal-type → skill selection map
# Skills are listed in priority order; only those present in the registry run.
# ---------------------------------------------------------------------------
SKILL_MAP: Dict[str, list[str]] = {
    "bug_fix":  [
        "symbol_indexer",
        "error_pattern_matcher",
        "git_history_analyzer",
        "type_checker",
        "linter_enforcer",
    ],
    "feature":  [
        "symbol_indexer",
        "architecture_validator",
        "api_contract_validator",
        "complexity_scorer",
        "dependency_analyzer",
    ],
    "refactor": [
        "symbol_indexer",
        "complexity_scorer",
        "code_clone_detector",
        "tech_debt_quantifier",
        "refactoring_advisor",
    ],
    "security": [
        "security_scanner",
        "dependency_analyzer",
        "type_checker",
        "linter_enforcer",
        "architecture_validator",
    ],
    "docs": [
        "doc_generator",
        "symbol_indexer",
    ],
    "default": [
        "symbol_indexer",
        "linter_enforcer",
    ],
}

# Keyword hints used to classify a goal string
_GOAL_TYPE_HINTS: Dict[str, list[str]] = {
    "bug_fix":  ["fix", "bug", "error", "crash", "broken", "fail", "issue",
                 "regression", "exception", "traceback", "panic"],
    "feature":  ["add", "implement", "feature", "new", "create", "build",
                 "support", "integrate", "introduce"],
    "refactor": ["refactor", "clean", "improve", "simplify", "restructure",
                 "reorganize", "extract", "dedup", "consolidate"],
    "security": ["security", "vulnerability", "auth", "permission", "sanitize",
                 "injection", "cve", "exploit", "xss", "sqli"],
    "docs":     ["doc", "docstring", "comment", "readme", "explain",
                 "document", "annotate"],
}


def classify_goal(goal: str) -> str:
    """Return the goal type string most appropriate for *goal*.

    Uses keyword-overlap scoring across goal types.  Falls back to
    ``"default"`` when no keywords match.
    """
    goal_lower = goal.lower()
    scores = {
        gt: sum(1 for kw in kws if kw in goal_lower)
        for gt, kws in _GOAL_TYPE_HINTS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "default"


def dispatch_skills(
    goal_type: str,
    skills: Dict[str, Any],
    project_root: str,
    timeout: float = 20.0,
) -> Dict[str, Any]:
    """Run goal-relevant skills concurrently and return their combined output.

    Args:
        goal_type:    One of the keys in :data:`SKILL_MAP` (or ``"default"``).
        skills:       Dict of ``{skill_name: SkillBase}`` from ``all_skills()``.
        project_root: Absolute or relative path passed to each skill.
        timeout:      Wall-clock seconds to wait for all skills to finish.
                      Skills that exceed the timeout are cancelled and their
                      result is recorded as ``{"error": "timeout"}``.

    Returns:
        ``{skill_name: result_dict}`` — never raises.
    """
    skill_names = SKILL_MAP.get(goal_type, SKILL_MAP["default"])
    available = [n for n in skill_names if n in skills]

    if not available:
        log_json("WARN", "skill_dispatch_no_skills", details={"goal_type": goal_type})
        return {}

    log_json(
        "INFO", "skill_dispatch_start",
        details={"goal_type": goal_type, "skills": available},
    )

    results: Dict[str, Any] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(available), 5), thread_name_prefix="skill"
    ) as pool:
        futures: Dict[concurrent.futures.Future, str] = {
            pool.submit(skills[name].run, {"project_root": project_root}): name
            for name in available
        }
        done, not_done = concurrent.futures.wait(futures, timeout=timeout)

        for fut in done:
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:  # skill raised despite SkillBase guard
                log_json("WARN", "skill_dispatch_error",
                         details={"skill": name, "error": str(exc)})
                results[name] = {"error": str(exc), "skill": name}

        for fut in not_done:
            name = futures[fut]
            log_json("WARN", "skill_dispatch_timeout", details={"skill": name})
            fut.cancel()
            results[name] = {"error": "timeout", "skill": name}

    log_json(
        "INFO", "skill_dispatch_done",
        details={
            "completed": [k for k, v in results.items() if "error" not in v],
            "failed":    [k for k, v in results.items() if "error" in v],
        },
    )
    return results
