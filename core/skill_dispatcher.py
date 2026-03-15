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
import time
from typing import Any, Dict

from core.logging_utils import log_json
from core.skill_classifier import SKILL_MAP, classify_goal, classify_goal_llm, _classify_goal_cache  # noqa: F401
from core.skill_metrics import SKILL_METRICS, SkillMetrics
from core.skill_chainer import chain_skill_results, SkillChainer  # noqa: F401


def dispatch_skills(
    goal_type: str,
    skills: Dict[str, Any],
    project_root: str,
    timeout: float = 20.0,
    corr_id: str = None,
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
        log_json("WARN", "skill_dispatch_no_skills", details={"goal_type": goal_type}, corr_id=corr_id, phase="skills", component="dispatcher")
        return {}

    log_json(
        "INFO", "skill_dispatch_start",
        details={"goal_type": goal_type, "skills": available},
        corr_id=corr_id,
        phase="skills",
        component="dispatcher",
    )

    results: Dict[str, Any] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(available), 5), thread_name_prefix="skill"
    ) as pool:
        def _timed_run(name: str) -> Any:
            t0 = time.monotonic()
            try:
                result = skills[name].run({"project_root": project_root})
                SKILL_METRICS.record(name, (time.monotonic() - t0) * 1000, error=False)
                return result
            except Exception as exc:
                SKILL_METRICS.record(name, (time.monotonic() - t0) * 1000, error=True)
                raise exc

        futures: Dict[concurrent.futures.Future, str] = {
            pool.submit(_timed_run, name): name
            for name in available
        }
        done, not_done = concurrent.futures.wait(futures, timeout=timeout)

        for fut in done:
            name = futures[fut]
            try:
                result = fut.result()
                # Distinguish a real skill error ({"error": ...}) from a
                # legitimate empty result so the orchestrator can tell them apart.
                if isinstance(result, dict) and "error" in result:
                    log_json(
                        "WARN",
                        "skill_returned_error",
                        details={"skill": name, "error": result["error"], "is_skill_fault": True},
                        corr_id=corr_id,
                        phase="skills",
                        component="dispatcher",
                        skill=name,
                        outcome="fail",
                        failure_reason=str(result["error"]),
                    )
                results[name] = result
            except Exception as exc:
                log_json(
                    "WARN",
                    "skill_dispatch_error",
                    details={"skill": name, "error": str(exc), "is_skill_fault": True},
                    corr_id=corr_id,
                    phase="skills",
                    component="dispatcher",
                    skill=name,
                    outcome="fail",
                    failure_reason=str(exc),
                )
                results[name] = {"error": str(exc), "skill": name,
                                 "is_skill_fault": True}

        for fut in not_done:
            name = futures[fut]
            SKILL_METRICS.record(name, timeout * 1000, error=True)
            log_json(
                "WARN",
                "skill_dispatch_timeout",
                details={"skill": name, "timeout_s": timeout, "is_skill_fault": True},
                corr_id=corr_id,
                phase="skills",
                component="dispatcher",
                skill=name,
                outcome="timeout",
                failure_reason="timeout",
                latency_ms=timeout * 1000,
            )
            fut.cancel()
            results[name] = {"error": "timeout", "skill": name,
                             "is_skill_fault": True}

    log_json(
        "INFO", "skill_dispatch_done",
        details={
            "completed": [k for k, v in results.items() if "error" not in v],
            "failed":    [k for k, v in results.items() if "error" in v],
        },
        corr_id=corr_id,
        phase="skills",
        component="dispatcher",
    )
    return results
