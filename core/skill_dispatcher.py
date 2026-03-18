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
import copy
import hashlib
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json

# Module-level cache for LLM-based goal classification results
_classify_goal_cache: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Goal-type → skill selection map
# Skills are listed in priority order; only those present in the registry run.
# ---------------------------------------------------------------------------
SKILL_MAP: Dict[str, list[str]] = {
    "bug_fix":  [
        "symbol_indexer",
        "test_and_observe",     # run failing tests, capture diagnostics
        "error_pattern_matcher",
        "git_history_analyzer",
        "type_checker",
        "linter_enforcer",
        "lint",                 # catch pre-existing violations before planning
    ],
    "feature":  [
        "symbol_indexer",
        "architecture_validator",
        "api_contract_validator",
        "complexity_scorer",
        "dependency_analyzer",
        "lint",                 # baseline quality gate before adding code
    ],
    "refactor": [
        "symbol_indexer",
        "complexity_scorer",
        "code_clone_detector",
        "tech_debt_quantifier",
        "refactoring_advisor",
        "lint",                 # verify cleanliness after restructuring
        "test_and_observe",     # confirm tests still pass post-refactor
    ],
    "security": [
        "security_scanner",
        "dependency_analyzer",
        "type_checker",
        "linter_enforcer",
        "architecture_validator",
        "lint",                 # surface style/hygiene issues in security context
    ],
    "docs": [
        "doc_generator",
        "symbol_indexer",
    ],
    "default": [
        "symbol_indexer",
        "linter_enforcer",
        "lint",                 # always run a baseline lint pass
    ],
}

# Keyword hints used to classify a goal string
_GOAL_TYPE_HINTS: Dict[str, list[str]] = {
    "bug_fix":  ["fix", "bug", "error", "crash", "broken", "fail", "issue",
                 "regression", "exception", "traceback", "panic",
                 "observe", "diagnose"],
    "feature":  ["add", "implement", "feature", "new", "create", "build",
                 "support", "integrate", "introduce"],
    "refactor": ["refactor", "clean", "improve", "simplify", "restructure",
                 "reorganize", "extract", "dedup", "consolidate"],
    "security": ["security", "vulnerability", "auth", "permission", "sanitize",
                 "injection", "cve", "exploit", "xss", "sqli"],
    "docs":     ["doc", "docstring", "comment", "readme", "explain",
                 "document", "annotate"],
}


class SkillMetrics:
    """Thread-safe per-skill metrics tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, float]] = {}

    def _ensure(self, skill: str) -> None:
        if skill not in self._data:
            self._data[skill] = {"call_count": 0, "total_latency_ms": 0.0, "error_count": 0}

    def record(self, skill: str, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self._ensure(skill)
            self._data[skill]["call_count"] += 1
            self._data[skill]["total_latency_ms"] += latency_ms
            if error:
                self._data[skill]["error_count"] += 1

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out = {}
            for k, v in self._data.items():
                entry = dict(v)
                # 'count' alias for backward compatibility with older consumers
                entry["count"] = entry["call_count"]
                out[k] = entry
            return out


SKILL_METRICS = SkillMetrics()

_CACHEABLE_SKILLS = {
    "architecture_validator",
    "code_clone_detector",
    "complexity_scorer",
    "refactoring_advisor",
    "security_scanner",
    "symbol_indexer",
    "tech_debt_quantifier",
    "test_coverage_analyzer",
}
_FINGERPRINT_SKIP_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".venv",
    "venv",
    "dist",
    "build",
}
_CACHE_TTL_S = 300  # 5 minutes
_CACHE_MAX_SIZE = 100
_SKILL_RESULT_CACHE: Dict[tuple[str, str, str], tuple[Dict[str, Any], float]] = {}
_SKILL_RESULT_CACHE_LOCK = threading.Lock()


def _normalize_project_root(project_root: str | Path) -> Path:
    return Path(project_root).resolve()


def _project_python_fingerprint(project_root: str | Path) -> str:
    root = _normalize_project_root(project_root)
    digest = hashlib.sha256()
    file_count = 0

    for py_file in sorted(root.rglob("*.py")):
        if any(part in _FINGERPRINT_SKIP_PARTS for part in py_file.parts):
            continue
        try:
            stat = py_file.stat()
        except OSError:
            continue
        digest.update(str(py_file.relative_to(root)).encode("utf-8", errors="replace"))
        digest.update(str(stat.st_mtime_ns).encode("ascii", errors="ignore"))
        digest.update(str(stat.st_size).encode("ascii", errors="ignore"))
        file_count += 1

    digest.update(f"count:{file_count}".encode("ascii"))
    return digest.hexdigest()


def _cache_key(skill_name: str, project_root: str | Path, fingerprint: str) -> tuple[str, str, str]:
    return (skill_name, str(_normalize_project_root(project_root)), fingerprint)


def _get_cached_skill_result(skill_name: str, project_root: str | Path, fingerprint: str) -> Optional[Dict[str, Any]]:
    key = _cache_key(skill_name, project_root, fingerprint)
    now = time.monotonic()
    with _SKILL_RESULT_CACHE_LOCK:
        entry = _SKILL_RESULT_CACHE.get(key)
        if entry is None:
            return None
        result, stored_at = entry
        if now - stored_at > _CACHE_TTL_S:
            del _SKILL_RESULT_CACHE[key]
            return None
    return copy.deepcopy(result)


def _store_cached_skill_result(skill_name: str, project_root: str | Path, fingerprint: str, result: Dict[str, Any]) -> None:
    if not isinstance(result, dict) or "error" in result:
        return
    key = _cache_key(skill_name, project_root, fingerprint)
    now = time.monotonic()
    with _SKILL_RESULT_CACHE_LOCK:
        # Evict oldest entries if cache is full
        if len(_SKILL_RESULT_CACHE) >= _CACHE_MAX_SIZE and key not in _SKILL_RESULT_CACHE:
            oldest_key = min(_SKILL_RESULT_CACHE, key=lambda k: _SKILL_RESULT_CACHE[k][1])
            del _SKILL_RESULT_CACHE[oldest_key]
        _SKILL_RESULT_CACHE[key] = (copy.deepcopy(result), now)


@lru_cache(maxsize=256)
def classify_goal(goal: str) -> str:
    """Return the goal type string most appropriate for *goal*.

    Uses keyword-overlap scoring across goal types.  Falls back to
    ``"default"`` when no keywords match.

    Results are memoised via :func:`functools.lru_cache` (maxsize=256).
    The 28x speedup on repeated calls (0.39µs vs 11µs) makes this safe
    to call in tight loops.
    """
    goal_lower = goal.lower()
    scores = {
        gt: sum(1 for kw in kws if kw in goal_lower)
        for gt, kws in _GOAL_TYPE_HINTS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "default"


_VALID_GOAL_TYPES = {"bug_fix", "feature", "refactor", "security", "docs", "default"}

_CLASSIFY_PROMPT = (
    "Classify the following software development goal into exactly one of these "
    "categories: bug_fix, feature, refactor, security, docs, default.\n"
    "Reply with ONLY the category name, no punctuation, no explanation.\n\n"
    "Goal: {goal}"
)


def classify_goal_llm(goal: str, model_adapter) -> str:
    """Classify *goal* using an LLM; falls back to :func:`classify_goal` on error.

    Results are cached in ``_classify_goal_cache`` so identical goal strings
    are only sent to the model once per process lifetime.

    Args:
        goal:          The goal description string.
        model_adapter: A :class:`~core.model_adapter.ModelAdapter` instance.

    Returns:
        One of ``"bug_fix"``, ``"feature"``, ``"refactor"``, ``"security"``,
        ``"docs"``, or ``"default"``.
    """
    if goal in _classify_goal_cache:
        return _classify_goal_cache[goal]

    try:
        prompt = _CLASSIFY_PROMPT.format(goal=goal)
        response = model_adapter.respond(prompt)
        goal_type = response.strip().lower().split()[0] if response else ""
        if goal_type not in _VALID_GOAL_TYPES:
            raise ValueError(f"unexpected goal type from LLM: {goal_type!r}")
        _classify_goal_cache[goal] = goal_type
        log_json("INFO", "classify_goal_llm", details={"goal_type": goal_type})
        return goal_type
    except Exception as exc:
        log_json("WARN", "classify_goal_llm_fallback",
                 details={"error": str(exc), "fallback": "keyword"})
        fallback = classify_goal(goal)
        _classify_goal_cache[goal] = fallback
        return fallback


def dispatch_skills(
    goal_type: str,
    skills: Dict[str, Any],
    project_root: str,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Run goal-relevant skills concurrently and return their combined output.

    Args:
        goal_type:    One of the keys in :data:`SKILL_MAP`. When a caller passes
                      an unknown goal type, the dispatcher runs the provided
                      ``skills`` dict as-is in insertion order. This keeps
                      specialized callers (for example health scans) from being
                      forced through the end-user goal routing table.
        skills:       Dict of ``{skill_name: SkillBase}`` from ``all_skills()``.
        project_root: Absolute or relative path passed to each skill.
        timeout:      Wall-clock seconds to wait for all skills to finish.
                      Skills that exceed the timeout are cancelled and their
                      result is recorded as ``{"error": "timeout"}``.

    Returns:
        ``{skill_name: result_dict}`` — never raises.
    """
    skill_names = SKILL_MAP.get(goal_type)
    if skill_names is None:
        skill_names = list(skills.keys())
    available = [n for n in skill_names if n in skills]

    if not available:
        log_json("WARN", "skill_dispatch_no_skills", details={"goal_type": goal_type})
        return {}

    log_json(
        "INFO", "skill_dispatch_start",
        details={"goal_type": goal_type, "skills": available},
    )

    results: Dict[str, Any] = {}
    fingerprint: Optional[str] = None
    submitted: Dict[concurrent.futures.Future, str] = {}
    cacheable_available = [name for name in available if name in _CACHEABLE_SKILLS]
    if cacheable_available:
        fingerprint = _project_python_fingerprint(project_root)

    for name in available:
        if fingerprint is None or name not in _CACHEABLE_SKILLS:
            continue
        cached = _get_cached_skill_result(name, project_root, fingerprint)
        if cached is None:
            continue
        log_json("INFO", "skill_dispatch_cache_hit", details={"skill": name})
        results[name] = cached

    remaining = [name for name in available if name not in results]
    if not remaining:
        log_json(
            "INFO", "skill_dispatch_done",
            details={
                "completed": list(results.keys()),
                "failed": [],
            },
        )
        return results

    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(available), 5), thread_name_prefix="skill"
    )
    try:
        def _timed_run(name: str) -> Any:
            t0 = time.monotonic()
            try:
                raw = skills[name].run({"project_root": project_root})
                # Normalize SkillResult → dict for downstream consumers
                result = raw.as_dict() if hasattr(raw, "as_dict") else raw
                SKILL_METRICS.record(name, (time.monotonic() - t0) * 1000, error=False)
                if fingerprint is not None and name in _CACHEABLE_SKILLS and isinstance(result, dict):
                    _store_cached_skill_result(name, project_root, fingerprint, result)
                return result
            except Exception as exc:
                SKILL_METRICS.record(name, (time.monotonic() - t0) * 1000, error=True)
                raise exc

        submitted = {
            pool.submit(_timed_run, name): name
            for name in remaining
        }
        done, not_done = concurrent.futures.wait(submitted, timeout=timeout)

        for fut in done:
            name = submitted[fut]
            try:
                result = fut.result()
                # Distinguish a real skill error ({"error": ...}) from a
                # legitimate empty result so the orchestrator can tell them apart.
                if isinstance(result, dict) and "error" in result:
                    log_json("WARN", "skill_returned_error",
                             details={"skill": name, "error": result["error"],
                                      "is_skill_fault": True})
                results[name] = result
            except Exception as exc:
                log_json("WARN", "skill_dispatch_error",
                         details={"skill": name, "error": str(exc),
                                  "is_skill_fault": True})
                results[name] = {"error": str(exc), "skill": name,
                                 "is_skill_fault": True}

        for fut in not_done:
            name = submitted[fut]
            SKILL_METRICS.record(name, timeout * 1000, error=True)
            log_json("WARN", "skill_dispatch_timeout",
                     details={"skill": name, "timeout_s": timeout,
                              "is_skill_fault": True})
            fut.cancel()
            results[name] = {"error": "timeout", "skill": name,
                             "is_skill_fault": True}
    finally:
        # Do not block on timed-out futures. If a running skill eventually
        # completes, its result can still populate the cache for the current
        # project fingerprint from inside _timed_run().
        pool.shutdown(wait=False, cancel_futures=True)

    log_json(
        "INFO", "skill_dispatch_done",
        details={
            "completed": [k for k, v in results.items() if "error" not in v],
            "failed":    [k for k, v in results.items() if "error" in v],
        },
    )
    return results


# ---------------------------------------------------------------------------
# Skill chaining — queue follow-up goals based on skill results
# ---------------------------------------------------------------------------

class SkillChainer:
    """Chains skill results into follow-up goals queued for the next cycle."""

    def maybe_chain(self, skill_name: str, result: dict, goal_queue) -> List[str]:
        """Inspect *result* from *skill_name* and optionally queue remediation goals.

        Currently handles:
        - ``security_scanner``: queues a remediation goal when
          ``critical_count > 0``.

        Args:
            skill_name:  Name of the skill that produced *result*.
            result:      The skill's output dict.
            goal_queue:  A :class:`~core.goal_queue.GoalQueue` instance.

        Returns:
            List of goal strings that were queued (may be empty).
        """
        queued: List[str] = []
        if skill_name == "security_scanner":
            critical_count = result.get("critical_count", 0)
            if critical_count and critical_count > 0:
                goal = (
                    f"Remediate {critical_count} critical security finding(s) "
                    f"identified by the security scanner: "
                    f"{result.get('scan_summary', '')}"
                )
                try:
                    goal_queue.add(goal)
                    queued.append(goal)
                    log_json("INFO", "skill_chainer_queued",
                             details={"skill": skill_name, "goal": goal})
                except Exception as exc:
                    log_json("WARN", "skill_chainer_queue_failed",
                             details={"skill": skill_name, "error": str(exc)})
        return queued


def chain_skill_results(skill_results: dict, goal_queue) -> List[str]:
    """Run :class:`SkillChainer` over all entries in *skill_results*.

    Args:
        skill_results: ``{skill_name: result_dict}`` as returned by
                       :func:`dispatch_skills`.
        goal_queue:    A :class:`~core.goal_queue.GoalQueue` instance.

    Returns:
        Flat list of all goal strings that were queued.
    """
    chainer = SkillChainer()
    all_queued: List[str] = []
    for skill_name, result in skill_results.items():
        all_queued.extend(chainer.maybe_chain(skill_name, result, goal_queue))
    return all_queued
