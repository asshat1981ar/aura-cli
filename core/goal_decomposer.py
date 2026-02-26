"""
Goal Decomposer.

Detects goals that are too large or ambiguous (via complexity heuristics),
breaks them into 3–5 ordered sub-goals using the LLM, and enqueues the
sub-goals with a ``parent_id`` tag so their completion can be tracked.

Goals prefixed with ``[DECOMPOSE]`` (placed there by ConvergenceEscapeLoop)
are always decomposed unconditionally.

Usage::

    from core.goal_decomposer import GoalDecomposer
    decomposer = GoalDecomposer(model)
    result = decomposer.maybe_decompose(goal, goal_queue, skills, project_root=".")
    # Returns {"decomposed": True, "sub_goals": [...]} or {"decomposed": False}
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json
from core.file_tools import _aura_safe_loads

# Heuristics — any of these in the goal text suggest decomposition is needed
_SCOPE_KEYWORDS = [
    "entire", "all", "whole", "every", "throughout", "across", "complete",
    "full", "refactor all", "migrate all", "rewrite",
]

# Goal character length above which we also consider decomposition
LENGTH_THRESHOLD: int = 180

# Estimated file-count above which the goal is considered "large"
FILE_COUNT_THRESHOLD: int = 8


class GoalDecomposer:
    """Detect and decompose large or ambiguous goals into sub-goals."""

    def __init__(self, model):
        self.model = model

    # ── Public API ───────────────────────────────────────────────────────────

    def maybe_decompose(
        self,
        goal: str,
        goal_queue,
        skills: Optional[Dict[str, Any]] = None,
        project_root: str = ".",
    ) -> Dict[str, Any]:
        """Conditionally decompose *goal*.

        Returns ``{"decomposed": True, "sub_goals": [...], "parent_id": str}``
        if decomposition happened, else ``{"decomposed": False}``.
        """
        try:
            return self._maybe_decompose(goal, goal_queue, skills, project_root)
        except Exception as exc:
            log_json("ERROR", "goal_decomposer_failed", details={"error": str(exc)})
            return {"decomposed": False, "error": str(exc)}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _maybe_decompose(
        self, goal: str, goal_queue, skills, project_root: str
    ) -> Dict[str, Any]:
        forced = goal.startswith("[DECOMPOSE]")
        clean_goal = goal.removeprefix("[DECOMPOSE]").strip()

        if not forced and not self._should_decompose(clean_goal, skills, project_root):
            return {"decomposed": False}

        log_json("INFO", "goal_decomposer_decomposing",
                 details={"goal": clean_goal[:80], "forced": forced})

        sub_goals = self._generate_sub_goals(clean_goal)
        if not sub_goals or len(sub_goals) < 2:
            log_json("WARN", "goal_decomposer_insufficient_sub_goals",
                     details={"count": len(sub_goals)})
            return {"decomposed": False, "reason": "llm_returned_too_few_sub_goals"}

        parent_id = uuid.uuid4().hex[:10]
        for i, sg in enumerate(sub_goals):
            tagged = f"[SUBTASK:{parent_id}:{i+1}/{len(sub_goals)}] {sg}"
            goal_queue.add(tagged)

        log_json("INFO", "goal_decomposer_done",
                 details={"parent_id": parent_id, "sub_goals": len(sub_goals)})
        return {
            "decomposed": True,
            "parent_id": parent_id,
            "sub_goals": sub_goals,
            "sub_goals_queued": len(sub_goals),
        }

    def _should_decompose(
        self, goal: str, skills: Optional[Dict], project_root: str
    ) -> bool:
        """Return True if this goal looks large enough to warrant decomposition."""
        goal_lower = goal.lower()

        # Keyword heuristics
        if any(kw in goal_lower for kw in _SCOPE_KEYWORDS):
            return True

        # Length heuristic
        if len(goal) > LENGTH_THRESHOLD:
            return True

        # File-count heuristic via symbol_indexer
        if skills and "symbol_indexer" in skills:
            try:
                result = skills["symbol_indexer"].run({"project_root": project_root})
                file_count = result.get("files", 0)
                # Heuristic: goal mentions >8 distinct file stems
                goal_words = set(goal_lower.split())
                # crude: count how many result symbols appear in goal text
                if file_count > FILE_COUNT_THRESHOLD * 10:
                    return True
            except Exception:
                pass

        return False

    def _generate_sub_goals(self, goal: str) -> List[str]:
        """Ask the LLM to decompose *goal* into 3–5 ordered sub-goals."""
        prompt = f"""You are a software planning agent.

Break the following high-level goal into 3 to 5 specific, independently
executable sub-goals.  Each sub-goal should be concrete, testable, and
small enough to be implemented in a single coding pass.

Goal:
{goal}

Return ONLY a JSON array of strings, e.g.:
["Sub-goal 1 description", "Sub-goal 2 description", ...]

No explanatory text outside the JSON.
"""
        try:
            raw = self.model.respond(prompt)
            parsed = _aura_safe_loads(raw, "goal_decomposer")
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                return [s.strip() for s in parsed if s.strip()]
        except Exception as exc:
            log_json("WARN", "goal_decomposer_llm_parse_failed",
                     details={"error": str(exc)})
        return []
