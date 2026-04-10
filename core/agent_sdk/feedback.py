# core/agent_sdk/feedback.py
"""Feedback collector and skill weight updater.

Dispatches goal outcomes to three systems: model router (EMA updates),
skill weights (JSON file), and brain memory (semantic storage).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHT = 0.5


class SkillWeightUpdater:
    """Adjust skill weights based on goal outcomes."""

    def __init__(
        self,
        weights_path: Path,
        success_delta: float = 0.1,
        failure_delta: float = -0.05,
        cap: float = 1.0,
        floor: float = 0.1,
    ) -> None:
        self._path = weights_path
        self._success_delta = success_delta
        self._failure_delta = failure_delta
        self._cap = cap
        self._floor = floor
        self._weights = self._load()

    def _load(self) -> Dict[str, float]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
            with open(fd, "w") as f:
                json.dump(self._weights, f, indent=2)
            Path(tmp).replace(self._path)
        except OSError as exc:
            logger.warning("Failed to save skill weights: %s", exc)
            if tmp and Path(tmp).exists():
                Path(tmp).unlink(missing_ok=True)

    def update(self, skills_used: List[str], success: bool) -> None:
        """Adjust weights for used skills based on outcome."""
        delta = self._success_delta if success else self._failure_delta
        for skill in skills_used:
            current = self._weights.get(skill, _DEFAULT_WEIGHT)
            new_weight = max(self._floor, min(self._cap, current + delta))
            self._weights[skill] = round(new_weight, 4)
        self._save()

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)


class FeedbackCollector:
    """Dispatch goal outcomes to model router, skill weights, and brain."""

    def __init__(
        self,
        model_router: Any,
        skill_updater: SkillWeightUpdater,
        brain: Any = None,
        session_store: Any = None,
    ) -> None:
        self.model_router = model_router
        self.skill_updater = skill_updater
        self._brain = brain
        self._session_store = session_store

    def on_goal_complete(
        self,
        session_pk: int,
        goal: str,
        goal_type: str,
        model: str,
        skills_used: List[str],
        success: bool,
        verification_result: Dict[str, Any],
        cost: float,
    ) -> Dict[str, Any]:
        """Dispatch outcome to all three feedback systems."""
        # 1. Model router
        self.model_router.record_outcome(goal_type, model, success)

        # 2. Skill weights
        self.skill_updater.update(skills_used, success)

        # 3. Brain memory
        if self._brain is not None:
            try:
                self._brain.remember(
                    {
                        "type": "goal_outcome",
                        "goal": goal,
                        "goal_type": goal_type,
                        "model": model,
                        "success": success,
                        "cost_usd": cost,
                        "verification": verification_result,
                    }
                )
            except (OSError, RuntimeError, TypeError) as exc:
                logger.warning("Failed to store outcome in brain: %s", exc)

        return {
            "model_updated": True,
            "skills_updated": skills_used,
            "brain_stored": self._brain is not None,
        }

    def get_failure_patterns(self, goal_type: str, limit: int = 3) -> List[str]:
        """Extract error patterns from recent failed sessions."""
        if self._session_store is None:
            return []

        try:
            recent = self._session_store.list_sessions(status="failed", limit=10)
        except (OSError, RuntimeError, KeyError):
            return []

        # Filter to this goal_type if possible
        failures = [s for s in recent if s.get("goal_type") == goal_type and s.get("error_summary")]
        if not failures:
            failures = [s for s in recent if s.get("error_summary")]

        if len(failures) < limit:
            return []

        return [f["error_summary"] for f in failures[:limit]]
