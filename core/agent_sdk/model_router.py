# core/agent_sdk/model_router.py
"""Adaptive model tier selection with EMA-based learning.

Selects the cheapest model tier that can reliably handle each goal type,
learning from historical outcomes. Persists stats to JSON file.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Tier ordering: cheapest first
MODEL_TIERS: List[Dict[str, str]] = [
    {"tier": "fast", "model": "claude-haiku-4-5"},
    {"tier": "standard", "model": "claude-sonnet-4-6"},
    {"tier": "powerful", "model": "claude-opus-4-6"},
]

MODEL_TO_TIER: Dict[str, str] = {t["model"]: t["tier"] for t in MODEL_TIERS}
TIER_TO_MODEL: Dict[str, str] = {t["tier"]: t["model"] for t in MODEL_TIERS}
TIER_ORDER: List[str] = [t["tier"] for t in MODEL_TIERS]

_EMPTY_TIER_STATS = {
    "attempts": 0,
    "successes": 0,
    "consecutive_failures": 0,
    "consecutive_successes": 0,
    "ema_score": 0.5,
}


class AdaptiveModelRouter:
    """Select model tier based on historical goal-type performance."""

    def __init__(
        self,
        stats_path: Path,
        ema_alpha: float = 0.2,
        min_success_rate: float = 0.7,
        escalation_threshold: int = 2,
        de_escalation_threshold: int = 5,
    ) -> None:
        self._stats_path = stats_path
        self._alpha = ema_alpha
        self._min_rate = min_success_rate
        self._esc_threshold = escalation_threshold
        self._deesc_threshold = de_escalation_threshold
        self._stats: Dict[str, Dict[str, Dict[str, Any]]] = self._load()

    def _load(self) -> Dict:
        """Load stats from file. Returns empty dict on missing/corrupt file."""
        if not self._stats_path.exists():
            return {}
        try:
            return json.loads(self._stats_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt model stats at %s: %s", self._stats_path, exc)
            return {}

    def _save(self) -> None:
        """Atomically persist stats to file."""
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self._stats_path.parent), suffix=".tmp")
            with open(fd, "w") as f:
                json.dump(self._stats, f, indent=2)
            Path(tmp).replace(self._stats_path)
        except OSError as exc:
            logger.warning("Failed to save model stats: %s", exc)
            if tmp and Path(tmp).exists():
                Path(tmp).unlink(missing_ok=True)

    def _get_tier_stats(self, goal_type: str, tier: str) -> Dict[str, Any]:
        """Get stats for a goal_type + tier, creating defaults if absent."""
        return self._stats.get(goal_type, {}).get(tier, dict(_EMPTY_TIER_STATS))

    def select_model(self, goal_type: str) -> str:
        """Select the cheapest viable model tier for a goal type.

        Checks tiers cheapest-first. A tier qualifies if:
        - EMA score >= min_success_rate
        - consecutive_failures < escalation_threshold

        De-escalation: if a higher tier has consecutive_successes >= threshold,
        we also check the tier below it (already covered by cheapest-first order,
        but de-escalation resets the consecutive_failures counter for the lower
        tier to give it another chance).
        """
        for i, entry in enumerate(MODEL_TIERS):
            tier = entry["tier"]
            stats = self._get_tier_stats(goal_type, tier)
            ema = stats.get("ema_score", 0.5)
            consec_fail = stats.get("consecutive_failures", 0)

            # Check de-escalation: if a HIGHER tier has enough successes,
            # reset this tier's consecutive failure count to give it a chance
            if i < len(MODEL_TIERS) - 1:
                higher_tier = MODEL_TIERS[i + 1]["tier"]
                higher_stats = self._get_tier_stats(goal_type, higher_tier)
                if higher_stats.get("consecutive_successes", 0) >= self._deesc_threshold:
                    consec_fail = 0  # give lower tier another shot

            attempts = stats.get("attempts", 0)
            if attempts == 0:
                # No data for this tier yet. Only start here if it's standard or
                # higher — "fast" must earn its place with a proven track record.
                if tier != "fast":
                    return entry["model"]
                # fast tier with no data: skip to standard
                continue
            if ema >= self._min_rate and consec_fail < self._esc_threshold:
                return entry["model"]
        # Fallback: most powerful
        return TIER_TO_MODEL["powerful"]

    def record_outcome(self, goal_type: str, model: str, success: bool) -> None:
        """Record a goal outcome and update EMA + counters."""
        tier = MODEL_TO_TIER.get(model, "standard")
        if goal_type not in self._stats:
            self._stats[goal_type] = {}
        if tier not in self._stats[goal_type]:
            self._stats[goal_type][tier] = dict(_EMPTY_TIER_STATS)

        s = self._stats[goal_type][tier]
        s["attempts"] += 1
        outcome = 1.0 if success else 0.0
        s["ema_score"] = self._alpha * outcome + (1 - self._alpha) * s["ema_score"]

        if success:
            s["successes"] += 1
            s["consecutive_successes"] += 1
            s["consecutive_failures"] = 0
        else:
            s["consecutive_failures"] += 1
            s["consecutive_successes"] = 0

        self._save()

    def escalate(self, goal_type: str, current_model: str) -> str:
        """Return the next tier up from current_model."""
        current_tier = MODEL_TO_TIER.get(current_model, "standard")
        idx = TIER_ORDER.index(current_tier) if current_tier in TIER_ORDER else 1
        next_idx = min(idx + 1, len(TIER_ORDER) - 1)
        return TIER_TO_MODEL[TIER_ORDER[next_idx]]

    def get_stats(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return the full stats dict for display."""
        return dict(self._stats)
