"""
Skill Weight Adapter (EMA-based).

After each cycle, scores each skill's output for the current goal_type using
an Exponential Moving Average (EMA).  Low-signal skills are deprioritised in
``SKILL_MAP``; high-signal skills are promoted to the front.

Weights are persisted to ``memory/skill_weights.json`` and loaded at startup
so the system improves across sessions.

Usage::

    from core.skill_weight_adapter import SkillWeightAdapter
    adapter = SkillWeightAdapter(memory_root="memory")
    adapter.on_cycle_complete(cycle_entry)   # call after every run_cycle()
    ordered = adapter.ranked_skills("bug_fix")
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from core.logging_utils import log_json
from core.skill_dispatcher import SKILL_MAP

# EMA smoothing factor.  Closer to 1 = faster adaptation; closer to 0 = more stable.
EMA_ALPHA: float = 0.15

# Skills whose EMA drops below this after MIN_RUNS are suspended (removed from dispatch).
SUSPEND_THRESHOLD: float = 0.12
MIN_RUNS_BEFORE_SUSPEND: int = 5

# Maximum skills returned per goal_type (avoids dispatching too many)
MAX_SKILLS_PER_TYPE: int = 5


class SkillWeightAdapter:
    """Track per-skill EMA signal scores and reorder SKILL_MAP accordingly.

    State file: ``<memory_root>/skill_weights.json``
    Schema::

        {
          "goal_type": {
            "skill_name": {"ema": 0.72, "runs": 14, "suspended": false}
          }
        }

    Args:
        memory_root: Directory for the weights JSON file (default ``"memory"``).
        momento:     Optional :class:`MomentoAdapter` for L1 hot cache reads.
    """

    def __init__(self, memory_root: str = "memory", momento=None):
        self._path = Path(memory_root) / "skill_weights.json"
        self._momento = momento
        self._weights: Dict[str, Dict[str, Dict]] = self._load()

    # ── Public API ───────────────────────────────────────────────────────────

    def on_cycle_complete(self, cycle_entry: Dict) -> None:
        """Update EMA scores from a completed cycle entry.  Never raises."""
        try:
            self._update(cycle_entry)
        except Exception as exc:
            log_json("ERROR", "skill_weight_adapter_failed", details={"error": str(exc)})

    def ranked_skills(self, goal_type: str) -> List[str]:
        """Return skills for *goal_type* ordered by EMA score (best first).

        Suspended skills are excluded.  Falls back to the static SKILL_MAP
        if no weight data exists yet.
        """
        base = list(SKILL_MAP.get(goal_type, SKILL_MAP["default"]))
        gt_weights = self._weights.get(goal_type, {})
        if not gt_weights:
            return base[:MAX_SKILLS_PER_TYPE]

        def _key(skill: str) -> float:
            info = gt_weights.get(skill, {})
            if info.get("suspended"):
                return -1.0
            return info.get("ema", 0.5)  # default 0.5 for untracked skills

        active = [s for s in base if not gt_weights.get(s, {}).get("suspended")]
        ranked = sorted(active, key=_key, reverse=True)
        return ranked[:MAX_SKILLS_PER_TYPE]

    def get_weights_summary(self) -> Dict:
        """Return a snapshot of all EMA weights (useful for reflection reports)."""
        return {
            gt: {
                sk: {"ema": round(v["ema"], 3), "runs": v["runs"],
                     "suspended": v.get("suspended", False)}
                for sk, v in skills.items()
            }
            for gt, skills in self._weights.items()
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update(self, cycle_entry: Dict) -> None:
        goal_type = cycle_entry.get("goal_type", "default")
        po = cycle_entry.get("phase_outputs", {})
        skill_context = po.get("skill_context", {})
        if not skill_context:
            return

        gt_weights = self._weights.setdefault(goal_type, {})
        updated: List[str] = []

        for skill_name, result in skill_context.items():
            signal = self._measure_signal(result)
            info = gt_weights.setdefault(skill_name, {"ema": 0.5, "runs": 0, "suspended": False})

            # EMA update
            old_ema = info["ema"]
            info["ema"] = EMA_ALPHA * signal + (1 - EMA_ALPHA) * old_ema
            info["runs"] += 1

            # Suspension check
            if (
                info["runs"] >= MIN_RUNS_BEFORE_SUSPEND
                and info["ema"] < SUSPEND_THRESHOLD
                and not info["suspended"]
            ):
                info["suspended"] = True
                log_json("INFO", "skill_weight_adapter_suspended",
                         details={"skill": skill_name, "goal_type": goal_type,
                                  "ema": round(info["ema"], 3), "runs": info["runs"]})

            # Un-suspend if signal improves significantly
            elif info.get("suspended") and info["ema"] > SUSPEND_THRESHOLD * 2:
                info["suspended"] = False
                log_json("INFO", "skill_weight_adapter_unsuspended",
                         details={"skill": skill_name, "goal_type": goal_type,
                                  "ema": round(info["ema"], 3)})

            updated.append(skill_name)

        self._save()
        log_json("INFO", "skill_weight_adapter_updated",
                 details={"goal_type": goal_type, "skills_updated": updated})

    def _measure_signal(self, result: Dict) -> float:
        """Score a skill result 0.0–1.0 based on how much useful content it has."""
        if not isinstance(result, dict) or "error" in result:
            return 0.0

        # Count non-empty, non-trivial values
        useful = sum(
            1 for k, v in result.items()
            if k != "error" and v not in (None, [], {}, "", 0, False)
        )
        total = max(len(result), 1)
        base_score = useful / total

        # Bonus for skills that explicitly flag issues (high-signal)
        bonus_keys = [
            "findings", "violations", "type_errors", "circular_deps",
            "debt_items", "suggestions", "high_risk_count", "critical_count",
        ]
        has_findings = any(
            result.get(k) not in (None, [], {}, 0)
            for k in bonus_keys
        )
        if has_findings:
            base_score = min(1.0, base_score + 0.25)

        return round(base_score, 3)

    def _load(self) -> Dict:
        # L1: Momento — check hot cache first
        if self._momento and self._momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE
                raw = self._momento.cache_get(WORKING_MEMORY_CACHE, "skill_weights:all")
                if raw:
                    log_json("INFO", "skill_weights_l1_hit")
                    return json.loads(raw)
            except Exception:
                pass
        # L2: JSON file
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        serialized = json.dumps(self._weights, indent=2)
        # L1: Momento write-through (no TTL — weights are persistent)
        if self._momento and self._momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE
                self._momento.cache_set(
                    WORKING_MEMORY_CACHE, "skill_weights:all",
                    serialized, ttl_seconds=0,
                )
            except Exception as exc:
                log_json("WARN", "skill_weights_l1_save_failed", details={"error": str(exc)})
        # L2: JSON file
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(serialized, encoding="utf-8")
        except Exception as exc:
            log_json("WARN", "skill_weight_save_failed", details={"error": str(exc)})
