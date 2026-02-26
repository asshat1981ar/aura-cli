"""Skill: track and recommend execution strategies based on past performance."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_STATS_FILE = Path("memory/strategy_stats.json")

_STRATEGY_KEYWORDS: Dict[str, List[str]] = {
    "sliding_window": ["refactor", "test", "fix", "bug", "small", "quick"],
    "time_bound": ["deadline", "fast", "urgent", "quickly", "asap"],
    "exhaustive": ["architecture", "security", "audit", "full", "complete", "deep"],
    "iterative": ["evolve", "improve", "optimize", "performance", "gradual"],
    "single_pass": ["generate", "scaffold", "create", "new", "add"],
}


def _load_stats(path: Path) -> List[Dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_stats(path: Path, stats: List[Dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats[-1000:], indent=2), encoding="utf-8")
    except Exception:
        pass


def _compute_rates(stats: List[Dict]) -> List[Dict]:
    groups: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for entry in stats:
        s = entry.get("strategy", "unknown")
        g = entry.get("goal_type", "general")
        groups[s][g].append(entry.get("success", False))
    results = []
    for strategy, goal_types in groups.items():
        for goal_type, outcomes in goal_types.items():
            cycles_list = [e.get("cycles_used", 5) for e in stats if e.get("strategy") == strategy and e.get("goal_type") == goal_type]
            success_rate = round(sum(1 for o in outcomes if o) / max(len(outcomes), 1), 2)
            avg_cycles = round(sum(cycles_list) / max(len(cycles_list), 1), 1)
            results.append({"strategy": strategy, "goal_type": goal_type, "success_rate": success_rate, "avg_cycles": avg_cycles, "sample_size": len(outcomes)})
    return results


def _classify_goal(goal: str) -> str:
    goal_lower = goal.lower()
    keyword_hits: Dict[str, int] = defaultdict(int)
    for cat, kws in _STRATEGY_KEYWORDS.items():
        for kw in kws:
            if kw in goal_lower:
                keyword_hits[cat] += 1
    if not keyword_hits:
        return "general"
    return max(keyword_hits, key=keyword_hits.__getitem__)


class AdaptiveStrategySelectorSkill(SkillBase):
    name = "adaptive_strategy_selector"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        goal: str = input_data.get("goal", "")
        available: Optional[List[str]] = input_data.get("available_strategies")
        record: Optional[Dict] = input_data.get("record_result")

        stats = _load_stats(_STATS_FILE)

        # Record new result if provided
        if record:
            entry = {
                "strategy": record.get("strategy", "unknown"),
                "goal_type": _classify_goal(goal),
                "success": bool(record.get("success", False)),
                "cycles_used": int(record.get("cycles", 5)),
                "stop_reason": record.get("stop_reason", ""),
            }
            stats.append(entry)
            _save_stats(_STATS_FILE, stats)
            log_json("INFO", "adaptive_strategy_recorded", details=entry)

        rates = _compute_rates(stats)
        goal_type = _classify_goal(goal)

        # Find best strategy for this goal type
        candidates = [r for r in rates if r["goal_type"] == goal_type and r["sample_size"] >= 2]
        if not candidates:
            candidates = [r for r in rates]  # fall back to global stats

        if available:
            candidates = [c for c in candidates if c["strategy"] in available]

        # Score: success_rate * (1 / avg_cycles normalized)
        def _score(r: Dict) -> float:
            return r["success_rate"] - (r["avg_cycles"] / 10)

        best = max(candidates, key=_score) if candidates else None

        if not best:
            # Default recommendation by keyword matching
            default = _classify_goal(goal)
            best_name = default if default in (available or list(_STRATEGY_KEYWORDS)) else "sliding_window"
            reasoning = f"No historical data yet. Recommending '{best_name}' based on goal keywords."
            confidence = 0.4
        else:
            best_name = best["strategy"]
            reasoning = f"Strategy '{best_name}' had {best['success_rate']*100:.0f}% success on '{best['goal_type']}' goals (n={best['sample_size']}, avg {best['avg_cycles']} cycles)."
            confidence = min(0.95, best["success_rate"] * (min(best["sample_size"], 10) / 10))

        log_json("INFO", "adaptive_strategy_selector_complete", details={"recommended": best_name, "confidence": confidence})
        return {"recommended_strategy": best_name, "confidence": round(confidence, 2), "stats": rates[:20], "reasoning": reasoning, "goal_type": goal_type}
