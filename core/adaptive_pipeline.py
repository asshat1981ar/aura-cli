"""
Adaptive Pipeline — Dynamic phase selection and configuration.

Replaces the static 7-phase list with a context-aware configuration that:
- Selects which phases to run based on goal_type
- Reorders phases when context signals suggest it
- Injects skill-derived context into phase inputs
- Skips expensive phases when signals indicate they won't help
- Intensifies phases when past failure patterns demand it

The resulting ``PipelineConfig`` is consumed by ``LoopOrchestrator.run_cycle()``.

Adaptation signals (in priority order)
----------------------------------------
1. ContextGraph — what worked/failed for this goal_type before
2. SkillWeightAdapter — which skills are high-signal for this goal_type
3. ReflectionReports — which phases have high failure rates
4. ConsecutiveFailures — escalate context depth
5. GoalType — base phase selection

Usage::

    from core.adaptive_pipeline import AdaptivePipeline
    pipeline = AdaptivePipeline(context_graph, skill_weight_adapter, memory_store)
    config = pipeline.configure(goal, goal_type, consecutive_fails=0)
    # config.phases    — ordered list of phase names to run
    # config.skill_set — skills to dispatch for enrichment
    # config.hints     — extra context injected into plan phase
    # config.intensity — "normal" | "deep" | "minimal"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json
from core.skill_dispatcher import SKILL_MAP

# Base phase sequence for each goal type
_BASE_PHASES: Dict[str, List[str]] = {
    "bug_fix":  ["ingest", "plan", "critique", "synthesize", "act", "verify", "reflect"],
    "feature":  ["ingest", "plan", "critique", "synthesize", "act", "verify", "reflect"],
    "refactor": ["ingest", "plan", "synthesize", "act", "verify", "reflect"],
    "security": ["ingest", "plan", "critique", "synthesize", "act", "verify", "reflect"],
    "docs":     ["ingest", "plan", "synthesize", "act", "reflect"],
    "default":  ["ingest", "plan", "synthesize", "act", "verify", "reflect"],
}

# Phases that can be skipped when intensity is "minimal"
_SKIPPABLE_PHASES = {"critique"}

# Phases that get injected when intensity is "deep"
_DEEP_EXTRA_HINTS = {
    "bug_fix":  ["Use symbol_indexer results to trace call graph before patching",
                 "Check git_history_analyzer for hotspot context"],
    "refactor": ["Check code_clone_detector for duplicate patterns first",
                 "Validate architecture coupling after every file change"],
    "security": ["Treat every user-controlled input as untrusted",
                 "Cross-reference dependency_analyzer for transitive vulns"],
}


@dataclass
class PipelineConfig:
    """Resolved pipeline configuration for one run_cycle() call.

    Attributes:
        phases:        Ordered list of phases to execute.
        skill_set:     Skills to dispatch during triage (by name).
        hints:         Extra context strings injected into the plan phase.
        intensity:     ``"minimal"`` | ``"normal"`` | ``"deep"``.
        skip_verify:   True if verification should be skipped (e.g. dry-run docs goals).
        max_act_attempts: How many times the act loop may retry.
        plan_retries:  How many times to re-plan on structural failure.
        extra_plan_ctx: Free-form dict merged into the plan phase input.
    """
    phases: List[str]
    skill_set: List[str]
    hints: List[str] = field(default_factory=list)
    intensity: str = "normal"
    skip_verify: bool = False
    max_act_attempts: int = 3
    plan_retries: int = 2
    extra_plan_ctx: Dict[str, Any] = field(default_factory=dict)


class AdaptivePipeline:
    """Produce a ``PipelineConfig`` tuned to the current cycle's context."""

    def __init__(
        self,
        context_graph=None,
        skill_weight_adapter=None,
        memory_store=None,
        brain=None,
    ):
        self.graph = context_graph
        self.weights = skill_weight_adapter
        self.memory = memory_store
        self._brain = brain

    # ── Public API ───────────────────────────────────────────────────────────

    def configure(
        self,
        goal: str,
        goal_type: str,
        consecutive_fails: int = 0,
        past_failures: Optional[List[str]] = None,
    ) -> PipelineConfig:
        """Build a ``PipelineConfig`` for the next cycle.  Never raises."""
        try:
            return self._configure(goal, goal_type, consecutive_fails, past_failures or [])
        except Exception as exc:
            log_json("WARN", "adaptive_pipeline_fallback", details={"error": str(exc)})
            return self._default_config(goal_type)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _configure(
        self,
        goal: str,
        goal_type: str,
        consecutive_fails: int,
        past_failures: List[str],
    ) -> PipelineConfig:
        # ── 1. Intensity decision ────────────────────────────────────────────
        intensity, confidence = self._choose_intensity_with_confidence(goal_type, consecutive_fails, past_failures)

        # ── 2. Phase selection ───────────────────────────────────────────────
        base = list(_BASE_PHASES.get(goal_type, _BASE_PHASES["default"]))
        if intensity == "minimal":
            base = [p for p in base if p not in _SKIPPABLE_PHASES]
        elif intensity == "deep":
            # For deep runs, insert an extra critique pass before act
            if "act" in base and "critique" not in base:
                base.insert(base.index("act"), "critique")

        # ── 3. Skip verify for doc-only goals ───────────────────────────────
        skip_verify = goal_type == "docs"

        # ── 4. Skill set selection ───────────────────────────────────────────
        if self.weights:
            skills = self.weights.ranked_skills(goal_type)
        else:
            skills = list(SKILL_MAP.get(goal_type, SKILL_MAP["default"]))

        # Boost skill set on deep runs
        if intensity == "deep":
            extra = [s for s in SKILL_MAP.get(goal_type, [])
                     if s not in skills][:2]
            skills = skills + extra

        # ── 5. Hints from context graph ──────────────────────────────────────
        hints: List[str] = list(_DEEP_EXTRA_HINTS.get(goal_type, []))
        extra_ctx: Dict[str, Any] = {}

        if self.graph:
            # Inject similar past resolutions as planner hints
            similar = self.graph.query_similar_resolutions(goal[:80])
            if similar:
                hints.append(
                    "Similar past resolutions: "
                    + "; ".join(r["goal"][:60] for r in similar[:3])
                )
            # Inject best skills info
            best_skills = self.graph.best_skills_for_goal_type(goal_type, limit=3)
            if best_skills:
                hints.append(f"High-value skills for this goal type: {', '.join(best_skills)}")
            # Inject weakness context
            weaknesses = self.graph.weaknesses_for_goal_type(goal_type, limit=3)
            if weaknesses:
                extra_ctx["known_weaknesses_for_type"] = weaknesses

        # ── 6. Retry counts scaled by intensity ─────────────────────────────
        act_attempts = {"minimal": 2, "normal": 3, "deep": 5}.get(intensity, 3)
        plan_retries = {"minimal": 1, "normal": 2, "deep": 3}.get(intensity, 2)

        # ── 7. Failure-specific adjustments ──────────────────────────────────
        if "old_code_not_found" in " ".join(past_failures):
            extra_ctx["prefer_overwrite"] = True
            hints.append("Prefer overwrite_file=True for files that previously had old_code_not_found errors")
        if "syntax" in " ".join(past_failures).lower():
            extra_ctx["model_preference"] = "quality"
            hints.append("Previous cycle had syntax errors — use high-quality model")

        config = PipelineConfig(
            phases=base,
            skill_set=skills,
            hints=hints,
            intensity=intensity,
            confidence=confidence,
            skip_verify=skip_verify,
            max_act_attempts=act_attempts,
            plan_retries=plan_retries,
            extra_plan_ctx=extra_ctx,
        )
        log_json("INFO", "adaptive_pipeline_configured", details={
            "goal_type": goal_type,
            "intensity": intensity,
            "confidence": f"{confidence:.2f}",
            "phases": base,
            "skills": skills[:4],
            "hints_count": len(hints),
        })
        return config

    def _choose_intensity_with_confidence(self, goal_type: str, consecutive_fails: int, past_failures: List[str]) -> Tuple[str, float]:
        """Map failure state and historical data to intensity level."""
        # Baseline heuristics
        baseline = "normal"
        if consecutive_fails >= 3:
            baseline = "deep"
        elif consecutive_fails == 0 and not past_failures:
            # Check if we've had recent phase failure spikes
            if self.memory:
                reports = self.memory.query("reflection_reports", limit=1)
                if reports:
                    high_count = sum(
                        1 for ins in reports[-1].get("insights", [])
                        if ins.get("severity") == "HIGH"
                    )
                    if high_count >= 2:
                        baseline = "deep"
            
        # Data-driven override
        rates = {
            "minimal": self.win_rate(goal_type, "minimal"),
            "normal":  self.win_rate(goal_type, "normal"),
            "deep":    self.win_rate(goal_type, "deep"),
        }
        
        # If we have enough data (e.g. non-zero rates), prefer the winner
        if any(r > 0 for r in rates.values()):
            best_strategy = max(rates, key=rates.get)
            # If the winner is better than baseline, or baseline is failing, switch
            if rates[best_strategy] > rates.get(baseline, 0) or rates.get(baseline, 0) < 0.3:
                log_json("INFO", "adaptive_strategy_override", details={"baseline": baseline, "override": best_strategy, "win_rate": rates[best_strategy]})
                return best_strategy, rates[best_strategy]
        
        return baseline, rates.get(baseline, 1.0 if consecutive_fails == 0 else 0.5)

    def _choose_intensity(self, consecutive_fails: int, past_failures: List[str]) -> str:
        """Legacy method for internal calls."""
        return self._choose_intensity_with_confidence("default", consecutive_fails, past_failures)[0]

    def _default_config(self, goal_type: str) -> PipelineConfig:
        return PipelineConfig(
            phases=list(_BASE_PHASES.get(goal_type, _BASE_PHASES["default"])),
            skill_set=list(SKILL_MAP.get(goal_type, SKILL_MAP["default"])),
        )

    def record_outcome(self, goal_type: str, strategy: str, success: bool) -> None:
        """Persist win/loss for (goal_type, strategy) to Brain if brain is available."""
        if not hasattr(self, '_brain') or self._brain is None:
            return
        key = f"__strategy_stats__:{goal_type}:{strategy}"
        existing = {"wins": 0, "losses": 0}
        try:
            for entry in reversed(self._brain.recall_recent(limit=100)):
                if entry.startswith(key + ":"):
                    try:
                        existing = json.loads(entry[len(key) + 1:])
                    except Exception:
                        pass
                    break
        except Exception:
            pass
        if success:
            existing["wins"] += 1
        else:
            existing["losses"] += 1
        try:
            self._brain.remember(f"{key}:{json.dumps(existing)}")
        except Exception:
            pass

    def win_rate(self, goal_type: str, strategy: str) -> float:
        """Return win rate 0.0-1.0 for this (goal_type, strategy) pair."""
        if not hasattr(self, '_brain') or self._brain is None:
            return 0.0
        key = f"__strategy_stats__:{goal_type}:{strategy}"
        try:
            for entry in reversed(self._brain.recall_recent(limit=100)):
                if entry.startswith(key + ":"):
                    try:
                        stats = json.loads(entry[len(key) + 1:])
                        wins = stats.get("wins", 0)
                        losses = stats.get("losses", 0)
                        total = wins + losses
                        return wins / total if total > 0 else 0.0
                    except Exception:
                        return 0.0
        except Exception:
            pass
        return 0.0
