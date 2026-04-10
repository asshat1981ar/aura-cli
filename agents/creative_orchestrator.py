"""Unified orchestrator combining AURA's execution with creative techniques.

Integrates the 10-phase AURA loop with creative innovation techniques
for comprehensive problem-solving: ideation → validation → implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum, auto

from core.logging_utils import log_json
from core.creative_bridge import (
    CreativeIdea,
    ImplementationResult,
    CreativeImplementationBridge,
)
from core.creative_memory import CreativePatternMemory, CreativePattern


class CreativePhase(Enum):
    """Phases of the unified creative-execution loop."""

    INGEST = auto()  # Problem analysis
    IDEATE = auto()  # Generate ideas (RPE, SCAMPER, etc.)
    REVIEW = auto()  # Review Council evaluation
    SELECT = auto()  # Select winning idea
    PLAN = auto()  # AURA: Plan implementation
    CRITIQUE = auto()  # AURA: Critique plan
    SYNTHESIZE = auto()  # AURA: Merge plan + critique
    IMPLEMENT = auto()  # AURA: Generate code
    VERIFY = auto()  # AURA: Test & validate
    REFLECT = auto()  # Learn & store patterns


@dataclass
class UnifiedCycleResult:
    """Result of a unified creative-execution cycle."""

    problem: str
    ideas: List[CreativeIdea] = field(default_factory=list)
    selected_idea: Optional[CreativeIdea] = None
    implementation: Optional[ImplementationResult] = None
    phase_results: Dict[CreativePhase, Any] = field(default_factory=dict)
    success: bool = False
    cycle_time_seconds: float = 0.0
    lessons_learned: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem": self.problem,
            "idea_count": len(self.ideas),
            "selected_technique": self.selected_idea.technique if self.selected_idea else None,
            "implementation_success": self.implementation.success if self.implementation else False,
            "total_time": self.cycle_time_seconds,
            "success": self.success,
        }


class CreativeOrchestrator:
    """Unified orchestrator for creative problem-solving + implementation.

        Combines AURA's autonomous coding capabilities with creative innovation
    t    techniques to provide end-to-end problem solving:

        1. **Creative Phase**: Generate diverse solutions using RPE, SCAMPER, etc.
        2. **Review Phase**: Multi-perspective evaluation via Review Council
        3. **Execution Phase**: AURA implements the selected solution
        4. **Learning Phase**: Store successful patterns for future recall

        Example:
            >>> orchestrator = CreativeOrchestrator(
            ...     aura_orchestrator=aura_orchestrator,
            ...     brain=brain,
            ... )
            >>> result = await orchestrator.solve(
            ...     problem="Reduce API latency",
            ...     techniques=["SCAMPER", "RPE"],
            ... )
            >>> print(f"Solution: {result.implementation.files_changed}")
    """

    def __init__(
        self,
        aura_orchestrator: Any,
        brain: Any,
        enable_creative_phase: bool = True,
        enable_review_council: bool = True,
        max_ideas_per_technique: int = 3,
    ):
        """Initialize the unified orchestrator.

        Args:
            aura_orchestrator: AURA's LoopOrchestrator instance
            brain: AURA Brain for memory
            enable_creative_phase: Whether to use creative techniques
            enable_review_council: Whether to use multi-perspective review
            max_ideas_per_technique: Maximum ideas per technique
        """
        self.aura_orchestrator = aura_orchestrator
        self.brain = brain
        self.enable_creative_phase = enable_creative_phase
        self.enable_review_council = enable_review_council
        self.max_ideas_per_technique = max_ideas_per_technique

        # Initialize subsystems
        self.implementation_bridge = CreativeImplementationBridge(orchestrator=aura_orchestrator)
        self.pattern_memory = CreativePatternMemory(brain=brain)

        # Available creative techniques
        self.techniques: Dict[str, Callable] = {
            "RPE": self._run_rpe,
            "SCAMPER": self._run_scamper,
            "SixHats": self._run_six_hats,
            "AutoTRIZ": self._run_autotriz,
        }

    async def solve(
        self,
        problem: str,
        techniques: Optional[List[str]] = None,
        requirements: Optional[List[str]] = None,
        domain: str = "general",
    ) -> UnifiedCycleResult:
        """Solve a problem using unified creative-execution approach.

        Args:
            problem: Problem description
            techniques: Creative techniques to use (default: all)
            requirements: Solution requirements
            domain: Problem domain

        Returns:
            Full cycle result with implementation
        """
        start_time = datetime.now()
        result = UnifiedCycleResult(problem=problem)

        try:
            # Phase 1: INGEST - Problem analysis
            log_json("INFO", "creative_orchestrator_ingest", {"problem": problem[:100]})
            result.phase_results[CreativePhase.INGEST] = {"problem": problem}

            # Phase 2: IDEATE - Generate ideas using creative techniques
            if self.enable_creative_phase:
                result.ideas = await self._generate_ideas(
                    problem=problem,
                    techniques=techniques or list(self.techniques.keys()),
                    domain=domain,
                )
                result.phase_results[CreativePhase.IDEATE] = {
                    "idea_count": len(result.ideas),
                    "techniques_used": list(set(i.technique for i in result.ideas)),
                }
            else:
                # Skip creative phase, create single direct idea
                result.ideas = [
                    CreativeIdea(
                        content=problem,
                        requirements=requirements or [],
                        technique="direct",
                        domain=domain,
                    )
                ]

            # Phase 3: REVIEW - Council evaluation (if enabled)
            if self.enable_review_council and len(result.ideas) > 1:
                result.ideas = await self._review_council(result.ideas)
                result.phase_results[CreativePhase.REVIEW] = {
                    "ideas_after_review": len(result.ideas),
                }

            # Phase 4: SELECT - Choose winning idea
            result.selected_idea = self._select_best_idea(result.ideas)
            result.phase_results[CreativePhase.SELECT] = {
                "selected_technique": result.selected_idea.technique,
                "confidence": result.selected_idea.confidence,
            }

            # Add requirements if provided
            if requirements:
                result.selected_idea.requirements.extend(requirements)

            # Phases 5-9: AURA execution (PLAN → VERIFY)
            result.implementation = await self.implementation_bridge.implement(result.selected_idea)
            result.phase_results[CreativePhase.IMPLEMENT] = {
                "success": result.implementation.success,
                "files_changed": len(result.implementation.files_changed),
                "cycles_used": result.implementation.cycles_used,
            }

            # Phase 10: REFLECT - Store patterns and learn
            await self._reflect_and_learn(result)
            result.phase_results[CreativePhase.REFLECT] = {
                "lessons_count": len(result.lessons_learned),
            }

            # Calculate overall success
            result.success = result.implementation.success
            result.cycle_time_seconds = (datetime.now() - start_time).total_seconds()

            log_json(
                "INFO",
                "creative_orchestrator_complete",
                {
                    "success": result.success,
                    "total_time": result.cycle_time_seconds,
                    "technique": result.selected_idea.technique,
                },
            )

        except Exception as e:
            log_json("ERROR", "creative_orchestrator_failed", {"error": str(e)})
            result.success = False

        return result

    async def _generate_ideas(
        self,
        problem: str,
        techniques: List[str],
        domain: str,
    ) -> List[CreativeIdea]:
        """Generate ideas using specified techniques."""
        ideas = []

        for technique_name in techniques:
            if technique_name not in self.techniques:
                continue

            try:
                technique_ideas = await self.techniques[technique_name](
                    problem=problem,
                    domain=domain,
                )
                ideas.extend(technique_ideas)
            except Exception as e:
                log_json("WARN", "technique_failed", {"technique": technique_name, "error": str(e)})

        return ideas

    async def _run_rpe(self, problem: str, domain: str) -> List[CreativeIdea]:
        """Run Recursive Prompt Expansion technique."""
        # Simplified RPE implementation
        ideas = []
        expansions = [
            f"Optimize {problem}",
            f"Simplify {problem}",
            f"Automate {problem}",
        ]

        for expansion in expansions[: self.max_ideas_per_technique]:
            ideas.append(
                CreativeIdea(
                    content=expansion,
                    technique="RPE",
                    domain=domain,
                    confidence=0.7,
                )
            )

        return ideas

    async def _run_scamper(self, problem: str, domain: str) -> List[CreativeIdea]:
        """Run SCAMPER technique."""
        scamper_verbs = ["Substitute", "Combine", "Adapt", "Modify", "Put to other uses"]
        ideas = []

        for verb in scamper_verbs[: self.max_ideas_per_technique]:
            ideas.append(
                CreativeIdea(
                    content=f"{verb} components to solve: {problem}",
                    technique="SCAMPER",
                    domain=domain,
                    confidence=0.6,
                )
            )

        return ideas

    async def _run_six_hats(self, problem: str, domain: str) -> List[CreativeIdea]:
        """Run Six Thinking Hats technique."""
        hats = ["White (facts)", "Red (emotion)", "Black (caution)", "Yellow (optimism)"]
        ideas = []

        for hat in hats[: self.max_ideas_per_technique]:
            ideas.append(
                CreativeIdea(
                    content=f"From {hat} perspective: {problem}",
                    technique="SixHats",
                    domain=domain,
                    confidence=0.65,
                )
            )

        return ideas

    async def _run_autotriz(self, problem: str, domain: str) -> List[CreativeIdea]:
        """Run AutoTRIZ technique."""
        principles = ["Segmentation", "Taking out", "Local quality", "Asymmetry"]
        ideas = []

        for principle in principles[: self.max_ideas_per_technique]:
            ideas.append(
                CreativeIdea(
                    content=f"Apply TRIZ principle '{principle}' to: {problem}",
                    technique="AutoTRIZ",
                    domain=domain,
                    confidence=0.75,
                )
            )

        return ideas

    async def _review_council(self, ideas: List[CreativeIdea]) -> List[CreativeIdea]:
        """Multi-perspective review of ideas."""
        # Simplified council: filter by confidence, boost diversity
        min_confidence = 0.5
        filtered = [i for i in ideas if i.confidence >= min_confidence]

        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)

        return filtered[:10]  # Return top 10

    def _select_best_idea(self, ideas: List[CreativeIdea]) -> CreativeIdea:
        """Select the best idea from candidates."""
        if not ideas:
            raise ValueError("No ideas to select from")

        # Prioritize by confidence, but also check pattern memory
        for idea in sorted(ideas, key=lambda x: x.confidence, reverse=True):
            # Check if similar patterns have succeeded
            try:
                similar = self.pattern_memory.recall(
                    domain=idea.domain,
                    query=idea.content,
                    top_k=1,
                    min_success_rate=0.7,
                )
                if similar:
                    # Boost confidence for proven patterns
                    idea.confidence = min(1.0, idea.confidence + 0.1)
            except (OSError, TypeError, AttributeError):
                pass

        # Return highest confidence
        return max(ideas, key=lambda x: x.confidence)

    async def _reflect_and_learn(self, result: UnifiedCycleResult) -> None:
        """Store patterns and extract lessons."""
        if not result.selected_idea or not result.implementation:
            return

        # Store pattern if successful
        if result.implementation.success:
            pattern = CreativePattern(
                id=f"{result.selected_idea.technique}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                content=result.selected_idea.content,
                domain=result.selected_idea.domain,
                technique=result.selected_idea.technique,
                success_rate=1.0 if result.implementation.success else 0.0,
            )

            try:
                self.pattern_memory.record(pattern)
                result.lessons_learned.append(f"Stored successful pattern: {pattern.technique}")
            except Exception as e:
                log_json("WARN", "pattern_storage_failed", {"error": str(e)})

        # Update success rate for existing patterns
        self.pattern_memory.update_success_rate(
            pattern_id=result.selected_idea.technique,
            success=result.implementation.success,
        )


def create_unified_orchestrator(
    aura_orchestrator: Any,
    brain: Any,
    **kwargs,
) -> CreativeOrchestrator:
    """Factory function for CreativeOrchestrator.

    Args:
        aura_orchestrator: AURA's orchestrator
        brain: AURA Brain
        **kwargs: Additional configuration

    Returns:
        Configured CreativeOrchestrator
    """
    return CreativeOrchestrator(
        aura_orchestrator=aura_orchestrator,
        brain=brain,
        **kwargs,
    )
