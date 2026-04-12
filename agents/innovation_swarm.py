"""
InnovationSwarm Agent - Multi-technique brainstorming orchestrator.

This agent coordinates multiple brainstorming technique bots to generate
diverse, novel ideas through the Innovation Catalyst methodology.

Now with LLM-powered idea generation via OpenRouter.
"""

import uuid
from typing import List, Dict, Any, Optional

from agents.schemas import Idea, InnovationOutput, TechniqueResult, InnovationPhase
from agents.brainstorming_bots import BRAINSTORMING_BOTS, get_bot
from core.logging_utils import log_json


class InnovationSwarm:
    """
    Orchestrates multiple brainstorming techniques to generate diverse ideas.

    Implements the DIVERGENCE and CONVERGENCE phases of the Innovation
    Catalyst methodology, coordinating technique bots in parallel.

    Supports both LLM-powered generation (via OpenRouter) and template-based
    fallback for offline/cost-sensitive scenarios.
    """

    capabilities = ["innovation", "brainstorming", "divergence", "convergence", "creativity", "ideation"]

    description = "Multi-technique brainstorming agent that generates diverse, novel ideas"

    def __init__(self, brain=None, model=None, llm_client=None, use_llm: bool = True):
        """
        Initialize the InnovationSwarm.

        Args:
            brain: Optional memory/brain instance for context
            model: Optional model adapter for LLM interactions (legacy)
            llm_client: Optional LLM client for idea generation
            use_llm: Whether to use LLM if available
        """
        self.brain = brain
        self.model = model
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.session_id = None

        # Initialize LLM client if not provided but enabled
        if self.use_llm and not self.llm_client:
            try:
                from agents.llm_brainstorming import get_llm_client

                self.llm_client = get_llm_client()
                log_json("INFO", "llm_client_initialized")
            except Exception as e:
                log_json("WARN", "llm_client_init_failed", details={"error": str(e)})

    def run(self, input_data: dict) -> dict:
        """
        Standard agent interface for orchestrator integration.

        Args:
            input_data: Dict with keys:
                - "task" or "problem": The problem statement
                - "techniques": List of technique names to use (optional)
                - "constraints": Dict with convergence criteria (optional)
                - "context": Additional context (optional)

        Returns:
            Dict with structured innovation output
        """
        problem = input_data.get("task") or input_data.get("problem", "")
        techniques = input_data.get("techniques", list(BRAINSTORMING_BOTS.keys()))
        constraints = input_data.get("constraints", {})
        context = input_data.get("context", "")

        # Run the brainstorming process
        output = self.brainstorm(problem_statement=problem, techniques=techniques, constraints=constraints, context=context)

        # Return as dict for orchestrator compatibility
        return output.dict()

    def brainstorm(self, problem_statement: str, techniques: Optional[List[str]] = None, constraints: Optional[Dict[str, Any]] = None, context: str = "") -> InnovationOutput:
        """
        Execute full brainstorming workflow: divergence + convergence.

        Args:
            problem_statement: The challenge or problem to solve
            techniques: List of technique names (defaults to all available)
            constraints: Convergence criteria (diversity_threshold, novelty_threshold, etc.)
            context: Additional context for the brainstorming session

        Returns:
            InnovationOutput with all ideas, selected ideas, and metrics
        """
        self.session_id = str(uuid.uuid4())[:8]
        constraints = constraints or {}

        # Default to all available techniques if none specified
        if not techniques:
            techniques = list(BRAINSTORMING_BOTS.keys())

        log_json("INFO", "innovation_swarm_started", details={"session_id": self.session_id, "problem": problem_statement[:50], "techniques": techniques, "phase": InnovationPhase.DIVERGENCE.value})

        # Phase 1: DIVERGENCE - Generate ideas from all techniques
        technique_results = self._divergence_phase(problem_statement, techniques, context)

        # Collect all ideas
        all_ideas = []
        for result in technique_results.values():
            all_ideas.extend(result.ideas)

        log_json("INFO", "innovation_divergence_complete", details={"session_id": self.session_id, "total_ideas": len(all_ideas), "techniques_used": len(technique_results)})

        # Phase 2: CONVERGENCE - Select best ideas
        selected_ideas = self._convergence_phase(all_ideas, constraints)

        log_json("INFO", "innovation_convergence_complete", details={"session_id": self.session_id, "selected_ideas": len(selected_ideas), "convergence_rate": len(selected_ideas) / len(all_ideas) if all_ideas else 0})

        # Calculate metrics
        diversity_score = self._calculate_diversity(technique_results)
        novelty_score = self._calculate_novelty(selected_ideas)
        feasibility_score = self._calculate_feasibility(selected_ideas)

        # Build output
        output = InnovationOutput(
            session_id=self.session_id,
            problem_statement=problem_statement,
            phase=InnovationPhase.CONVERGENCE,
            techniques_used=techniques,
            all_ideas=all_ideas,
            selected_ideas=selected_ideas,
            technique_results=technique_results,
            diversity_score=diversity_score,
            novelty_score=novelty_score,
            feasibility_score=feasibility_score,
            total_ideas_generated=len(all_ideas),
            total_ideas_selected=len(selected_ideas),
            reasoning={"divergence": f"Generated {len(all_ideas)} ideas across {len(techniques)} techniques", "convergence": f"Selected top {len(selected_ideas)} ideas based on novelty, feasibility, and impact", "diversity": f"Diversity score: {diversity_score:.2f} across techniques"},
        )

        # Store in memory if brain available
        if self.brain:
            self.brain.remember(f"Innovation session {self.session_id}: {len(selected_ideas)} ideas for '{problem_statement[:40]}...'")

        return output

    def _divergence_phase(self, problem_statement: str, techniques: List[str], context: str) -> Dict[str, TechniqueResult]:
        """
        Generate ideas using all specified techniques.

        Args:
            problem_statement: The challenge to solve
            techniques: List of technique names
            context: Additional context

        Returns:
            Dict mapping technique names to TechniqueResult
        """
        results = {}

        # Run each technique (sequential for now, could be parallel)
        for technique_name in techniques:
            try:
                bot = get_bot(technique_name, llm_client=self.llm_client, use_llm=self.use_llm)
                ideas = bot.generate(problem_statement, context)

                results[technique_name] = TechniqueResult(technique=bot.technique_name, ideas=ideas, idea_count=len(ideas))

                log_json("DEBUG", "technique_complete", details={"session_id": self.session_id, "technique": technique_name, "ideas_generated": len(ideas)})

            except Exception as e:
                log_json("ERROR", "technique_failed", details={"session_id": self.session_id, "technique": technique_name, "error": str(e)})
                # Continue with other techniques
                continue

        return results

    def _convergence_phase(self, ideas: List[Idea], constraints: Dict[str, Any]) -> List[Idea]:
        """
        Select best ideas based on scoring and constraints.

        Args:
            ideas: All generated ideas
            constraints: Convergence criteria including:
                - selection_ratio: Fraction of ideas to select (default: 0.2)
                - min_novelty: Minimum novelty threshold (default: 0.5)
                - min_feasibility: Minimum feasibility threshold (default: 0.4)
                - max_ideas: Maximum number of ideas to select (default: 20)

        Returns:
            List of selected ideas
        """
        if not ideas:
            return []

        # Get constraints with defaults
        selection_ratio = constraints.get("selection_ratio", 0.2)
        min_novelty = constraints.get("min_novelty", 0.5)
        min_feasibility = constraints.get("min_feasibility", 0.4)
        max_ideas = constraints.get("max_ideas", 20)

        # Score each idea
        scored_ideas = []
        for idea in ideas:
            # Filter by minimum thresholds
            if idea.novelty < min_novelty or idea.feasibility < min_feasibility:
                continue

            # Composite score: novelty (40%) + feasibility (30%) + impact (30%)
            score = idea.novelty * 0.4 + idea.feasibility * 0.3 + idea.impact * 0.3
            scored_ideas.append((idea, score))

        # Sort by score descending
        scored_ideas.sort(key=lambda x: x[1], reverse=True)

        # Select top ideas
        num_to_select = min(max(1, int(len(ideas) * selection_ratio)), max_ideas, len(scored_ideas))

        selected = [idea for idea, score in scored_ideas[:num_to_select]]
        return selected

    def _calculate_diversity(self, technique_results: Dict[str, TechniqueResult]) -> float:
        """
        Calculate diversity score across techniques.

        Higher score when techniques produce different types of ideas.
        Based on: number of techniques used, distribution of ideas, and technique variety.

        Args:
            technique_results: Results from each technique

        Returns:
            Diversity score (0-1)
        """
        if not technique_results:
            return 0.0

        # Base diversity on technique count (more techniques = more diversity)
        technique_count = len(technique_results)
        base_diversity = min(1.0, technique_count / 6 * 1.2)

        # Adjust based on idea distribution
        idea_counts = [r.idea_count for r in technique_results.values()]
        if idea_counts:
            avg_count = sum(idea_counts) / len(idea_counts)
            # Penalize if one technique dominates
            max_count = max(idea_counts)
            balance = 1.0 - (max_count - avg_count) / max_count if max_count > 0 else 0
        else:
            balance = 0

        # Combine scores
        diversity = (base_diversity * 0.6) + (balance * 0.4)
        return min(1.0, diversity)

    def _calculate_novelty(self, ideas: List[Idea]) -> float:
        """Calculate average novelty of selected ideas."""
        if not ideas:
            return 0.0
        return sum(i.novelty for i in ideas) / len(ideas)

    def _calculate_feasibility(self, ideas: List[Idea]) -> float:
        """Calculate average feasibility of selected ideas."""
        if not ideas:
            return 0.0
        return sum(i.feasibility for i in ideas) / len(ideas)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the last brainstorming session."""
        return {"session_id": self.session_id, "capabilities": self.capabilities, "available_techniques": list(BRAINSTORMING_BOTS.keys())}
