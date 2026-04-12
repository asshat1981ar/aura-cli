"""
Structured output schemas for AURA agents with Chain-of-Thought reasoning.

These Pydantic models enforce consistent, parseable responses from LLMs
while encouraging step-by-step reasoning through CoT prompts.

For prompt templates and role-based system prompts, see: agents/prompt_manager.py
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DebugStrategy(BaseModel):
    """Typed diagnosis and fix strategy produced by DebuggerAgent.

    Mirrors the pattern from the external multi-agent template: structured,
    schema-enforced debugger output instead of a loosely-typed dict.  The
    field names are kept consistent with AURA's existing ``fix_strategy`` key
    so that ``model.model_dump()`` remains backward-compatible with callers
    that access ``result["fix_strategy"]``.
    """

    summary: str = Field(description="Concise summary of the error")
    diagnosis: str = Field(description="Detailed explanation of the probable cause")
    fix_strategy: str = Field(description="Step-by-step plan or code suggestion to resolve the issue")
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(description="Error severity level")

    @property
    def fix_instructions(self) -> str:
        """Alias for ``fix_strategy`` — matches the external template's field name."""
        return self.fix_strategy


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    step_number: int = Field(description="Sequential step number (1-indexed)")
    description: str = Field(description="Clear, actionable description of the step")
    target_file: Optional[str] = Field(default=None, description="Primary file this step modifies (if any)")
    verification: str = Field(description="How to verify this step completed successfully")


class PlannerOutput(BaseModel):
    """Structured output from the PlannerAgent with Chain-of-Thought reasoning."""

    # Chain-of-Thought reasoning
    analysis: str = Field(description="Step 1: Analyze the goal and current codebase state")
    gap_assessment: str = Field(description="Step 2: Identify structural gaps and requirements")
    approach: str = Field(description="Step 3: Describe the overall approach and strategy")
    risk_assessment: str = Field(description="Step 4: Predict potential failure modes and risks")

    # Final structured output
    plan: List[PlanStep] = Field(description="The generated execution plan as ordered steps")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0) in this plan")
    total_steps: int = Field(description="Total number of steps in the plan")
    estimated_complexity: Literal["low", "medium", "high"] = Field(description="Estimated complexity of the goal")


class CriticIssue(BaseModel):
    """A specific issue identified during critique."""

    severity: Literal["critical", "major", "minor", "suggestion"] = Field(description="Severity of the issue")
    category: Literal["completeness", "clarity", "feasibility", "alignment", "safety", "correctness", "maintainability", "performance", "security", "other"] = Field(description="Category of the issue")
    description: str = Field(description="Detailed description of the issue")
    recommendation: str = Field(description="Specific recommendation to address this issue")


class CriticOutput(BaseModel):
    """Structured output from the CriticAgent with Chain-of-Thought reasoning."""

    # Chain-of-Thought reasoning
    initial_assessment: str = Field(description="Step 1: Initial impression and high-level assessment")
    completeness_check: str = Field(description="Step 2: Check if all requirements are addressed")
    feasibility_analysis: str = Field(description="Step 3: Analyze feasibility of each component")
    risk_identification: str = Field(description="Step 4: Identify potential risks and edge cases")

    # Final structured output
    overall_assessment: Literal["approve", "approve_with_changes", "request_changes", "reject"] = Field(description="Overall assessment of the plan/code")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this assessment")
    issues: List[CriticIssue] = Field(default_factory=list, description="List of identified issues")
    positive_aspects: List[str] = Field(default_factory=list, description="What's done well")
    summary: str = Field(description="Executive summary of the critique")


class CodeChange(BaseModel):
    """A single code change to be applied."""

    file_path: str = Field(description="Target file path for this change")
    search_block: str = Field(description="Exact code to find and replace (for matching)")
    replace_block: str = Field(description="New code to insert")
    reasoning: str = Field(description="Why this change is needed")


class CoderOutput(BaseModel):
    """Structured output from the CoderAgent with Chain-of-Thought reasoning."""

    # Chain-of-Thought reasoning
    problem_analysis: str = Field(description="Step 1: Analyze the problem and requirements")
    approach_selection: str = Field(description="Step 2: Select the best implementation approach")
    design_considerations: str = Field(description="Step 3: Consider edge cases and design patterns")
    testing_strategy: str = Field(description="Step 4: Plan how to verify the implementation")

    # Final structured output
    aura_target: str = Field(description="Target file path for the main code")
    code: str = Field(description="The generated Python code")
    explanation: str = Field(description="High-level explanation of the solution")
    dependencies: List[str] = Field(default_factory=list, description="Required imports/dependencies")
    edge_cases_handled: List[str] = Field(default_factory=list, description="Edge cases addressed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this solution")


class MutationValidationOutput(BaseModel):
    """Structured output for mutation validation (used by CriticAgent)."""

    # Chain-of-Thought reasoning
    impact_analysis: str = Field(description="Step 1: Analyze the potential impact of this mutation")
    safety_assessment: str = Field(description="Step 2: Assess safety concerns and risks")
    effectiveness_evaluation: str = Field(description="Step 3: Evaluate likelihood of success")

    # Final structured output
    decision: Literal["APPROVED", "REJECTED", "NEEDS_REVISION"] = Field(description="Validation decision")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in this decision")
    impact_assessment: str = Field(description="Summary of positive and negative impacts")
    reasoning: str = Field(description="Detailed reasoning for the decision")
    recommendations: Optional[str] = Field(default=None, description="Recommendations if revision needed")


# ============================================================================
# INNOVATION CATALYST SCHEMAS
# ============================================================================

from enum import Enum
from datetime import datetime
from typing import Dict, Any


class InnovationPhase(str, Enum):
    """Phases of the Innovation Catalyst methodology."""

    IMMERSION = "immersion"
    DIVERGENCE = "divergence"
    CONVERGENCE = "convergence"
    INCUBATION = "incubation"
    TRANSFORMATION = "transformation"


class BrainstormingTechnique(str, Enum):
    """Supported brainstorming techniques."""

    SCAMPER = "SCAMPER"
    SIX_THINKING_HATS = "Six Thinking Hats"
    MIND_MAPPING = "Mind Mapping"
    REVERSE_BRAINSTORMING = "Reverse Brainstorming"
    WORST_IDEA = "Worst Idea"
    LOTUS_BLOSSOM = "Lotus Blossom"
    STAR_BRAINSTORMING = "Star Brainstorming"
    BISOCIATIVE_ASSOCIATION = "Bisociative Association"


class Idea(BaseModel):
    """A single idea generated during brainstorming."""

    description: str = Field(description="Detailed description of the idea")
    technique: str = Field(description="Brainstorming technique that generated this idea")
    novelty: float = Field(ge=0.0, le=1.0, description="Novelty score (0-1)")
    feasibility: float = Field(ge=0.0, le=1.0, description="Feasibility score (0-1)")
    impact: float = Field(ge=0.0, le=1.0, description="Potential impact score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Technique-specific metadata")


class TechniqueResult(BaseModel):
    """Results from a single brainstorming technique."""

    technique: str = Field(description="Name of the technique")
    ideas: List[Idea] = Field(description="Ideas generated by this technique")
    idea_count: int = Field(description="Number of ideas generated")
    execution_time_ms: Optional[int] = Field(default=None, description="Execution time in milliseconds")


class InnovationOutput(BaseModel):
    """Structured output from the InnovationSwarm agent."""

    session_id: str = Field(description="Unique session identifier")
    problem_statement: str = Field(description="Original problem being solved")
    phase: InnovationPhase = Field(description="Current innovation phase")
    techniques_used: List[str] = Field(description="Techniques applied in this session")
    all_ideas: List[Idea] = Field(description="All raw ideas generated")
    selected_ideas: List[Idea] = Field(description="Ideas selected after convergence")
    technique_results: Dict[str, TechniqueResult] = Field(description="Per-technique results")
    diversity_score: float = Field(ge=0.0, le=1.0, description="Diversity across techniques (0-1)")
    novelty_score: float = Field(ge=0.0, le=1.0, description="Average novelty of selected ideas (0-1)")
    feasibility_score: float = Field(ge=0.0, le=1.0, description="Average feasibility of selected ideas (0-1)")
    total_ideas_generated: int = Field(description="Total count of raw ideas")
    total_ideas_selected: int = Field(description="Count of ideas after convergence")
    reasoning: Dict[str, str] = Field(default_factory=dict, description="Chain-of-thought reasoning")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the session completed")


class InnovationSessionState(BaseModel):
    """Tracks the complete state of an innovation session."""

    session_id: str = Field(description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    problem_statement: str = Field(description="Original problem")
    current_phase: InnovationPhase = Field(default=InnovationPhase.IMMERSION)
    phases_completed: List[InnovationPhase] = Field(default_factory=list)
    techniques: List[str] = Field(default_factory=list)  # Technique identifiers
    constraints: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[InnovationOutput] = Field(default=None)
    status: Literal["active", "completed", "paused"] = Field(default="active")
    ideas_generated: int = Field(default=0)
    ideas_selected: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetaConductorOutput(BaseModel):
    """Output from the MetaConductor orchestrating innovation sessions."""

    session_id: str = Field(description="Session identifier")
    problem_statement: str = Field(description="Original problem")
    phases: List[InnovationPhase] = Field(description="All phases in the workflow")
    current_phase: InnovationPhase = Field(description="Current phase")
    phase_tasks: Dict[str, Any] = Field(description="Tasks for each phase")
    convergence_criteria: Dict[str, float] = Field(description="Criteria for moving between phases")
    catalyst_methodology: bool = Field(default=True, description="Whether using Innovation Catalyst")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in session setup")


# ============================================================================
# BACKWARD COMPATIBILITY: Legacy prompt templates
# These are deprecated. Use agents.prompt_manager.render_prompt() instead.
# ============================================================================

# Import from prompt_manager for backward compatibility
try:
    from agents.prompt_manager import (
        PLANNER_PROMPT_TEMPLATE as PLANNER_COT_PROMPT_TEMPLATE,
        CRITIC_PROMPT_TEMPLATE as CRITIC_COT_PROMPT_TEMPLATE,
        CODER_PROMPT_TEMPLATE as CODER_COT_PROMPT_TEMPLATE,
    )
except ImportError:
    # Fallback if prompt_manager not available
    PLANNER_COT_PROMPT_TEMPLATE = ""
    CRITIC_COT_PROMPT_TEMPLATE = ""
    CODER_COT_PROMPT_TEMPLATE = ""
