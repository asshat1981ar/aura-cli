"""Tree of Thoughts (ToT) decomposition strategy.

Implements the Tree of Thoughts reasoning approach from the paper by
Yao et al. (2023) for complex goal decomposition.

Key concepts:
- Deliberate reasoning: Explore multiple reasoning paths
- Self-evaluation: Score different approaches
- Lookahead/backtracking: Simulate and choose best path
- Global decision: Select optimal decomposition

Best for: Complex multi-step problems requiring planning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum

from core.decomposition.base import DecompositionStrategy, DecompositionResult, SubTask, TaskPriority
from core.logging_utils import log_json


class ThoughtStatus(Enum):
    """Status of a thought in the tree."""

    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    SELECTED = "selected"
    REJECTED = "rejected"


@dataclass
class Thought:
    """A single thought/state in the reasoning tree.

    In Tree of Thoughts, each node represents a partial solution
    or reasoning step toward the final goal.

    Attributes:
        id: Unique thought identifier
        content: The thought content/description
        parent_id: Parent thought ID (None for root)
        children: Child thought IDs
        status: Current status
        score: Evaluation score (0-1)
        depth: Tree depth
        reasoning: Why this thought was proposed
    """

    id: str
    content: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    status: ThoughtStatus = ThoughtStatus.PROPOSED
    score: float = 0.0
    depth: int = 0
    reasoning: str = ""

    def __post_init__(self):
        if not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")


@dataclass
class ReasoningPath:
    """A complete path from root to leaf in the thought tree."""

    thoughts: List[Thought] = field(default_factory=list)
    total_score: float = 0.0

    def __post_init__(self):
        if self.thoughts:
            # Score is average of thought scores
            self.total_score = sum(t.score for t in self.thoughts) / len(self.thoughts)


class TreeOfThoughtsDecomposition(DecompositionStrategy):
    """Tree of Thoughts decomposition for complex goals.

    This strategy explores multiple reasoning paths for decomposing
    a goal, evaluating each path, and selecting the best one.

    Algorithm:
    1. Generate multiple initial thoughts (approaches)
    2. Evaluate each thought's potential
    3. Expand promising thoughts with sub-thoughts
    4. Continue until reaching complete decompositions
    5. Select best path based on cumulative scores

    Reference:
        Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., & Narasimhan, K. (2023).
        Tree of Thoughts: Deliberate Problem Solving with Large Language Models.
        arXiv:2305.10601.
    """

    name = "tree_of_thoughts"

    def __init__(
        self,
        max_depth: int = 5,
        branching_factor: int = 3,
        evaluation_threshold: float = 0.3,
    ):
        """Initialize ToT decomposition.

        Args:
            max_depth: Maximum depth of thought tree
            branching_factor: Number of children per thought
            evaluation_threshold: Minimum score to expand a thought
        """
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.evaluation_threshold = evaluation_threshold
        self.thoughts: Dict[str, Thought] = {}
        self.evaluator: Optional[Callable[[Thought], float]] = None

    def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecompositionResult:
        """Decompose goal using Tree of Thoughts reasoning."""
        context = context or {}
        self.thoughts.clear()

        # Step 1: Generate initial thoughts (approaches)
        root_thoughts = self._generate_initial_thoughts(goal)

        log_json(
            "DEBUG",
            "tot_initial_thoughts",
            {
                "count": len(root_thoughts),
                "thoughts": [t.content[:50] for t in root_thoughts],
            },
        )

        # Step 2: Build tree by expanding promising thoughts
        complete_paths: List[ReasoningPath] = []

        for root in root_thoughts:
            paths = self._expand_thought(root, goal, context)
            complete_paths.extend(paths)

        # Step 3: Select best path
        if not complete_paths:
            # Fallback to simple sequential
            return self._fallback_decomposition(goal)

        best_path = max(complete_paths, key=lambda p: p.total_score)

        log_json(
            "DEBUG",
            "tot_path_selected",
            {
                "path_length": len(best_path.thoughts),
                "score": best_path.total_score,
            },
        )

        # Step 4: Convert thought path to subtasks
        subtasks = self._thoughts_to_subtasks(best_path.thoughts)

        return DecompositionResult(
            original_goal=goal,
            subtasks=subtasks,
            strategy=self.name,
            reasoning=f"Selected best path from {len(complete_paths)} complete paths using ToT reasoning",
            confidence=best_path.total_score,
            metadata={
                "thoughts_explored": len(self.thoughts),
                "paths_evaluated": len(complete_paths),
                "best_path_score": best_path.total_score,
                "tree_depth": max(t.depth for t in self.thoughts.values()) if self.thoughts else 0,
            },
        )

    def _generate_initial_thoughts(self, goal: str) -> List[Thought]:
        """Generate initial approach thoughts.

        In a real implementation, this would use an LLM to generate
        diverse approaches. Here we use rule-based generation.
        """
        thoughts: List[Thought] = []
        goal_lower = goal.lower()

        # Approach 1: Sequential decomposition
        if any(kw in goal_lower for kw in ["step", "sequence", "first", "then"]):
            thoughts.append(
                Thought(
                    id="t1",
                    content="Sequential approach: Break into ordered steps",
                    reasoning="Goal implies sequential execution",
                    score=0.8,
                )
            )

        # Approach 2: Parallel decomposition
        if any(kw in goal_lower for kw in ["all", "every", "each", "multiple"]):
            thoughts.append(
                Thought(
                    id="t2",
                    content="Parallel approach: Identify independent workstreams",
                    reasoning="Goal suggests parallelizable work",
                    score=0.7,
                )
            )

        # Approach 3: Hierarchical decomposition
        thoughts.append(
            Thought(
                id="t3",
                content="Hierarchical approach: Top-down refinement",
                reasoning="General-purpose decomposition strategy",
                score=0.6,
            )
        )

        # Approach 4: Pattern-based
        if any(kw in goal_lower for kw in ["refactor", "fix", "update"]):
            thoughts.append(
                Thought(
                    id="t4",
                    content="Pattern-based approach: Apply known solution patterns",
                    reasoning="Goal matches known refactoring pattern",
                    score=0.75,
                )
            )

        for t in thoughts:
            self.thoughts[t.id] = t

        return thoughts

    def _expand_thought(
        self,
        thought: Thought,
        goal: str,
        context: Dict[str, Any],
    ) -> List[ReasoningPath]:
        """Expand a thought into sub-thoughts.

        Returns complete paths from this thought to leaves.
        """
        # Base case: max depth reached
        if thought.depth >= self.max_depth:
            thought.status = ThoughtStatus.SELECTED
            return [ReasoningPath(thoughts=[thought])]

        # Evaluate if we should expand
        if thought.score < self.evaluation_threshold:
            thought.status = ThoughtStatus.REJECTED
            return []

        thought.status = ThoughtStatus.SELECTED

        # Generate child thoughts (sub-steps)
        children = self._generate_children(thought, goal, context)

        if not children:
            # Leaf node - complete path
            return [ReasoningPath(thoughts=[thought])]

        # Recursively expand children
        complete_paths: List[ReasoningPath] = []
        for child in children:
            child_paths = self._expand_thought(child, goal, context)
            for path in child_paths:
                path.thoughts.insert(0, thought)
                complete_paths.append(path)

        return complete_paths

    def _generate_children(
        self,
        parent: Thought,
        goal: str,
        context: Dict[str, Any],
    ) -> List[Thought]:
        """Generate child thoughts for a parent thought."""
        children: List[Thought] = []

        # Generate based on approach type
        if "Sequential" in parent.content:
            steps = [
                "Analyze current state",
                "Identify issues/opportunities",
                "Plan changes",
                "Execute changes",
                "Verify results",
            ]
        elif "Parallel" in parent.content:
            steps = [
                "Identify workstreams",
                "Assign workstream 1",
                "Assign workstream 2",
                "Assign workstream 3",
                "Synchronize results",
            ]
        elif "Hierarchical" in parent.content:
            steps = [
                "Define high-level structure",
                "Refine level 1 components",
                "Refine level 2 components",
                "Define interfaces",
                "Verify consistency",
            ]
        else:
            steps = [
                "Understand requirements",
                "Design solution",
                "Implement",
                "Test",
            ]

        # Create child thoughts
        for i, step in enumerate(steps[: self.branching_factor], 1):
            child_id = f"{parent.id}.{i}"
            child = Thought(
                id=child_id,
                content=step,
                parent_id=parent.id,
                depth=parent.depth + 1,
                score=self._evaluate_thought(step, goal),
                reasoning=f"Sub-step of: {parent.content[:30]}...",
            )
            parent.children.append(child_id)
            self.thoughts[child_id] = child
            children.append(child)

        return children

    def _evaluate_thought(self, thought_content: str, goal: str) -> float:
        """Evaluate the quality of a thought.

        In a real implementation, this would use an LLM or
        learned model to score thoughts.
        """
        score = 0.5  # Base score

        # Keyword-based scoring
        content_lower = thought_content.lower()
        goal_lower = goal.lower()

        # Boost for relevant keywords
        relevant_keywords = ["analyze", "plan", "verify", "test"]
        for kw in relevant_keywords:
            if kw in content_lower:
                score += 0.1

        # Boost for goal relevance
        goal_words = set(goal_lower.split())
        content_words = set(content_lower.split())
        overlap = len(goal_words & content_words)
        score += overlap * 0.05

        # Penalize very generic steps
        generic_words = ["do", "make", "process", "handle"]
        if any(gw in content_lower for gw in generic_words):
            score -= 0.1

        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]

    def _thoughts_to_subtasks(self, thoughts: List[Thought]) -> List[SubTask]:
        """Convert a thought path to subtasks."""
        subtasks: List[SubTask] = []

        # Skip root thoughts (approaches), use actual steps
        step_thoughts = [t for t in thoughts if t.depth > 0]

        prev_id = None
        for i, thought in enumerate(step_thoughts, 1):
            task_id = f"tot_step_{i}"

            subtask = SubTask(
                id=task_id,
                description=thought.content,
                priority=TaskPriority.HIGH if thought.score > 0.7 else TaskPriority.NORMAL,
                dependencies=[prev_id] if prev_id else [],
                estimated_effort=int(thought.score * 10),
                context={
                    "thought_id": thought.id,
                    "thought_score": thought.score,
                    "reasoning": thought.reasoning,
                },
                acceptance_criteria=[f"Successfully: {thought.content}"],
            )
            subtasks.append(subtask)
            prev_id = task_id

        return subtasks

    def _fallback_decomposition(self, goal: str) -> DecompositionResult:
        """Fallback to simple sequential decomposition."""
        return DecompositionResult(
            original_goal=goal,
            subtasks=[
                SubTask(
                    id="step_1",
                    description=goal,
                    priority=TaskPriority.NORMAL,
                    dependencies=[],
                    estimated_effort=5,
                )
            ],
            strategy="sequential_fallback",
            reasoning="Tree of Thoughts failed to find valid paths, falling back to sequential",
            confidence=0.4,
        )


class BeamSearchTreeOfThoughts(TreeOfThoughtsDecomposition):
    """ToT with beam search for more efficient exploration.

    Instead of expanding all thoughts, only keep the top-k at each level.
    """

    name = "tot_beam_search"

    def __init__(
        self,
        max_depth: int = 5,
        branching_factor: int = 3,
        evaluation_threshold: float = 0.3,
        beam_width: int = 2,
    ):
        super().__init__(max_depth, branching_factor, evaluation_threshold)
        self.beam_width = beam_width

    def _expand_thought(
        self,
        thought: Thought,
        goal: str,
        context: Dict[str, Any],
    ) -> List[ReasoningPath]:
        """Expand with beam search pruning."""
        paths = super()._expand_thought(thought, goal, context)

        # Prune to top beam_width paths at each depth
        if len(paths) > self.beam_width:
            paths.sort(key=lambda p: p.total_score, reverse=True)
            return paths[: self.beam_width]

        return paths
