"""Sequential decomposition strategy.

Breaks goals into a linear sequence of steps.
Best for: Well-defined procedures, step-by-step tasks
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, List

from core.decomposition.base import DecompositionStrategy, DecompositionResult, SubTask, TaskPriority


class SequentialDecomposition(DecompositionStrategy):
    """Decompose goals into sequential steps.

        This strategy breaks down goals into a linear sequence where
    each step depends on the previous one.

        Example:
            Goal: "Refactor the auth module"
            Steps:
            1. Analyze current auth implementation
            2. Identify code smells and issues
            3. Create refactoring plan
            4. Apply refactoring changes
            5. Run tests and verify
    """

    name = "sequential"

    def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecompositionResult:
        """Decompose goal into sequential steps.

        Uses keyword analysis to identify implied sequence.
        """
        context = context or {}

        # Parse goal for action keywords
        steps = self._extract_steps(goal)

        subtasks: List[SubTask] = []
        prev_id = None

        for i, step_desc in enumerate(steps, 1):
            task_id = f"step_{i}"

            # Determine priority based on position
            priority = TaskPriority.CRITICAL if i == 1 else (TaskPriority.HIGH if i == len(steps) else TaskPriority.NORMAL)

            subtask = SubTask(
                id=task_id,
                description=step_desc,
                priority=priority,
                dependencies=[prev_id] if prev_id else [],
                estimated_effort=self._estimate_effort(step_desc),
                acceptance_criteria=[f"Complete: {step_desc}"],
            )
            subtasks.append(subtask)
            prev_id = task_id

        return DecompositionResult(
            original_goal=goal,
            subtasks=subtasks,
            strategy=self.name,
            reasoning=f"Decomposed into {len(subtasks)} sequential steps based on action keywords",
            confidence=0.85 if len(subtasks) > 1 else 0.6,
            metadata={"extracted_steps": steps},
        )

    def _extract_steps(self, goal: str) -> List[str]:
        """Extract sequential steps from goal description."""
        # Look for numbered lists or sequential keywords
        numbered_pattern = r"(?:\d+[.)]|\([\d]\))\s*(.+)"
        numbered = re.findall(numbered_pattern, goal, re.IGNORECASE)

        if numbered:
            return [s.strip() for s in numbered]

        # Split on sequential keywords
        sequential_keywords = [
            r"\bfirst\b",
            r"\bthen\b",
            r"\bnext\b",
            r"\bafter\b",
            r"\bfinally\b",
            r"\blastly\b",
            r"\bfollowed by\b",
        ]

        parts = re.split("|".join(sequential_keywords), goal, flags=re.IGNORECASE)
        parts = [p.strip(" ,.:;") for p in parts if p.strip()]

        if len(parts) > 1:
            return parts

        # Default: single task
        return [goal]

    def _estimate_effort(self, step: str) -> int:
        """Estimate effort for a step (1-10)."""
        step_lower = step.lower()

        # Keywords indicating complexity
        high_effort = ["refactor", "rewrite", "architecture", "design", "implement"]
        medium_effort = ["update", "modify", "add", "create", "fix"]
        low_effort = ["test", "verify", "check", "review", "document"]

        if any(kw in step_lower for kw in high_effort):
            return 8
        elif any(kw in step_lower for kw in medium_effort):
            return 5
        elif any(kw in step_lower for kw in low_effort):
            return 3

        return 5  # Default
