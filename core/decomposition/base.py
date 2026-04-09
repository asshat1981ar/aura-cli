"""Base classes for decomposition strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TaskPriority(Enum):
    """Priority levels for subtasks."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class SubTask:
    """A decomposed subtask.
    
    Attributes:
        id: Unique subtask identifier
        description: Task description
        priority: Task priority
        dependencies: IDs of tasks this depends on
        estimated_effort: Estimated effort (1-10)
        context: Additional context for the task
        acceptance_criteria: How to verify completion
    """
    id: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    estimated_effort: int = 5
    context: Dict[str, Any] = field(default_factory=dict)
    acceptance_criteria: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not 1 <= self.estimated_effort <= 10:
            raise ValueError("estimated_effort must be between 1 and 10")


@dataclass
class DecompositionResult:
    """Result of goal decomposition.
    
    Attributes:
        original_goal: The original goal
        subtasks: List of decomposed subtasks
        strategy: Name of strategy used
        reasoning: Why this decomposition was chosen
        confidence: Confidence score (0-1)
        metadata: Additional metadata
    """
    original_goal: str
    subtasks: List[SubTask]
    strategy: str
    reasoning: str = ""
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_goal": self.original_goal,
            "subtasks": [
                {
                    "id": s.id,
                    "description": s.description,
                    "priority": s.priority.value,
                    "dependencies": s.dependencies,
                    "estimated_effort": s.estimated_effort,
                    "context": s.context,
                    "acceptance_criteria": s.acceptance_criteria,
                }
                for s in self.subtasks
            ],
            "strategy": self.strategy,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class DecompositionStrategy(ABC):
    """Abstract base class for decomposition strategies."""
    
    name: str = "base"
    
    @abstractmethod
    def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecompositionResult:
        """Decompose a goal into subtasks.
        
        Args:
            goal: The goal to decompose
            context: Additional context
            
        Returns:
            Decomposition result with subtasks
        """
        pass
    
    def validate(self, result: DecompositionResult) -> bool:
        """Validate a decomposition result.
        
        Args:
            result: Decomposition to validate
            
        Returns:
            True if valid
        """
        # Check for circular dependencies
        visited = set()
        
        def has_cycle(task_id: str, path: set) -> bool:
            if task_id in path:
                return True
            if task_id in visited:
                return False
            
            path.add(task_id)
            task = next((t for t in result.subtasks if t.id == task_id), None)
            if task:
                for dep_id in task.dependencies:
                    if has_cycle(dep_id, path):
                        return True
            path.remove(task_id)
            visited.add(task_id)
            return False
        
        for task in result.subtasks:
            if has_cycle(task.id, set()):
                return False
        
        # Check that all dependencies exist
        task_ids = {t.id for t in result.subtasks}
        for task in result.subtasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    return False
        
        return True
