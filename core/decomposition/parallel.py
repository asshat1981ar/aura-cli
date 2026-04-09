"""Parallel decomposition strategy.

Breaks goals into independent subtasks that can execute concurrently.
Best for: Bulk operations, independent workstreams, batch processing
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import re

from core.decomposition.base import DecompositionStrategy, DecompositionResult, SubTask, TaskPriority


class ParallelDecomposition(DecompositionStrategy):
    """Decompose goals into parallel independent subtasks.
    
    This strategy identifies independent work items that can be
    executed in parallel without dependencies.
    
    Example:
        Goal: "Update all config files in the project"
        Tasks:
        - Update package.json
        - Update pyproject.toml
        - Update setup.py
        - Update requirements.txt
    """
    
    name = "parallel"
    
    def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecompositionResult:
        """Decompose goal into parallel independent tasks."""
        context = context or {}
        
        # Extract items that can be processed in parallel
        items = self._extract_parallel_items(goal, context)
        
        subtasks: List[SubTask] = []
        for i, item in enumerate(items, 1):
            subtask = SubTask(
                id=f"task_{i}",
                description=item["description"],
                priority=TaskPriority(item.get("priority", "normal")),
                dependencies=[],  # No dependencies for parallel execution
                estimated_effort=item.get("effort", 5),
                context=item.get("context", {}),
                acceptance_criteria=item.get("criteria", [f"Complete: {item['description']}"]),
            )
            subtasks.append(subtask)
        
        # If no items extracted, treat as single task
        if not subtasks:
            subtasks.append(SubTask(
                id="task_1",
                description=goal,
                priority=TaskPriority.NORMAL,
                dependencies=[],
                estimated_effort=5,
            ))
        
        return DecompositionResult(
            original_goal=goal,
            subtasks=subtasks,
            strategy=self.name,
            reasoning=f"Decomposed into {len(subtasks)} parallel independent tasks",
            confidence=0.8 if len(subtasks) > 1 else 0.5,
            metadata={"parallel_items": len(subtasks)},
        )
    
    def _extract_parallel_items(
        self,
        goal: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract items that can be processed in parallel."""
        items: List[Dict[str, Any]] = []
        
        # Look for lists with commas or "and"
        if ',' in goal or ' and ' in goal:
            # Split on commas and 'and'
            parts = re.split(r',\s*|\s+and\s+', goal)
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
            
            for part in parts:
                items.append({
                    "description": part,
                    "priority": "normal",
                    "effort": 4,
                })
        
        # Check for "all" or "every" patterns (bulk operations)
        bulk_patterns = [
            r'(?:update|fix|refactor|test)\s+all\s+(\w+)',
            r'(?:update|fix|refactor|test)\s+every\s+(\w+)',
        ]
        
        for pattern in bulk_patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                target_type = match.group(1)
                # Would expand based on context (e.g., files of type)
                if "files" in context:
                    for file_path in context["files"]:
                        items.append({
                            "description": f"Process {target_type}: {file_path}",
                            "priority": "normal",
                            "effort": 3,
                            "context": {"file": file_path},
                        })
        
        return items
