"""Bridge between Creative Innovation System and AURA CLI.

Transforms creative ideas into AURA-executable goals and manages
the full implementation pipeline.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.logging_utils import log_json
from core.exceptions import (
    AURAError,
)


@dataclass
class CreativeIdea:
    """A creative idea ready for implementation."""
    content: str
    requirements: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    technique: str = "unknown"  # RPE, SCAMPER, SixHats, etc.
    confidence: float = 0.5
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_goal(self) -> str:
        """Convert idea to AURA goal format."""
        req_section = "\n".join(f"- {r}" for r in self.requirements) or "- Implement the solution"
        criteria_section = "\n".join(f"- {c}" for c in self.validation_criteria) or "- Solution works correctly"
        
        return f"""## Task
{self.content}

## Requirements
{req_section}

## Success Criteria
{criteria_section}

## Context
- Creative Technique: {self.technique}
- Confidence: {self.confidence:.2f}
- Domain: {self.domain}
"""


@dataclass
class ImplementationResult:
    """Result of implementing a creative idea."""
    idea: CreativeIdea
    success: bool
    files_changed: List[str] = field(default_factory=list)
    tests_passed: bool = False
    cycles_used: int = 0
    execution_time_seconds: float = 0.0
    reflection: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "idea": {
                "content": self.idea.content,
                "technique": self.idea.technique,
                "confidence": self.idea.confidence,
            },
            "success": self.success,
            "files_changed": self.files_changed,
            "tests_passed": self.tests_passed,
            "cycles_used": self.cycles_used,
            "execution_time_seconds": self.execution_time_seconds,
            "reflection": self.reflection,
            "errors": self.errors,
        }


class CreativeBridgeError(AURAError):
    """Error in creative bridge operations."""
    pass


class CreativeImplementationBridge:
    """Bridge creative ideas to AURA implementation.
    
    Example:
        >>> bridge = CreativeImplementationBridge(orchestrator)
        >>> idea = CreativeIdea(
        ...     content="Add caching to API client",
        ...     requirements=["Use Redis", "TTL 5 minutes"],
        ...     technique="SCAMPER"
        ... )
        >>> result = await bridge.implement(idea)
        >>> print(f"Success: {result.success}, Files: {result.files_changed}")
    """
    
    def __init__(
        self,
        orchestrator: Any,
        max_cycles: int = 5,
        enable_critique: bool = True,
    ):
        self.orchestrator = orchestrator
        self.max_cycles = max_cycles
        self.enable_critique = enable_critique
        self._implementation_history: List[ImplementationResult] = []
    
    async def implement(self, idea: CreativeIdea) -> ImplementationResult:
        """Implement a creative idea through AURA orchestration.
        
        Args:
            idea: The creative idea to implement
            
        Returns:
            ImplementationResult with full execution details
            
        Raises:
            CreativeBridgeError: If implementation fails catastrophically
        """
        start_time = datetime.now()
        log_json(
            "INFO",
            "creative_implementation_start",
            {"technique": idea.technique, "domain": idea.domain}
        )
        
        try:
            # Convert idea to goal
            goal = idea.to_goal()
            
            # Run AURA orchestration
            result = await self._run_orchestration(goal, idea)
            
            # Build result
            execution_time = (datetime.now() - start_time).total_seconds()
            impl_result = ImplementationResult(
                idea=idea,
                success=result.get("success", False),
                files_changed=result.get("files_changed", []),
                tests_passed=result.get("tests_passed", False),
                cycles_used=result.get("cycles", 0),
                execution_time_seconds=execution_time,
                reflection=result.get("reflection", {}),
                errors=result.get("errors", []),
            )
            
            self._implementation_history.append(impl_result)
            
            log_json(
                "INFO",
                "creative_implementation_complete",
                {
                    "success": impl_result.success,
                    "files_changed": len(impl_result.files_changed),
                    "cycles": impl_result.cycles_used,
                    "execution_time": execution_time,
                }
            )
            
            return impl_result
            
        except Exception as e:
            log_json("ERROR", "creative_implementation_failed", {"error": str(e)})
            raise CreativeBridgeError(f"Implementation failed: {e}") from e
    
    async def implement_batch(
        self,
        ideas: List[CreativeIdea],
        max_parallel: int = 3,
    ) -> List[ImplementationResult]:
        """Implement multiple ideas in parallel.
        
        Args:
            ideas: List of creative ideas
            max_parallel: Maximum concurrent implementations
            
        Returns:
            List of implementation results
        """
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def _impl(idea: CreativeIdea) -> ImplementationResult:
            async with semaphore:
                return await self.implement(idea)
        
        tasks = [_impl(idea) for idea in ideas]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_orchestration(
        self,
        goal: str,
        idea: CreativeIdea,
    ) -> Dict[str, Any]:
        """Run AURA orchestration loop.
        
        Args:
            goal: Formatted goal string
            idea: Original idea for context
            
        Returns:
            Orchestration result dictionary
        """
        # Prepare context with creative metadata
        context = {
            "creative_technique": idea.technique,
            "creative_confidence": idea.confidence,
            "creative_domain": idea.domain,
            "enable_critique": self.enable_critique,
        }
        
        # Run orchestrator
        if hasattr(self.orchestrator, 'run_loop'):
            result = await self.orchestrator.run_loop(
                goal=goal,
                max_cycles=self.max_cycles,
                context=context,
            )
        elif hasattr(self.orchestrator, 'run'):
            # Synchronous fallback
            result = self.orchestrator.run(
                goal=goal,
                max_cycles=self.max_cycles,
                context=context,
            )
        else:
            raise CreativeBridgeError(
                "Orchestrator must have run_loop() or run() method"
            )
        
        return self._normalize_result(result)
    
    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """Normalize orchestrator result to standard format."""
        if isinstance(result, dict):
            return {
                "success": result.get("success", result.get("verify_success", False)),
                "files_changed": result.get("files_changed", result.get("changes", [])),
                "tests_passed": result.get("tests_passed", result.get("verify_success", False)),
                "cycles": result.get("cycles", result.get("cycles_used", 0)),
                "reflection": result.get("reflection", {}),
                "errors": result.get("errors", []),
            }
        
        # Handle object results
        return {
            "success": getattr(result, "success", False),
            "files_changed": getattr(result, "files_changed", []),
            "tests_passed": getattr(result, "tests_passed", False),
            "cycles": getattr(result, "cycles", 0),
            "reflection": getattr(result, "reflection", {}),
            "errors": [],
        }
    
    def get_implementation_history(
        self,
        technique: Optional[str] = None,
        success_only: bool = False,
    ) -> List[ImplementationResult]:
        """Get history of implementations.
        
        Args:
            technique: Filter by technique
            success_only: Only successful implementations
            
        Returns:
            Filtered implementation history
        """
        results = self._implementation_history
        
        if technique:
            results = [r for r in results if r.idea.technique == technique]
        
        if success_only:
            results = [r for r in results if r.success]
        
        return results
    
    def get_success_rate(self, technique: Optional[str] = None) -> float:
        """Calculate success rate for implementations.
        
        Args:
            technique: Filter by technique, or overall if None
            
        Returns:
            Success rate as float (0.0-1.0)
        """
        history = self.get_implementation_history(technique=technique)
        if not history:
            return 0.0
        
        successful = sum(1 for r in history if r.success)
        return successful / len(history)


def create_bridge(
    orchestrator: Any,
    **kwargs,
) -> CreativeImplementationBridge:
    """Factory function to create a CreativeImplementationBridge.
    
    Args:
        orchestrator: AURA orchestrator instance
        **kwargs: Additional config options
        
    Returns:
        Configured CreativeImplementationBridge
    """
    return CreativeImplementationBridge(
        orchestrator=orchestrator,
        **kwargs,
    )
