"""Prompt optimization based on feedback."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .feedback import ExecutionOutcome, ExecutionStatus


@dataclass
class PromptTemplate:
    """A prompt template with optimization metadata."""
    name: str
    template: str
    version: int = 1
    success_count: int = 0
    total_count: int = 0
    avg_quality: float = 0.0
    variations: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count


class PromptOptimizer:
    """Optimize prompts based on execution feedback."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        self.templates["plan"] = PromptTemplate(
            name="plan",
            template="""Given the goal: {goal}

Context:
{context}

Generate a step-by-step plan to accomplish this goal.
""",
        )
        
        self.templates["code"] = PromptTemplate(
            name="code",
            template="""Generate code to: {goal}

Requirements:
- Follow best practices
- Include error handling
- Add appropriate documentation

Context:
{context}
""",
        )
    
    def optimize_prompt(
        self,
        template_name: str,
        outcomes: List[ExecutionOutcome],
    ) -> Optional[str]:
        """Optimize a prompt template based on outcomes."""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        
        # Filter outcomes for this template type
        relevant = [o for o in outcomes if template_name in o.metadata.get("template", "")]
        
        if len(relevant) < 5:
            return None  # Not enough data
        
        successes = [o for o in relevant if o.status == ExecutionStatus.SUCCESS]
        failures = [o for o in relevant if o.status == ExecutionStatus.FAILURE]
        
        if len(successes) < 3:
            return None
        
        # Analyze what makes successful prompts work
        improvements = self._analyze_success_patterns(successes, failures)
        
        if not improvements:
            return None
        
        # Apply improvements to template
        optimized = self._apply_improvements(template.template, improvements)
        
        # Store as new variation
        template.variations.append(optimized)
        template.version += 1
        
        return optimized
    
    def _analyze_success_patterns(
        self,
        successes: List[ExecutionOutcome],
        failures: List[ExecutionOutcome],
    ) -> Dict[str, Any]:
        """Analyze patterns in successful vs failed prompts."""
        improvements = {}
        
        # Check for common failure reasons
        failure_reasons = defaultdict(int)
        for outcome in failures:
            if outcome.error_message:
                # Categorize error
                if "too long" in outcome.error_message.lower():
                    failure_reasons["length"] += 1
                elif "ambiguous" in outcome.error_message.lower():
                    failure_reasons["clarity"] += 1
                elif "context" in outcome.error_message.lower():
                    failure_reasons["context"] += 1
                else:
                    failure_reasons["other"] += 1
        
        # Suggest improvements based on failure patterns
        if failure_reasons.get("length", 0) >= 2:
            improvements["reduce_length"] = True
        
        if failure_reasons.get("clarity", 0) >= 2:
            improvements["add_clarity"] = True
        
        if failure_reasons.get("context", 0) >= 2:
            improvements["improve_context"] = True
        
        # Analyze successful patterns
        success_keywords = self._extract_keywords_from_outcomes(successes)
        failure_keywords = self._extract_keywords_from_outcomes(failures)
        
        distinguishing = success_keywords - failure_keywords
        if distinguishing:
            improvements["emphasize_keywords"] = list(distinguishing)[:5]
        
        return improvements
    
    def _apply_improvements(
        self,
        template: str,
        improvements: Dict[str, Any],
    ) -> str:
        """Apply improvements to a template."""
        optimized = template
        
        if improvements.get("reduce_length"):
            # Add instruction to be concise
            optimized = self._add_instruction(
                optimized,
                "Be concise and focus on essential steps."
            )
        
        if improvements.get("add_clarity"):
            # Add clarity instruction
            optimized = self._add_instruction(
                optimized,
                "Be specific and avoid ambiguity."
            )
        
        if improvements.get("improve_context"):
            # Add context formatting instruction
            optimized = self._add_instruction(
                optimized,
                "Use the provided context to inform your response."
            )
        
        if "emphasize_keywords" in improvements:
            # Add keywords to emphasize
            keywords = ", ".join(improvements["emphasize_keywords"])
            optimized = self._add_instruction(
                optimized,
                f"Focus on: {keywords}"
            )
        
        return optimized
    
    def _add_instruction(self, template: str, instruction: str) -> str:
        """Add an instruction to a template."""
        # Find a good place to add the instruction
        lines = template.split('\n')
        
        # Add before the main content or at the end
        if len(lines) > 3:
            # Insert before the last section
            insert_pos = len(lines) - 2
            lines.insert(insert_pos, f"\n{instruction}")
        else:
            lines.append(f"\n{instruction}")
        
        return '\n'.join(lines)
    
    def _extract_keywords_from_outcomes(
        self,
        outcomes: List[ExecutionOutcome],
    ) -> set:
        """Extract keywords from outcome goals."""
        keywords = set()
        for outcome in outcomes:
            words = outcome.goal.lower().split()
            keywords.update(w for w in words if len(w) >= 4)
        return keywords
    
    def get_best_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get the best performing version of a template."""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        
        # If we have variations, find the best one
        if template.variations:
            # For now, return the main template
            # In future, could track per-variation performance
            pass
        
        return template
    
    def record_template_performance(
        self,
        template_name: str,
        outcome: ExecutionOutcome,
    ):
        """Record performance of a template."""
        if template_name not in self.templates:
            return
        
        template = self.templates[template_name]
        template.total_count += 1
        
        if outcome.status == ExecutionStatus.SUCCESS:
            template.success_count += 1
        
        # Update running average
        template.avg_quality = (
            (template.avg_quality * (template.total_count - 1) + outcome.output_quality)
            / template.total_count
        )
    
    def suggest_template_for_goal(self, goal: str) -> Optional[str]:
        """Suggest the best template for a given goal."""
        goal_lower = goal.lower()
        
        # Simple keyword matching
        if any(kw in goal_lower for kw in ["plan", "steps", "approach"]):
            return "plan"
        
        if any(kw in goal_lower for kw in ["code", "implement", "function", "class"]):
            return "code"
        
        # Default to most successful template
        best = max(
            self.templates.values(),
            key=lambda t: t.success_rate,
            default=None
        )
        
        return best.name if best else None
    
    def export_templates(self) -> Dict[str, Any]:
        """Export all templates as dictionary."""
        return {
            name: {
                "name": t.name,
                "version": t.version,
                "success_rate": t.success_rate,
                "avg_quality": t.avg_quality,
                "total_count": t.total_count,
            }
            for name, t in self.templates.items()
        }
