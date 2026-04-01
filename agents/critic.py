import inspect
from typing import List
import json
from core.file_tools import _aura_safe_loads
from core.logging_utils import log_json
from pydantic import ValidationError

try:
    from agents.schemas import CriticOutput, CriticIssue, MutationValidationOutput
    from agents.prompt_manager import render_prompt, get_cached_prompt_stats
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False


class CriticAgent:
    """
    The CriticAgent evaluates plans and code using Chain-of-Thought reasoning
    and structured outputs for consistent, actionable feedback.
    
    Uses role-based system prompts (Principal Engineer) and prompt caching.
    """
    
    capabilities = ["critique", "review", "validation", "analysis"]

    def __init__(self, brain, model):
        self.brain = brain
        self.model = model
        self.use_structured = SCHEMAS_AVAILABLE

    def _respond(self, route_key: str, prompt: str) -> str:
        try:
            inspect.getattr_static(self.model, "respond_for_role")
        except AttributeError:
            return self.model.respond(prompt)
        responder = getattr(self.model, "respond_for_role", None)
        if callable(responder):
            return responder(route_key, prompt)
        return self.model.respond(prompt)

    def critique_plan(self, task: str, plan: List[str]) -> dict:
        """Critiques a plan with structured output and CoT reasoning."""
        plan_text = "\n".join(plan)
        memory_text = "\n".join(self.brain.recall_with_budget(max_tokens=1500))
        
        if self.use_structured:
            return self._critique_plan_structured(task, plan_text, memory_text)
        else:
            return self._critique_plan_legacy(task, plan_text, memory_text)

    def _critique_plan_structured(self, task: str, plan_text: str, memory_text: str) -> dict:
        """Structured critique with CoT and role-based prompt."""
        prompt = render_prompt(
            template_name="critic",
            role="critic",
            params={
                "target_type": "plan",
                "task": task,
                "target_content": f"Plan to critique:\n{plan_text}",
                "memory": memory_text
            }
        )
        
        response = self._respond("critique", prompt)
        self.brain.remember(f"Structured critique of plan: {task[:50]}...")
        
        try:
            parsed = _aura_safe_loads(response, "critic_plan_structured")
            critic_output = CriticOutput(**parsed)
            
            log_json("INFO", "critic_cot_reasoning", details={
                "task": task[:50],
                "assessment": critic_output.overall_assessment,
                "confidence": critic_output.confidence,
                "issue_count": len(critic_output.issues),
                "positive_count": len(critic_output.positive_aspects)
            })
            
            critical = [i for i in critic_output.issues if i.severity == "critical"]
            major = [i for i in critic_output.issues if i.severity == "major"]
            
            return {
                "structured_output": critic_output.dict(),
                "assessment": critic_output.overall_assessment,
                "confidence": critic_output.confidence,
                "summary": critic_output.summary,
                "feedback_text": self._format_feedback_text(critic_output),
                "requires_changes": len(critical) > 0 or len(major) > 0 or 
                                   critic_output.overall_assessment in ["request_changes", "reject"],
                "critical_issues": len(critical),
                "reasoning": {
                    "initial_assessment": critic_output.initial_assessment,
                    "completeness_check": critic_output.completeness_check,
                    "feasibility_analysis": critic_output.feasibility_analysis,
                    "risk_identification": critic_output.risk_identification
                }
            }
            
        except (json.JSONDecodeError, ValidationError) as e:
            log_json("WARN", "critic_structured_parse_failed", details={"error": str(e)})
            feedback = self._critique_plan_legacy(task, plan_text, memory_text)
            return {"feedback_text": feedback, "structured_output": None}
        except Exception as e:
            log_json("ERROR", "critic_structured_error", details={"error": str(e)})
            return {"feedback_text": f"Critique error: {e}", "structured_output": None}

    def _format_feedback_text(self, output: CriticOutput) -> str:
        """Convert structured critique to readable text."""
        lines = [f"Assessment: {output.overall_assessment.upper()} (confidence: {output.confidence:.2f})"]
        lines.append(f"\nSummary: {output.summary}\n")
        
        if output.issues:
            lines.append("Issues Found:")
            for issue in output.issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.category}: {issue.description}")
                lines.append(f"    → {issue.recommendation}")
        
        if output.positive_aspects:
            lines.append("\nPositive Aspects:")
            for aspect in output.positive_aspects:
                lines.append(f"  ✓ {aspect}")
        
        return "\n".join(lines)

    def _critique_plan_legacy(self, task: str, plan_text: str, memory_text: str) -> str:
        """Legacy critique method."""
        prompt = f"""Evaluate this plan.

Task:
{task}

Plan:
{plan_text}

Previous memory:
{memory_text}

Evaluate for completeness, clarity, feasibility, and alignment."""

        response = self._respond("critique", prompt)
        self.brain.remember(f"Legacy critique of plan: {task[:50]}...")
        return response

    def critique_code(self, task: str, code: str, requirements: str = "") -> dict:
        """Critiques code with structured output and CoT reasoning."""
        memory_text = "\n".join(self.brain.recall_with_budget(max_tokens=1500))
        
        if self.use_structured:
            return self._critique_code_structured(task, code, requirements, memory_text)
        else:
            return self._critique_code_legacy(task, code, requirements, memory_text)

    def _critique_code_structured(self, task: str, code: str, requirements: str, memory_text: str) -> dict:
        """Structured code critique with CoT."""
        target_content = f"""Code to critique:
```python
{code}
```

Requirements:
{requirements if requirements else "No specific requirements."}"""

        prompt = render_prompt(
            template_name="critic",
            role="critic",
            params={
                "target_type": "code",
                "task": task,
                "target_content": target_content,
                "memory": memory_text
            }
        )
        
        response = self._respond("critique", prompt)
        self.brain.remember(f"Structured code critique: {task[:50]}...")
        
        try:
            parsed = _aura_safe_loads(response, "critic_code_structured")
            critic_output = CriticOutput(**parsed)
            
            security_issues = [i for i in critic_output.issues if i.category == "safety"]
            
            return {
                "structured_output": critic_output.dict(),
                "assessment": critic_output.overall_assessment,
                "confidence": critic_output.confidence,
                "summary": critic_output.summary,
                "feedback_text": self._format_feedback_text(critic_output),
                "security_concerns": len(security_issues) > 0,
                "requires_changes": critic_output.overall_assessment in ["request_changes", "reject"],
                "reasoning": {
                    "initial_assessment": critic_output.initial_assessment,
                    "completeness_check": critic_output.completeness_check,
                    "feasibility_analysis": critic_output.feasibility_analysis,
                    "risk_identification": critic_output.risk_identification
                }
            }
            
        except Exception as e:
            log_json("WARN", "critic_code_structured_failed", details={"error": str(e)})
            feedback = self._critique_code_legacy(task, code, requirements, memory_text)
            return {"feedback_text": feedback, "structured_output": None}

    def _critique_code_legacy(self, task: str, code: str, requirements: str, memory_text: str) -> str:
        """Legacy code critique."""
        prompt = f"""Evaluate this code.

Task:
{task}

Code:
```python
{code}
```

Requirements:
{requirements if requirements else "None"}

Previous memory:
{memory_text}

Evaluate for correctness, efficiency, readability, and adherence."""

        response = self._respond("critique", prompt)
        self.brain.remember(f"Legacy code critique: {task[:50]}...")
        return response

    def validate_mutation(self, mutation_proposal: str) -> dict:
        """Validates a mutation with structured output and CoT reasoning."""
        if self.use_structured:
            return self._validate_mutation_structured(mutation_proposal)
        else:
            return self._validate_mutation_legacy(mutation_proposal)

    def _validate_mutation_structured(self, mutation_proposal: str) -> dict:
        """Structured mutation validation with CoT."""
        prompt = f"""{render_prompt('critic', 'critic', {})}

Validate this system mutation:

{mutation_proposal}

Think step-by-step:
1. Impact Analysis: Analyze potential impact
2. Safety Assessment: Assess safety concerns  
3. Effectiveness Evaluation: Evaluate likelihood of success

Then provide JSON:
{{
    "impact_analysis": "your analysis",
    "safety_assessment": "your safety assessment",
    "effectiveness_evaluation": "your evaluation",
    "decision": "APPROVED|REJECTED|NEEDS_REVISION",
    "confidence_score": 0.0-1.0,
    "impact_assessment": "summary of impacts",
    "reasoning": "detailed reasoning",
    "recommendations": "optional revision recommendations"
}}"""

        response = self._respond("analysis", prompt)
        self.brain.remember(f"Structured mutation validation: {mutation_proposal[:50]}...")
        
        try:
            parsed = _aura_safe_loads(response, "critic_mutation_structured")
            validation = MutationValidationOutput(**parsed)
            
            log_json("INFO", "critic_mutation_validation", details={
                "decision": validation.decision,
                "confidence": validation.confidence_score
            })
            
            return {
                "structured_output": validation.dict(),
                "decision": validation.decision,
                "confidence": validation.confidence_score,
                "approved": validation.decision == "APPROVED",
                "impact": validation.impact_assessment,
                "reasoning": validation.reasoning,
                "recommendations": validation.recommendations
            }
            
        except Exception as e:
            log_json("WARN", "critic_mutation_structured_failed", details={"error": str(e)})
            return self._validate_mutation_legacy(mutation_proposal)

    def _validate_mutation_legacy(self, mutation_proposal: str) -> dict:
        """Legacy mutation validation."""
        prompt = f"""Validate this mutation:

{mutation_proposal}

Respond with JSON:
{{"decision": "APPROVED|REJECTED", "confidence_score": "0.0-1.0", "impact_assessment": "...", "reasoning": "..."}}"""

        response = self._respond("analysis", prompt)
        self.brain.remember(f"Legacy mutation validation: {mutation_proposal[:50]}...")
        
        try:
            parsed = _aura_safe_loads(response, "critic_mutation_legacy")
            return {
                "decision": parsed.get("decision", "REJECTED"),
                "confidence": float(parsed.get("confidence_score", 0)),
                "approved": parsed.get("decision") == "APPROVED",
                "impact": parsed.get("impact_assessment", ""),
                "reasoning": parsed.get("reasoning", ""),
                "structured_output": None
            }
        except Exception as e:
            log_json("ERROR", "critic_mutation_legacy_failed", details={"error": str(e)})
            return {
                "decision": "REJECTED",
                "confidence": 0.0,
                "approved": False,
                "impact": "Parse error",
                "reasoning": str(e),
                "structured_output": None
            }

    def get_cache_stats(self) -> dict:
        """Get prompt cache statistics."""
        if SCHEMAS_AVAILABLE:
            return get_cached_prompt_stats()
        return {"error": "Prompt manager not available"}
