import inspect
from typing import List, Union
import json
from core.file_tools import _aura_safe_loads
from core.logging_utils import log_json
from pydantic import ValidationError

try:
    from agents.schemas import PlannerOutput
    from agents.prompt_manager import render_prompt, get_cached_prompt_stats
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False


class PlannerAgent:
    """
    The PlannerAgent generates and updates plans using Chain-of-Thought reasoning
    and structured outputs for improved reliability and transparency.
    
    Uses role-based system prompts (Senior Software Architect) and prompt caching
    for efficient token usage.
    """

    capabilities = ["planning", "decomposition", "design", "tree_of_thought", "strategy"]

    def __init__(self, brain, model, vector_store=None):
        self.brain = brain
        self.model = model
        self.vector_store = vector_store
        self.use_structured = SCHEMAS_AVAILABLE

    def run(self, input_data: dict) -> dict:
        """Standard agent interface with structured output support."""
        goal = input_data.get("goal", "")
        mem = input_data.get("memory_snapshot", "")
        sim = input_data.get("similar_past_problems", "")
        weak = input_data.get("known_weaknesses", "")
        backfill_ctx = input_data.get("backfill_context", [])

        hints: list = []
        if self.vector_store and goal:
            try:
                hints = self.vector_store.query(goal, top_k=3) or []
            except Exception as exc:
                log_json("WARN", "planner_vector_hints_failed", details={"error": str(exc)})
                hints = []
            log_json("INFO", "planner_vector_hints", details={"hint_count": len(hints), "goal": goal})

        result = self.plan(goal, mem, sim, weak, backfill_context=backfill_ctx, hints=hints)

        if isinstance(result, dict) and "plan" in result:
            return result
        return {"steps": result}

    def _respond(self, prompt: str) -> str:
        try:
            inspect.getattr_static(self.model, "respond_for_role")
        except AttributeError:
            return self.model.respond(prompt)
        responder = getattr(self.model, "respond_for_role", None)
        if callable(responder):
            return responder("planning", prompt)
        return self.model.respond(prompt)

    def plan(self, goal: str, memory_snapshot: str, similar_past_problems: str, 
             known_weaknesses: str, backfill_context: list = None,
             hints: list = None) -> Union[List[str], dict]:
        """Generate a detailed plan with Chain-of-Thought reasoning and structured output."""
        hints = hints or []
        backfill_instr = ""
        if backfill_context:
            backfill_instr = "CRITICAL: Modules with LOW/ZERO test coverage:\n"
            for item in backfill_context:
                pct = item.get("coverage_pct", item.get("coverage", 0.0))
                backfill_instr += f"- {item['file']} ({pct}% coverage)\n"
            backfill_instr += "\nPRIORITIZE with 'Test Backfill' steps at BEGINNING."

        if self.use_structured:
            return self._plan_structured(goal, memory_snapshot, similar_past_problems, 
                                         known_weaknesses, backfill_instr, hints)
        else:
            return self._plan_legacy(goal, memory_snapshot, similar_past_problems, 
                                     known_weaknesses, backfill_instr, hints)

    def _plan_structured(self, goal: str, memory: str, similar: str, 
                         weakness: str, backfill_instr: str,
                         hints: list = None) -> dict:
        """Generate plan using structured output with CoT reasoning and role-based prompt."""
        hints = hints or []
        past_reflections_section = ""
        if hints:
            items = "\n".join(f"- {h}" for h in hints)
            past_reflections_section = f"\n## Past Reflections\n{items}\n"

        # Use cached prompt with role-based system context
        prompt = render_prompt(
            template_name="planner",
            role="planner",
            params={
                "goal": goal,
                "memory": memory,
                "similar": similar,
                "weakness": weakness,
                "backfill_instr": backfill_instr
            }
        )

        if past_reflections_section:
            prompt = past_reflections_section + prompt
        
        response = self._respond(prompt)
        self.brain.remember(f"Structured plan for: {goal[:50]}...")
        
        try:
            parsed = _aura_safe_loads(response, "planner_structured_response")
            planner_output = PlannerOutput(**parsed)
            
            # Log CoT reasoning for observability
            log_json("INFO", "planner_cot_reasoning", details={
                "analysis": planner_output.analysis[:200],
                "approach": planner_output.approach[:200],
                "risk_assessment": planner_output.risk_assessment[:200],
                "confidence": planner_output.confidence,
                "complexity": planner_output.estimated_complexity
            })
            
            # Convert PlanStep objects to step strings for backward compatibility
            steps = []
            for step in planner_output.plan:
                step_str = f"Step {step.step_number}: {step.description}"
                if step.target_file:
                    step_str += f" [{step.target_file}]"
                steps.append(step_str)
            
            return {
                "steps": steps,
                "structured_output": planner_output.dict(),
                "confidence": planner_output.confidence,
                "complexity": planner_output.estimated_complexity,
                "reasoning": {
                    "analysis": planner_output.analysis,
                    "gap_assessment": planner_output.gap_assessment,
                    "approach": planner_output.approach,
                    "risk_assessment": planner_output.risk_assessment
                }
            }
            
        except (json.JSONDecodeError, ValidationError, TypeError, KeyError) as e:
            log_json("WARN", "planner_structured_parse_failed", details={
                "error": str(e),
                "response_snippet": response[:200]
            })
            # Fallback to legacy format
            return self._parse_legacy_response(response, goal)
        except Exception as e:
            log_json("ERROR", "planner_structured_unexpected_error", details={"error": str(e)})
            return [f"ERROR: Plan generation failed: {e}"]

    def _plan_legacy(self, goal: str, memory: str, similar: str, 
                     weakness: str, backfill_instr: str,
                     hints: list = None) -> List[str]:
        """Fallback legacy planning method."""
        hints = hints or []
        past_reflections_section = ""
        if hints:
            items = "\n".join(f"- {h}" for h in hints)
            past_reflections_section = f"## Past Reflections\n{items}\n\n"

        prompt = f"""{past_reflections_section}You are AURA — an autonomous recursive engineering system.

Current Goal:
{goal}

System State:
- Memory: {memory}
- Similar Past Problems: {similar}
- Known Weaknesses: {weakness}

{backfill_instr}

Provide response as JSON array of strings: ["Step 1: ...", "Step 2: ..."]"""
        
        response = self._respond(prompt)
        return self._parse_legacy_response(response, goal)

    def _parse_legacy_response(self, response: str, goal: str) -> List[str]:
        """Parse legacy JSON array response."""
        self.brain.remember(f"Legacy plan for: {goal[:50]}...")
        try:
            plan = _aura_safe_loads(response, "planner_legacy_response")
            if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                return plan
            else:
                log_json("ERROR", "planner_legacy_invalid_format")
                return [f"ERROR: Invalid plan format. Raw: {response[:200]}"]
        except Exception as e:
            log_json("ERROR", "planner_legacy_parse_error", details={"error": str(e)})
            return [f"ERROR: Failed to parse plan: {e}"]

    def _update_plan(self, original_plan: List[str], feedback: str) -> List[str]:
        """Revises an existing plan based on feedback."""
        if isinstance(original_plan, dict) and "steps" in original_plan:
            original_plan = original_plan["steps"]
            
        prompt = f"""Revise this plan based on feedback.

Original Plan:
{json.dumps(original_plan)}

Feedback:
{feedback}

Provide revised plan as JSON array: ["Step 1: ...", "Step 2: ..."]"""

        response = self._respond(prompt)
        self.brain.remember(f"Plan revision: {feedback[:50]}...")
        
        try:
            plan = _aura_safe_loads(response, "planner_update_response")
            if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                return plan
            else:
                log_json("ERROR", "planner_update_invalid_format")
                return original_plan
        except Exception as e:
            log_json("ERROR", "planner_update_error", details={"error": str(e)})
            return original_plan

    def get_cache_stats(self) -> dict:
        """Get prompt cache statistics."""
        if SCHEMAS_AVAILABLE:
            return get_cached_prompt_stats()
        return {"error": "Prompt manager not available"}
