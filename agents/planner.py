import inspect
import json
from typing import List

from core.file_tools import _aura_safe_loads
from core.logging_utils import log_json


EVOLUTION_PROMPT = """
You are AURA — an autonomous recursive engineering system.

Current Goal:
{goal}

System State:
- Memory Snapshot: {memory}
- Similar Past Problems: {similar}
- Known Weaknesses: {weakness}

{backfill_instr}

You must:

1. Analyze structural gaps.
2. Propose capability upgrades.
3. Design execution steps.
4. Predict failure modes.
5. Suggest improvements to your own architecture.

Think recursively.
Optimize long-term intelligence.

Provide your response as a JSON array of strings, where each string is a planning step.
Example: ["Step 1: ...", "Step 2: ..."]
Ensure your response contains ONLY the JSON array, with no conversational text or other explanations.
"""


class PlannerAgent:
    """Generate and revise implementation plans for the orchestrator."""

    def __init__(self, brain, model):
        self.brain = brain
        self.model = model

    def run(self, input_data: dict) -> dict:
        """Standard agent interface used by direct callers/tests."""
        goal = input_data.get("goal", "")
        mem = input_data.get("memory_snapshot", "")
        sim = input_data.get("similar_past_problems", "")
        weak = input_data.get("known_weaknesses", "")
        backfill_ctx = input_data.get("backfill_context", [])

        steps = self.plan(
            goal,
            mem,
            sim,
            weak,
            backfill_context=backfill_ctx,
        )
        return {"steps": steps}

    def _respond(self, prompt: str) -> str:
        try:
            inspect.getattr_static(self.model, "respond_for_role")
        except AttributeError:
            return self.model.respond(prompt)
        responder = getattr(self.model, "respond_for_role", None)
        if callable(responder):
            return responder("planning", prompt)
        return self.model.respond(prompt)

    def _remember(self, text: str) -> None:
        remember = getattr(self.brain, "remember", None)
        if not callable(remember):
            return
        try:
            remember(text)
        except Exception as exc:  # pylint: disable=broad-except
            log_json("WARN", "planner_brain_remember_failed", details={"error": str(exc)})

    def _parse_steps(self, response: str, event_prefix: str) -> List[str]:
        try:
            plan = _aura_safe_loads(response, f"{event_prefix}_response")
            if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                return plan
            log_json(
                "ERROR",
                f"{event_prefix}_invalid_format",
                details={"response_snippet": response[:200], "parsed_type": type(plan).__name__},
            )
            return [f"ERROR: Planner response was not a valid JSON array of strings. Raw response: {response}"]
        except json.JSONDecodeError as exc:
            log_json(
                "ERROR",
                f"{event_prefix}_json_decode_error",
                details={"error": str(exc), "response_snippet": response[:200]},
            )
            return [f"ERROR: Failed to decode JSON from planner response: {exc}. Raw response: {response}"]
        except Exception as exc:  # pylint: disable=broad-except
            log_json(
                "ERROR",
                f"{event_prefix}_unexpected_error",
                details={"error": str(exc), "response_snippet": response[:200]},
            )
            return [f"ERROR: An unexpected error occurred during plan parsing: {exc}. Raw response: {response}"]

    def plan(
        self,
        goal: str,
        memory_snapshot: str,
        similar_past_problems: str,
        known_weaknesses: str,
        backfill_context: list | None = None,
    ) -> List[str]:
        """Generate a plan as a JSON array of step strings."""
        backfill_instr = ""
        if backfill_context:
            backfill_instr = (
                "CRITICAL: The following modules have LOW/ZERO test coverage "
                "and are considered HIGH RISK:\n"
            )
            for item in backfill_context:
                pct = item.get("coverage_pct", item.get("coverage", 0.0))
                backfill_instr += f"- {item['file']} ({pct}% coverage)\n"
            backfill_instr += (
                "\nPRIORITIZE these modules by adding 'Test Backfill' steps "
                "at the BEGINNING of your plan."
            )

        prompt = EVOLUTION_PROMPT.format(
            goal=goal,
            memory=memory_snapshot,
            similar=similar_past_problems,
            weakness=known_weaknesses,
            backfill_instr=backfill_instr,
        )

        try:
            response = self._respond(prompt)
        except Exception as exc:  # pylint: disable=broad-except
            log_json("ERROR", "planner_plan_model_error", details={"error": str(exc), "goal": goal})
            return [f"ERROR: Planner model failed: {exc}"]

        self._remember(f"Planned for goal: {goal} with raw response: {response[:100]}...")
        return self._parse_steps(response, "planner_plan")

    def _update_plan(self, original_plan: List[str], feedback: str) -> List[str]:
        """Revise an existing plan based on feedback."""
        prompt = f"""
You are an autonomous planning agent. You have an existing plan and have received feedback on it. Your task is to revise the plan based on the feedback.

Original Plan:
{json.dumps(original_plan)}

Feedback:
{feedback}

Provide the revised plan as a JSON array of strings, where each string is a step.
Example: ["Revised Step 1: ...", "Revised Step 2: ..."]
Ensure your response contains ONLY the JSON array, with no conversational text or other explanations.
"""
        try:
            response = self._respond(prompt)
        except Exception as exc:  # pylint: disable=broad-except
            log_json("ERROR", "planner_update_plan_model_error", details={"error": str(exc)})
            return [f"ERROR: Planner model failed while updating plan: {exc}"]

        self._remember(f"Revised plan based on feedback: {feedback}. Raw response: {response[:100]}...")
        return self._parse_steps(response, "planner_update_plan")
