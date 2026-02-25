from typing import List
import json
import re # Import re
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
    """
    The PlannerAgent is responsible for generating and updating plans based on a given goal,
    system state, and feedback. It leverages the LLM to analyze problems, propose solutions,
    and decompose complex tasks into atomic steps.
    """
    def __init__(self, brain, model):
        """
        Initializes the PlannerAgent with access to the system's brain and model.

        Args:
            brain: An instance of the system's memory (Brain).
            model: An instance of the model adapter for LLM interactions.
        """
        self.brain = brain
        self.model = model

    def plan(self, goal: str, memory_snapshot: str, similar_past_problems: str, known_weaknesses: str) -> List[str]:
        """
        Generates a detailed plan based on the current goal, system memory,
        similar past problems, and known system weaknesses. The LLM is prompted
        to output the plan as a JSON array of strings, each representing a step.

        Args:
            goal (str): The current objective for which a plan is needed.
            memory_snapshot (str): A summary of the system's current memory.
            similar_past_problems (str): Information about problems similar to the current goal.
            known_weaknesses (str): Identified weaknesses of the AURA system.

        Returns:
            List[str]: A list of strings, where each string is a step in the generated plan.
                       Returns an error message within the list if parsing fails.
        """
        prompt = EVOLUTION_PROMPT.format(
            goal=goal,
            memory=memory_snapshot,
            similar=similar_past_problems,
            weakness=known_weaknesses
        )
        response = self.model.respond(prompt)
        self.brain.remember(f"Planned for goal: {goal} with raw response: {response[:100]}...")
        try:
            plan = _aura_safe_loads(response, "planner_plan_response")
            if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                return plan
            else:
                log_json("ERROR", "planner_plan_response_invalid_format", details={"response_snippet": response[:200], "parsed_type": type(plan).__name__})
                return [f"ERROR: Planner response was not a valid JSON array of strings. Raw response: {response}"]
        except json.JSONDecodeError as e:
            log_json("ERROR", "planner_plan_json_decode_error", details={"error": str(e), "response_snippet": response[:200]})
            return [f"ERROR: Failed to decode JSON from planner response: {e}. Raw response: {response}"]
        except Exception as e:
            log_json("ERROR", "planner_plan_unexpected_error", details={"error": str(e), "response_snippet": response[:200]})
            return [f"ERROR: An unexpected error occurred during plan parsing: {e}. Raw response: {response}"]

    def _update_plan(self, original_plan: List[str], feedback: str) -> List[str]:
        """
        Revises an existing plan based on new feedback. The LLM is prompted
        to return the revised plan as a JSON array of strings.

        Args:
            original_plan (List[str]): The existing plan that needs revision.
            feedback (str): Feedback received on the original plan.

        Returns:
            List[str]: A list of strings, where each string is a step in the revised plan.
                       Returns an error message within the list if parsing fails.
        """
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
        response = self.model.respond(prompt)
        self.brain.remember(f"Revised plan based on feedback: {feedback}. Raw response: {response[:100]}...")
        try:
            plan = _aura_safe_loads(response, "planner_update_plan_response")
            if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                return plan
            else:
                log_json("ERROR", "planner_update_plan_response_invalid_format", details={"response_snippet": response[:200], "parsed_type": type(plan).__name__})
                return [f"ERROR: Planner update response was not a valid JSON array of strings. Raw response: {response}"]
        except json.JSONDecodeError as e:
            log_json("ERROR", "planner_update_plan_json_decode_error", details={"error": str(e), "response_snippet": response[:200]})
            return [f"ERROR: Failed to decode JSON from planner update response: {e}. Raw response: {response}"]
        except Exception as e:
            log_json("ERROR", "planner_update_plan_unexpected_error", details={"error": str(e), "response_snippet": response[:200]})
            return [f"ERROR: An unexpected error occurred during plan update parsing: {e}. Raw response: {response}"]
