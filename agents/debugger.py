import inspect
import json
from agents.schemas import DebugStrategy
from core.schema import parse_llm_to_model
from core.logging_utils import log_json

class DebuggerAgent:
    """
    The DebuggerAgent is responsible for analyzing error messages and providing
    a diagnosis and fix strategy. It interacts with the LLM to get insights
    into potential issues and how to resolve them.
    """
    def __init__(self, brain, model):
        self.brain = brain
        self.model = model

    def _respond(self, prompt: str) -> str:
        try:
            inspect.getattr_static(self.model, "respond_for_role")
        except AttributeError:
            return self.model.respond(prompt)
        responder = getattr(self.model, "respond_for_role", None)
        if callable(responder):
            return responder("analysis", prompt)
        return self.model.respond(prompt)

    def diagnose(self, error_message: str, current_goal: str = "", context: str = "", improve_plan: str = "", implement_details: dict = None) -> dict:
        """Analyze an error and return a typed diagnosis as a plain dict.

        Internally parses the LLM response into a :class:`~agents.schemas.DebugStrategy`
        Pydantic model (schema-enforced), then returns ``model.model_dump()`` so
        all existing callers that access ``result["fix_strategy"]`` continue to work.

        Args:
            error_message: The specific error message encountered.
            current_goal: The current goal being pursued when the error occurred.
            context: Relevant code context or other information for debugging.
            improve_plan: The previous IMPROVE plan, if any.
            implement_details: Details of the implementation step that led to the error.

        Returns:
            dict with keys: ``summary``, ``diagnosis``, ``fix_strategy``, ``severity``.
            The dict is the serialised form of a validated :class:`DebugStrategy` object.
        """
        if implement_details is None:
            implement_details = {}

        implement_str = json.dumps(implement_details, indent=2) if implement_details else "No implementation details available."

        diagnosis_prompt = f"""You are an expert debugging AI. Your task is to analyze the provided error message and context, then offer a diagnosis and a clear strategy for fixing the issue.

Error Message:
{error_message}

Current Goal:
{current_goal}

Relevant Code Context (if available):
{context}

Previous IMPROVE plan (if any):
{improve_plan}

Implementation details from previous step (if any):
{implement_str}

Provide your response as a JSON object with the following keys:
- "summary": A concise summary of the error.
- "diagnosis": A detailed explanation of the probable cause.
- "fix_strategy": A step-by-step plan or code suggestion to resolve the issue.
- "severity": one of "CRITICAL", "HIGH", "MEDIUM", "LOW"
"""
        raw_llm_response = ""
        try:
            raw_llm_response = self._respond(diagnosis_prompt)
            strategy = parse_llm_to_model(raw_llm_response, DebugStrategy)
            return strategy.model_dump()
        except Exception as e:
            log_json("ERROR", "debugger_llm_diagnosis_failed", details={"error": str(e), "raw_response_snippet": raw_llm_response[:200] if raw_llm_response else "N/A"})
            return {
                "summary": "LLM diagnosis failed.",
                "diagnosis": f"Failed to get LLM diagnosis or parse response: {e}",
                "fix_strategy": "Manually inspect the error message and context.",
                "severity": "CRITICAL",
            }

    def run(self, input_data: dict) -> dict:
        """Uniform execution interface for the orchestrator loop."""
        error_message = input_data.get("error_message", "")
        current_goal = input_data.get("current_goal", "")
        context = input_data.get("context", "")
        improve_plan = input_data.get("improve_plan", "")
        implement_details = input_data.get("implement_details", {})

        result = self.diagnose(
            error_message=error_message,
            current_goal=current_goal,
            context=context,
            improve_plan=improve_plan,
            implement_details=implement_details
        )
        return {
            "status": "success",
            "diagnosis": result
        }
