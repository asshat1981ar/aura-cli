import json
from core.file_tools import _aura_safe_loads
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

    def diagnose(self, error_message: str, current_goal: str = "", context: str = "", improve_plan: str = "", implement_details: dict = None) -> dict:
        """
        Analyzes an error message and relevant context to provide a diagnosis
        and a strategy for fixing the issue. It prompts the LLM for a structured
        JSON response containing a summary, diagnosis, fix strategy, and severity.

        Args:
            error_message (str): The specific error message encountered.
            current_goal (str, optional): The current goal being pursued when the error occurred. Defaults to "".
            context (str, optional): Relevant code context or other information for debugging. Defaults to "".
            improve_plan (str, optional): The previous IMPROVE plan, if any. Defaults to "".
            implement_details (dict, optional): Details of the implementation step that led to the error. Defaults to None.

        Returns:
            dict: A dictionary containing the LLM's diagnosis (summary, diagnosis, fix_strategy, severity)
                  or a fallback error dictionary if LLM interaction fails.
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
- "severity": "CRITICAL", "HIGH", "MEDIUM", "LOW"
"""
        try:
            raw_llm_response = self.model.respond(diagnosis_prompt)
            parsed_diagnosis = _aura_safe_loads(raw_llm_response, "debugger_diagnosis")
            # Ensure the parsed response has expected keys and types if necessary
            if isinstance(parsed_diagnosis, dict) and all(k in parsed_diagnosis for k in ["summary", "diagnosis", "fix_strategy", "severity"]):
                return parsed_diagnosis
            else:
                log_json("ERROR", "debugger_diagnosis_invalid_format", details={"response_snippet": raw_llm_response[:200], "parsed_type": type(parsed_diagnosis).__name__})
                raise ValueError("LLM response did not contain expected diagnosis structure.")
        except Exception as e:
            log_json("ERROR", "debugger_llm_diagnosis_failed", details={"error": str(e), "raw_response_snippet": raw_llm_response[:200] if 'raw_llm_response' in locals() else 'N/A'})
            # Fallback if LLM fails or returns unparseable JSON
            return {
                "summary": "LLM diagnosis failed.",
                "diagnosis": f"Failed to get LLM diagnosis or parse response: {e}",
                "fix_strategy": "Manually inspect the error message and context.",
                "severity": "CRITICAL"
            }
