import json
import re
from core.logging_utils import log_json

class SelfCorrectionAgent:
    """
    Intercepts failed tool calls or invalid LLM responses, analyzes errors,
    and generates adjusted prompts to prevent repetitive failures.
    """
    def __init__(self, brain=None):
        self.brain = brain
        self.error_log = []

    def analyze_error(self, error_message: str, context: dict) -> str:
        """
        Analyzes the error and returns a suggestion for prompt adjustment.
        Combines heuristic pattern matching with optional LLM reasoning.
        """
        log_json("INFO", "self_correction_analysis_start", details={"error": error_message, "phase": context.get("phase")})
        
        # 1. Heuristic Analysis for common Python/System errors
        heuristics = [
            (r"IndentationError", "Fix the indentation in the generated code. Ensure consistency between tabs and spaces."),
            (r"NameError: name '(\w+)' is not defined", "The variable or class '{0}' is used but not defined. Check imports or local definitions."),
            (r"ImportError: cannot import name '(\w+)'", "The name '{0}' cannot be imported. Verify the module path and symbol existence."),
            (r"ModuleNotFoundError: No module named '(\w+)'", "The module '{0}' is missing. Check if it needs to be installed or if the path is wrong."),
            (r"SyntaxError: (.*)", "There is a syntax error: {0}. Review the code structure and fix the invalid syntax."),
            (r"syntax error(?:[:\s]+(.*))?", "There is a syntax error: {0}. Review the code structure and fix the invalid syntax."),
            (r"TypeError: '(\w+)' object is not subscriptable", "You are trying to access an index on '{0}', which is not a list or dict. Check the variable type."),
            (r"timeout|TimeoutError", "The task timed out. Try to break it down into smaller sub-tasks or increase the timeout limit."),
            (r"quota|rate limit", "API rate limit exceeded. Consider switching model providers or reducing request frequency.")
        ]

        for pattern, suggestion in heuristics:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                if "{0}" in suggestion:
                    groups = match.groups()
                    val = groups[0] if groups else "unknown"
                    suggestion = suggestion.format(val)
                return f"Self-Correction Suggestion: {suggestion}"

        # 2. LLM-based Analysis (if brain/model is available and heuristics fail)
        if self.brain and hasattr(self.brain, 'ask'):
            try:
                # This would be a lightweight call to diagnose complex tracebacks
                prompt = f"Analyze this error message from an autonomous agent cycle and provide a concise one-sentence fix instruction.\nError: {error_message}\nContext: {json.dumps(context)}"
                analysis = self.brain.ask(prompt, system="You are an expert debugger for the AURA system.")
                return f"AI Analysis: {analysis}"
            except Exception as exc:
                log_json("WARN", "self_correction_llm_failed", details={"error": str(exc)})

        # 3. Generic Fallback
        return f"Self-Correction: An error occurred in the '{context.get('phase', 'unknown')}' phase. Please review the logs and adjust the implementation strategy."

    def execute_tool_call(self, prompt: str):
        """Mock method for testing purposes."""
        if "error" in prompt.lower() or "this will error" in prompt.lower():
            self.error_log.append(prompt)
            return "Error detected. Please clarify your input."
        return f"Successful response for: {prompt}"

    def call_tool(self, *args, **kwargs):
        """Mock method for patch-testing."""
        pass
