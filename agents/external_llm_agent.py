"""External LLM/Service agent — bridges to external APIs and hosted LLMs.

Wraps core/model_adapter.py's multi-provider support to expose it as a
routable agent for tasks requiring external AI services.
"""
from __future__ import annotations

from typing import Any, Dict


# Model routing categories
_ROUTING_CATEGORIES = {
    "code": "code_generation",
    "plan": "planning",
    "analyze": "analysis",
    "critique": "critique",
    "fast": "fast",
    "quality": "quality",
}


class ExternalLLMAgentAdapter:
    """Pipeline adapter for external LLM service calls.

    Routes requests to the appropriate model provider (Gemini, Claude,
    OpenAI, local) based on task type and model routing configuration.
    """

    name = "external_llm"

    def __init__(self, model_adapter=None):
        self.model = model_adapter

    def run(self, input_data: dict) -> dict:
        """Execute an external LLM call.

        Args:
            input_data: Dict with keys:
                - task (str): Task description or prompt.
                - prompt (str, optional): Direct prompt override.
                - category (str, optional): Routing category —
                  "code", "plan", "analyze", "critique", "fast", "quality".
                  Defaults to "fast".
                - model (str, optional): Explicit model override.
                - max_tokens (int, optional): Max response tokens.
                - temperature (float, optional): Sampling temperature.
                - system_prompt (str, optional): System prompt.
                - context (str, optional): Additional context.

        Returns:
            Dict with keys: response, model_used, category, tokens_used.
        """
        task = input_data.get("task", "")
        prompt = input_data.get("prompt", "") or task
        category = input_data.get("category", "fast")
        model_override = input_data.get("model")
        context = input_data.get("context", "")

        if context:
            prompt = f"{context}\n\n{prompt}"

        result: Dict[str, Any] = {
            "category": category,
            "task": task,
            "response": "",
            "model_used": "",
            "error": None,
        }

        if self.model is None:
            result["error"] = "No model adapter configured"
            return result

        try:
            # Use the model adapter's routing if available
            routing_key = _ROUTING_CATEGORIES.get(category, category)

            if model_override and hasattr(self.model, "generate_with_model"):
                response = self.model.generate_with_model(model_override, prompt)
            elif hasattr(self.model, "generate_for_task"):
                response = self.model.generate_for_task(routing_key, prompt)
            elif hasattr(self.model, "generate"):
                response = self.model.generate(prompt)
            else:
                result["error"] = "Model adapter has no generate method"
                return result

            result["response"] = response
            result["model_used"] = model_override or routing_key
            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result
