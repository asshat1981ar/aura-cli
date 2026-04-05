"""Unit tests for agents/external_llm_agent.py."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agents.external_llm_agent import ExternalLLMAgentAdapter, _ROUTING_CATEGORIES


class TestRoutingCategories(unittest.TestCase):
    def test_known_categories_present(self):
        for key in ("code", "plan", "analyze", "critique", "fast", "quality"):
            self.assertIn(key, _ROUTING_CATEGORIES)


class TestExternalLLMAgentAdapter(unittest.TestCase):
    """Tests for ExternalLLMAgentAdapter.run()."""

    def _make_adapter(self, model=None):
        return ExternalLLMAgentAdapter(model_adapter=model)

    def test_agent_name(self):
        self.assertEqual(ExternalLLMAgentAdapter.name, "external_llm")

    def test_no_model_returns_error(self):
        adapter = self._make_adapter(model=None)
        result = adapter.run({"task": "do something"})

        self.assertEqual(result["error"], "No model adapter configured")
        self.assertEqual(result["response"], "")

    def test_run_with_generate_method(self):
        model = MagicMock(spec=["generate"])
        model.generate.return_value = "generated text"

        adapter = self._make_adapter(model=model)
        result = adapter.run({"task": "hello"})

        self.assertEqual(result["response"], "generated text")
        self.assertIsNone(result["error"])
        model.generate.assert_called_once()

    def test_run_with_generate_for_task_method(self):
        model = MagicMock(spec=["generate_for_task"])
        model.generate_for_task.return_value = "task response"

        adapter = self._make_adapter(model=model)
        result = adapter.run({"task": "plan something", "category": "plan"})

        self.assertEqual(result["response"], "task response")
        model.generate_for_task.assert_called_once_with("planning", "plan something")

    def test_model_override_uses_generate_with_model(self):
        model = MagicMock(spec=["generate_with_model"])
        model.generate_with_model.return_value = "specific model output"

        adapter = self._make_adapter(model=model)
        result = adapter.run({"task": "test", "model": "claude-3"})

        self.assertEqual(result["response"], "specific model output")
        self.assertEqual(result["model_used"], "claude-3")
        model.generate_with_model.assert_called_once_with("claude-3", "test")

    def test_context_prepended_to_prompt(self):
        model = MagicMock(spec=["generate"])
        model.generate.return_value = "ok"

        adapter = self._make_adapter(model=model)
        adapter.run({"task": "my prompt", "context": "extra context"})

        called_prompt = model.generate.call_args[0][0]
        self.assertIn("extra context", called_prompt)
        self.assertIn("my prompt", called_prompt)
        # Context should come first
        self.assertLess(called_prompt.index("extra context"), called_prompt.index("my prompt"))

    def test_prompt_field_overrides_task_as_prompt(self):
        model = MagicMock(spec=["generate"])
        model.generate.return_value = "ok"

        adapter = self._make_adapter(model=model)
        adapter.run({"task": "task text", "prompt": "explicit prompt"})

        called_prompt = model.generate.call_args[0][0]
        self.assertIn("explicit prompt", called_prompt)

    def test_model_exception_captured_in_error(self):
        model = MagicMock(spec=["generate"])
        model.generate.side_effect = RuntimeError("timeout")

        adapter = self._make_adapter(model=model)
        result = adapter.run({"task": "go"})

        self.assertIsNotNone(result["error"])
        self.assertIn("timeout", result["error"])
        self.assertEqual(result["response"], "")

    def test_no_generate_method_returns_error(self):
        model = MagicMock(spec=[])  # no generate methods at all

        adapter = self._make_adapter(model=model)
        result = adapter.run({"task": "do it"})

        self.assertIn("no generate method", result["error"].lower())

    def test_category_passed_through_in_result(self):
        model = MagicMock(spec=["generate"])
        model.generate.return_value = "x"

        adapter = self._make_adapter(model=model)
        result = adapter.run({"task": "t", "category": "code"})

        self.assertEqual(result["category"], "code")

    def test_task_field_in_result(self):
        model = MagicMock(spec=["generate"])
        model.generate.return_value = "y"

        adapter = self._make_adapter(model=model)
        result = adapter.run({"task": "my task"})

        self.assertEqual(result["task"], "my task")


if __name__ == "__main__":
    unittest.main()
