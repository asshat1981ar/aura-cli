from unittest.mock import MagicMock

from agents.refactor_agent import RefactorAgent
from agents.test_generator import TestGeneratorAgent


def test_test_generator_agent_prefers_quality_route():
    brain = MagicMock()
    model = MagicMock()
    model.respond_for_role.return_value = "def test_example():\n    assert True\n"

    agent = TestGeneratorAgent(brain, model)
    result = agent.generate(goal="Add regression coverage", code="def foo(): return 1")

    assert "test_example" in result
    model.respond_for_role.assert_called_once()
    brain.remember.assert_called_once()


def test_refactor_agent_records_review_and_uses_analysis_route():
    brain = MagicMock()
    model = MagicMock()
    model.respond_for_role.return_value = "Extract a helper and keep the public API unchanged."

    agent = RefactorAgent(brain, model)
    result = agent.suggest(goal="Reduce duplication", code="def a(): ...", constraints=["keep API stable"])

    assert "keep the public API unchanged" in result
    model.respond_for_role.assert_called_once()
    brain.remember.assert_called_once()
