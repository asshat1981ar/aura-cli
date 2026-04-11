"""Tests for agents/synthesizer.py — SynthesizerAgent."""

import pytest
from agents.synthesizer import SynthesizerAgent


@pytest.fixture
def agent():
    return SynthesizerAgent()


class TestSynthesizerAgentIdentity:
    def test_name(self, agent):
        assert agent.name == "synthesize"


class TestSynthesizerAgentRun:
    def test_empty_input_produces_task(self, agent):
        result = agent.run({})
        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_task_uses_goal_as_title(self, agent):
        result = agent.run({"goal": "Add login feature"})
        assert result["tasks"][0]["title"] == "Add login feature"

    def test_empty_goal_title_fallback(self, agent):
        result = agent.run({})
        assert result["tasks"][0]["title"] == "Unnamed goal"

    def test_task_id_is_task_1(self, agent):
        result = agent.run({"goal": "x"})
        assert result["tasks"][0]["id"] == "task_1"

    def test_plan_steps_become_intent(self, agent):
        plan = {"steps": ["Step A", "Step B"]}
        result = agent.run({"goal": "x", "plan": plan})
        intent = result["tasks"][0]["intent"]
        assert "Step A" in intent
        assert "Step B" in intent

    def test_critique_issues_appended_to_intent(self, agent):
        plan = {"steps": ["Do thing"]}
        critique = {"issues": ["Too vague", "Missing tests"]}
        result = agent.run({"goal": "x", "plan": plan, "critique": critique})
        intent = result["tasks"][0]["intent"]
        assert "Too vague" in intent
        assert "Missing tests" in intent

    def test_critique_with_dict_issues(self, agent):
        plan = {"steps": ["Do thing"]}
        critique = {"issues": [{"description": "Security hole"}]}
        result = agent.run({"goal": "x", "plan": plan, "critique": critique})
        assert "Security hole" in result["tasks"][0]["intent"]

    def test_empty_critique_issues_no_critique_section(self, agent):
        result = agent.run({"goal": "x", "plan": {"steps": ["S1"]}, "critique": {"issues": []}})
        assert "Critique:" not in result["tasks"][0]["intent"]

    def test_default_files(self, agent):
        result = agent.run({"goal": "x"})
        assert result["tasks"][0]["files"] == ["core/", "agents/", "memory/"]

    def test_explicit_files_override_defaults(self, agent):
        result = agent.run({"goal": "x", "files": ["my_module.py"]})
        assert result["tasks"][0]["files"] == ["my_module.py"]

    def test_default_tests(self, agent):
        result = agent.run({"goal": "x"})
        assert result["tasks"][0]["tests"] == ["python3 -m pytest -q"]

    def test_explicit_tests_override_defaults(self, agent):
        result = agent.run({"goal": "x", "tests": ["pytest -k smoke"]})
        assert result["tasks"][0]["tests"] == ["pytest -k smoke"]

    def test_no_plan_produces_fallback_intent(self, agent):
        result = agent.run({"goal": "x"})
        assert result["tasks"][0]["intent"] == "No plan provided."

    def test_empty_steps_fallback_intent(self, agent):
        result = agent.run({"goal": "x", "plan": {"steps": []}})
        assert result["tasks"][0]["intent"] == "No plan provided."

    def test_none_tests_defaults(self, agent):
        # tests=None in input_data should trigger default
        result = agent.run({"goal": "x", "tests": None})
        assert result["tasks"][0]["tests"] == ["python3 -m pytest -q"]
