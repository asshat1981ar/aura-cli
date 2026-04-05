"""Unit tests for agents/typescript_agent.py."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agents.typescript_agent import TypeScriptAgentAdapter


class TestTypeScriptAgentAdapterInit:
    def test_default_init(self):
        agent = TypeScriptAgentAdapter()
        assert agent.name == "typescript_agent"
        assert agent.model is None
        assert agent.skills == {}

    def test_custom_init(self):
        mock_model = MagicMock()
        skills = {"api_contract_validator": MagicMock()}
        agent = TypeScriptAgentAdapter(model_adapter=mock_model, skills=skills)
        assert agent.model is mock_model
        assert "api_contract_validator" in agent.skills


class TestRunMethod:
    def setup_method(self):
        self.agent = TypeScriptAgentAdapter()

    @patch("agents.typescript_agent.subprocess.run")
    def test_analyze_action_calls_lint_and_type_check(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        result = self.agent.run({"task": "check", "action": "analyze"})
        assert result["action"] == "analyze"
        assert "lint_results" in result
        assert "type_check_results" in result
        assert "api_contract" in result
        assert "schema" in result
        assert mock_run.call_count == 2  # eslint + tsc

    @patch("agents.typescript_agent.subprocess.run")
    def test_lint_action_only_runs_eslint(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = self.agent.run({"task": "lint check", "action": "lint"})
        assert result["action"] == "lint"
        assert "lint_results" in result
        assert "type_check_results" not in result
        mock_run.assert_called_once()

    @patch("agents.typescript_agent.subprocess.run")
    def test_type_check_action_only_runs_tsc(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = self.agent.run({"task": "type check", "action": "type_check"})
        assert "type_check_results" in result
        assert "lint_results" not in result

    @patch("agents.typescript_agent.subprocess.run")
    def test_build_action(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="built", stderr="")
        result = self.agent.run({"task": "build", "action": "build"})
        assert "build_results" in result
        assert result["build_results"]["exit_code"] == 0

    def test_generate_action_no_model_returns_empty(self):
        result = self.agent.run({"task": "generate types", "action": "generate"})
        assert result["action"] == "generate"
        assert result["generated_code"] == ""

    def test_generate_action_with_model(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = "const x: string = 'hello';"
        agent = TypeScriptAgentAdapter(model_adapter=mock_model)
        result = agent.run({"task": "generate a string var", "action": "generate"})
        assert result["generated_code"] == "const x: string = 'hello';"
        mock_model.generate.assert_called_once()


class TestSubprocessErrors:
    @patch("agents.typescript_agent.subprocess.run", side_effect=FileNotFoundError("npx not found"))
    def test_eslint_file_not_found_returns_error(self, _):
        agent = TypeScriptAgentAdapter()
        result = agent._run_eslint(".")
        assert "error" in result

    @patch("agents.typescript_agent.subprocess.run", side_effect=subprocess.TimeoutExpired("npx", 60))
    def test_tsc_timeout_returns_error(self, _):
        agent = TypeScriptAgentAdapter()
        result = agent._run_tsc(".")
        assert "error" in result

    @patch("agents.typescript_agent.subprocess.run", side_effect=FileNotFoundError("npm not found"))
    def test_build_file_not_found_returns_error(self, _):
        agent = TypeScriptAgentAdapter()
        result = agent._run_build(".")
        assert "error" in result


class TestSkillExecution:
    def test_skill_not_available_returns_status(self):
        agent = TypeScriptAgentAdapter(skills={})
        result = agent._run_skill("api_contract_validator", {})
        assert result["status"] == "skill_not_available"
        assert result["skill"] == "api_contract_validator"

    def test_skill_run_called_when_available(self):
        mock_skill = MagicMock()
        mock_skill.run.return_value = {"valid": True}
        agent = TypeScriptAgentAdapter(skills={"api_contract_validator": mock_skill})
        result = agent._run_skill("api_contract_validator", {"task": "validate"})
        assert result == {"valid": True}
        mock_skill.run.assert_called_once()

    def test_skill_run_exception_returns_error(self):
        mock_skill = MagicMock()
        mock_skill.run.side_effect = RuntimeError("boom")
        agent = TypeScriptAgentAdapter(skills={"api_contract_validator": mock_skill})
        result = agent._run_skill("api_contract_validator", {})
        assert "error" in result
