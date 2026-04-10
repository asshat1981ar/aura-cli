"""Unit tests for agents/handlers/applicator.py.

Sprint 4: Unit tests for core pipeline — applicator handler.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agents.handlers.applicator import handle
from agents.handlers import applicator as applicator_module


class TestResolveAgent:
    """Tests for _resolve_agent helper."""

    def test_returns_agent_from_context(self):
        """Should return agent if present in context."""
        mock_agent = MagicMock()
        context = {"agent": mock_agent}

        result = applicator_module._resolve_agent(context)

        assert result is mock_agent

    def test_constructs_from_brain(self):
        """Should construct ApplicatorAgent from brain if no agent in context."""
        mock_brain = MagicMock()
        context = {"brain": mock_brain}

        with patch("agents.applicator.ApplicatorAgent") as mock_agent_class:
            mock_instance = MagicMock()
            mock_agent_class.return_value = mock_instance
            result = applicator_module._resolve_agent(context)

            mock_agent_class.assert_called_once_with(
                brain=mock_brain, backup_dir=".aura/backups"
            )
            assert result is mock_instance

    def test_constructs_with_custom_backup_dir(self):
        """Should use custom backup_dir from context."""
        mock_brain = MagicMock()
        context = {"brain": mock_brain, "backup_dir": "/custom/backups"}

        with patch("agents.handlers.applicator.ApplicatorAgent") as mock_agent_class:
            applicator_module._resolve_agent(context)

            mock_agent_class.assert_called_once_with(
                brain=mock_brain, backup_dir="/custom/backups"
            )

    def test_raises_on_missing_context(self):
        """Should raise ValueError if neither agent nor brain in context."""
        context = {}

        with pytest.raises(ValueError) as exc_info:
            applicator_module._resolve_agent(context)

        assert "must contain 'agent' or 'brain'" in str(exc_info.value)


class TestHandleApply:
    """Tests for _handle_apply helper."""

    def test_applies_code_with_target_path(self):
        """Should call agent.apply with correct parameters."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.target_path = "/path/to/file.py"
        mock_result.backup_path = "/path/to/backup.py"
        mock_result.code = "print('hello')"
        mock_result.error = None
        mock_result.metadata = {"parsed": True}
        mock_agent.apply.return_value = mock_result

        task = {
            "llm_output": '```python\nprint("hello")\n```',
            "target_path": "/path/to/file.py",
            "allow_overwrite": False,
        }

        result = applicator_module._handle_apply(mock_agent, task)

        mock_agent.apply.assert_called_once_with(
            llm_output='```python\nprint("hello")\n```',
            target_path="/path/to/file.py",
            allow_overwrite=False,
        )
        assert result["success"] is True
        assert result["target_path"] == "/path/to/file.py"
        assert result["backup_path"] == "/path/to/backup.py"
        assert result["code"] == "print('hello')"
        assert result["error"] is None
        assert result["metadata"] == {"parsed": True}

    def test_uses_defaults_for_optional_params(self):
        """Should use defaults when optional params not provided."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.target_path = None
        mock_result.backup_path = None
        mock_result.code = None
        mock_result.error = None
        mock_result.metadata = {}
        mock_agent.apply.return_value = mock_result

        task = {"llm_output": "code"}

        applicator_module._handle_apply(mock_agent, task)

        mock_agent.apply.assert_called_once_with(
            llm_output="code", target_path=None, allow_overwrite=True
        )


class TestHandleRollback:
    """Tests for _handle_rollback helper."""

    def test_rollback_with_apply_result(self):
        """Should reconstruct ApplyResult and call agent.rollback."""
        mock_agent = MagicMock()
        mock_agent.rollback.return_value = True

        task = {
            "apply_result": {
                "success": True,
                "target_path": "/path/to/file.py",
                "backup_path": "/path/to/backup.py",
                "code": "original code",
                "error": None,
                "metadata": {"key": "value"},
            }
        }

        with patch("agents.applicator.ApplyResult") as mock_apply_result_class:
            mock_apply_result_instance = MagicMock()
            mock_apply_result_class.return_value = mock_apply_result_instance

            result = applicator_module._handle_rollback(mock_agent, task)

            mock_apply_result_class.assert_called_once_with(
                success=True,
                target_path="/path/to/file.py",
                backup_path="/path/to/backup.py",
                code="original code",
                error=None,
                metadata={"key": "value"},
            )
            mock_agent.rollback.assert_called_once_with(mock_apply_result_instance)
            assert result == {"rolled_back": True}

    def test_rollback_with_defaults(self):
        """Should use defaults when apply_result fields missing."""
        mock_agent = MagicMock()
        mock_agent.rollback.return_value = False

        task = {"apply_result": {}}

        with patch("agents.applicator.ApplyResult") as mock_apply_result_class:
            mock_apply_result_class.return_value = MagicMock()

            applicator_module._handle_rollback(mock_agent, task)

            mock_apply_result_class.assert_called_once_with(
                success=False,
                target_path=None,
                backup_path=None,
                code=None,
                error=None,
                metadata={},
            )


class TestHandle:
    """Tests for main handle function."""

    def test_handle_apply_action(self):
        """Should route to _handle_apply for default action."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.target_path = "/file.py"
        mock_result.backup_path = None
        mock_result.code = None
        mock_result.error = None
        mock_result.metadata = {}
        mock_agent.apply.return_value = mock_result

        task = {"llm_output": "code", "action": "apply"}
        context = {"agent": mock_agent}

        result = handle(task, context)

        assert result["success"] is True
        mock_agent.apply.assert_called_once()

    def test_handle_rollback_action(self):
        """Should route to _handle_rollback for rollback action."""
        mock_agent = MagicMock()
        mock_agent.rollback.return_value = True

        task = {"action": "rollback", "apply_result": {"success": True}}
        context = {"agent": mock_agent}

        with patch("agents.applicator.ApplyResult", MagicMock()):
            result = handle(task, context)

        assert result["rolled_back"] is True

    def test_handle_catches_exceptions(self):
        """Should catch exceptions and return error dict."""
        task = {"action": "apply"}
        context = {}  # No agent or brain

        result = handle(task, context)

        assert "error" in result
        assert "must contain 'agent' or 'brain'" in result["error"]
