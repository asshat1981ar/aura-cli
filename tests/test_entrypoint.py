"""
Unit tests for aura_cli/entrypoint.py

Tests for CLI entry point, project root resolution, and history management.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from types import SimpleNamespace

# Import the module under test
from aura_cli.entrypoint import _resolve_project_root, _ensure_project_on_path, _load_history, _save_history, main


class TestResolveProjectRoot:
    """Tests for _resolve_project_root function."""

    def test_with_override(self):
        """Test resolving with explicit override."""
        result = _resolve_project_root("/custom/path")

        assert result == Path("/custom/path")

    def test_with_skip_chdir_env(self):
        """Test resolving when AURA_SKIP_CHDIR is set."""
        with patch.dict(os.environ, {"AURA_SKIP_CHDIR": "1"}):
            result = _resolve_project_root()

            assert result == Path.cwd()

    @patch.dict(os.environ, {"AURA_SKIP_CHDIR": "0"})
    @patch("os.chdir")
    def test_default_resolution(self, mock_chdir):
        """Test default project root resolution."""
        result = _resolve_project_root()

        # Should return the project root (parent of aura_cli)
        assert "aura-cli" in str(result) or "aura_cli" in str(result)
        mock_chdir.assert_called_once()

    def test_path_object_override(self):
        """Test resolving with Path object override."""
        override_path = Path("/some/path")

        result = _resolve_project_root(override_path)

        assert result == Path("/some/path")


class TestEnsureProjectOnPath:
    """Tests for _ensure_project_on_path function."""

    def test_adds_to_path(self):
        """Test that project root is added to sys.path."""
        project_root = Path("/test/project")
        original_path = sys.path.copy()

        _ensure_project_on_path(project_root)

        assert str(project_root) in sys.path
        assert sys.path[0] == str(project_root)  # Should be at index 0

        # Cleanup
        sys.path = original_path

    def test_does_not_duplicate(self):
        """Test that project root is not added twice."""
        project_root = Path("/test/project")
        original_path = sys.path.copy()

        # Add it first
        _ensure_project_on_path(project_root)
        path_after_first = sys.path.copy()

        # Add again
        _ensure_project_on_path(project_root)
        path_after_second = sys.path.copy()

        # Count occurrences
        count = path_after_second.count(str(project_root))
        assert count == 1

        # Cleanup
        sys.path = original_path


class TestLoadHistory:
    """Tests for _load_history function."""

    @patch("aura_cli.entrypoint.readline")
    def test_loads_existing_history(self, mock_readline):
        """Test loading existing history file."""
        project_root = Path("/test/project")
        history_file = project_root / "memory" / ".aura_history"

        with patch.object(Path, "exists", return_value=True):
            _load_history(project_root)

            mock_readline.read_history_file.assert_called_once_with(str(history_file))
            mock_readline.set_history_length.assert_called_once_with(1000)

    @patch("aura_cli.entrypoint.readline")
    def test_handles_missing_history(self, mock_readline):
        """Test handling when history file doesn't exist."""
        project_root = Path("/test/project")

        with patch.object(Path, "exists", return_value=False):
            _load_history(project_root)

            mock_readline.read_history_file.assert_not_called()
            mock_readline.set_history_length.assert_called_once_with(1000)

    @patch("aura_cli.entrypoint.readline")
    def test_handles_readline_exception(self, mock_readline):
        """Test handling exceptions during history load."""
        project_root = Path("/test/project")
        mock_readline.read_history_file.side_effect = Exception("Read error")

        with patch.object(Path, "exists", return_value=True):
            # Should not raise
            _load_history(project_root)

    def test_no_readline_module(self):
        """Test behavior when readline is not available."""
        with patch("aura_cli.entrypoint.readline", None):
            project_root = Path("/test/project")

            # Should not raise
            _load_history(project_root)


class TestSaveHistory:
    """Tests for _save_history function."""

    @patch("aura_cli.entrypoint.readline")
    def test_saves_history(self, mock_readline):
        """Test saving history to file."""
        project_root = Path("/test/project")
        history_file = project_root / "memory" / ".aura_history"

        _save_history(project_root)

        mock_readline.write_history_file.assert_called_once_with(str(history_file))

    @patch("aura_cli.entrypoint.readline")
    def test_handles_write_exception(self, mock_readline):
        """Test handling exceptions during history save."""
        project_root = Path("/test/project")
        mock_readline.write_history_file.side_effect = Exception("Write error")

        # Should not raise
        _save_history(project_root)

    def test_no_readline_module(self):
        """Test behavior when readline is not available."""
        with patch("aura_cli.entrypoint.readline", None):
            project_root = Path("/test/project")

            # Should not raise
            _save_history(project_root)


class TestMain:
    """Tests for main function."""

    @patch("aura_cli.entrypoint._resolve_project_root")
    @patch("aura_cli.entrypoint._ensure_project_on_path")
    @patch("aura_cli.entrypoint._load_history")
    @patch("aura_cli.entrypoint.parse_cli_args")
    @patch("aura_cli.entrypoint.dispatch_command")
    @patch("aura_cli.entrypoint._save_history")
    def test_successful_execution(self, mock_save, mock_dispatch, mock_parse, mock_load, mock_ensure, mock_resolve):
        """Test successful main execution."""
        mock_resolve.return_value = Path("/test/project")
        mock_parse.return_value = SimpleNamespace(action="test")
        mock_dispatch.return_value = 0

        result = main()

        assert result == 0
        mock_resolve.assert_called_once()
        mock_ensure.assert_called_once_with(Path("/test/project"))
        mock_load.assert_called_once_with(Path("/test/project"))
        mock_parse.assert_called_once()
        mock_dispatch.assert_called_once()
        mock_save.assert_called_once()

    @patch("aura_cli.entrypoint._resolve_project_root")
    @patch("aura_cli.entrypoint._ensure_project_on_path")
    @patch("aura_cli.entrypoint._load_history")
    @patch("aura_cli.entrypoint.parse_cli_args")
    @patch("aura_cli.entrypoint.dispatch_command")
    @patch("aura_cli.entrypoint._save_history")
    def test_with_custom_argv(self, mock_save, mock_dispatch, mock_parse, mock_load, mock_ensure, mock_resolve):
        """Test main with custom argv."""
        mock_resolve.return_value = Path("/test/project")
        mock_parse.return_value = SimpleNamespace(action="test")

        main(argv=["goal", "add", "test goal"])

        mock_parse.assert_called_once_with(["goal", "add", "test goal"])

    @patch("aura_cli.entrypoint._resolve_project_root")
    @patch("aura_cli.entrypoint._ensure_project_on_path")
    @patch("aura_cli.entrypoint._load_history")
    @patch("aura_cli.entrypoint.parse_cli_args")
    @patch("aura_cli.entrypoint.dispatch_command")
    @patch("aura_cli.entrypoint._save_history")
    def test_with_project_root_override(self, mock_save, mock_dispatch, mock_parse, mock_load, mock_ensure, mock_resolve):
        """Test main with project root override."""
        mock_parse.return_value = SimpleNamespace(action="test")

        main(project_root_override="/custom/path")

        mock_resolve.assert_called_once_with("/custom/path")

    @patch("aura_cli.entrypoint._resolve_project_root")
    @patch("aura_cli.entrypoint._ensure_project_on_path")
    @patch("aura_cli.entrypoint._load_history")
    @patch("aura_cli.entrypoint.parse_cli_args")
    @patch("builtins.print")
    @patch("aura_cli.entrypoint._save_history")
    def test_parse_error(self, mock_save, mock_print, mock_parse, mock_load, mock_ensure, mock_resolve):
        """Test handling of CLIParseError."""
        from aura_cli.cli_options import CLIParseError

        mock_resolve.return_value = Path("/test/project")
        error = CLIParseError("Invalid arguments", code=2, usage="usage: ...")
        mock_parse.side_effect = error

        result = main()

        assert result == 2
        mock_print.assert_any_call("Error: Invalid arguments", file=sys.stderr)

    @patch("aura_cli.entrypoint._resolve_project_root")
    @patch("aura_cli.entrypoint._ensure_project_on_path")
    @patch("aura_cli.entrypoint._load_history")
    @patch("aura_cli.entrypoint.parse_cli_args")
    @patch("builtins.print")
    @patch("aura_cli.entrypoint._save_history")
    def test_parse_error_json_output(self, mock_save, mock_print, mock_parse, mock_load, mock_ensure, mock_resolve):
        """Test JSON output for parse error when --json in argv."""
        from aura_cli.cli_options import CLIParseError

        mock_resolve.return_value = Path("/test/project")
        error = CLIParseError("Invalid arguments", code=2)
        mock_parse.side_effect = error

        with patch.object(sys, "argv", ["aura", "--json"]):
            result = main()

        assert result == 2
        # Should print JSON
        printed_args = list(mock_print.call_args_list)
        assert len(printed_args) > 0

    @patch("aura_cli.entrypoint._resolve_project_root")
    @patch("aura_cli.entrypoint._ensure_project_on_path")
    @patch("aura_cli.entrypoint._load_history")
    @patch("aura_cli.entrypoint.parse_cli_args")
    @patch("aura_cli.entrypoint.dispatch_command")
    @patch("aura_cli.entrypoint._save_history")
    def test_non_zero_return_code(self, mock_save, mock_dispatch, mock_parse, mock_load, mock_ensure, mock_resolve):
        """Test that non-zero return codes are passed through."""
        mock_resolve.return_value = Path("/test/project")
        mock_parse.return_value = SimpleNamespace(action="test")
        mock_dispatch.return_value = 1

        result = main()

        assert result == 1

    @patch("aura_cli.entrypoint._resolve_project_root")
    @patch("aura_cli.entrypoint._ensure_project_on_path")
    @patch("aura_cli.entrypoint._load_history")
    @patch("aura_cli.entrypoint.parse_cli_args")
    @patch("aura_cli.entrypoint.dispatch_command")
    @patch("aura_cli.entrypoint._save_history")
    def test_save_history_called_in_finally(self, mock_save, mock_dispatch, mock_parse, mock_load, mock_ensure, mock_resolve):
        """Test that history is saved even when dispatch raises."""
        mock_resolve.return_value = Path("/test/project")
        mock_parse.return_value = SimpleNamespace(action="test")
        mock_dispatch.side_effect = Exception("Dispatch error")

        with pytest.raises(Exception):
            main()

        # _save_history should still be called
        mock_save.assert_called_once()


class TestIntegration:
    """Integration tests for entrypoint module."""

    def test_full_workflow_mocked(self):
        """Test full workflow with all components mocked."""
        with (
            patch("aura_cli.entrypoint._resolve_project_root") as mock_resolve,
            patch("aura_cli.entrypoint._ensure_project_on_path") as mock_ensure,
            patch("aura_cli.entrypoint._load_history") as mock_load,
            patch("aura_cli.entrypoint.parse_cli_args") as mock_parse,
            patch("aura_cli.entrypoint.dispatch_command") as mock_dispatch,
            patch("aura_cli.entrypoint._save_history") as mock_save,
        ):
            # Setup mocks
            mock_resolve.return_value = Path("/test/project")
            mock_parse.return_value = SimpleNamespace(action="goal", command="status")
            mock_dispatch.return_value = 0

            # Execute
            result = main(project_root_override="/test/project")

            # Verify chain of calls
            assert result == 0
            mock_resolve.assert_called_once_with("/test/project")
            mock_ensure.assert_called_once()
            mock_load.assert_called_once()
            mock_parse.assert_called_once()
            mock_dispatch.assert_called_once()
            mock_save.assert_called_once()

    def test_project_root_chain(self):
        """Test the project root resolution chain."""
        # Test that override takes precedence
        result = _resolve_project_root("/explicit/path")
        assert result == Path("/explicit/path")

        # Test with env var
        with patch.dict(os.environ, {"AURA_SKIP_CHDIR": "1"}):
            result = _resolve_project_root()
            assert result == Path.cwd()

    def test_path_manipulation(self):
        """Test sys.path manipulation."""
        test_path = Path("/tmp/test_project")
        original_path = sys.path.copy()

        try:
            _ensure_project_on_path(test_path)
            assert str(test_path) in sys.path

            # Verify at beginning
            assert sys.path[0] == str(test_path)
        finally:
            sys.path = original_path
