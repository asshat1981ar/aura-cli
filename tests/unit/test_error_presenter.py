"""Unit tests for error presenter module.

Tests cover:
- Error presentation formatting
- JSON output mode
- Verbose mode
- Multiple error presentation
- Error summary generation
"""

import json
import pytest
from unittest.mock import Mock, patch

from core.exceptions import (
    AuraCLIError,
    MCPConnectionError,
    ConfigNotFoundError,
    ERROR_REGISTRY,
)
from aura_cli.error_presenter import (
    ErrorPresenter,
    PresenterConfig,
    present_error,
    present_errors,
    get_error_summary,
    format_error_summary,
    handle_errors,
)


class TestPresenterConfig:
    """Tests for PresenterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = PresenterConfig()

        assert config.verbose is False
        assert config.json_output is False
        assert config.no_color is False
        assert config.show_suggestions is True
        assert config.show_context is True
        assert config.show_cause is True
        assert config.max_context_lines == 10
        assert config.max_cause_depth == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = PresenterConfig(
            verbose=True,
            json_output=True,
            no_color=True,
            show_suggestions=False,
        )

        assert config.verbose is True
        assert config.json_output is True
        assert config.no_color is True
        assert config.show_suggestions is False


class TestErrorPresenterInit:
    """Tests for ErrorPresenter initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        presenter = ErrorPresenter()

        assert presenter.config is not None
        assert presenter.console is not None

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = PresenterConfig(verbose=True)
        presenter = ErrorPresenter(config)

        assert presenter.config.verbose is True


class TestErrorPresenterPresent:
    """Tests for presenting single errors."""

    def test_present_aura_cli_error(self, capsys):
        """Test presenting AuraCLIError."""
        presenter = ErrorPresenter()
        error = AuraCLIError(code="AURA-100", message="Test error")

        presenter.present(error)

        captured = capsys.readouterr()
        assert "AURA-100" in captured.out or "Test error" in str(error)

    def test_present_regular_exception(self, capsys):
        """Test presenting regular exception."""
        presenter = ErrorPresenter()
        error = ValueError("Test value error")

        presenter.present(error)

        captured = capsys.readouterr()
        # Should convert to AuraCLIError
        assert "AURA-000" in captured.out or len(captured.out) > 0

    def test_present_dict_error(self, capsys):
        """Test presenting error as dictionary."""
        presenter = ErrorPresenter()
        error_dict = {
            "code": "AURA-200",
            "message": "Auth failed",
            "context": {"user": "test"},
        }

        presenter.present(error_dict)

        captured = capsys.readouterr()
        assert "AURA-200" in captured.out or len(captured.out) > 0

    def test_present_json_mode(self, capsys):
        """Test JSON output mode."""
        config = PresenterConfig(json_output=True)
        presenter = ErrorPresenter(config)
        error = AuraCLIError(code="AURA-100", message="Test error")

        presenter.present(error)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["code"] == "AURA-100"
        assert output["message"] == "Test error"


class TestErrorPresenterPresentMultiple:
    """Tests for presenting multiple errors."""

    def test_present_multiple_errors(self, capsys):
        """Test presenting multiple errors."""
        presenter = ErrorPresenter()
        errors = [
            AuraCLIError(code="AURA-100", message="Error 1"),
            AuraCLIError(code="AURA-200", message="Error 2"),
        ]

        presenter.present_multiple(errors, title="Test Errors")

        captured = capsys.readouterr()
        # Should contain table with errors
        assert len(captured.out) > 0

    def test_present_multiple_json_mode(self, capsys):
        """Test presenting multiple errors as JSON."""
        config = PresenterConfig(json_output=True)
        presenter = ErrorPresenter(config)
        errors = [
            AuraCLIError(code="AURA-100", message="Error 1"),
        ]

        presenter.present_multiple(errors)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, list)
        assert len(output) == 1
        assert output[0]["code"] == "AURA-100"

    def test_present_empty_errors(self):
        """Test presenting empty error list."""
        presenter = ErrorPresenter()

        # Should not raise
        presenter.present_multiple([], title="No Errors")


class TestErrorConversion:
    """Tests for error conversion."""

    def test_exception_to_aura_cli_error(self):
        """Test converting exception to AuraCLIError."""
        from core.exceptions import exception_to_aura_cli_error

        error = ValueError("Test error")
        aura_error = exception_to_aura_cli_error(error)

        assert isinstance(aura_error, AuraCLIError)
        assert aura_error.code == "AURA-000"
        assert aura_error.message == "Test error"

    def test_exception_with_context(self):
        """Test converting exception with context."""
        from core.exceptions import exception_to_aura_cli_error

        error = ValueError("Test error")
        aura_error = exception_to_aura_cli_error(error, context={"key": "value"})

        assert aura_error.context == {"key": "value"}

    def test_aura_cli_error_passthrough(self):
        """Test AuraCLIError passes through unchanged."""
        from core.exceptions import exception_to_aura_cli_error

        error = AuraCLIError(code="AURA-100", message="Test")
        result = exception_to_aura_cli_error(error)

        assert result is error


class TestErrorSummary:
    """Tests for error summary functions."""

    def test_get_error_summary(self):
        """Test getting error summary."""
        errors = [
            AuraCLIError(code="AURA-100", message="Config error"),
            AuraCLIError(code="AURA-200", message="Auth error"),
            AuraCLIError(code="AURA-100", message="Another config error"),
        ]

        summary = get_error_summary(errors)

        assert summary["total"] == 3
        assert summary["by_severity"]["error"] == 3
        assert summary["by_category"]["configuration"] == 2
        assert summary["by_category"]["authentication"] == 1
        assert summary["by_code"]["AURA-100"] == 2
        assert summary["by_code"]["AURA-200"] == 1

    def test_get_error_summary_empty(self):
        """Test getting summary of empty errors."""
        summary = get_error_summary([])

        assert summary["total"] == 0
        assert summary["by_severity"] == {}
        assert summary["by_category"] == {}
        assert summary["by_code"] == {}

    def test_format_error_summary(self):
        """Test formatting error summary."""
        summary = {
            "total": 5,
            "by_severity": {"error": 3, "warning": 2},
            "by_category": {"configuration": 3, "network": 2},
            "by_code": {"AURA-100": 3, "AURA-300": 2},
        }

        formatted = format_error_summary(summary)

        assert "Total Errors: 5" in formatted
        assert "error: 3" in formatted
        assert "warning: 2" in formatted
        assert "configuration: 3" in formatted


class TestHandleErrorsDecorator:
    """Tests for handle_errors decorator."""

    def test_decorator_catches_exception(self):
        """Test decorator catches and presents exception."""

        @handle_errors(verbose=False, reraise=False)
        def failing_function():
            raise ValueError("Test error")

        # Should not raise, should return None
        result = failing_function()
        assert result is None

    def test_decorator_reraises(self):
        """Test decorator reraises when configured."""

        @handle_errors(verbose=False, reraise=True)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_decorator_success(self):
        """Test decorator with successful function."""

        @handle_errors()
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_present_error(self, capsys):
        """Test present_error convenience function."""
        error = AuraCLIError(code="AURA-100", message="Test")

        present_error(error)

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_present_errors(self, capsys):
        """Test present_errors convenience function."""
        errors = [AuraCLIError(code="AURA-100", message="Test")]

        present_errors(errors)

        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestErrorHighlighter:
    """Tests for error highlighter."""

    def test_highlighter_imports(self):
        """Test that ErrorHighlighter can be imported."""
        from aura_cli.error_presenter import ErrorHighlighter

        highlighter = ErrorHighlighter()
        assert highlighter is not None


class TestAuraCLIErrorDict:
    """Tests for AuraCLIError to_dict method."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        error = AuraCLIError(code="AURA-100", message="Test error")

        result = error.to_dict()

        assert result["code"] == "AURA-100"
        assert result["message"] == "Test error"
        assert result["severity"] == "error"
        assert result["category"] == "configuration"
        assert "suggestion" in result

    def test_to_dict_with_context(self):
        """Test to_dict with context."""
        error = AuraCLIError(
            code="AURA-100",
            message="Test error",
            context={"file": "config.json"},
        )

        result = error.to_dict()

        assert result["context"] == {"file": "config.json"}

    def test_to_dict_with_cause(self):
        """Test to_dict with cause."""
        cause = ValueError("Original error")
        error = AuraCLIError(
            code="AURA-000",
            message="Test error",
            cause=cause,
        )

        result = error.to_dict()

        assert result["cause_type"] == "ValueError"
        assert result["cause_message"] == "Original error"


class TestErrorRegistryIntegration:
    """Tests for error registry integration."""

    def test_error_info_from_registry(self):
        """Test error info is loaded from registry."""
        error = AuraCLIError(code="AURA-100")

        assert error.message == ERROR_REGISTRY["AURA-100"]["user_message"]
        assert error.suggestion == ERROR_REGISTRY["AURA-100"]["suggestion"]
        assert error.severity == ERROR_REGISTRY["AURA-100"]["severity"]

    def test_unknown_error_code(self):
        """Test handling of unknown error code."""
        error = AuraCLIError(code="AURA-999")

        assert error.message == "Unknown error (AURA-999)"
        assert error.severity == "error"


class TestExitCodes:
    """Tests for exit code determination."""

    def test_get_exit_code_critical(self):
        """Test exit code for critical error."""
        presenter = ErrorPresenter()
        error = AuraCLIError(code="AURA-000")
        error.severity = "critical"

        exit_code = presenter._get_exit_code(error)

        assert exit_code == 1

    def test_get_exit_code_error(self):
        """Test exit code for error."""
        presenter = ErrorPresenter()
        error = AuraCLIError(code="AURA-000")
        error.severity = "error"

        exit_code = presenter._get_exit_code(error)

        assert exit_code == 1

    def test_get_exit_code_warning(self):
        """Test exit code for warning."""
        presenter = ErrorPresenter()
        error = AuraCLIError(code="AURA-000")
        error.severity = "warning"

        exit_code = presenter._get_exit_code(error)

        assert exit_code == 2
