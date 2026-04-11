"""Unit tests for core/exception_handler.py — ExceptionHandler, common handlers."""

import pytest

from core.exception_handler import (
    ExceptionHandler,
    common_io_handler,
    common_value_handler,
    default_handler,
)


class TestExceptionHandlerRegisterAndHandle:
    def test_registered_handler_is_called(self):
        handler = ExceptionHandler()
        results = []
        handler.register_pattern(KeyError, lambda e: results.append("key_error"))
        handler.handle(KeyError("missing"))
        assert results == ["key_error"]

    def test_registered_handler_return_value_propagated(self):
        handler = ExceptionHandler()
        handler.register_pattern(ValueError, lambda e: "handled")
        result = handler.handle(ValueError("bad value"))
        assert result == "handled"

    def test_unregistered_exception_is_reraised(self):
        handler = ExceptionHandler()
        with pytest.raises(RuntimeError):
            handler.handle(RuntimeError("unexpected"))

    def test_multiple_patterns_dispatch_correctly(self):
        handler = ExceptionHandler()
        handler.register_pattern(IOError, lambda e: "io")
        handler.register_pattern(ValueError, lambda e: "value")
        assert handler.handle(IOError()) == "io"
        assert handler.handle(ValueError()) == "value"

    def test_overwriting_pattern_uses_latest_handler(self):
        handler = ExceptionHandler()
        handler.register_pattern(KeyError, lambda e: "first")
        handler.register_pattern(KeyError, lambda e: "second")
        assert handler.handle(KeyError()) == "second"


class TestCommonHandlers:
    def test_common_io_handler_returns_none(self, capsys):
        result = common_io_handler(IOError("disk full"))
        assert result is None

    def test_common_io_handler_prints_message(self, capsys):
        common_io_handler(IOError("disk full"))
        captured = capsys.readouterr()
        assert "disk full" in captured.out

    def test_common_value_handler_returns_none(self, capsys):
        result = common_value_handler(ValueError("bad input"))
        assert result is None

    def test_common_value_handler_prints_message(self, capsys):
        common_value_handler(ValueError("bad input"))
        captured = capsys.readouterr()
        assert "bad input" in captured.out


class TestDefaultHandler:
    def test_default_handler_handles_io_error(self):
        result = default_handler.handle(IOError("test"))
        assert result is None

    def test_default_handler_handles_value_error(self):
        result = default_handler.handle(ValueError("test"))
        assert result is None

    def test_default_handler_reraises_unregistered(self):
        with pytest.raises(KeyError):
            default_handler.handle(KeyError("not registered"))
