"""Tests for shell models."""

import pytest
from datetime import datetime

from aura.shell.models import (
    CommandCategory,
    CommandResult,
    ShellCommand,
    ShellSession,
)


class TestCommandCategory:
    def test_category_values(self):
        assert CommandCategory.SYSTEM.value == "system"
        assert CommandCategory.CONFIG.value == "config"
        assert CommandCategory.GOAL.value == "goal"
        assert CommandCategory.AGENT.value == "agent"
        assert CommandCategory.UTILITY.value == "utility"


class TestCommandResult:
    def test_ok_result(self):
        result = CommandResult.ok("output")

        assert result.success is True
        assert result.output == "output"
        assert result.error is None
        assert result.exit_shell is False

    def test_error_result(self):
        result = CommandResult.failure("something failed")

        assert result.success is False
        assert result.error == "something failed"
        assert result.output is None

    def test_exit_result(self):
        result = CommandResult.exit("farewell")

        assert result.success is True
        assert result.output == "farewell"
        assert result.exit_shell is True


class TestShellCommand:
    def test_basic_command(self):
        def handler(x):
            return x

        cmd = ShellCommand(
            name="test",
            handler=handler,
            description="Test command",
        )

        assert cmd.name == "test"
        assert cmd.handler == handler
        assert cmd.description == "Test command"
        assert cmd.category == CommandCategory.UTILITY
        assert cmd.aliases == []

    def test_full_command(self):
        def handler(x):
            return x

        cmd = ShellCommand(
            name="test",
            handler=handler,
            description="Test command",
            category=CommandCategory.SYSTEM,
            aliases=["t", "tst"],
            args_help="<arg1>",
            examples=["example1", "example2"],
        )

        assert cmd.category == CommandCategory.SYSTEM
        assert cmd.aliases == ["t", "tst"]
        assert cmd.args_help == "<arg1>"
        assert cmd.examples == ["example1", "example2"]

    def test_default_description(self):
        def handler(x):
            return x

        cmd = ShellCommand(name="mycommand", handler=handler)
        # Default description is set in __post_init__
        assert cmd.description == "Execute mycommand command"


class TestShellSession:
    def test_default_creation(self):
        session = ShellSession()

        assert isinstance(session.start_time, datetime)
        assert session.command_count == 0
        assert session.variables == {}
        assert session.last_result is None

    def test_record_command(self):
        session = ShellSession()

        session.record_command()
        assert session.command_count == 1

        session.record_command()
        assert session.command_count == 2

    def test_duration_seconds(self):
        import time

        session = ShellSession()

        time.sleep(0.01)
        duration = session.duration_seconds

        assert duration > 0

    def test_last_result(self):
        session = ShellSession()
        result = CommandResult.ok("test")

        session.last_result = result
        assert session.last_result == result
