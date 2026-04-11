"""Tests for built-in commands."""

import pytest

from aura.shell.builtin_commands import create_builtin_commands
from aura.shell.models import CommandResult, ShellSession


class TestBuiltinCommands:
    @pytest.fixture
    def commands(self):
        commands_dict = {}
        builtins = create_builtin_commands(commands_dict)
        for cmd in builtins:
            commands_dict[cmd.name] = cmd
            for alias in cmd.aliases:
                commands_dict[alias] = cmd
        return commands_dict

    def test_help_command(self, commands):
        cmd = commands["help"]
        result = cmd.handler()

        assert result.success is True
        assert "AURA" in result.output

    def test_exit_command(self, commands):
        cmd = commands["exit"]
        result = cmd.handler()

        assert result.success is True
        assert result.exit_shell is True

    def test_quit_alias(self, commands):
        cmd = commands["quit"]
        result = cmd.handler()

        assert result.exit_shell is True

    def test_echo_command(self, commands):
        cmd = commands["echo"]
        result = cmd.handler("Hello", "World")

        assert result.success is True
        assert result.output == "Hello World"

    def test_status_command(self, commands):
        cmd = commands["status"]
        session = ShellSession()
        session.record_command()

        result = cmd.handler(session=session)

        assert result.success is True
        assert result.output["commands_executed"] == 1

    def test_status_no_session(self, commands):
        cmd = commands["status"]
        result = cmd.handler()

        assert result.success is False
        assert "No active session" in result.error

    def test_vars_list(self, commands):
        cmd = commands["vars"]
        session = ShellSession()
        session.variables["key1"] = "value1"

        result = cmd.handler(session=session)

        assert result.success is True
        assert result.output == {"key1": "value1"}

    def test_vars_get(self, commands):
        cmd = commands["vars"]
        session = ShellSession()
        session.variables["myvar"] = "myvalue"

        result = cmd.handler("myvar", session=session)

        assert result.success is True
        assert result.output == {"myvar": "myvalue"}

    def test_vars_get_not_found(self, commands):
        cmd = commands["vars"]
        session = ShellSession()

        result = cmd.handler("missing", session=session)

        assert result.success is False
        assert "not found" in result.error

    def test_vars_set(self, commands):
        cmd = commands["vars"]
        session = ShellSession()

        result = cmd.handler("newvar", "newvalue", session=session)

        assert result.success is True
        assert session.variables["newvar"] == "newvalue"

    def test_last_command(self, commands):
        cmd = commands["last"]
        session = ShellSession()
        session.last_result = CommandResult.ok("previous output")

        result = cmd.handler(session=session)

        assert result.success is True
        assert result.output["output"] == "previous output"

    def test_last_no_previous(self, commands):
        cmd = commands["last"]
        session = ShellSession()

        result = cmd.handler(session=session)

        assert result.success is False
