"""Tests for command completer."""

import pytest

from aura.shell.completer import CommandCompleter
from aura.shell.models import ShellCommand, CommandCategory


class TestCommandCompleter:
    @pytest.fixture
    def completer(self):
        return CommandCompleter()

    @pytest.fixture
    def sample_commands(self):
        return [
            ShellCommand(name="help", handler=lambda: None, aliases=["h"]),
            ShellCommand(name="exit", handler=lambda: None, aliases=["quit"]),
            ShellCommand(name="status", handler=lambda: None),
            ShellCommand(name="config", handler=lambda: None),
        ]

    def test_complete_empty_prefix(self, completer, sample_commands):
        completer.register_commands(sample_commands)

        # Empty prefix should return all command names
        result = completer.complete("")

        assert "help" in result
        assert "exit" in result
        assert "status" in result
        assert "config" in result

    def test_complete_with_prefix(self, completer, sample_commands):
        completer.register_commands(sample_commands)

        result = completer.complete("ex")

        assert "exit" in result
        assert "help" not in result

    def test_complete_aliases(self, completer, sample_commands):
        completer.register_commands(sample_commands)

        result = completer.complete("h")

        assert "help" in result  # 'h' is alias for help
        assert "h" in result  # Alias itself

    def test_complete_no_match(self, completer, sample_commands):
        completer.register_commands(sample_commands)

        result = completer.complete("xyz")

        assert result == []

    def test_add_to_history(self, completer):
        completer.add_to_history("help")
        completer.add_to_history("status")

        assert completer._history == ["help", "status"]

    def test_add_duplicate_to_history(self, completer):
        completer.add_to_history("help")
        completer.add_to_history("help")

        # Duplicates should not be added consecutively
        assert completer._history == ["help"]

    def test_get_history(self, completer):
        completer.add_to_history("cmd1")
        completer.add_to_history("cmd2")
        completer.add_to_history("cmd3")

        history = completer.get_history(limit=2)
        assert history == ["cmd2", "cmd3"]

    def test_search_history(self, completer):
        completer.add_to_history("help config")
        completer.add_to_history("status show")
        completer.add_to_history("help agents")

        result = completer.search_history("help")

        assert "help config" in result
        assert "help agents" in result
        assert "status show" not in result

    def test_max_history_limit(self, completer):
        completer._max_history = 3

        completer.add_to_history("cmd1")
        completer.add_to_history("cmd2")
        completer.add_to_history("cmd3")
        completer.add_to_history("cmd4")

        assert completer._history == ["cmd2", "cmd3", "cmd4"]
