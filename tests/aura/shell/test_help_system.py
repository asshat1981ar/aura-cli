"""Tests for help system."""

import pytest

from aura.shell.help_system import HelpSystem
from aura.shell.models import ShellCommand, CommandCategory


class TestHelpSystem:
    @pytest.fixture
    def commands(self):
        return {
            "help": ShellCommand(
                name="help",
                handler=lambda **kw: None,
                description="Show help",
                category=CommandCategory.SYSTEM,
                aliases=["h"],
                args_help="[command]",
                examples=["", "exit"],
            ),
            "exit": ShellCommand(
                name="exit",
                handler=lambda **kw: None,
                description="Exit shell",
                category=CommandCategory.SYSTEM,
            ),
            "status": ShellCommand(
                name="status",
                handler=lambda **kw: None,
                description="Show status",
                category=CommandCategory.UTILITY,
            ),
        }

    @pytest.fixture
    def help_system(self, commands):
        return HelpSystem(commands)

    def test_general_help(self, help_system):
        text = help_system.get_help()

        assert "AURA" in text
        assert "help" in text
        assert "exit" in text
        assert "status" in text
        assert "[SYSTEM]" in text

    def test_command_help(self, help_system):
        text = help_system.get_help("help")

        assert "help" in text
        assert "Show help" in text
        assert "Aliases:" in text
        assert "h" in text
        assert "Usage:" in text
        assert "Examples:" in text

    def test_command_help_not_found(self, help_system):
        text = help_system.get_help("unknown")

        assert "Unknown command" in text

    def test_command_help_suggestion(self, help_system):
        text = help_system.get_help("exi")  # Typo for exit

        assert "Unknown command" in text
        assert "Did you mean" in text

    def test_levenshtein_distance(self, help_system):
        # Same string
        assert help_system._levenshtein("hello", "hello") == 0

        # One insertion
        assert help_system._levenshtein("hello", "hellos") == 1

        # One deletion
        assert help_system._levenshtein("hello", "hell") == 1

        # One substitution
        assert help_system._levenshtein("hello", "hallo") == 1

        # Multiple edits
        assert help_system._levenshtein("kitten", "sitting") == 3

    def test_find_similar(self, help_system):
        similar = help_system._find_similar("exi")

        assert "exit" in similar
