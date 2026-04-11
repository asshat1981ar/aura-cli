"""Interactive shell with REPL, completion, and help system."""

from .builtin_commands import create_builtin_commands
from .completer import CommandCompleter
from .help_system import HelpSystem
from .models import (
    CommandCategory,
    CommandResult,
    ShellCommand,
    ShellSession,
)
from .repl import REPL

__all__ = [
    "REPL",
    "CommandCompleter",
    "HelpSystem",
    "ShellCommand",
    "CommandResult",
    "CommandCategory",
    "ShellSession",
    "create_builtin_commands",
]
