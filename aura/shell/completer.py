"""Command completion for interactive shell."""

from typing import List

from .models import ShellCommand


class CommandCompleter:
    """Provides command completion suggestions."""
    
    def __init__(self):
        self._commands: List[ShellCommand] = []
        self._history: List[str] = []
        self._max_history = 1000
    
    def register_commands(self, commands: List[ShellCommand]):
        """Register available commands for completion."""
        self._commands = commands
    
    def complete(self, text: str, cursor_pos: int = None) -> List[str]:
        """Get completion suggestions for the given text."""
        if cursor_pos is None:
            cursor_pos = len(text)
        
        # Get the word being completed
        before_cursor = text[:cursor_pos]
        words = before_cursor.split()
        
        if len(words) <= 1 and not before_cursor.endswith(" "):
            # Completing command name
            return self._complete_command(words[0] if words else "")
        else:
            # Completing arguments - could be file paths, history, etc.
            return self._complete_arguments(words, text)
    
    def _complete_command(self, prefix: str) -> List[str]:
        """Complete command names and aliases."""
        matches = []
        prefix_lower = prefix.lower()
        
        for cmd in self._commands:
            if cmd.name.lower().startswith(prefix_lower):
                matches.append(cmd.name)
            for alias in cmd.aliases:
                if alias.lower().startswith(prefix_lower):
                    matches.append(alias)
        
        return sorted(set(matches))
    
    def _complete_arguments(self, words: List[str], full_text: str) -> List[str]:
        """Complete command arguments."""
        # For now, return empty list - can be extended for file completion
        return []
    
    def add_to_history(self, command: str):
        """Add a command to history."""
        if command.strip() and (not self._history or self._history[-1] != command):
            self._history.append(command)
            if len(self._history) > self._max_history:
                self._history.pop(0)
    
    def get_history(self, limit: int = 50) -> List[str]:
        """Get command history."""
        return self._history[-limit:]
    
    def search_history(self, query: str) -> List[str]:
        """Search command history."""
        query_lower = query.lower()
        return [cmd for cmd in self._history if query_lower in cmd.lower()]
    
    def get_suggestions(self, current_input: str) -> List[str]:
        """Get contextual suggestions based on current input."""
        if not current_input:
            # Show recently used commands
            return self._history[-5:] if self._history else []
        
        return self.complete(current_input)
