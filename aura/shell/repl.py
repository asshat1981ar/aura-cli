"""REPL engine for interactive shell."""

import shlex
import sys
from typing import Dict, List, Optional

from .completer import CommandCompleter
from .models import CommandCategory, CommandResult, ShellCommand, ShellSession


class REPL:
    """Read-Eval-Print Loop for interactive shell."""
    
    def __init__(self, prompt: str = "aura> ", banner: Optional[str] = None):
        self.prompt = prompt
        self.banner = banner or self._default_banner()
        self.session = ShellSession()
        self.completer = CommandCompleter()
        self._commands: Dict[str, ShellCommand] = {}
        self._running = False
    
    def _default_banner(self) -> str:
        return """\
╔═══════════════════════════════════════╗
║     AURA Interactive Shell v1.0       ║
║  Type 'help' for available commands   ║
╚═══════════════════════════════════════╝
"""
    
    def register_command(self, command: ShellCommand):
        """Register a command."""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._commands[alias] = command
        self.completer.register_commands(list(self._commands.values()))
    
    def register_commands(self, commands: List[ShellCommand]):
        """Register multiple commands."""
        for cmd in commands:
            self.register_command(cmd)
    
    def run(self):
        """Start the REPL loop."""
        self._running = True
        print(self.banner)
        
        while self._running:
            try:
                user_input = self._read_input()
                if not user_input.strip():
                    continue
                
                self.completer.add_to_history(user_input)
                result = self._eval(user_input)
                self._print_result(result)
                
                if result.exit_shell:
                    break
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' or Ctrl+D to quit")
            except EOFError:
                print()
                break
        
        print(f"\nGoodbye! ({self.session.command_count} commands executed)")
    
    def run_once(self, command_line: str) -> CommandResult:
        """Execute a single command."""
        return self._eval(command_line)
    
    def _read_input(self) -> str:
        """Read user input with completion support."""
        try:
            # Try to use readline for better experience
            import readline
            return input(self.prompt)
        except (ImportError, EOFError):
            return input(self.prompt)
    
    def _eval(self, command_line: str) -> CommandResult:
        """Evaluate a command line."""
        parts = self._parse_command(command_line)
        if not parts:
            return CommandResult.ok()
        
        command_name = parts[0]
        args = parts[1:]
        
        command = self._commands.get(command_name)
        if not command:
            return CommandResult.failure(f"Unknown command: {command_name}")
        
        self.session.record_command()
        
        try:
            result = command.handler(*args, session=self.session)
            if not isinstance(result, CommandResult):
                result = CommandResult.ok(result)
            self.session.last_result = result
            return result
        except Exception as e:
            return CommandResult.failure(str(e))
    
    def _parse_command(self, command_line: str) -> List[str]:
        """Parse command line into parts."""
        command_line = command_line.strip()
        if not command_line:
            return []
        
        try:
            return shlex.split(command_line)
        except ValueError:
            # Fallback for unclosed quotes
            return command_line.split()
    
    def _print_result(self, result: CommandResult):
        """Print command result."""
        if result.error:
            print(f"Error: {result.error}", file=sys.stderr)
        elif result.output is not None:
            if isinstance(result.output, str):
                print(result.output)
            else:
                import json
                print(json.dumps(result.output, indent=2, default=str))
    
    def stop(self):
        """Stop the REPL loop."""
        self._running = False
    
    def get_commands_by_category(self) -> Dict[CommandCategory, List[ShellCommand]]:
        """Get commands grouped by category."""
        result: Dict[CommandCategory, List[ShellCommand]] = {}
        seen = set()
        
        for cmd in self._commands.values():
            if cmd.name in seen:
                continue
            seen.add(cmd.name)
            
            if cmd.category not in result:
                result[cmd.category] = []
            result[cmd.category].append(cmd)
        
        return result
