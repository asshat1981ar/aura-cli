"""Built-in commands for interactive shell."""

from typing import List

from .help_system import HelpSystem
from .models import CommandCategory, CommandResult, ShellCommand, ShellSession


def create_builtin_commands(commands_dict: dict) -> List[ShellCommand]:
    """Create built-in shell commands."""

    def cmd_help(*args, session: ShellSession = None) -> CommandResult:
        """Show help information."""
        help_system = HelpSystem(commands_dict)
        topic = args[0] if args else None
        return CommandResult.ok(help_system.get_help(topic))

    def cmd_exit(*args, session: ShellSession = None) -> CommandResult:
        """Exit the shell."""
        message = "Goodbye!"
        return CommandResult.exit(message)

    def cmd_quit(*args, session: ShellSession = None) -> CommandResult:
        """Quit the shell (alias for exit)."""
        return cmd_exit(*args, session=session)

    def cmd_history(*args, session: ShellSession = None) -> CommandResult:
        """Show command history."""
        # This would need access to the completer's history
        return CommandResult.ok("History feature requires shell context")

    def cmd_clear(*args, session: ShellSession = None) -> CommandResult:
        """Clear the screen."""
        import os

        os.system("clear" if os.name != "nt" else "cls")
        return CommandResult.ok()

    def cmd_status(*args, session: ShellSession = None) -> CommandResult:
        """Show shell session status."""
        if not session:
            return CommandResult.failure("No active session")

        return CommandResult.ok(
            {
                "commands_executed": session.command_count,
                "session_duration_seconds": round(session.duration_seconds, 2),
                "variables": len(session.variables),
            }
        )

    def cmd_vars(*args, session: ShellSession = None) -> CommandResult:
        """List or set shell variables."""
        if not session:
            return CommandResult.failure("No active session")

        if not args:
            return CommandResult.ok(session.variables)

        if len(args) == 1:
            # Get variable
            key = args[0]
            if key in session.variables:
                return CommandResult.ok({key: session.variables[key]})
            return CommandResult.failure(f"Variable not found: {key}")

        # Set variable
        key, value = args[0], args[1]
        session.variables[key] = value
        return CommandResult.ok(f"Set {key} = {value}")

    def cmd_echo(*args, session: ShellSession = None) -> CommandResult:
        """Echo arguments."""
        return CommandResult.ok(" ".join(args))

    def cmd_last(*args, session: ShellSession = None) -> CommandResult:
        """Show result of last command."""
        if not session or not session.last_result:
            return CommandResult.failure("No previous command")

        last = session.last_result
        return CommandResult.ok(
            {
                "success": last.success,
                "output": last.output,
                "error": last.error,
            }
        )

    return [
        ShellCommand(
            name="help",
            handler=cmd_help,
            description="Show help for commands",
            category=CommandCategory.SYSTEM,
            aliases=["h", "?"],
            args_help="[command]",
            examples=["", "exit", "status"],
        ),
        ShellCommand(
            name="exit",
            handler=cmd_exit,
            description="Exit the shell",
            category=CommandCategory.SYSTEM,
            aliases=["q"],
        ),
        ShellCommand(
            name="quit",
            handler=cmd_quit,
            description="Quit the shell (alias for exit)",
            category=CommandCategory.SYSTEM,
        ),
        ShellCommand(
            name="history",
            handler=cmd_history,
            description="Show command history",
            category=CommandCategory.SYSTEM,
            aliases=["hist"],
        ),
        ShellCommand(
            name="clear",
            handler=cmd_clear,
            description="Clear the screen",
            category=CommandCategory.SYSTEM,
            aliases=["cls"],
        ),
        ShellCommand(
            name="status",
            handler=cmd_status,
            description="Show session status",
            category=CommandCategory.SYSTEM,
        ),
        ShellCommand(
            name="vars",
            handler=cmd_vars,
            description="Manage shell variables",
            category=CommandCategory.UTILITY,
            args_help="[key] [value]",
            examples=["", "api_key", "name John"],
        ),
        ShellCommand(
            name="echo",
            handler=cmd_echo,
            description="Echo arguments to output",
            category=CommandCategory.UTILITY,
            args_help="<message>",
            examples=["Hello World"],
        ),
        ShellCommand(
            name="last",
            handler=cmd_last,
            description="Show result of last command",
            category=CommandCategory.UTILITY,
        ),
    ]
