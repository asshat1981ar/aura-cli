"""Data models for interactive shell."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CommandCategory(Enum):
    """Command category for grouping."""
    SYSTEM = "system"
    CONFIG = "config"
    GOAL = "goal"
    AGENT = "agent"
    UTILITY = "utility"


@dataclass
class ShellCommand:
    """A shell command definition."""
    name: str
    handler: Callable
    description: str = ""
    category: CommandCategory = CommandCategory.UTILITY
    aliases: List[str] = field(default_factory=list)
    args_help: str = ""
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.description:
            self.description = f"Execute {self.name} command"


@dataclass
class CommandResult:
    """Result of executing a shell command."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    exit_shell: bool = False
    
    @classmethod
    def ok(cls, output: Any = None) -> "CommandResult":
        return cls(success=True, output=output)
    
    @classmethod
    def failure(cls, message: str) -> "CommandResult":
        """Create a failure result (named 'failure' to avoid conflict with field)."""
        return cls(success=False, error=message)
    
    @classmethod
    def exit(cls, output: Any = None) -> "CommandResult":
        return cls(success=True, output=output, exit_shell=True)


@dataclass
class ShellSession:
    """Shell session state."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    command_count: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)
    last_result: Optional[CommandResult] = None
    
    def record_command(self):
        self.command_count += 1
    
    @property
    def duration_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()
