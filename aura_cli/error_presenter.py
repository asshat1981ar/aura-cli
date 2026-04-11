"""Error presentation layer for AURA CLI.

Provides user-friendly error display with Rich formatting, including:
- Color-coded severity levels
- Contextual suggestions
- Verbose mode for debugging
- JSON output mode for programmatic consumption
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from core.exceptions import AuraCLIError

# Rich imports
from rich import box
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# AURA error imports
from core.exceptions import (
    AuraCLIError,
    exception_to_aura_cli_error,
    get_error_info,
)


# =============================================================================
# Custom Highlighter for Error Messages
# =============================================================================

class ErrorHighlighter(RegexHighlighter):
    """Highlighter for error messages."""
    
    base_style = "error."
    highlights = [
        r"(?P<error_code>AURA-\d+)",
        r"(?P<path>/[\w/\-\.]+)",
        r"(?P<url>https?://[\w\./\-\?=&]+)",
    ]


# =============================================================================
# Themes and Styles
# =============================================================================

AURA_ERROR_THEME = Theme({
    # Severity styles
    "error.critical": "bold white on red",
    "error.error": "bold red",
    "error.warning": "bold yellow",
    "error.info": "bold blue",
    "error.debug": "dim cyan",
    
    # Component styles
    "error.code": "bold cyan",
    "error.category": "italic magenta",
    "error.suggestion": "green",
    "error.context.key": "dim yellow",
    "error.context.value": "white",
    "error.cause": "dim red",
    "error.title": "bold underline",
    "error.header": "bold blue",
    "error.traceback": "dim",
    
    # Highlighter styles
    "error.error_code": "bold cyan",
    "error.path": "green",
    "error.url": "underline blue",
})


# Severity color mapping
SEVERITY_COLORS = {
    "critical": "red",
    "error": "red",
    "warning": "yellow",
    "info": "blue",
    "debug": "dim",
}

SEVERITY_ICONS = {
    "critical": "💥",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "debug": "🔍",
}


# =============================================================================
# Error Presenter Configuration
# =============================================================================

@dataclass
class PresenterConfig:
    """Configuration for error presenter."""
    
    verbose: bool = False
    json_output: bool = False
    no_color: bool = False
    show_suggestions: bool = True
    show_context: bool = True
    show_cause: bool = True
    max_context_lines: int = 10
    max_cause_depth: int = 3


# =============================================================================
# Error Presenter
# =============================================================================

class ErrorPresenter:
    """Rich-based error presenter for AURA CLI.
    
    Provides formatted error output with severity-based styling,
    contextual suggestions, and multiple output modes.
    
    Example:
        >>> presenter = ErrorPresenter()
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     presenter.present(e)
        
        >>> # With custom config
        >>> config = PresenterConfig(verbose=True, json_output=False)
        >>> presenter = ErrorPresenter(config)
        >>> presenter.present(error)
    """
    
    def __init__(self, config: Optional[PresenterConfig] = None, console: Optional[Console] = None):
        """Initialize error presenter.
        
        Args:
            config: Presenter configuration
            console: Rich console instance (creates default if None)
        """
        self.config = config or PresenterConfig()
        
        if console:
            self.console = console
        else:
            # Create console with theme
            self.console = Console(
                theme=AURA_ERROR_THEME,
                color_system=None if self.config.no_color else "auto",
                highlighter=ErrorHighlighter(),
            )
    
    def present(self, error: Union[Exception, AuraCLIError, Dict[str, Any]], exit_after: bool = False) -> None:
        """Present an error to the user.
        
        Args:
            error: Error to present (Exception, AuraCLIError, or dict)
            exit_after: Whether to exit after presenting (default: False)
        """
        # Convert to AuraCLIError if needed
        if isinstance(error, dict):
            aura_error = self._dict_to_error(error)
        elif isinstance(error, AuraCLIError):
            aura_error = error
        else:
            aura_error = exception_to_aura_cli_error(error)
        
        # Route to appropriate presenter
        if self.config.json_output:
            self._present_json(aura_error)
        else:
            self._present_rich(aura_error)
        
        if exit_after:
            sys.exit(self._get_exit_code(aura_error))
    
    def present_multiple(self, errors: List[Union[Exception, AuraCLIError]], title: str = "Multiple Errors") -> None:
        """Present multiple errors in a formatted list.
        
        Args:
            errors: List of errors to present
            title: Title for the error panel
        """
        if not errors:
            return
        
        if self.config.json_output:
            output = [self._error_to_dict(exception_to_aura_cli_error(e)) for e in errors]
            # Use regular print to avoid Rich console wrapping
            print(json.dumps(output, indent=2))
            return
        
        # Rich multi-error presentation
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Code", style="cyan", width=10)
        table.add_column("Severity", style="bold", width=10)
        table.add_column("Message", style="white")
        table.add_column("Category", style="magenta", width=15)
        
        for error in errors:
            aura_error = exception_to_aura_cli_error(error)
            icon = SEVERITY_ICONS.get(aura_error.severity, "❌")
            table.add_row(
                aura_error.code,
                f"{icon} {aura_error.severity.upper()}",
                aura_error.message[:80] + "..." if len(aura_error.message) > 80 else aura_error.message,
                aura_error.category,
            )
        
        self.console.print()
        self.console.print(table)
        self.console.print()
    
    def _present_rich(self, error: AuraCLIError) -> None:
        """Present error using Rich formatting.
        
        Args:
            error: AuraCLIError to present
        """
        severity_color = SEVERITY_COLORS.get(error.severity, "red")
        icon = SEVERITY_ICONS.get(error.severity, "❌")
        
        # Build main error panel
        content = Text()
        
        # Error code and severity
        content.append(f"{icon} ", style="")
        content.append(f"[{error.code}]", style="bold cyan")
        content.append(f" ({error.severity.upper()})\n", style=f"bold {severity_color}")
        content.append("─" * 60 + "\n", style="dim")
        
        # Error message
        content.append(f"\n{error.message}\n", style="bold white")
        
        # Suggestion
        if self.config.show_suggestions and error.suggestion:
            content.append("\n💡 Suggestion: ", style="bold green")
            content.append(f"{error.suggestion}\n", style="green")
        
        # Context (if verbose or explicitly shown)
        if self.config.show_context and error.context:
            content.append("\n📋 Context:\n", style="bold yellow")
            for i, (key, value) in enumerate(error.context.items()):
                if i >= self.config.max_context_lines:
                    content.append(f"  ... and {len(error.context) - i} more\n", style="dim")
                    break
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                content.append(f"  • {key}: ", style="dim yellow")
                content.append(f"{value_str}\n", style="white")
        
        # Cause chain (if verbose)
        if self.config.verbose and self.config.show_cause and error.cause:
            content.append("\n🔍 Cause:\n", style="bold red")
            self._format_cause_chain(content, error.cause, depth=0)
        
        # Create panel
        panel = Panel(
            content,
            title="[bold]AURA CLI Error[/bold]",
            subtitle=f"[dim]{error.category}[/dim]",
            border_style=severity_color,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()
        
        # Additional verbose info
        if self.config.verbose:
            self._present_verbose_info(error)
    
    def _present_json(self, error: AuraCLIError) -> None:
        """Present error as JSON.
        
        Args:
            error: AuraCLIError to present
        """
        error_dict = self._error_to_dict(error)
        # Use regular print to avoid Rich console wrapping
        print(json.dumps(error_dict, indent=2, default=str))
    
    def _present_verbose_info(self, error: AuraCLIError) -> None:
        """Present additional verbose information.
        
        Args:
            error: AuraCLIError to present
        """
        # Full error info from registry
        info = get_error_info(error.code)
        
        table = Table(
            title="Verbose Error Information",
            box=box.SIMPLE,
            show_header=False,
        )
        table.add_column("Property", style="bold yellow")
        table.add_column("Value", style="white")
        
        table.add_row("Error Code", error.code)
        table.add_row("Severity", error.severity)
        table.add_row("Category", error.category)
        table.add_row("Registry Message", info.get("user_message", "N/A"))
        table.add_row("Suggestion", info.get("suggestion", "N/A"))
        
        if error.context:
            context_json = json.dumps(error.context, indent=2, default=str)
            table.add_row("Full Context", f"[dim]{context_json}[/dim]")
        
        self.console.print(table)
        self.console.print()
    
    def _format_cause_chain(self, content: Text, cause: Exception, depth: int) -> None:
        """Format cause chain for display.
        
        Args:
            content: Text object to append to
            cause: Exception in the cause chain
            depth: Current depth in the chain
        """
        if depth >= self.config.max_cause_depth:
            content.append(f"  {'  ' * depth}... (truncated)\n", style="dim")
            return
        
        indent = "  " * depth
        content.append(f"{indent}↳ {type(cause).__name__}: ", style="dim red")
        content.append(f"{str(cause)}\n", style="red")
        
        if hasattr(cause, "__cause__") and cause.__cause__:
            self._format_cause_chain(content, cause.__cause__, depth + 1)
    
    def _error_to_dict(self, error: AuraCLIError) -> Dict[str, Any]:
        """Convert error to dictionary.
        
        Args:
            error: AuraCLIError to convert
            
        Returns:
            Dictionary representation
        """
        result = error.to_dict()
        result["timestamp"] = self._get_timestamp()
        # Sanitize any string values that might contain control characters
        for key in ["message", "suggestion", "cause_message"]:
            if key in result and result[key]:
                result[key] = str(result[key]).replace("\n", " ").replace("\r", " ")
        return result
    
    def _dict_to_error(self, error_dict: Dict[str, Any]) -> AuraCLIError:
        """Convert dictionary to AuraCLIError.
        
        Args:
            error_dict: Dictionary with error data
            
        Returns:
            AuraCLIError instance
        """
        return AuraCLIError(
            code=error_dict.get("code", "AURA-000"),
            message=error_dict.get("message"),
            context=error_dict.get("context"),
        )
    
    def _get_exit_code(self, error: AuraCLIError) -> int:
        """Get exit code for error severity.
        
        Args:
            error: AuraCLIError to get exit code for
            
        Returns:
            Exit code integer
        """
        severity_exit_codes = {
            "critical": 1,
            "error": 1,
            "warning": 2,
            "info": 0,
            "debug": 0,
        }
        return severity_exit_codes.get(error.severity, 1)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# =============================================================================
# Convenience Functions
# =============================================================================

def present_error(
    error: Union[Exception, AuraCLIError],
    verbose: bool = False,
    json_output: bool = False,
    exit_after: bool = False,
) -> None:
    """Convenience function to present an error.
    
    Args:
        error: Error to present
        verbose: Show verbose information
        json_output: Output as JSON
        exit_after: Exit after presenting
    """
    config = PresenterConfig(verbose=verbose, json_output=json_output)
    presenter = ErrorPresenter(config)
    presenter.present(error, exit_after=exit_after)


def present_errors(
    errors: List[Union[Exception, AuraCLIError]],
    title: str = "Multiple Errors",
    verbose: bool = False,
    json_output: bool = False,
) -> None:
    """Convenience function to present multiple errors.
    
    Args:
        errors: List of errors to present
        title: Title for error list
        verbose: Show verbose information
        json_output: Output as JSON
    """
    config = PresenterConfig(verbose=verbose, json_output=json_output)
    presenter = ErrorPresenter(config)
    presenter.present_multiple(errors, title=title)


def get_error_summary(errors: List[Union[Exception, AuraCLIError]]) -> Dict[str, int]:
    """Get summary statistics for a list of errors.
    
    Args:
        errors: List of errors
        
    Returns:
        Dictionary with error counts by severity and category
    """
    severity_counts = {}
    category_counts = {}
    code_counts = {}
    
    for error in errors:
        aura_error = exception_to_aura_cli_error(error)
        
        severity_counts[aura_error.severity] = severity_counts.get(aura_error.severity, 0) + 1
        category_counts[aura_error.category] = category_counts.get(aura_error.category, 0) + 1
        code_counts[aura_error.code] = code_counts.get(aura_error.code, 0) + 1
    
    return {
        "total": len(errors),
        "by_severity": severity_counts,
        "by_category": category_counts,
        "by_code": code_counts,
    }


def format_error_summary(summary: Dict[str, Any]) -> str:
    """Format error summary as a string.
    
    Args:
        summary: Error summary from get_error_summary
        
    Returns:
        Formatted string
    """
    lines = [
        f"Total Errors: {summary['total']}",
        "",
        "By Severity:",
    ]
    
    for severity, count in sorted(summary['by_severity'].items()):
        icon = SEVERITY_ICONS.get(severity, "❌")
        lines.append(f"  {icon} {severity}: {count}")
    
    lines.append("")
    lines.append("By Category:")
    for category, count in sorted(summary['by_category'].items()):
        lines.append(f"  • {category}: {count}")
    
    return "\n".join(lines)


# =============================================================================
# Error Handler Decorator
# =============================================================================

def handle_errors(
    verbose: bool = False,
    json_output: bool = False,
    reraise: bool = False,
    default_message: str = "An error occurred",
):
    """Decorator to handle errors in functions.
    
    Args:
        verbose: Show verbose error information
        json_output: Output errors as JSON
        reraise: Re-raise the exception after presenting
        default_message: Default message for unhandled exceptions
        
    Example:
        >>> @handle_errors(verbose=True)
        ... def my_function():
        ...     raise ValueError("Something went wrong")
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                config = PresenterConfig(verbose=verbose, json_output=json_output)
                presenter = ErrorPresenter(config)
                
                if not isinstance(e, AuraCLIError):
                    aura_error = AuraCLIError(
                        code="AURA-000",
                        message=str(e) or default_message,
                        context={"function": func.__name__},
                        cause=e,
                    )
                else:
                    aura_error = e
                
                presenter.present(aura_error)
                
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


# =============================================================================
# CLI Integration Helpers
# =============================================================================

def add_error_options(parser):
    """Add error presentation options to an argument parser.
    
    Args:
        parser: ArgumentParser instance
    """
    error_group = parser.add_argument_group("Error Display Options")
    error_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose error information",
    )
    error_group.add_argument(
        "--json-errors",
        action="store_true",
        help="Output errors as JSON",
    )
    error_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    error_group.add_argument(
        "--no-suggestions",
        action="store_true",
        help="Hide error suggestions",
    )


def presenter_from_args(args) -> ErrorPresenter:
    """Create an ErrorPresenter from parsed arguments.
    
    Args:
        args: Parsed arguments with error options
        
    Returns:
        ErrorPresenter instance
    """
    config = PresenterConfig(
        verbose=getattr(args, "verbose", False),
        json_output=getattr(args, "json_errors", False),
        no_color=getattr(args, "no_color", False),
        show_suggestions=not getattr(args, "no_suggestions", False),
    )
    return ErrorPresenter(config)


# =============================================================================
# Legacy Compatibility
# =============================================================================

def print_error(message: str, code: str = "AURA-000", **kwargs) -> None:
    """Legacy function for simple error printing.
    
    Args:
        message: Error message
        code: Error code
        **kwargs: Additional context
    """
    error = AuraCLIError(code=code, message=message, context=kwargs)
    present_error(error)


def print_warning(message: str, **kwargs) -> None:
    """Legacy function for printing warnings.
    
    Args:
        message: Warning message
        **kwargs: Additional context
    """
    error = AuraCLIError(
        code="AURA-000",
        message=message,
        context=kwargs,
    )
    error.severity = "warning"
    present_error(error)
