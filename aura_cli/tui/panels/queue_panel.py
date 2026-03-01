"""Goal queue panel for AURA TUI."""
from __future__ import annotations

from typing import Any, Optional

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def build_queue_panel(goal_queue: Optional[Any] = None, goal_archive: Optional[Any] = None) -> "Panel":
    """Build the goal queue panel."""
    if not _RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Goal", overflow="fold")

    # Pending Goals
    goals: list = []
    if goal_queue is not None:
        try:
            goals = list(goal_queue.queue)
        except Exception:
            pass

    if goals:
        for i, goal in enumerate(goals[:5]):
            truncated = (str(goal)[:52] + "…") if len(str(goal)) > 53 else str(goal)
            prefix = "[bold green]▶[/bold green]" if i == 0 else " ·"
            table.add_row(f"{prefix} {truncated}")
        if len(goals) > 5:
            table.add_row(f"[dim]  … and {len(goals) - 5} more pending[/dim]")
    
    # Completed Goals
    completed: list = []
    if goal_archive is not None:
        try:
            completed = list(goal_archive.completed)
        except Exception:
            pass
            
    if completed:
        table.add_section()
        table.add_row("[bold cyan]Completed[/bold cyan]")
        for goal, score in reversed(completed[-5:]):
            truncated = (str(goal)[:50] + "…") if len(str(goal)) > 51 else str(goal)
            score_str = f"[dim]({score:.2f})[/dim]" if score is not None else ""
            table.add_row(f" [dim]✓[/dim] {truncated} {score_str}")

    if not goals and not completed:
        table.add_row("[dim](no goals active or finished)[/dim]")

    title_count = f"[bold green]Goal Queue[/bold green] [dim]({len(goals)} pending, {len(completed)} done)[/dim]"
    return Panel(table, title=title_count, border_style="green")
