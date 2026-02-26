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


def build_queue_panel(goal_queue: Optional[Any] = None) -> "Panel":
    """Build the goal queue panel."""
    if not _RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Goal", overflow="fold")

    goals: list = []
    if goal_queue is not None:
        try:
            goals = list(goal_queue.queue)
        except Exception:
            pass

    if not goals:
        table.add_row("[dim](queue empty)[/dim]")
    else:
        for i, goal in enumerate(goals[:10]):
            truncated = (str(goal)[:52] + "…") if len(str(goal)) > 53 else str(goal)
            prefix = "[bold green]▶[/bold green]" if i == 0 else " ·"
            table.add_row(f"{prefix} {truncated}")
        if len(goals) > 10:
            table.add_row(f"[dim]  … and {len(goals) - 10} more[/dim]")

    title_count = f"[bold green]Goal Queue[/bold green] [dim]({len(goals)} goals)[/dim]"
    return Panel(table, title=title_count, border_style="green")
