"""Goal queue panel for AURA TUI."""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from core.operator_runtime import build_queue_summary


def build_queue_panel(
    goal_queue: Optional[Any] = None,
    goal_archive: Optional[Any] = None,
    queue_summary: Optional[Dict[str, Any]] = None,
) -> "Panel":
    """Build the goal queue panel."""
    if not _RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Goal", overflow="fold")

    summary = queue_summary or build_queue_summary(goal_queue, goal_archive)
    goals = summary.get("pending", [])
    completed = summary.get("completed", [])

    if goals:
        for i, item in enumerate(goals[:5]):
            goal = item.get("goal", "")
            truncated = (str(goal)[:52] + "…") if len(str(goal)) > 53 else str(goal)
            prefix = "[bold green]▶[/bold green]" if i == 0 else " ·"
            table.add_row(f"{prefix} {truncated}")
        
        if len(goals) > 5:
            table.add_row(f"[dim]  … and {len(goals) - 5} more pending[/dim]")

    if completed:
        table.add_section()
        table.add_row("[bold cyan]Completed[/bold cyan]")
        for item in reversed(completed[-5:]):
            goal = item.get("goal", "")
            score = item.get("score")
            truncated = (str(goal)[:50] + "…") if len(str(goal)) > 51 else str(goal)
            score_str = f"[dim]({score:.2f})[/dim]" if score is not None else ""
            table.add_row(f" [dim]✓[/dim] {truncated} {score_str}")

    if not goals and not completed:
        table.add_row("[dim](no goals active or finished)[/dim]")

    title_count = (
        f"[bold green]Goal Queue[/bold green] "
        f"[dim]({summary.get('pending_count', len(goals))} pending, {summary.get('completed_count', len(completed))} done)[/dim]"
    )
    return Panel(table, title=title_count, border_style="green")
