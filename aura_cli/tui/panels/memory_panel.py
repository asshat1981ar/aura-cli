"""Memory browser panel for AURA TUI."""
from __future__ import annotations

from typing import Any, Optional

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def build_memory_panel(brain: Optional[Any] = None, limit: int = 5) -> "Panel":
    """Build the memory entries panel."""
    if not _RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Entry", overflow="fold")

    count = 0
    recent: list = []

    if brain is not None:
        try:
            count = brain.count_memories()
        except Exception:
            try:
                count = len(brain.recall_all())
            except Exception:
                pass
        try:
            recent = brain.recall_recent(limit=limit)
        except Exception:
            pass

    for entry in recent[-limit:]:
        text = str(entry)
        truncated = (text[:55] + "…") if len(text) > 56 else text
        table.add_row(f"[dim]>[/dim] {truncated}")

    if not recent:
        table.add_row("[dim](no recent entries)[/dim]")

    title = f"[bold magenta]Brain Memory[/bold magenta] [dim]({count:,} entries)[/dim]"
    return Panel(table, title=title, border_style="magenta")
