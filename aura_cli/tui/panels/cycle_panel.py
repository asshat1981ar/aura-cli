"""Cycle pipeline panel for AURA TUI."""
from __future__ import annotations

from typing import Dict, Optional

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_PHASES = ["ingest", "plan", "critique", "synthesize", "act", "verify", "reflect"]
_ICONS = {"✅": "✅", "⟳": "[yellow]⟳[/yellow]", "○": "[dim]○[/dim]", "❌": "[red]❌[/red]"}


def build_cycle_panel(
    current_goal: str,
    phases_status: Dict[str, str],
    current_phase: str,
) -> "Panel":
    """Build the pipeline progress panel."""
    if not _RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Phase", style="bold", width=12)
    table.add_column("Status", width=4)

    for phase in _PHASES:
        status = phases_status.get(phase, "○")
        icon = _ICONS.get(status, status)
        phase_label = f"[cyan]{phase}[/cyan]" if phase == current_phase else phase
        table.add_row(phase_label, icon)

    goal_text = (current_goal[:55] + "…") if len(current_goal) > 56 else (current_goal or "(waiting)")

    return Panel(
        table,
        title=f"[bold blue]Pipeline[/bold blue] [dim]{goal_text}[/dim]",
        border_style="blue",
    )
