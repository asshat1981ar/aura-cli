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

_PHASES = ["ingest", "skill_dispatch", "plan", "critique", "synthesize", "act", "sandbox", "apply", "verify", "reflect", "measure", "learn", "discover"]
_ICONS = {"✅": "✅", "⟳": "[yellow]⟳[/yellow]", "○": "[dim]○[/dim]", "❌": "[red]❌[/red]"}


def build_cycle_panel(
    current_goal: str,
    phases_status: Dict[str, str],
    current_phase: str,
    confidence: Optional[float] = None,
    last_summary: Optional[Dict[str, Any]] = None,
) -> "Panel":
    """Build the pipeline progress panel."""
    if not _RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Phase", style="bold", width=12)
    table.add_column("Status", width=4)

    # Active Pipeline
    for phase in _PHASES:
        status = phases_status.get(phase, "○")
        icon = _ICONS.get(status, status)
        phase_label = f"[cyan]{phase}[/cyan]" if phase == current_phase else phase
        table.add_row(phase_label, icon)

    # Outcome summary if idle
    if not current_goal and last_summary:
        outcome = last_summary.get("outcome", "UNKNOWN")
        stop = last_summary.get("stop_reason", "N/A")
        follow_ups = last_summary.get("queued_follow_up_goals", [])
        
        color = "green" if outcome == "SUCCESS" else "red" if outcome == "FAILED" else "yellow"
        table.add_section()
        table.add_row("Last Outcome", f"[bold {color}]{outcome}[/bold {color}]")
        table.add_row("Stop Reason", f"[dim]{stop}[/dim]")
        if follow_ups:
            table.add_row("Follow-ups", f"[cyan]{len(follow_ups)} goals enqueued[/cyan]")

    goal_text = (current_goal[:55] + "…") if len(current_goal) > 56 else (current_goal or "(idle)")
    title = f"[bold blue]Pipeline[/bold blue] [dim]{goal_text}[/dim]"
    if confidence is not None and current_goal:
        color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
        title += f" [dim]|[/dim] [bold {color}]Conf: {confidence*100:.0f}%[/bold {color}]"

    return Panel(
        table,
        title=title,
        border_style="blue",
    )
