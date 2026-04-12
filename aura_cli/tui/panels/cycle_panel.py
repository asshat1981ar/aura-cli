"""Cycle pipeline panel for AURA TUI."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_PHASES = ["ingest", "skill_dispatch", "plan", "critique", "synthesize", "act", "sandbox", "apply", "verify", "reflect", "measure", "learn", "discover", "evolve"]
_ICONS = {"✅": "✅", "⟳": "[yellow]⟳[/yellow]", "○": "[dim]○[/dim]", "❌": "[red]❌[/red]"}


def build_cycle_panel(
    current_goal: str,
    phases_status: Dict[str, str],
    current_phase: str,
    confidence: Optional[float] = None,
    last_summary: Optional[Dict[str, Any]] = None,
    beads_runtime: Optional[Dict[str, Any]] = None,
) -> "Panel":
    """Build the pipeline progress panel."""
    if not _RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Phase", style="bold", width=12)
    table.add_column("Status", width=4)

    summary_phase_status = {}
    if last_summary and isinstance(last_summary.get("phase_status"), dict):
        summary_phase_status = {phase: ("✅" if status == "pass" else "❌" if status == "fail" else "⟳" if status == "running" else "○") for phase, status in last_summary["phase_status"].items()}

    display_status = phases_status or summary_phase_status

    for phase in _PHASES:
        status = display_status.get(phase, "○")
        icon = _ICONS.get(status, status)
        phase_label = f"[cyan]{phase}[/cyan]" if phase == current_phase else phase
        table.add_row(phase_label, icon)

    if not current_goal and last_summary:
        outcome = last_summary.get("outcome", "UNKNOWN")
        stop = last_summary.get("stop_reason", "N/A")
        follow_ups = last_summary.get("queued_follow_up_goals", [])

        color = "green" if outcome == "SUCCESS" else "red" if outcome == "FAILED" else "yellow"
        table.add_section()
        table.add_row("Last Outcome", f"[bold {color}]{outcome}[/bold {color}]")
        table.add_row("Stop Reason", f"[dim]{stop}[/dim]")
        if last_summary.get("beads_status"):
            beads_detail = str(last_summary["beads_status"])
            if last_summary.get("beads_decision_id"):
                beads_detail += f" ({last_summary['beads_decision_id']})"
            table.add_row("BEADS", f"[magenta]{beads_detail}[/magenta]")
            if last_summary.get("beads_summary"):
                table.add_row("BEADS Note", str(last_summary["beads_summary"]))
        if follow_ups:
            table.add_row("Follow-ups", f"[cyan]{len(follow_ups)} goals enqueued[/cyan]")

    if beads_runtime:
        gate_label = "required" if beads_runtime.get("required") else "optional"
        enabled = "on" if beads_runtime.get("enabled") else "off"
        table.add_section()
        table.add_row("Gate", f"[magenta]{enabled} ({gate_label})[/magenta]")
        table.add_row("Scope", f"[dim]{beads_runtime.get('scope', 'goal_run')}[/dim]")

    goal_text = (current_goal[:55] + "…") if len(current_goal) > 56 else (current_goal or "(idle)")
    title = f"[bold blue]Pipeline[/bold blue] [dim]{goal_text}[/dim]"
    if confidence is not None and current_goal:
        color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
        title += f" [dim]|[/dim] [bold {color}]Conf: {confidence * 100:.0f}%[/bold {color}]"

    return Panel(
        table,
        title=title,
        border_style="blue",
    )
