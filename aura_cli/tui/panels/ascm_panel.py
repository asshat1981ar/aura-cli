"""ASCM Context Inspector panel for AURA TUI."""
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

def build_ascm_panel(bundle: Optional[Dict[str, Any]]) -> "Panel":
    """Build the context inspector panel."""
    if not _RICH_AVAILABLE:
        return ""

    if not bundle:
        return Panel(
            Text("Waiting for context assembly...", justify="center", style="dim"),
            title="[bold magenta]Context Inspector[/bold magenta]",
            border_style="magenta",
        )

    # 1. Summary Info
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Snippets", str(len(bundle.get("snippets", []))))
    table.add_row("Insights", str(len(bundle.get("related_insights", []))))
    table.add_row("Memory",   str(len(bundle.get("memory", []))))
    
    # 2. Budget Breakdown
    report = bundle.get("budget_report", {})
    limit = report.get("total_limit", 8000)
    used = report.get("total_used", 0)
    
    usage_pct = (used / limit * 100) if limit > 0 else 0
    bar_width = 20
    filled = int(usage_pct / 100 * bar_width)
    bar = "[" + "=" * filled + " " * (bar_width - filled) + "]"
    
    color = "green"
    if usage_pct > 90: color = "red"
    elif usage_pct > 70: color = "yellow"
    
    table.add_row("Token Budget", f"[{color}]{bar}[/{color}] {usage_pct:.1f}%")
    table.add_row("Used/Limit", f"{used} / {limit}")

    return Panel(
        table,
        title="[bold magenta]Context Inspector[/bold magenta]",
        border_style="magenta",
    )
