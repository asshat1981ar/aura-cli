"""Performance metrics panel for AURA TUI."""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def _sparkline(values: List[float], width: int = 10) -> str:
    """Generate a Unicode sparkline from a list of floats."""
    bars = "▁▂▃▄▅▆▇█"
    if not values:
        return "─" * width
    mn, mx = min(values), max(values)
    if mx == mn:
        return bars[3] * min(len(values), width)
    normalized = [(v - mn) / (mx - mn) for v in values[-width:]]
    return "".join(bars[int(n * (len(bars) - 1))] for n in normalized)


def build_metrics_panel(cycle_log: Optional[List[Dict]] = None) -> "Panel":
    """Build the performance metrics panel."""
    if not _RICH_AVAILABLE:
        return ""

    cycle_log = cycle_log or []

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Metric", style="bold", width=14)
    table.add_column("Value")

    recent = cycle_log[-10:]
    durations = [c.get("duration_s", 0) for c in recent if "duration_s" in c]
    successes = [c for c in recent if c.get("success")]

    if recent:
        avg_s = statistics.mean(durations) if durations else 0
        p95_s = sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 2 else (durations[0] if durations else 0)
        success_pct = int(len(successes) / len(recent) * 100) if recent else 0
        spark = _sparkline(durations)

        table.add_row("total cycles", str(len(cycle_log)))
        table.add_row("avg (last 10)", f"{avg_s:.1f}s")
        table.add_row("p95 (last 10)", f"{p95_s:.1f}s")
        table.add_row("success rate", f"{success_pct}%")
        table.add_row("trend", f"[cyan]{spark}[/cyan]")
    else:
        table.add_row("[dim]No cycles recorded yet[/dim]", "")

    title = f"[bold yellow]Metrics[/bold yellow] [dim](last {min(10, len(cycle_log))} cycles)[/dim]"
    return Panel(table, title=title, border_style="yellow")
