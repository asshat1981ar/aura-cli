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


def _run_status_label(entry: Dict[str, Any]) -> str:
    if entry.get("timed_out"):
        return "timeout"
    if entry.get("truncated"):
        return "truncated"
    code = entry.get("code")
    if code in (None, 0):
        return "ok"
    return f"exit {code}"


def build_metrics_panel(cycle_log: Optional[List[Dict]] = None, *, run_tool_audit: Optional[Dict[str, Any]] = None) -> "Panel":
    """Build the performance metrics panel."""
    if not _RICH_AVAILABLE:
        return ""

    cycle_log = cycle_log or []

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Metric", style="bold", width=14)
    table.add_column("Value")

    recent = cycle_log[-10:]
    durations = [c.get("duration_s", 0) for c in recent if c.get("duration_s") is not None]
    successes = [c for c in recent if c.get("outcome") == "SUCCESS"]
    skipped = [c for c in recent if c.get("outcome") == "SKIPPED"]

    if recent:
        avg_s = statistics.mean(durations) if durations else 0
        p95_s = sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 2 else (durations[0] if durations else 0)
        success_pct = int(len(successes) / len(recent) * 100) if recent else 0
        spark = _sparkline(durations)

        table.add_row("total cycles", str(len(cycle_log)))
        table.add_row("avg (last 10)", f"{avg_s:.1f}s")
        table.add_row("p95 (last 10)", f"{p95_s:.1f}s")
        table.add_row("success rate", f"{success_pct}% ({len(successes)} pass, {len(skipped)} skip)")
        table.add_row("trend", f"[cyan]{spark}[/cyan]")
    else:
        table.add_row("[dim]No cycles recorded yet[/dim]", "")

    if run_tool_audit:
        table.add_row("", "")
        table.add_row("[bold]run tool[/bold]", "")
        table.add_row("tracked", str(run_tool_audit.get("count", 0)))
        table.add_row("last cmd", str(run_tool_audit.get("last_command") or "n/a"))
        table.add_row(
            "outcomes",
            (
                f"{run_tool_audit.get('success_count', 0)} ok, "
                f"{run_tool_audit.get('error_count', 0)} err, "
                f"{run_tool_audit.get('timeout_count', 0)} timeout, "
                f"{run_tool_audit.get('truncated_count', 0)} trunc"
            ),
        )
        for entry in list(run_tool_audit.get("recent", []))[-3:]:
            command = str(entry.get("command") or "")
            compact_command = command if len(command) <= 32 else f"{command[:29]}..."
            table.add_row("recent", f"{_run_status_label(entry)} | {compact_command}")

    title = f"[bold yellow]Metrics[/bold yellow] [dim](last {min(10, len(cycle_log))} cycles)[/dim]"
    return Panel(table, title=title, border_style="yellow")
