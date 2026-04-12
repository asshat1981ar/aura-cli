"""Observability panel for AURA TUI — real-time metrics and traces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.tree import Tree

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def _format_duration(ms: Optional[float]) -> str:
    """Format duration in human-readable form."""
    if ms is None:
        return "N/A"
    if ms < 1:
        return f"{ms * 1000:.0f}μs"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.1f}s"


def _sparkline(values: List[float], width: int = 15) -> str:
    """Generate Unicode sparkline."""
    bars = "▁▂▃▄▅▆▇█"
    if not values:
        return "─" * width
    mn, mx = min(values), max(values)
    if mx == mn:
        return bars[3] * min(len(values), width)
    normalized = [(v - mn) / (mx - mn) for v in values[-width:]]
    return "".join(bars[int(n * (len(bars) - 1))] for n in normalized)


def build_observability_panel(
    metrics_store: Optional[Any] = None,
    tracer: Optional[Any] = None,
    max_metrics: int = 8,
    max_traces: int = 5,
) -> "Panel":
    """Build the observability panel showing metrics and traces.
    
    Args:
        metrics_store: MetricsStore instance
        tracer: Tracer instance  
        max_metrics: Max number of metrics to display
        max_traces: Max number of recent traces to display
    """
    if not _RICH_AVAILABLE:
        return ""

    # Main layout table
    layout = Table.grid(expand=True)
    layout.add_column("left", ratio=1)
    layout.add_column("right", ratio=1)

    # Left: Metrics
    metrics_panel = _build_metrics_section(metrics_store, max_metrics)
    
    # Right: Active Traces
    traces_panel = _build_traces_section(tracer, max_traces)
    
    layout.add_row(metrics_panel, traces_panel)

    return Panel(
        layout,
        title="[bold cyan]Observability[/bold cyan]",
        border_style="cyan",
    )


def _build_metrics_section(
    metrics_store: Optional[Any],
    max_metrics: int,
) -> Panel:
    """Build the metrics subsection."""
    if not metrics_store:
        return Panel(
            "[dim]Metrics not initialized[/dim]",
            title="[bold]Metrics[/bold]",
            border_style="dim",
        )

    table = Table(box=box.SIMPLE, show_header=True, expand=True)
    table.add_column("Metric", style="bold", width=20)
    table.add_column("Type", width=8)
    table.add_column("Value", width=12)
    table.add_column("Trend", width=15)

    try:
        snapshot = metrics_store.snapshot()
        sorted_metrics = sorted(
            snapshot.items(),
            key=lambda x: x[1].get("stats", {}).get("count", 0),
            reverse=True,
        )[:max_metrics]

        for name, data in sorted_metrics:
            metric_type = data.get("type", "GAUGE")
            stats = data.get("stats", {})
            latest = data.get("latest")
            
            # Format value based on type
            if metric_type == "COUNTER":
                value_str = f"{stats.get('sum', 0):.0f}"
            elif metric_type == "TIMER":
                value_str = _format_duration(latest)
            else:
                value_str = f"{latest:.2f}" if latest is not None else "N/A"
            
            # Get trend sparkline for timers/gauges
            trend = ""
            if metric_type in ("TIMER", "HISTOGRAM", "GAUGE") and stats.get("count", 0) > 1:
                # Get recent values if available
                series = metrics_store.get_series(name)
                if series and len(series.values) > 1:
                    values = [v.value for v in list(series.values)[-15:]]
                    trend = f"[cyan]{_sparkline(values)}[/cyan]"
            
            table.add_row(name[:20], metric_type[:8], value_str, trend)

    except Exception as e:
        table.add_row("[red]Error loading metrics[/red]", "", "", str(e)[:20])

    return Panel(table, title="[bold]Metrics[/bold]", border_style="blue")


def _build_traces_section(
    tracer: Optional[Any],
    max_traces: int,
) -> Panel:
    """Build the active traces subsection."""
    # Get current span info if available
    current_span = None
    if tracer:
        try:
            current_span = tracer.get_current_span()
        except Exception:
            pass

    if not current_span:
        return Panel(
            "[dim]No active traces[/dim]",
            title="[bold]Traces[/bold]",
            border_style="dim",
        )

    # Build trace tree
    tree = Tree(f"[bold]{current_span.name}[/bold]")
    
    # Add span details
    duration = current_span.duration_ms
    if duration is not None:
        tree.add(f"Duration: {_format_duration(duration)}")
    
    status = current_span.status.name if hasattr(current_span.status, 'name') else str(current_span.status)
    status_color = "green" if status == "OK" else "red" if status == "ERROR" else "yellow"
    tree.add(f"Status: [{status_color}]{status}[/{status_color}]")
    
    # Add attributes (limited)
    if current_span.attributes:
        attrs_tree = tree.add("Attributes")
        for key, value in list(current_span.attributes.items())[:5]:
            attrs_tree.add(f"{key}: {str(value)[:30]}")
    
    # Add recent events
    if current_span.events:
        events_tree = tree.add("Recent Events")
        for event in current_span.events[-3:]:
            events_tree.add(f"• {event.name}")

    return Panel(tree, title="[bold]Active Trace[/bold]", border_style="green")


def build_health_panel(
    health_status: Optional[Dict[str, Any]] = None,
) -> Panel:
    """Build a health status panel.
    
    Args:
        health_status: Dict with component health info
    """
    if not _RICH_AVAILABLE:
        return ""

    if not health_status:
        return Panel(
            "[dim]Health data not available[/dim]",
            title="[bold yellow]Health[/bold yellow]",
            border_style="yellow",
        )

    table = Table(box=box.SIMPLE, show_header=True, expand=True)
    table.add_column("Component", style="bold")
    table.add_column("Status", width=10)
    table.add_column("Details")

    status_colors = {
        "healthy": "green",
        "degraded": "yellow", 
        "unhealthy": "red",
        "unknown": "dim",
    }

    for component, info in health_status.items():
        status = info.get("status", "unknown")
        color = status_colors.get(status, "white")
        details = info.get("message", "")
        
        table.add_row(
            component,
            f"[{color}]{status}[/{color}]",
            details[:40],
        )

    return Panel(
        table,
        title="[bold yellow]Health[/bold yellow]",
        border_style="yellow",
    )
