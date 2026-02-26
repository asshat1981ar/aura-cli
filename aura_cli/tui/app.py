"""
AURA Studio — real-time TUI dashboard.

Usage:
    from aura_cli.tui.app import AuraStudio
    app = AuraStudio(runtime=runtime_dict)
    app.run()          # blocking — starts live display
    app.render_once()  # print single snapshot (no live update)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from aura_cli.tui.panels.cycle_panel import build_cycle_panel
from aura_cli.tui.panels.queue_panel import build_queue_panel
from aura_cli.tui.panels.memory_panel import build_memory_panel
from aura_cli.tui.panels.metrics_panel import build_metrics_panel


class AuraStudio:
    """
    Main AURA TUI application.

    Displays a 4-panel live dashboard:
      ┌─ Pipeline ──┬─ Goal Queue ──┐
      │             │               │
      ├─ Memory ────┼─ Metrics ─────┤
      │             │               │
      └─────────────┴───────────────┘
    """

    def __init__(
        self,
        runtime: Optional[Dict[str, Any]] = None,
        refresh_rate: float = 1.0,
    ):
        self.runtime = runtime or {}
        self.refresh_rate = refresh_rate
        self._console = Console() if _RICH_AVAILABLE else None
        self._cycle_log: List[Dict] = []
        self._current_goal: str = ""
        self._current_phase: str = ""
        self._phases_status: Dict[str, str] = {}  # phase → "✅"|"⟳"|"○"|"❌"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_phase_start(self, phase: str) -> None:
        """Call at start of each pipeline phase."""
        self._current_phase = phase
        self._phases_status[phase] = "⟳"

    def on_phase_complete(self, phase: str, elapsed_ms: float, success: bool = True) -> None:
        """Call at end of each pipeline phase."""
        self._phases_status[phase] = "✅" if success else "❌"

    def on_cycle_start(self, goal: str) -> None:
        """Call when a new cycle begins."""
        self._current_goal = goal
        self._phases_status = {}

    def on_cycle_complete(self, outcome: Dict) -> None:
        """Call when a cycle finishes — append to cycle log."""
        self._cycle_log.append({
            "goal": outcome.get("goal", ""),
            "success": outcome.get("success", False),
            "duration_s": outcome.get("duration_s", 0),
            "ts": time.time(),
        })
        # Keep only last 50 cycles
        self._cycle_log = self._cycle_log[-50:]

    def render_once(self) -> None:
        """Print a single snapshot of all panels (no live update)."""
        if not _RICH_AVAILABLE:
            self._render_plain()
            return
        layout = self._build_layout()
        self._console.print(layout)

    def run(self, seconds: Optional[float] = None) -> None:
        """
        Start the live TUI dashboard.

        Args:
            seconds: If set, auto-exit after this many seconds. Otherwise runs
                     until Ctrl-C.
        """
        if not _RICH_AVAILABLE:
            print("rich not installed. Run: pip install rich")
            return

        start = time.time()
        try:
            with Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=1 / self.refresh_rate,
                screen=True,
            ) as live:
                while True:
                    time.sleep(self.refresh_rate)
                    live.update(self._build_layout())
                    if seconds and (time.time() - start) >= seconds:
                        break
        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------
    # Layout assembly
    # ------------------------------------------------------------------

    def _build_layout(self):
        """Assemble the 4-panel layout."""
        if not _RICH_AVAILABLE:
            return ""

        layout = Layout()
        layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="pipeline"),
            Layout(name="memory"),
        )
        layout["right"].split_column(
            Layout(name="queue"),
            Layout(name="metrics"),
        )

        goal_queue = self.runtime.get("goal_queue")
        brain = self.runtime.get("brain")
        cycle_log = self._cycle_log

        layout["pipeline"].update(
            build_cycle_panel(
                current_goal=self._current_goal,
                phases_status=self._phases_status,
                current_phase=self._current_phase,
            )
        )
        layout["queue"].update(build_queue_panel(goal_queue=goal_queue))
        layout["memory"].update(build_memory_panel(brain=brain))
        layout["metrics"].update(build_metrics_panel(cycle_log=cycle_log))

        return layout

    def _render_plain(self) -> None:
        """Fallback plain-text render when rich is unavailable."""
        print("=== AURA Studio ===")
        print(f"Current goal: {self._current_goal or '(none)'}")
        print(f"Current phase: {self._current_phase or '(none)'}")
        if self._phases_status:
            for phase, status in self._phases_status.items():
                print(f"  {status} {phase}")
        print(f"Cycles completed: {len(self._cycle_log)}")
