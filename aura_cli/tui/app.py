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
import threading
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
from aura_cli.tui.panels.ascm_panel import build_ascm_panel
from core.logging_utils import log_json


class AuraStudio:
    """
    Main AURA TUI application.

    Displays a live dashboard and can drive the autonomous orchestrator loop.
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
        self._last_context_bundle: Optional[Dict[str, Any]] = None
        self._strategy_confidence: Optional[float] = None
        self._is_running: bool = False
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API (Callbacks)
    # ------------------------------------------------------------------

    def on_pipeline_configured(self, config: Dict[str, Any]) -> None:
        """Call when AdaptivePipeline produces a new config."""
        self._strategy_confidence = config.get("confidence")

    def on_context_assembled(self, bundle: Dict[str, Any]) -> None:
        """Call when ASCM produces a new context bundle."""
        self._last_context_bundle = bundle

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

    # ------------------------------------------------------------------
    # Execution Logic
    # ------------------------------------------------------------------

    def start_autonomous_loop(self):
        """Begin processing the goal queue in a background thread."""
        if self._is_running:
            return
        self._is_running = True
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def _worker_loop(self):
        orchestrator = self.runtime.get("orchestrator")
        goal_queue = self.runtime.get("goal_queue")
        if not orchestrator or not goal_queue:
            return

        orchestrator.attach_ui_callback(self)
        
        while not self._stop_event.is_set():
            if not goal_queue.has_goals():
                time.sleep(5)
                continue
            
            try:
                goal = goal_queue.pop()
                self._current_goal = str(goal)
                orchestrator.run_loop(self._current_goal, max_cycles=5)
            except Exception as e:
                log_json("ERROR", "tui_worker_error", details={"error": str(e)})
            
            self._current_goal = ""
            self._phases_status = {}

    def stop(self):
        """Stop both the TUI and the background loop."""
        self._stop_event.set()
        self._is_running = False

    def render_once(self) -> None:
        """Print a single snapshot of all panels (no live update)."""
        if not _RICH_AVAILABLE:
            self._render_plain()
            return
        layout = self._build_layout()
        self._console.print(layout)

    def run(self, seconds: Optional[float] = None, autonomous: bool = False) -> None:
        """
        Start the live TUI dashboard.

        Args:
            seconds: If set, auto-exit after this many seconds.
            autonomous: If True, start processing the goal queue in background.
        """
        if not _RICH_AVAILABLE:
            print("rich not installed. Run: pip install rich")
            return

        if autonomous:
            self.start_autonomous_loop()

        start = time.time()
        try:
            with Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=1 / self.refresh_rate,
                screen=True,
            ) as live:
                while not self._stop_event.is_set():
                    time.sleep(self.refresh_rate)
                    live.update(self._build_layout())
                    if seconds and (time.time() - start) >= seconds:
                        break
        except KeyboardInterrupt:
            self.stop()

    # ------------------------------------------------------------------
    # Layout assembly
    # ------------------------------------------------------------------

    def _build_layout(self):
        """Assemble the multi-panel layout."""
        if not _RICH_AVAILABLE:
            return ""

        layout = Layout()
        layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="pipeline", ratio=1),
            Layout(name="ascm", ratio=1),
            Layout(name="memory", ratio=2),
        )
        layout["right"].split_column(
            Layout(name="queue", ratio=2),
            Layout(name="metrics", ratio=1),
        )

        goal_queue = self.runtime.get("goal_queue")
        brain = self.runtime.get("brain")
        cycle_log = self._cycle_log

        layout["pipeline"].update(
            build_cycle_panel(
                current_goal=self._current_goal,
                phases_status=self._phases_status,
                current_phase=self._current_phase,
                confidence=self._strategy_confidence,
            )
        )
        layout["ascm"].update(build_ascm_panel(bundle=self._last_context_bundle))
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
