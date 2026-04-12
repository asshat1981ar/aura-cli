"""
Heartbeat manager for periodic maintenance tasks.

Inspired by openclaw-workspace batching patterns.  Registers named
health-checks with configurable intervals, persists state between
runs, and executes only the most overdue checks per tick.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.logging_utils import log_json

_DEFAULT_STATE_PATH = Path.home() / ".aura" / "heartbeat_state.json"


@dataclass
class HeartbeatCheck:
    """A single registered periodic check."""

    name: str
    callback: Callable[[], Any]
    interval_seconds: float = 3600.0  # default: once per hour
    enabled: bool = True


@dataclass
class HeartbeatResult:
    """Outcome of a single check execution."""

    name: str
    success: bool
    duration_seconds: float = 0.0
    detail: Any = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class HeartbeatManager:
    """Manages periodic maintenance checks for AURA.

    Features:
      - Register named checks with callbacks and intervals
      - ``tick()`` runs the most overdue checks (capped per call)
      - Persists last-run timestamps to ``~/.aura/heartbeat_state.json``
      - Supports ``force_run()``, ``is_due()``, and ``status()``
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        max_checks_per_tick: int = 2,
    ):
        self.state_path: Path = state_path or _DEFAULT_STATE_PATH
        self.max_checks_per_tick = max_checks_per_tick
        self._checks: Dict[str, HeartbeatCheck] = {}
        self._last_run: Dict[str, float] = {}
        self._results: Dict[str, HeartbeatResult] = {}
        self._load_state()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        callback: Callable[[], Any],
        interval_seconds: float = 3600.0,
        enabled: bool = True,
    ) -> None:
        """Register a new periodic check."""
        self._checks[name] = HeartbeatCheck(
            name=name,
            callback=callback,
            interval_seconds=interval_seconds,
            enabled=enabled,
        )
        if name not in self._last_run:
            self._last_run[name] = 0.0
        log_json(
            "INFO",
            "heartbeat_check_registered",
            details={"name": name, "interval": interval_seconds, "enabled": enabled},
        )

    # ------------------------------------------------------------------
    # Tick — run due checks
    # ------------------------------------------------------------------
    def tick(self) -> List[HeartbeatResult]:
        """Execute up to ``max_checks_per_tick`` overdue checks.

        Checks are prioritised by how overdue they are (most overdue first).
        """
        now = time.time()
        due: List[tuple] = []  # (overdue_seconds, name)

        for name, check in self._checks.items():
            if not check.enabled:
                continue
            elapsed = now - self._last_run.get(name, 0.0)
            if elapsed >= check.interval_seconds:
                due.append((elapsed - check.interval_seconds, name))

        # Sort most-overdue first
        due.sort(key=lambda t: t[0], reverse=True)
        to_run = [name for _, name in due[: self.max_checks_per_tick]]

        results: List[HeartbeatResult] = []
        for name in to_run:
            result = self._execute_check(name)
            results.append(result)

        if results:
            self._save_state()

        return results

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def is_due(self, name: str) -> bool:
        """Return True if the named check is overdue."""
        check = self._checks.get(name)
        if check is None or not check.enabled:
            return False
        elapsed = time.time() - self._last_run.get(name, 0.0)
        return elapsed >= check.interval_seconds

    def force_run(self, name: str) -> Optional[HeartbeatResult]:
        """Run a check immediately regardless of its schedule."""
        if name not in self._checks:
            log_json("WARN", "heartbeat_force_run_unknown", details={"name": name})
            return None
        result = self._execute_check(name)
        self._save_state()
        return result

    def status(self) -> Dict[str, Any]:
        """Return a summary of all registered checks and their state."""
        now = time.time()
        entries: Dict[str, Any] = {}
        for name, check in self._checks.items():
            last = self._last_run.get(name, 0.0)
            last_result = self._results.get(name)
            entries[name] = {
                "enabled": check.enabled,
                "interval_seconds": check.interval_seconds,
                "last_run": last,
                "seconds_since_last_run": round(now - last, 2) if last else None,
                "is_due": self.is_due(name),
                "last_success": last_result.success if last_result else None,
            }
        return entries

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------
    def _execute_check(self, name: str) -> HeartbeatResult:
        check = self._checks[name]
        start = time.time()
        try:
            detail = check.callback()
            duration = time.time() - start
            result = HeartbeatResult(
                name=name,
                success=True,
                duration_seconds=round(duration, 4),
                detail=detail,
            )
            log_json(
                "INFO",
                "heartbeat_check_passed",
                details={"name": name, "duration": result.duration_seconds},
            )
        except Exception as exc:
            duration = time.time() - start
            result = HeartbeatResult(
                name=name,
                success=False,
                duration_seconds=round(duration, 4),
                error=str(exc),
            )
            log_json(
                "WARN",
                "heartbeat_check_failed",
                details={"name": name, "error": str(exc)},
            )

        self._last_run[name] = time.time()
        self._results[name] = result
        return result

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        """Load last-run timestamps from disk."""
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._last_run = {k: float(v) for k, v in data.items()}
            log_json("INFO", "heartbeat_state_loaded",
                     details={"path": str(self.state_path), "checks": len(data)})
        except Exception as exc:
            log_json("WARN", "heartbeat_state_load_failed",
                     details={"error": str(exc)})

    def _save_state(self) -> None:
        """Persist last-run timestamps to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(self._last_run, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log_json("WARN", "heartbeat_state_save_failed",
                     details={"error": str(exc)})
