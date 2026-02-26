"""
Autonomous Discovery — Self-propagating work discovery from codebase signals.

Scans the project for latent signals that represent work opportunities,
converts them into prioritised goals, and enqueues them.  Runs every N
cycles as an improvement loop.

Discovery signals
-----------------
todo_fixme       Lines containing TODO / FIXME / HACK / XXX
type_ignore      Lines with ``# type: ignore`` or ``# noqa``
missing_tests    Python modules in non-test dirs that lack a corresponding
                 test file in ``tests/``
dead_imports     Import statements in __init__.py files that reference
                 non-existent modules
low_coverage     Files listed in health snapshot with coverage < threshold
duplicate_code   Files flagged by code_clone_detector (if available)

All discovered items are content-hashed to prevent re-queuing the same item
across sessions.  Hashes persisted in ``memory/discovery_seen.json``.

Usage::

    from core.autonomous_discovery import AutonomousDiscovery
    discovery = AutonomousDiscovery(goal_queue, memory_store, project_root=".")
    discovery.on_cycle_complete(cycle_entry)   # auto-trigger every N cycles
    report = discovery.run_scan()              # or trigger manually
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.logging_utils import log_json

# Lines to scan for (pattern, goal_type, priority)
_CODE_SIGNAL_PATTERNS: List[Tuple[str, str, str]] = [
    (r"#\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.+)",  "todo_fixme",   "low"),
    (r"#\s*type:\s*ignore",                       "type_ignore",  "medium"),
    (r"#\s*noqa",                                  "noqa",         "low"),
]

# Directories that contain source (not tests, not vendor)
_SOURCE_DIRS = ["agents", "core", "aura_cli", "memory", "tools"]
_TEST_DIR = "tests"

# Skip files matching these patterns
_SKIP_PATTERNS = [
    "__pycache__", ".git", "node_modules", "*.pyc",
    "test_", "_test.py",
]

# Max goals emitted per scan to avoid flooding the queue
MAX_GOALS_PER_SCAN = 8
TRIGGER_EVERY_N = 15


class AutonomousDiscovery:
    """Scan the codebase for work opportunities and queue them as goals."""

    def __init__(
        self,
        goal_queue,
        memory_store,
        project_root: str = ".",
    ):
        self.queue = goal_queue
        self.memory = memory_store
        self.root = Path(project_root)
        self._seen_path = self.root / "memory" / "discovery_seen.json"
        self._seen: set = self._load_seen()
        self._cycle_count = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def on_cycle_complete(self, _entry: Dict[str, Any]) -> None:
        """Trigger a scan every TRIGGER_EVERY_N cycles."""
        self._cycle_count += 1
        if self._cycle_count % TRIGGER_EVERY_N == 0:
            self.run_scan()

    def run_scan(self) -> Dict[str, Any]:
        """Run all discovery scanners.  Never raises."""
        try:
            return self._scan()
        except Exception as exc:
            log_json("ERROR", "autonomous_discovery_failed", details={"error": str(exc)})
            return {"error": str(exc)}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _scan(self) -> Dict[str, Any]:
        log_json("INFO", "autonomous_discovery_scan_start",
                 details={"root": str(self.root)})

        findings: List[Dict] = []
        findings += self._scan_code_signals()
        findings += self._scan_missing_tests()

        # Sort by priority (high first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        findings.sort(key=lambda f: priority_order.get(f.get("priority", "low"), 2))

        # Deduplicate and cap
        new_goals: List[str] = []
        for f in findings:
            if len(new_goals) >= MAX_GOALS_PER_SCAN:
                break
            item_hash = f["hash"]
            if item_hash in self._seen:
                continue
            self._seen.add(item_hash)
            self.queue.add(f["goal"])
            new_goals.append(f["goal"])
            log_json("INFO", "autonomous_discovery_goal_queued",
                     details={"goal": f["goal"][:80], "signal": f["signal"]})

        self._save_seen()
        report = {
            "scan_timestamp": time.time(),
            "findings_total": len(findings),
            "new_goals": len(new_goals),
            "goals": new_goals,
        }
        self.memory.put("discovery_reports", report)
        log_json("INFO", "autonomous_discovery_scan_complete",
                 details={"new_goals": len(new_goals), "seen_total": len(self._seen)})
        return report

    def _scan_code_signals(self) -> List[Dict]:
        """Scan Python files for TODO, FIXME, type:ignore patterns."""
        findings: List[Dict] = []
        for src_dir in _SOURCE_DIRS:
            dir_path = self.root / src_dir
            if not dir_path.is_dir():
                continue
            for py_file in dir_path.rglob("*.py"):
                if self._should_skip(py_file):
                    continue
                try:
                    self._scan_file_signals(py_file, findings)
                except Exception:
                    continue
        return findings

    def _scan_file_signals(self, path: Path, findings: List[Dict]) -> None:
        rel = str(path.relative_to(self.root))
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        for lineno, line in enumerate(content.splitlines(), 1):
            for pattern, signal, priority in _CODE_SIGNAL_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if not match:
                    continue
                if signal == "todo_fixme":
                    description = match.group(2).strip()[:80] if match.lastindex >= 2 else line.strip()[:60]
                    goal = f"Address {match.group(1).upper()} in {rel}:{lineno} — {description}"
                    priority = "medium" if "fixme" in match.group(1).lower() else "low"
                else:
                    goal = f"Fix {signal} annotation in {rel}:{lineno}"
                item_hash = hashlib.sha256(f"{rel}:{lineno}:{signal}".encode()).hexdigest()[:16]
                findings.append({
                    "signal": signal,
                    "priority": priority,
                    "goal": goal,
                    "hash": item_hash,
                    "file": rel,
                    "line": lineno,
                })

    def _scan_missing_tests(self) -> List[Dict]:
        """Find source modules without a corresponding test file."""
        findings: List[Dict] = []
        test_dir = self.root / _TEST_DIR
        if not test_dir.is_dir():
            return findings

        existing_tests = {
            f.stem.removeprefix("test_")
            for f in test_dir.glob("test_*.py")
        }

        for src_dir in _SOURCE_DIRS:
            dir_path = self.root / src_dir
            if not dir_path.is_dir():
                continue
            for py_file in dir_path.glob("*.py"):
                if self._should_skip(py_file):
                    continue
                module_name = py_file.stem
                if module_name.startswith("_") or module_name in existing_tests:
                    continue
                # Only suggest tests for files with actual function/class definitions
                if not self._has_definitions(py_file):
                    continue
                rel = str(py_file.relative_to(self.root))
                goal = f"Add unit tests for {rel} (no test file found in {_TEST_DIR}/)"
                item_hash = hashlib.sha256(f"missing_test:{rel}".encode()).hexdigest()[:16]
                findings.append({
                    "signal": "missing_tests",
                    "priority": "low",
                    "goal": goal,
                    "hash": item_hash,
                    "file": rel,
                    "line": 0,
                })
        return findings

    def _has_definitions(self, path: Path) -> bool:
        """Return True if file contains at least one function or class definition."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       for n in ast.walk(tree))
        except Exception:
            return False

    def _should_skip(self, path: Path) -> bool:
        parts = str(path)
        return any(skip in parts for skip in _SKIP_PATTERNS)

    def _load_seen(self) -> set:
        try:
            if self._seen_path.exists():
                data = json.loads(self._seen_path.read_text(encoding="utf-8"))
                return set(data) if isinstance(data, list) else set()
        except Exception:
            pass
        return set()

    def _save_seen(self) -> None:
        try:
            self._seen_path.parent.mkdir(parents=True, exist_ok=True)
            self._seen_path.write_text(
                json.dumps(sorted(self._seen), indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log_json("WARN", "autonomous_discovery_save_failed",
                     details={"error": str(exc)})
