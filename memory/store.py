"""
MemoryStore — simple JSON-backed persistent storage.

Provides two storage primitives:

1. **Tier store** — named JSON files holding a list of records.
   Used for summaries, health snapshots, cycle history, etc.
   ``put(tier, record)`` appends; ``query(tier, limit)`` retrieves the last N.

2. **Decision log** — a single append-only JSONL file (``decision_log.jsonl``).
   ``append_log(entry)`` writes one entry; ``read_log()`` returns all entries.

Usage::

    from memory.store import MemoryStore
    from pathlib import Path

    store = MemoryStore(Path("memory/store"))
    store.put("cycle_summaries", {"goal": "...", "status": "pass"})
    store.append_log({"cycle_id": "abc", "goal_type": "bug_fix", "phase_outputs": {}})
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class MemoryStore:
    """JSON-file-backed persistent memory store.

    Args:
        root: Directory in which tier files and the decision log are kept.
              Created automatically if it does not exist.
    """

    _LOG_FILE = "decision_log.jsonl"

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ── Tier store ────────────────────────────────────────────────────────────

    def put(self, tier: str, record: Dict[str, Any]) -> None:
        """Append *record* to the JSON list stored in ``<tier>.json``."""
        path = self._tier_path(tier)
        records = self._read_json_list(path)
        records.append(record)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def query(self, tier: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the last *limit* records from ``<tier>.json``."""
        records = self._read_json_list(self._tier_path(tier))
        return records[-limit:]

    # ── Decision log ──────────────────────────────────────────────────────────

    def append_log(self, entry: Dict[str, Any]) -> None:
        """Append *entry* as a single JSON line to the decision log."""
        log_path = self._root / self._LOG_FILE
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def read_log(self) -> List[Dict[str, Any]]:
        """Return all entries in the decision log."""
        log_path = self._root / self._LOG_FILE
        if not log_path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for lineno, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    from core.logging_utils import log_json
                    log_json("WARN", "decision_log_parse_error",
                             details={"line": lineno, "error": str(exc)})
        return entries

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tier_path(self, tier: str) -> Path:
        return self._root / f"{tier}.json"

    def _read_json_list(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return []
