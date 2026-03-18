import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from core.runtime_state import atomic_write_json, backup_runtime_file, load_json_with_repair

# Maximum size (bytes) of decision_log.jsonl before rotation
_LOG_MAX_BYTES = int(os.getenv("AURA_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
_LOG_KEEP_ROTATIONS = 3  # number of rotated files to keep


@contextmanager
def _file_lock(path: Path):
    """Acquire an exclusive file lock for the duration of the context."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


_SCHEMA_VERSION = 1


def _unwrap_versioned(raw: Any) -> list:
    """Extract data from a versioned envelope, migrating if needed."""
    if isinstance(raw, dict) and "schema_version" in raw:
        return list(raw.get("data", []))
    # Legacy format: bare list
    if isinstance(raw, list):
        return raw
    return []


def _wrap_versioned(data: list) -> dict:
    """Wrap data in a versioned envelope."""
    return {"schema_version": _SCHEMA_VERSION, "data": data}


class MemoryStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "decision_log.jsonl"

    def _tier_path(self, tier: str) -> Path:
        return self.root / f"{tier}.json"

    def put(self, tier: str, record: Dict[str, Any]) -> None:
        path = self._tier_path(tier)
        with _file_lock(path):
            data: list = []
            if path.exists():
                raw = load_json_with_repair(
                    path,
                    default=[],
                    validator=lambda value: isinstance(value, (dict, list)),
                    state_name=f"memory_store_{tier}",
                )
                data = _unwrap_versioned(raw)
            data.append(record)
            atomic_write_json(path, _wrap_versioned(data), indent=2)

    def query(self, tier: str, limit: int = 100) -> List[Dict[str, Any]]:
        path = self._tier_path(tier)
        if not path.exists():
            return []
        raw = load_json_with_repair(
            path,
            default=[],
            validator=lambda value: isinstance(value, (dict, list)),
            state_name=f"memory_store_{tier}",
        )
        data = _unwrap_versioned(raw)
        return data[-limit:]

    def _rotate_log_if_needed(self) -> None:
        """Rotate decision_log.jsonl when it exceeds *_LOG_MAX_BYTES*.

        Keeps up to *_LOG_KEEP_ROTATIONS* compressed copies named
        ``decision_log.jsonl.1``, ``.2``, …  Older files are deleted.
        """
        if not self.log_path.exists():
            return
        if self.log_path.stat().st_size < _LOG_MAX_BYTES:
            return

        # Shift existing rotations down: .2 → .3, .1 → .2, etc.
        for i in range(_LOG_KEEP_ROTATIONS - 1, 0, -1):
            src = self.log_path.with_suffix(f".jsonl.{i}")
            dst = self.log_path.with_suffix(f".jsonl.{i + 1}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)

        # Rotate current log to .1
        rotated = self.log_path.with_suffix(".jsonl.1")
        self.log_path.rename(rotated)

        # Prune anything beyond the keep window
        excess = self.log_path.with_suffix(f".jsonl.{_LOG_KEEP_ROTATIONS + 1}")
        if excess.exists():
            excess.unlink()

    def append_log(self, entry: Dict[str, Any]) -> None:
        with _file_lock(self.log_path):
            self._rotate_log_if_needed()
            if self.log_path.exists():
                backup_runtime_file(self.log_path)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

    def read_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        entries = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries[-limit:]
