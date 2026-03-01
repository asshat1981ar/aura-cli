import json
import os
from pathlib import Path
from typing import Any, Dict, List

# Maximum size (bytes) of decision_log.jsonl before rotation
_LOG_MAX_BYTES = int(os.getenv("AURA_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
_LOG_KEEP_ROTATIONS = 3  # number of rotated files to keep


class MemoryStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "decision_log.jsonl"

    def _tier_path(self, tier: str) -> Path:
        return self.root / f"{tier}.json"

    def put(self, tier: str, record: Dict[str, Any]) -> None:
        path = self._tier_path(tier)
        data = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = []
        data.append(record)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def query(self, tier: str, limit: int = 100) -> List[Dict[str, Any]]:
        path = self._tier_path(tier)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
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
        self._rotate_log_if_needed()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def read_log(self, limit: int = 0) -> List[Dict[str, Any]]:
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
        return entries[-limit:] if limit > 0 else entries
