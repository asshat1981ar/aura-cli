import json
from pathlib import Path
from typing import Any, Dict, List


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

    def append_log(self, entry: Dict[str, Any]) -> None:
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
