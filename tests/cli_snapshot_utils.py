import json
from pathlib import Path


def snapshot_dir_for(test_file: str) -> Path:
    return Path(test_file).resolve().parent / "snapshots"


def normalized_json_text(raw_json: str) -> str:
    return json.dumps(json.loads(raw_json), indent=2, sort_keys=True) + "\n"


def read_snapshot_text(snapshot_dir: Path, name: str) -> str:
    return (snapshot_dir / name).read_text(encoding="utf-8")


def read_snapshot_json(snapshot_dir: Path, name: str) -> dict:
    return json.loads(read_snapshot_text(snapshot_dir, name))
