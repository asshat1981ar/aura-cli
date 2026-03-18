"""Helpers for validating and repairing runtime-managed state files."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable

from core.logging_utils import log_json

JsonValidator = Callable[[Any], bool]


def _tmp_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".bak")


def backup_runtime_file(path: Path | str) -> Path | None:
    source = Path(path)
    if not source.exists():
        return None
    backup = _backup_path(source)
    _tmp_write(backup, source.read_text(encoding="utf-8"))
    return backup


def atomic_write_json(
    path: Path | str,
    payload: Any,
    *,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    ensure_backup: bool = True,
) -> None:
    target = Path(path)
    if ensure_backup and target.exists():
        backup_runtime_file(target)
    _tmp_write(target, json.dumps(payload, indent=indent, separators=separators))


def _restore_runtime_backup(path: Path) -> Any | None:
    backup = _backup_path(path)
    if not backup.exists():
        return None
    raw = backup.read_text(encoding="utf-8")
    _tmp_write(path, raw)
    return json.loads(raw)


def load_json_with_repair(
    path: Path | str,
    *,
    default: Any,
    validator: JsonValidator | None,
    state_name: str,
) -> Any:
    target = Path(path)
    if not target.exists():
        return default

    def _validate(payload: Any) -> bool:
        return validator(payload) if validator is not None else True

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if _validate(payload):
            return payload
        raise ValueError("invalid_json_shape")
    except (json.JSONDecodeError, ValueError) as exc:
        restored = _restore_runtime_backup(target)
        if restored is not None and _validate(restored):
            log_json(
                "WARN",
                f"{state_name}_repaired_from_backup",
                details={"path": str(target), "backup": str(_backup_path(target)), "error": str(exc)},
            )
            return restored
        log_json(
            "WARN",
            f"{state_name}_corrupted",
            details={"path": str(target), "error": str(exc), "message": "Starting with empty state."},
        )
        return default


def validate_brain_schema(
    db_path: Path | str,
    *,
    required_tables: tuple[str, ...] = ("schema_version", "memory", "weaknesses", "vector_store_data", "kv_store"),
) -> dict[str, Any]:
    path = Path(db_path)
    result = {
        "path": str(path),
        "ok": False,
        "tables": [],
        "missing_tables": list(required_tables),
        "schema_version": None,
        "error": None,
    }
    if not path.exists():
        result["error"] = "missing_db"
        return result

    try:
        with sqlite3.connect(str(path)) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = sorted(row[0] for row in rows)
            result["tables"] = table_names
            result["missing_tables"] = [name for name in required_tables if name not in table_names]
            if "schema_version" in table_names:
                row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                result["schema_version"] = row[0] if row else None
            result["ok"] = not result["missing_tables"]
    except sqlite3.Error as exc:
        result["error"] = str(exc)
    return result
