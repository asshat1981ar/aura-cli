"""SQLite schema and helpers for the semantic codebase index."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_BASE_SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    module_name TEXT,
    cluster TEXT,
    line_count INTEGER NOT NULL DEFAULT 0,
    last_modified TEXT,
    last_scan_sha TEXT,
    module_summary TEXT,
    coupling_score REAL NOT NULL DEFAULT 0.0,
    scanned_at TEXT
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    signature TEXT,
    docstring TEXT,
    decorators TEXT,
    intent_summary TEXT
);

CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    imported_module TEXT NOT NULL,
    imported_name TEXT,
    is_from_import INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS call_sites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    callee_name TEXT NOT NULL,
    line INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    to_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    rel_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.0,
    UNIQUE(from_file_id, to_file_id, rel_type)
);

CREATE TABLE IF NOT EXISTS scan_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_sha TEXT,
    scan_time TEXT NOT NULL,
    files_scanned INTEGER NOT NULL DEFAULT 0,
    symbols_found INTEGER NOT NULL DEFAULT 0,
    llm_calls_made INTEGER NOT NULL DEFAULT 0,
    llm_cost_usd REAL NOT NULL DEFAULT 0.0,
    scan_type TEXT NOT NULL
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    docstring,
    intent_summary,
    content='symbols',
    content_rowid='id'
);
"""


class SemanticDB:
    """Thin SQLite wrapper for the semantic index."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.fts_enabled = True
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(_BASE_SCHEMA)
        try:
            self.conn.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError as exc:
            self.fts_enabled = False
            logger.warning("sqlite_fts5_unavailable: %s", exc)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _row_to_dict(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        return dict(row)

    def _rows_to_dicts(self, rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
        return [dict(row) for row in rows]

    def list_tables(self) -> List[str]:
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table') ORDER BY name").fetchall()
        return [row["name"] for row in rows]

    def checkpoint(self, mode: str = "TRUNCATE") -> None:
        self.conn.execute(f"PRAGMA wal_checkpoint({mode})")

    def clear_all(self) -> None:
        for table in ("relationships", "call_sites", "imports", "symbols", "files", "scan_meta"):
            self.conn.execute(f"DELETE FROM {table}")
        if self.fts_enabled:
            self.conn.execute("DELETE FROM symbols_fts")
        self.conn.commit()

    def clear_file_data(self, file_id: int) -> None:
        self.conn.execute("DELETE FROM relationships WHERE from_file_id = ? OR to_file_id = ?", (file_id, file_id))
        self.conn.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
        self.conn.execute(
            "DELETE FROM call_sites WHERE caller_symbol_id IN (SELECT id FROM symbols WHERE file_id = ?)",
            (file_id,),
        )
        self.conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
        if self.fts_enabled:
            self.conn.execute("DELETE FROM symbols_fts WHERE rowid NOT IN (SELECT id FROM symbols)")
        self.conn.commit()

    def upsert_file(
        self,
        path: str,
        module_name: str,
        cluster: str,
        line_count: int,
        last_modified: str,
        last_scan_sha: str | None,
        module_summary: str | None = None,
        scanned_at: str | None = None,
    ) -> int:
        existing = self.get_file_by_path(path)
        if existing:
            self.conn.execute(
                """
                UPDATE files
                SET module_name = ?, cluster = ?, line_count = ?, last_modified = ?,
                    last_scan_sha = ?, module_summary = COALESCE(?, module_summary),
                    scanned_at = COALESCE(?, scanned_at)
                WHERE id = ?
                """,
                (
                    module_name,
                    cluster,
                    line_count,
                    last_modified,
                    last_scan_sha,
                    module_summary,
                    scanned_at,
                    existing["id"],
                ),
            )
            self.conn.commit()
            return int(existing["id"])

        cursor = self.conn.execute(
            """
            INSERT INTO files(path, module_name, cluster, line_count, last_modified, last_scan_sha, module_summary, scanned_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (path, module_name, cluster, line_count, last_modified, last_scan_sha, module_summary, scanned_at),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def update_file_summary(self, file_id: int, summary: str | None) -> None:
        self.conn.execute("UPDATE files SET module_summary = ? WHERE id = ?", (summary, file_id))
        self.conn.commit()

    def update_file_coupling(self, file_id: int, coupling_score: float) -> None:
        self.conn.execute("UPDATE files SET coupling_score = ? WHERE id = ?", (coupling_score, file_id))
        self.conn.commit()

    def get_file_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return self._row_to_dict(row)

    def get_file_by_module(self, module_name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM files WHERE module_name = ?", (module_name,)).fetchone()
        return self._row_to_dict(row)

    def get_file_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return self._row_to_dict(row)

    def get_all_files(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM files ORDER BY path").fetchall()
        return self._rows_to_dicts(rows)

    def delete_file(self, file_id: int) -> None:
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        if self.fts_enabled:
            self.conn.execute("DELETE FROM symbols_fts WHERE rowid NOT IN (SELECT id FROM symbols)")
        self.conn.commit()

    def delete_missing_paths(self, keep_paths: Iterable[str]) -> None:
        keep = list(keep_paths)
        if not keep:
            self.clear_all()
            return
        placeholders = ", ".join("?" for _ in keep)
        self.conn.execute(f"DELETE FROM files WHERE path NOT IN ({placeholders})", keep)
        if self.fts_enabled:
            self.conn.execute("DELETE FROM symbols_fts WHERE rowid NOT IN (SELECT id FROM symbols)")
        self.conn.commit()

    def insert_symbol(
        self,
        file_id: int,
        name: str,
        kind: str,
        line_start: int,
        line_end: int,
        signature: str,
        docstring: str | None,
        decorators: str,
        intent_summary: str | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO symbols(file_id, name, kind, line_start, line_end, signature, docstring, decorators, intent_summary)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (file_id, name, kind, line_start, line_end, signature, docstring, decorators, intent_summary),
        )
        symbol_id = int(cursor.lastrowid)
        if self.fts_enabled:
            self.conn.execute(
                "INSERT INTO symbols_fts(rowid, name, docstring, intent_summary) VALUES(?, ?, ?, ?)",
                (symbol_id, name, docstring or "", intent_summary or ""),
            )
        self.conn.commit()
        return symbol_id

    def update_symbol_summary(self, symbol_id: int, summary: str | None) -> None:
        row = self.conn.execute("SELECT name, docstring FROM symbols WHERE id = ?", (symbol_id,)).fetchone()
        self.conn.execute("UPDATE symbols SET intent_summary = ? WHERE id = ?", (summary, symbol_id))
        if self.fts_enabled and row is not None:
            self.conn.execute("DELETE FROM symbols_fts WHERE rowid = ?", (symbol_id,))
            self.conn.execute(
                "INSERT INTO symbols_fts(rowid, name, docstring, intent_summary) VALUES(?, ?, ?, ?)",
                (symbol_id, row["name"], row["docstring"] or "", summary or ""),
            )
        self.conn.commit()

    def get_symbols_for_file(self, file_id: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM symbols WHERE file_id = ? ORDER BY line_start, id",
            (file_id,),
        ).fetchall()
        return self._rows_to_dicts(rows)

    def get_symbol_by_name(self, name: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM symbols WHERE name = ? ORDER BY id", (name,)).fetchall()
        return self._rows_to_dicts(rows)

    def get_symbol_by_id(self, symbol_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM symbols WHERE id = ?", (symbol_id,)).fetchone()
        return self._row_to_dict(row)

    def insert_import(
        self,
        file_id: int,
        imported_module: str,
        imported_name: str | None,
        is_from_import: bool,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO imports(file_id, imported_module, imported_name, is_from_import)
            VALUES(?, ?, ?, ?)
            """,
            (file_id, imported_module, imported_name, int(is_from_import)),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def get_imports_for_file(self, file_id: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM imports WHERE file_id = ? ORDER BY id",
            (file_id,),
        ).fetchall()
        return self._rows_to_dicts(rows)

    def insert_call_site(self, caller_symbol_id: int, callee_name: str, line: int) -> int:
        cursor = self.conn.execute(
            "INSERT INTO call_sites(caller_symbol_id, callee_name, line) VALUES(?, ?, ?)",
            (caller_symbol_id, callee_name, line),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def get_call_sites_for_symbol(self, caller_symbol_id: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM call_sites WHERE caller_symbol_id = ? ORDER BY line, id",
            (caller_symbol_id,),
        ).fetchall()
        return self._rows_to_dicts(rows)

    def get_call_sites_with_callers(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT call_sites.*, symbols.file_id, symbols.name AS caller_name
            FROM call_sites
            JOIN symbols ON symbols.id = call_sites.caller_symbol_id
            """
        ).fetchall()
        return self._rows_to_dicts(rows)

    def upsert_relationship(self, from_file_id: int, to_file_id: int, rel_type: str, strength: float) -> None:
        self.conn.execute(
            """
            INSERT INTO relationships(from_file_id, to_file_id, rel_type, strength)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(from_file_id, to_file_id, rel_type)
            DO UPDATE SET strength = excluded.strength
            """,
            (from_file_id, to_file_id, rel_type, strength),
        )
        self.conn.commit()

    def delete_relationships_by_type(self, rel_type: str) -> None:
        self.conn.execute("DELETE FROM relationships WHERE rel_type = ?", (rel_type,))
        self.conn.commit()

    def get_relationships_from(self, from_file_id: int, rel_type: str | None = None) -> List[Dict[str, Any]]:
        if rel_type is None:
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE from_file_id = ? ORDER BY id",
                (from_file_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE from_file_id = ? AND rel_type = ? ORDER BY id",
                (from_file_id, rel_type),
            ).fetchall()
        return self._rows_to_dicts(rows)

    def get_relationships_to(self, to_file_id: int, rel_type: str | None = None) -> List[Dict[str, Any]]:
        if rel_type is None:
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE to_file_id = ? ORDER BY id",
                (to_file_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE to_file_id = ? AND rel_type = ? ORDER BY id",
                (to_file_id, rel_type),
            ).fetchall()
        return self._rows_to_dicts(rows)

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM relationships ORDER BY id").fetchall()
        return self._rows_to_dicts(rows)

    def record_scan(
        self,
        scan_sha: str | None,
        files_scanned: int,
        symbols_found: int,
        llm_calls_made: int,
        llm_cost_usd: float,
        scan_type: str,
        scan_time: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO scan_meta(scan_sha, scan_time, files_scanned, symbols_found, llm_calls_made, llm_cost_usd, scan_type)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (scan_sha, scan_time, files_scanned, symbols_found, llm_calls_made, llm_cost_usd, scan_type),
        )
        self.conn.commit()

    def get_last_scan(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM scan_meta ORDER BY id DESC LIMIT 1").fetchone()
        return self._row_to_dict(row)

    def fts_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.fts_enabled:
            raise RuntimeError("FTS5 is not enabled for this SQLite build")
        rows = self.conn.execute(
            """
            SELECT symbols.*, files.path
            FROM symbols_fts
            JOIN symbols ON symbols_fts.rowid = symbols.id
            JOIN files ON files.id = symbols.file_id
            WHERE symbols_fts MATCH ?
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return self._rows_to_dicts(rows)

    def keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        like = f"%{query.lower()}%"
        rows = self.conn.execute(
            """
            SELECT symbols.*, files.path
            FROM symbols
            JOIN files ON files.id = symbols.file_id
            WHERE lower(symbols.name) LIKE ?
               OR lower(COALESCE(symbols.docstring, '')) LIKE ?
               OR lower(COALESCE(symbols.intent_summary, '')) LIKE ?
            LIMIT ?
            """,
            (like, like, like, limit),
        ).fetchall()
        return self._rows_to_dicts(rows)
