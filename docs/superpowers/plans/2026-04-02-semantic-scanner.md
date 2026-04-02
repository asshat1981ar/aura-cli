# Semantic Codebase Scanner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-layer semantic scanner that produces a SQLite index of the codebase with LLM-augmented summaries, a query tool for the agent-sdk, and context injection into the system prompt.

**Architecture:** Three new modules (semantic_schema, semantic_scanner, semantic_querier) form the core. Schema handles SQLite storage, scanner runs the 3-layer pipeline, querier provides the read interface. Five existing modules gain small modifications for integration. Bottom-up build order: schema → scanner → querier → config → integration.

**Tech Stack:** Python `ast` (stdlib), `sqlite3` (stdlib), `gitpython` (existing), `ModelAdapter` (existing for Layer 3 LLM calls).

**Spec:** `docs/superpowers/specs/2026-04-02-semantic-scanner-design.md`

---

## File Structure

**New files:**

| File | Responsibility |
|------|---------------|
| `core/agent_sdk/semantic_schema.py` | SQLite schema DDL, CRUD operations, FTS5 management |
| `core/agent_sdk/semantic_scanner.py` | Three-layer pipeline: AST extraction, relationship analysis, LLM intent |
| `core/agent_sdk/semantic_querier.py` | 7 query methods + architecture_overview for the agent-sdk |

**Modified files:**

| File | Change |
|------|--------|
| `core/agent_sdk/config.py` | Add 7 scan config fields + from_aura_config mappings |
| `core/agent_sdk/context_builder.py` | Add semantic_querier param + _get_codebase_context() + prompt rendering |
| `core/agent_sdk/controller.py` | Create querier in __init__, use self.context_builder in _build_prompt, pass querier to MCP server |
| `core/agent_sdk/tool_registry.py` | Add query_codebase tool + _handle_query_codebase handler + semantic_querier dep |
| `core/agent_sdk/cli_integration.py` | Add handle_agent_scan() |

---

## Task 1: Extend Config with Scan Fields

**Files:**
- Modify: `core/agent_sdk/config.py`
- Test: `tests/test_agent_sdk_config.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_sdk_config.py`:

```python
class TestAgentSDKConfigScan(unittest.TestCase):
    """Test scan config fields."""

    def test_default_scan_fields(self):
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        self.assertEqual(config.semantic_index_path, Path("memory/semantic_index.db"))
        self.assertAlmostEqual(config.scan_llm_budget_usd, 0.50)
        self.assertEqual(config.scan_llm_model, "claude-haiku-4-5")
        self.assertEqual(config.scan_min_function_lines, 10)
        self.assertEqual(config.scan_min_file_lines, 5)
        self.assertEqual(config.scan_batch_size, 10)
        self.assertIn(".git", config.scan_exclude_patterns)

    def test_from_aura_config_reads_scan_fields(self):
        from core.agent_sdk.config import AgentSDKConfig
        aura_config = {"agent_sdk": {"scan_batch_size": 20, "scan_min_file_lines": 10}}
        config = AgentSDKConfig.from_aura_config(aura_config)
        self.assertEqual(config.scan_batch_size, 20)
        self.assertEqual(config.scan_min_file_lines, 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_config.py::TestAgentSDKConfigScan -v`

- [ ] **Step 3: Add fields to AgentSDKConfig**

Add after the existing `skill_weight_floor` field in `config.py`:

```python
    # Semantic scanner config
    semantic_index_path: Path = field(default_factory=lambda: Path("memory/semantic_index.db"))
    scan_llm_budget_usd: float = 0.50
    scan_llm_model: str = "claude-haiku-4-5"
    scan_exclude_patterns: List[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "venv", "*.egg-info",
    ])
    scan_min_function_lines: int = 10
    scan_min_file_lines: int = 5
    scan_batch_size: int = 10
```

Add to `from_aura_config()` cls(...) call:

```python
            semantic_index_path=Path(sdk_section.get("semantic_index_path", "memory/semantic_index.db")),
            scan_llm_budget_usd=sdk_section.get("scan_llm_budget_usd", 0.50),
            scan_llm_model=sdk_section.get("scan_llm_model", "claude-haiku-4-5"),
            scan_exclude_patterns=sdk_section.get("scan_exclude_patterns", [
                ".git", "__pycache__", "node_modules", ".venv", "venv", "*.egg-info",
            ]),
            scan_min_function_lines=sdk_section.get("scan_min_function_lines", 10),
            scan_min_file_lines=sdk_section.get("scan_min_file_lines", 5),
            scan_batch_size=sdk_section.get("scan_batch_size", 10),
```

- [ ] **Step 4: Run ALL config tests**

Run: `python3 -m pytest tests/test_agent_sdk_config.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/config.py tests/test_agent_sdk_config.py
git commit -m "feat: add semantic scanner config fields"
```

---

## Task 2: SQLite Schema & Data Access Layer

**Files:**
- Create: `core/agent_sdk/semantic_schema.py`
- Test: `tests/test_semantic_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_semantic_schema.py
"""Tests for semantic index SQLite schema and CRUD."""
import tempfile
import unittest
from pathlib import Path


class TestSemanticSchema(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_index.db"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_creates_tables(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        tables = db.list_tables()
        for t in ["files", "symbols", "imports", "call_sites", "relationships", "scan_meta"]:
            self.assertIn(t, tables)

    def test_upsert_file(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        fid = db.upsert_file("core/foo.py", "core.foo", "core", 100, "2026-01-01", "abc123")
        self.assertIsInstance(fid, int)
        # Upsert again — same id
        fid2 = db.upsert_file("core/foo.py", "core.foo", "core", 105, "2026-01-02", "def456")
        self.assertEqual(fid, fid2)

    def test_insert_symbol(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        fid = db.upsert_file("test.py", "test", ".", 50, "2026-01-01", "abc")
        sid = db.insert_symbol(fid, "my_func", "function", 10, 25, "(x, y)", "Does stuff", "")
        self.assertIsInstance(sid, int)

    def test_insert_import(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        fid = db.upsert_file("test.py", "test", ".", 50, "2026-01-01", "abc")
        db.insert_import(fid, "os.path", "join", True)
        imports = db.get_imports_for_file(fid)
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0]["imported_name"], "join")

    def test_insert_call_site(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        fid = db.upsert_file("test.py", "test", ".", 50, "2026-01-01", "abc")
        sid = db.insert_symbol(fid, "caller", "function", 1, 10, "()", None, "")
        db.insert_call_site(sid, "callee_func", 5)
        calls = db.get_call_sites_for_symbol(sid)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["callee_name"], "callee_func")

    def test_upsert_relationship(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        f1 = db.upsert_file("a.py", "a", ".", 10, "2026-01-01", "abc")
        f2 = db.upsert_file("b.py", "b", ".", 20, "2026-01-01", "abc")
        db.upsert_relationship(f1, f2, "imports", 1.0)
        rels = db.get_relationships_from(f1)
        self.assertEqual(len(rels), 1)

    def test_delete_file_cascades(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        fid = db.upsert_file("gone.py", "gone", ".", 10, "2026-01-01", "abc")
        db.insert_symbol(fid, "func", "function", 1, 5, "()", None, "")
        db.insert_import(fid, "os", None, False)
        db.delete_file(fid)
        self.assertIsNone(db.get_file_by_path("gone.py"))

    def test_record_scan_meta(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        db.record_scan("abc123", 10, 50, 5, 0.15, "full")
        meta = db.get_last_scan()
        self.assertEqual(meta["scan_sha"], "abc123")
        self.assertEqual(meta["files_scanned"], 10)

    def test_fts_search(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        fid = db.upsert_file("test.py", "test", ".", 50, "2026-01-01", "abc")
        sid = db.insert_symbol(fid, "authenticate_user", "function", 1, 20,
                               "(username, password)", "Validate user credentials", "")
        db.update_symbol_summary(sid, "Validates username and password against the user database")
        results = db.fts_search("authenticate password", limit=5)
        self.assertGreater(len(results), 0)

    def test_get_all_files(self):
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        db.upsert_file("a.py", "a", ".", 10, "2026-01-01", "abc")
        db.upsert_file("b.py", "b", ".", 20, "2026-01-01", "abc")
        files = db.get_all_files()
        self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_semantic_schema.py -v`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/semantic_schema.py
"""SQLite schema and data access layer for the semantic codebase index."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA_DDL = """
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
    is_from_import BOOLEAN NOT NULL DEFAULT 0
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
    scan_sha TEXT NOT NULL,
    scan_time TEXT NOT NULL,
    files_scanned INTEGER NOT NULL DEFAULT 0,
    symbols_found INTEGER NOT NULL DEFAULT 0,
    llm_calls_made INTEGER NOT NULL DEFAULT 0,
    llm_cost_usd REAL NOT NULL DEFAULT 0.0,
    scan_type TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name, docstring, intent_summary, content=symbols, content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, docstring, intent_summary)
    VALUES (new.id, new.name, COALESCE(new.docstring, ''), COALESCE(new.intent_summary, ''));
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, docstring, intent_summary)
    VALUES ('delete', old.id, old.name, COALESCE(old.docstring, ''), COALESCE(old.intent_summary, ''));
END;
"""


class SemanticDB:
    """SQLite data access layer for the semantic index."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_DDL)

    def list_tables(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    # --- files ---

    def upsert_file(
        self, path: str, module_name: str, cluster: str,
        line_count: int, last_modified: str, scan_sha: str,
    ) -> int:
        now = datetime.utcnow().isoformat()
        existing = self.get_file_by_path(path)
        if existing:
            self._conn.execute(
                """UPDATE files SET module_name=?, cluster=?, line_count=?,
                   last_modified=?, last_scan_sha=?, scanned_at=? WHERE id=?""",
                (module_name, cluster, line_count, last_modified, scan_sha, now, existing["id"]),
            )
            self._conn.commit()
            return existing["id"]
        cur = self._conn.execute(
            """INSERT INTO files (path, module_name, cluster, line_count,
               last_modified, last_scan_sha, scanned_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (path, module_name, cluster, line_count, last_modified, scan_sha, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_file_by_path(self, path: str) -> Optional[Dict]:
        row = self._conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        return dict(row) if row else None

    def get_all_files(self) -> List[Dict]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM files").fetchall()]

    def delete_file(self, file_id: int) -> None:
        self._conn.execute("DELETE FROM files WHERE id=?", (file_id,))
        self._conn.commit()

    def update_file_summary(self, file_id: int, summary: str) -> None:
        self._conn.execute("UPDATE files SET module_summary=? WHERE id=?", (summary, file_id))
        self._conn.commit()

    def update_file_coupling(self, file_id: int, score: float) -> None:
        self._conn.execute("UPDATE files SET coupling_score=? WHERE id=?", (score, file_id))
        self._conn.commit()

    # --- symbols ---

    def insert_symbol(
        self, file_id: int, name: str, kind: str, line_start: int, line_end: int,
        signature: str, docstring: Optional[str], decorators: str,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO symbols (file_id, name, kind, line_start, line_end,
               signature, docstring, decorators)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id, name, kind, line_start, line_end, signature, docstring, decorators),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_symbol_summary(self, symbol_id: int, summary: str) -> None:
        self._conn.execute("UPDATE symbols SET intent_summary=? WHERE id=?", (summary, symbol_id))
        # Update FTS index
        row = self._conn.execute("SELECT name, docstring FROM symbols WHERE id=?", (symbol_id,)).fetchone()
        if row:
            self._conn.execute(
                "INSERT INTO symbols_fts(symbols_fts, rowid, name, docstring, intent_summary) VALUES ('delete', ?, ?, ?, '')",
                (symbol_id, row["name"], row["docstring"] or ""),
            )
            self._conn.execute(
                "INSERT INTO symbols_fts(rowid, name, docstring, intent_summary) VALUES (?, ?, ?, ?)",
                (symbol_id, row["name"], row["docstring"] or "", summary),
            )
        self._conn.commit()

    def get_symbols_for_file(self, file_id: int) -> List[Dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM symbols WHERE file_id=? ORDER BY line_start", (file_id,)
        ).fetchall()]

    def clear_symbols_for_file(self, file_id: int) -> None:
        self._conn.execute("DELETE FROM symbols WHERE file_id=?", (file_id,))
        self._conn.commit()

    # --- imports ---

    def insert_import(self, file_id: int, module: str, name: Optional[str], is_from: bool) -> None:
        self._conn.execute(
            "INSERT INTO imports (file_id, imported_module, imported_name, is_from_import) VALUES (?, ?, ?, ?)",
            (file_id, module, name, is_from),
        )
        self._conn.commit()

    def get_imports_for_file(self, file_id: int) -> List[Dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM imports WHERE file_id=?", (file_id,)
        ).fetchall()]

    def clear_imports_for_file(self, file_id: int) -> None:
        self._conn.execute("DELETE FROM imports WHERE file_id=?", (file_id,))
        self._conn.commit()

    # --- call_sites ---

    def insert_call_site(self, caller_symbol_id: int, callee_name: str, line: int) -> None:
        self._conn.execute(
            "INSERT INTO call_sites (caller_symbol_id, callee_name, line) VALUES (?, ?, ?)",
            (caller_symbol_id, callee_name, line),
        )
        self._conn.commit()

    def get_call_sites_for_symbol(self, symbol_id: int) -> List[Dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM call_sites WHERE caller_symbol_id=?", (symbol_id,)
        ).fetchall()]

    # --- relationships ---

    def upsert_relationship(self, from_id: int, to_id: int, rel_type: str, strength: float) -> None:
        self._conn.execute(
            """INSERT INTO relationships (from_file_id, to_file_id, rel_type, strength)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(from_file_id, to_file_id, rel_type) DO UPDATE SET strength=?""",
            (from_id, to_id, rel_type, strength, strength),
        )
        self._conn.commit()

    def get_relationships_from(self, file_id: int) -> List[Dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM relationships WHERE from_file_id=?", (file_id,)
        ).fetchall()]

    def get_relationships_to(self, file_id: int) -> List[Dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM relationships WHERE to_file_id=?", (file_id,)
        ).fetchall()]

    # --- scan_meta ---

    def record_scan(
        self, sha: str, files_scanned: int, symbols_found: int,
        llm_calls: int, llm_cost: float, scan_type: str,
    ) -> None:
        self._conn.execute(
            """INSERT INTO scan_meta (scan_sha, scan_time, files_scanned, symbols_found,
               llm_calls_made, llm_cost_usd, scan_type) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sha, datetime.utcnow().isoformat(), files_scanned, symbols_found,
             llm_calls, llm_cost, scan_type),
        )
        self._conn.commit()

    def get_last_scan(self) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM scan_meta ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # --- FTS ---

    def fts_search(self, query: str, limit: int = 10) -> List[Dict]:
        try:
            rows = self._conn.execute(
                """SELECT s.id, s.name, s.kind, s.file_id, s.signature, s.docstring,
                          s.intent_summary, f.path
                   FROM symbols_fts fts
                   JOIN symbols s ON s.id = fts.rowid
                   JOIN files f ON f.id = s.file_id
                   WHERE symbols_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def checkpoint(self) -> None:
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_semantic_schema.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/semantic_schema.py tests/test_semantic_schema.py
git commit -m "feat: add semantic index SQLite schema with FTS5"
```

---

## Task 3: AST Scanner (Layers 1 + 2)

**Files:**
- Create: `core/agent_sdk/semantic_scanner.py`
- Test: `tests/test_semantic_scanner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_semantic_scanner.py
"""Tests for the three-layer semantic scanner pipeline."""
import tempfile
import textwrap
import unittest
from pathlib import Path


class TestASTExtraction(unittest.TestCase):
    """Test Layer 1: AST extraction from Python files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_py(self, name: str, code: str) -> Path:
        p = self.project / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(code))
        return p

    def test_extract_functions(self):
        from core.agent_sdk.semantic_scanner import extract_symbols
        self._write_py("mod.py", '''
            def hello(name: str) -> str:
                """Greet someone."""
                return f"Hello {name}"

            def add(a, b):
                return a + b
        ''')
        symbols = extract_symbols(self.project / "mod.py")
        names = [s["name"] for s in symbols]
        self.assertIn("hello", names)
        self.assertIn("add", names)
        hello = next(s for s in symbols if s["name"] == "hello")
        self.assertEqual(hello["kind"], "function")
        self.assertIn("name: str", hello["signature"])
        self.assertEqual(hello["docstring"], "Greet someone.")

    def test_extract_classes_and_methods(self):
        from core.agent_sdk.semantic_scanner import extract_symbols
        self._write_py("cls.py", '''
            class Foo:
                """A foo class."""
                def bar(self):
                    pass

                @staticmethod
                def baz():
                    pass
        ''')
        symbols = extract_symbols(self.project / "cls.py")
        names = [s["name"] for s in symbols]
        self.assertIn("Foo", names)
        self.assertIn("bar", names)
        self.assertIn("baz", names)
        foo = next(s for s in symbols if s["name"] == "Foo")
        self.assertEqual(foo["kind"], "class")
        baz = next(s for s in symbols if s["name"] == "baz")
        self.assertIn("staticmethod", baz["decorators"])

    def test_extract_imports(self):
        from core.agent_sdk.semantic_scanner import extract_imports
        self._write_py("imp.py", '''
            import os
            from pathlib import Path
            from typing import Any, Dict
        ''')
        imports = extract_imports(self.project / "imp.py")
        modules = [i["imported_module"] for i in imports]
        self.assertIn("os", modules)
        self.assertIn("pathlib", modules)

    def test_extract_call_sites(self):
        from core.agent_sdk.semantic_scanner import extract_symbols, extract_call_sites
        self._write_py("calls.py", '''
            def caller():
                result = callee(1, 2)
                other_func()
                return result
        ''')
        symbols = extract_symbols(self.project / "calls.py")
        caller = next(s for s in symbols if s["name"] == "caller")
        calls = extract_call_sites(self.project / "calls.py", caller)
        callee_names = [c["callee_name"] for c in calls]
        self.assertIn("callee", callee_names)
        self.assertIn("other_func", callee_names)

    def test_extract_base_classes(self):
        from core.agent_sdk.semantic_scanner import extract_symbols
        self._write_py("inherit.py", '''
            class Base:
                pass
            class Child(Base):
                pass
        ''')
        symbols = extract_symbols(self.project / "inherit.py")
        child = next(s for s in symbols if s["name"] == "Child")
        self.assertIn("Base", child.get("bases", []))


class TestRelationshipAnalysis(unittest.TestCase):
    """Test Layer 2: relationship building."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_import_relationships(self):
        from core.agent_sdk.semantic_scanner import build_relationships
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        f1 = db.upsert_file("a.py", "a", ".", 10, "2026-01-01", "abc")
        f2 = db.upsert_file("b.py", "b", ".", 10, "2026-01-01", "abc")
        db.insert_import(f1, "b", None, False)
        build_relationships(db)
        rels = db.get_relationships_from(f1)
        self.assertGreater(len(rels), 0)
        self.assertEqual(rels[0]["rel_type"], "imports")

    def test_compute_coupling_scores(self):
        from core.agent_sdk.semantic_scanner import compute_coupling_scores
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        f1 = db.upsert_file("hub.py", "hub", ".", 100, "2026-01-01", "abc")
        f2 = db.upsert_file("leaf.py", "leaf", ".", 20, "2026-01-01", "abc")
        db.upsert_relationship(f1, f2, "imports", 1.0)
        db.upsert_relationship(f2, f1, "imports", 1.0)
        compute_coupling_scores(db)
        hub = db.get_file_by_path("hub.py")
        self.assertGreater(hub["coupling_score"], 0)


class TestFullScan(unittest.TestCase):
    """Test the full scan pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()
        self.db_path = Path(self.tmpdir) / "index.db"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_scan_without_llm(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        (self.project / "main.py").write_text("def main():\n    print('hello')\n")
        (self.project / "utils.py").write_text("import os\ndef helper():\n    return os.getcwd()\n")
        scanner = SemanticScanner(
            project_root=self.project, db_path=self.db_path,
            exclude_patterns=[".git", "__pycache__"],
        )
        scanner.scan_full()
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        files = db.get_all_files()
        self.assertEqual(len(files), 2)
        meta = db.get_last_scan()
        self.assertIsNotNone(meta)
        self.assertEqual(meta["scan_type"], "full")

    def test_scan_respects_exclude_patterns(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        (self.project / "good.py").write_text("x = 1\n")
        pycache = self.project / "__pycache__"
        pycache.mkdir()
        (pycache / "bad.py").write_text("x = 2\n")
        scanner = SemanticScanner(
            project_root=self.project, db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        scanner.scan_full()
        from core.agent_sdk.semantic_schema import SemanticDB
        db = SemanticDB(self.db_path)
        files = db.get_all_files()
        paths = [f["path"] for f in files]
        self.assertNotIn("__pycache__/bad.py", paths)


class TestIncrementalScan(unittest.TestCase):
    """Test incremental scan and refresh_if_needed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()
        self.db_path = Path(self.tmpdir) / "index.db"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scan_incremental_processes_only_listed_files(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        from core.agent_sdk.semantic_schema import SemanticDB
        # Create initial files and do full scan
        (self.project / "a.py").write_text("def a():\n    return 1\n")
        (self.project / "b.py").write_text("def b():\n    return 2\n")
        scanner = SemanticScanner(
            project_root=self.project, db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        scanner.scan_full()
        # Add a new file and do incremental scan with only that file
        (self.project / "c.py").write_text("def c():\n    return 3\n")
        result = scanner.scan_incremental(["c.py"], git_sha="new123")
        self.assertEqual(result["files_scanned"], 1)
        db = SemanticDB(self.db_path)
        files = db.get_all_files()
        paths = [f["path"] for f in files]
        self.assertIn("c.py", paths)

    def test_refresh_if_needed_noop_on_matching_sha(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        from unittest.mock import patch
        (self.project / "test.py").write_text("x = 1\n")
        scanner = SemanticScanner(
            project_root=self.project, db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        scanner.scan_full(git_sha="abc123")
        # Mock git to return same SHA
        with patch("subprocess.check_output", return_value=b"abc123\n"):
            result = scanner.refresh_if_needed()
        self.assertIsNone(result)  # No-op

    def test_incremental_deletes_removed_files(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        from core.agent_sdk.semantic_schema import SemanticDB
        (self.project / "temp.py").write_text("x = 1\n")
        scanner = SemanticScanner(
            project_root=self.project, db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        scanner.scan_full()
        # Delete the file
        (self.project / "temp.py").unlink()
        scanner.scan_incremental(["temp.py"], git_sha="del123")
        db = SemanticDB(self.db_path)
        self.assertIsNone(db.get_file_by_path("temp.py"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_semantic_scanner.py -v`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/semantic_scanner.py
"""Three-layer semantic scanner pipeline.

Layer 1: AST extraction (symbols, imports, call sites, class inheritance)
Layer 2: Relationship analysis (call graph, dependency chains, coupling)
Layer 3: LLM intent augmentation (module summaries, function intents)
"""
from __future__ import annotations

import ast
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: AST Extraction
# ---------------------------------------------------------------------------

def extract_symbols(file_path: Path) -> List[Dict[str, Any]]:
    """Extract functions, classes, methods from a Python file."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            decorators = [_decorator_name(d) for d in node.decorator_list]
            kind = "function"
            if any(d in ("staticmethod",) for d in decorators):
                kind = "staticmethod"
            elif any(d in ("classmethod",) for d in decorators):
                kind = "classmethod"
            elif any(d in ("property",) for d in decorators):
                kind = "property"
            # Check if inside a class
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    if node in ast.iter_child_nodes(parent):
                        if kind == "function":
                            kind = "method"
                        break

            symbols.append({
                "name": node.name,
                "kind": kind,
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
                "signature": _get_signature(node),
                "docstring": ast.get_docstring(node),
                "decorators": ",".join(decorators),
                "bases": [],
            })

        elif isinstance(node, ast.ClassDef):
            bases = [_base_name(b) for b in node.bases]
            symbols.append({
                "name": node.name,
                "kind": "class",
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
                "signature": "",
                "docstring": ast.get_docstring(node),
                "decorators": ",".join(_decorator_name(d) for d in node.decorator_list),
                "bases": bases,
            })

    return symbols


def extract_imports(file_path: Path) -> List[Dict[str, Any]]:
    """Extract import statements from a Python file."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "imported_module": alias.name,
                    "imported_name": None,
                    "is_from_import": False,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "imported_module": module,
                    "imported_name": alias.name,
                    "is_from_import": True,
                })
    return imports


def extract_call_sites(file_path: Path, symbol: Dict) -> List[Dict[str, Any]]:
    """Extract function/method calls within a symbol's body."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    calls = []
    start = symbol["line_start"]
    end = symbol["line_end"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and hasattr(node, "lineno"):
            if start <= node.lineno <= end:
                name = _call_name(node.func)
                if name:
                    calls.append({"callee_name": name, "line": node.lineno})
    return calls


# ---------------------------------------------------------------------------
# Layer 2: Relationship Analysis
# ---------------------------------------------------------------------------

def build_relationships(db: Any) -> None:
    """Build file-level relationships from imports and call sites."""
    files = db.get_all_files()
    path_to_id = {f["path"]: f["id"] for f in files}
    module_to_id = {f["module_name"]: f["id"] for f in files if f.get("module_name")}

    for f in files:
        imports = db.get_imports_for_file(f["id"])
        for imp in imports:
            mod = imp["imported_module"]
            # Try to resolve to a file in the project
            target_id = module_to_id.get(mod)
            if not target_id:
                # Try path-based: core.foo -> core/foo.py
                candidate = mod.replace(".", "/") + ".py"
                target_id = path_to_id.get(candidate)
            if target_id and target_id != f["id"]:
                db.upsert_relationship(f["id"], target_id, "imports", 1.0)


def compute_coupling_scores(db: Any) -> None:
    """Compute coupling score for each file."""
    files = db.get_all_files()
    total = max(len(files), 1)
    for f in files:
        inbound = len(db.get_relationships_to(f["id"]))
        outbound = len(db.get_relationships_from(f["id"]))
        score = min((inbound + outbound) / total, 1.0)
        db.update_file_coupling(f["id"], round(score, 4))


# ---------------------------------------------------------------------------
# Layer 3: LLM Intent Augmentation
# ---------------------------------------------------------------------------

def generate_module_summary(
    db: Any, file_id: int, file_path: str, model_adapter: Any,
) -> Optional[str]:
    """Generate an LLM summary for a module."""
    symbols = db.get_symbols_for_file(file_id)
    imports = db.get_imports_for_file(file_id)
    sym_list = ", ".join(f"{s['name']}({s['signature']})" for s in symbols[:20])
    imp_list = ", ".join(i["imported_module"] for i in imports[:15])

    prompt = (
        f"Summarize this Python module in 2-3 sentences. "
        f"Focus on: what it does, what role it plays, and key dependencies.\n"
        f"Module: {file_path}\n"
        f"Symbols: {sym_list}\n"
        f"Imports: {imp_list}"
    )
    try:
        return model_adapter.respond_for_role("code_generation", prompt)
    except Exception as exc:
        logger.warning("LLM summary failed for %s: %s", file_path, exc)
        return None


def generate_function_intents(
    symbols: List[Dict], file_path: Path, model_adapter: Any, batch_size: int = 10,
) -> Dict[int, str]:
    """Generate LLM intent summaries for functions, batched."""
    source = file_path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    results: Dict[int, str] = {}

    eligible = [s for s in symbols if s["kind"] != "class" and
                (s["line_end"] - s["line_start"]) >= 10]

    for i in range(0, len(eligible), batch_size):
        batch = eligible[i:i + batch_size]
        parts = []
        for s in batch:
            body = "\n".join(lines[s["line_start"] - 1:s["line_end"]])[:500]
            parts.append(f"{s['name']}({s['signature']})\n{body}")

        prompt = (
            "For each function below, write ONE sentence describing what it does and why.\n"
            "Format: function_name: description\n\n" + "\n\n".join(parts)
        )
        try:
            response = model_adapter.respond_for_role("code_generation", prompt)
            for line in response.strip().splitlines():
                if ":" in line:
                    fname, desc = line.split(":", 1)
                    fname = fname.strip()
                    matching = [s for s in batch if s["name"] == fname]
                    if matching:
                        results[matching[0].get("_db_id", 0)] = desc.strip()
        except Exception as exc:
            logger.warning("LLM intent batch failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Scanner Pipeline
# ---------------------------------------------------------------------------

class SemanticScanner:
    """Three-layer semantic scanner pipeline."""

    def __init__(
        self,
        project_root: Path,
        db_path: Path,
        exclude_patterns: Optional[List[str]] = None,
        model_adapter: Any = None,
        min_function_lines: int = 10,
        min_file_lines: int = 5,
        batch_size: int = 10,
        llm_budget: float = 0.50,
    ) -> None:
        self._root = project_root
        self._db_path = db_path
        self._excludes = exclude_patterns or [".git", "__pycache__", "node_modules"]
        self._model = model_adapter
        self._min_func_lines = min_function_lines
        self._min_file_lines = min_file_lines
        self._batch_size = batch_size
        self._budget = llm_budget

    def _find_py_files(self) -> List[Path]:
        """Find all .py files respecting exclude patterns."""
        results = []
        for root, dirs, files in os.walk(self._root):
            dirs[:] = [d for d in dirs if not any(
                pat in d for pat in self._excludes
            )]
            for f in files:
                if f.endswith(".py"):
                    results.append(Path(root) / f)
        return sorted(results)

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._root))
        except ValueError:
            return str(path)

    def _module_name(self, rel_path: str) -> str:
        return rel_path.replace("/", ".").replace("\\", ".").removesuffix(".py")

    def _cluster(self, rel_path: str) -> str:
        parts = rel_path.split("/")
        return "/".join(parts[:-1]) if len(parts) > 1 else "."

    def scan_full(self, git_sha: str = "unknown") -> Dict[str, Any]:
        """Run a full scan of all Python files."""
        from core.agent_sdk.semantic_schema import SemanticDB

        db = SemanticDB(self._db_path)
        py_files = self._find_py_files()
        total_symbols = 0
        llm_calls = 0
        llm_cost = 0.0

        for fpath in py_files:
            rel = self._relative(fpath)
            line_count = len(fpath.read_text(encoding="utf-8", errors="replace").splitlines())

            if line_count < self._min_file_lines:
                continue

            mtime = datetime.fromtimestamp(fpath.stat().st_mtime).isoformat()
            fid = db.upsert_file(rel, self._module_name(rel), self._cluster(rel),
                                 line_count, mtime, git_sha)

            # Clear old data for this file
            db.clear_symbols_for_file(fid)
            db.clear_imports_for_file(fid)

            # Layer 1: AST
            symbols = extract_symbols(fpath)
            for s in symbols:
                sid = db.insert_symbol(
                    fid, s["name"], s["kind"], s["line_start"], s["line_end"],
                    s["signature"], s["docstring"], s["decorators"],
                )
                s["_db_id"] = sid
                total_symbols += 1

                # Call sites
                if s["kind"] != "class":
                    for call in extract_call_sites(fpath, s):
                        db.insert_call_site(sid, call["callee_name"], call["line"])

            for imp in extract_imports(fpath):
                db.insert_import(fid, imp["imported_module"], imp["imported_name"],
                                 imp["is_from_import"])

            # Layer 3: LLM summaries (if available and within budget)
            if self._model and llm_cost < self._budget:
                summary = generate_module_summary(db, fid, rel, self._model)
                if summary:
                    db.update_file_summary(fid, summary)
                    llm_calls += 1
                    llm_cost += 0.001  # estimated

                intents = generate_function_intents(
                    symbols, fpath, self._model, self._batch_size
                )
                for sid, intent in intents.items():
                    if sid:
                        db.update_symbol_summary(sid, intent)
                if intents:
                    llm_calls += 1
                    llm_cost += 0.002  # estimated

        # Layer 2: Relationships
        build_relationships(db)
        compute_coupling_scores(db)

        # Record scan metadata
        db.record_scan(git_sha, len(py_files), total_symbols, llm_calls, llm_cost, "full")
        db.checkpoint()

        return {
            "files_scanned": len(py_files),
            "symbols_found": total_symbols,
            "llm_calls": llm_calls,
            "llm_cost_usd": llm_cost,
        }

    def scan_incremental(self, changed_files: List[str], git_sha: str = "unknown") -> Dict[str, Any]:
        """Re-scan only changed files."""
        from core.agent_sdk.semantic_schema import SemanticDB

        db = SemanticDB(self._db_path)
        total_symbols = 0

        for rel_path in changed_files:
            fpath = self._root / rel_path
            if not fpath.exists():
                # File was deleted
                existing = db.get_file_by_path(rel_path)
                if existing:
                    db.delete_file(existing["id"])
                continue

            if not rel_path.endswith(".py"):
                continue

            line_count = len(fpath.read_text(encoding="utf-8", errors="replace").splitlines())
            if line_count < self._min_file_lines:
                continue

            mtime = datetime.fromtimestamp(fpath.stat().st_mtime).isoformat()
            fid = db.upsert_file(rel_path, self._module_name(rel_path),
                                 self._cluster(rel_path), line_count, mtime, git_sha)
            db.clear_symbols_for_file(fid)
            db.clear_imports_for_file(fid)

            symbols = extract_symbols(fpath)
            for s in symbols:
                sid = db.insert_symbol(
                    fid, s["name"], s["kind"], s["line_start"], s["line_end"],
                    s["signature"], s["docstring"], s["decorators"],
                )
                total_symbols += 1
                if s["kind"] != "class":
                    for call in extract_call_sites(fpath, s):
                        db.insert_call_site(sid, call["callee_name"], call["line"])

            for imp in extract_imports(fpath):
                db.insert_import(fid, imp["imported_module"], imp["imported_name"],
                                 imp["is_from_import"])

        build_relationships(db)
        compute_coupling_scores(db)
        db.record_scan(git_sha, len(changed_files), total_symbols, 0, 0.0, "incremental")

        return {"files_scanned": len(changed_files), "symbols_found": total_symbols}

    def refresh_if_needed(self) -> Optional[Dict]:
        """Check git and run incremental scan if needed."""
        from core.agent_sdk.semantic_schema import SemanticDB
        import subprocess

        db = SemanticDB(self._db_path)
        last = db.get_last_scan()

        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(self._root),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        if last and last["scan_sha"] == head:
            return None  # No-op

        if not last:
            return self.scan_full(git_sha=head)

        # Try incremental
        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "--name-only", f"{last['scan_sha']}..{head}", "--", "*.py"],
                cwd=str(self._root), stderr=subprocess.DEVNULL,
            ).decode().strip()
        except subprocess.CalledProcessError:
            logger.warning("Last scan SHA %s is orphaned, falling back to full scan", last["scan_sha"])
            return self.scan_full(git_sha=head)

        if not diff_output:
            return None

        changed = [f for f in diff_output.splitlines() if f.strip()]
        return self.scan_incremental(changed, git_sha=head)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _get_signature(node: ast.FunctionDef) -> str:
    args = node.args
    parts = []
    for a in args.args:
        ann = ""
        if a.annotation:
            ann = f": {ast.unparse(a.annotation)}"
        parts.append(f"{a.arg}{ann}")
    return f"({', '.join(parts)})"


def _call_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_semantic_scanner.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/semantic_scanner.py tests/test_semantic_scanner.py
git commit -m "feat: add three-layer semantic scanner with AST extraction and relationships"
```

---

## Task 4: Semantic Querier

**Files:**
- Create: `core/agent_sdk/semantic_querier.py`
- Test: `tests/test_semantic_querier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_semantic_querier.py
"""Tests for semantic querier — all 7 query types."""
import tempfile
import unittest
from pathlib import Path


def _populate_test_db(db_path: Path):
    """Create a populated test database."""
    from core.agent_sdk.semantic_schema import SemanticDB
    db = SemanticDB(db_path)
    f1 = db.upsert_file("core/auth.py", "core.auth", "core", 100, "2026-01-01", "abc")
    f2 = db.upsert_file("core/server.py", "core.server", "core", 200, "2026-01-01", "abc")
    f3 = db.upsert_file("agents/coder.py", "agents.coder", "agents", 150, "2026-01-01", "abc")

    s1 = db.insert_symbol(f1, "authenticate", "function", 10, 30, "(user, pwd)", "Auth user", "")
    s2 = db.insert_symbol(f1, "create_token", "function", 35, 50, "(user)", "Create JWT", "")
    s3 = db.insert_symbol(f2, "handle_request", "function", 1, 40, "(req)", "Handle HTTP", "")
    s4 = db.insert_symbol(f3, "implement", "method", 46, 100, "(self, task)", "Generate code", "")

    db.update_symbol_summary(s1, "Validates username and password against the user database")
    db.update_symbol_summary(s4, "Generates code with CoT reasoning and structured output")

    db.insert_import(f2, "core.auth", "authenticate", True)
    db.insert_import(f3, "core.auth", "create_token", True)
    db.insert_call_site(s3, "authenticate", 15)
    db.insert_call_site(s3, "create_token", 20)

    db.upsert_relationship(f2, f1, "imports", 1.0)
    db.upsert_relationship(f3, f1, "imports", 1.0)

    db.update_file_coupling(f1, 0.67)
    db.update_file_coupling(f2, 0.33)
    db.update_file_coupling(f3, 0.33)

    db.update_file_summary(f1, "OAuth2 and JWT authentication system")
    db.update_file_summary(f2, "HTTP request handler for the API server")
    db.record_scan("abc", 3, 4, 0, 0.0, "full")
    return db


class TestSemanticQuerier(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        _populate_test_db(self.db_path)
        from core.agent_sdk.semantic_querier import SemanticQuerier
        self.q = SemanticQuerier(self.db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_what_calls(self):
        results = self.q.what_calls("authenticate")
        self.assertGreater(len(results), 0)
        callers = [r["caller"] for r in results]
        self.assertIn("handle_request", callers)

    def test_what_depends_on(self):
        results = self.q.what_depends_on("core/auth.py")
        paths = [r["path"] for r in results]
        self.assertIn("core/server.py", paths)
        self.assertIn("agents/coder.py", paths)

    def test_what_changes_break(self):
        results = self.q.what_changes_break("core/auth.py", depth=1)
        self.assertGreater(len(results), 0)

    def test_summarize_file(self):
        result = self.q.summarize("core/auth.py")
        self.assertIn("OAuth2", result)

    def test_summarize_symbol(self):
        result = self.q.summarize("authenticate")
        self.assertIn("Validates", result)

    def test_find_similar(self):
        results = self.q.find_similar("authentication password")
        self.assertGreater(len(results), 0)

    def test_architecture_overview(self):
        overview = self.q.architecture_overview()
        self.assertIn("clusters", overview)
        self.assertIn("top_coupled", overview)
        self.assertIn("total_files", overview)

    def test_recent_changes(self):
        results = self.q.recent_changes(n_commits=5)
        # Returns whatever's in the DB — at least our test data
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_semantic_querier.py -v`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/semantic_querier.py
"""Query interface for the semantic codebase index.

Provides 7 query types for the agent-sdk to understand code structure,
dependencies, and intent.
"""
from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SemanticQuerier:
    """Read-only query interface for the semantic index."""

    def __init__(self, db_path: Path) -> None:
        from core.agent_sdk.semantic_schema import SemanticDB
        self._db = SemanticDB(db_path)

    def what_calls(self, symbol_name: str) -> List[Dict]:
        """Find all callers of a symbol."""
        rows = self._db._conn.execute(
            """SELECT cs.callee_name, s.name as caller, s.kind, f.path, cs.line
               FROM call_sites cs
               JOIN symbols s ON s.id = cs.caller_symbol_id
               JOIN files f ON f.id = s.file_id
               WHERE cs.callee_name = ?""",
            (symbol_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def what_depends_on(self, file_path: str) -> List[Dict]:
        """Find files that import from the given file."""
        target = self._db.get_file_by_path(file_path)
        if not target:
            return []
        rows = self._db._conn.execute(
            """SELECT f.path, r.rel_type, r.strength
               FROM relationships r
               JOIN files f ON f.id = r.from_file_id
               WHERE r.to_file_id = ?""",
            (target["id"],),
        ).fetchall()
        return [dict(r) for r in rows]

    def what_changes_break(self, file_path: str, depth: int = 2) -> List[Dict]:
        """Transitive dependents — ripple analysis."""
        target = self._db.get_file_by_path(file_path)
        if not target:
            return []

        visited = set()
        result = []

        def _walk(file_id: int, dist: int) -> None:
            if dist > depth or file_id in visited:
                return
            visited.add(file_id)
            deps = self._db._conn.execute(
                """SELECT f.path, r.rel_type, r.from_file_id
                   FROM relationships r
                   JOIN files f ON f.id = r.from_file_id
                   WHERE r.to_file_id = ?""",
                (file_id,),
            ).fetchall()
            for d in deps:
                if d["from_file_id"] not in visited:
                    result.append({
                        "path": d["path"],
                        "distance": dist,
                        "relationship": d["rel_type"],
                    })
                    _walk(d["from_file_id"], dist + 1)

        _walk(target["id"], 1)
        return result

    def summarize(self, target: str) -> str:
        """Return summary for a file or symbol."""
        # Try as file first
        f = self._db.get_file_by_path(target)
        if f and f.get("module_summary"):
            return f["module_summary"]

        # Try as symbol
        row = self._db._conn.execute(
            "SELECT intent_summary, docstring FROM symbols WHERE name = ? LIMIT 1",
            (target,),
        ).fetchone()
        if row:
            return row["intent_summary"] or row["docstring"] or f"Symbol: {target}"

        return f"No summary available for: {target}"

    def find_similar(self, description: str, limit: int = 5) -> List[Dict]:
        """Find symbols matching a description via FTS5."""
        return self._db.fts_search(description, limit=limit)

    def architecture_overview(self) -> Dict:
        """Compact architectural overview."""
        files = self._db.get_all_files()
        clusters: Dict[str, List[str]] = defaultdict(list)
        for f in files:
            clusters[f.get("cluster", ".")].append(f["path"])

        top_coupled = sorted(files, key=lambda f: f.get("coupling_score", 0), reverse=True)[:5]

        return {
            "total_files": len(files),
            "clusters": {k: len(v) for k, v in clusters.items()},
            "top_coupled": [
                {"path": f["path"], "coupling": f.get("coupling_score", 0),
                 "summary": f.get("module_summary", "")}
                for f in top_coupled
            ],
            "summary": self._build_overview_text(clusters, top_coupled, len(files)),
        }

    def _build_overview_text(self, clusters, top_coupled, total) -> str:
        cluster_str = ", ".join(f"{k} ({v} files)" for k, v in
                                sorted(clusters.items(), key=lambda x: -len(x[1]) if isinstance(x[1], list) else -x[1])[:5])
        hub_str = ", ".join(f["path"] for f in top_coupled[:3])
        return f"{total} files in {len(clusters)} clusters: {cluster_str}. Key hubs: {hub_str}."

    def recent_changes(self, n_commits: int = 5) -> List[Dict]:
        """Changed files with summaries since N commits ago."""
        try:
            diff = subprocess.check_output(
                ["git", "log", f"--max-count={n_commits}", "--name-only", "--pretty=format:"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

        paths = list(set(p.strip() for p in diff.splitlines() if p.strip() and p.endswith(".py")))
        results = []
        for p in paths[:20]:
            f = self._db.get_file_by_path(p)
            if f:
                results.append({
                    "path": p,
                    "summary": f.get("module_summary", ""),
                    "coupling": f.get("coupling_score", 0),
                })
        return results
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_semantic_querier.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/semantic_querier.py tests/test_semantic_querier.py
git commit -m "feat: add semantic querier with 7 query types"
```

---

## Task 5: Integrate into Context Builder

**Files:**
- Modify: `core/agent_sdk/context_builder.py`
- Test: `tests/test_agent_sdk_context_builder.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_sdk_context_builder.py`:

```python
class TestContextBuilderSemantic(unittest.TestCase):
    """Test semantic querier integration in context builder."""

    def test_codebase_context_with_querier(self):
        from core.agent_sdk.context_builder import ContextBuilder
        mock_querier = MagicMock()
        mock_querier.architecture_overview.return_value = {
            "total_files": 50, "clusters": {"core": 20}, "summary": "50 files...",
            "top_coupled": [],
        }
        mock_querier.find_similar.return_value = [
            {"name": "auth", "file": "core/auth.py", "path": "core/auth.py"}
        ]
        mock_querier.what_depends_on.return_value = [
            {"path": "core/server.py", "rel_type": "imports"}
        ]
        builder = ContextBuilder(
            project_root=Path("/tmp/test"),
            semantic_querier=mock_querier,
        )
        ctx = builder.build(goal="Fix auth bug")
        self.assertIn("codebase_overview", ctx)
        self.assertIn("relevant_symbols", ctx)

    def test_codebase_context_without_querier(self):
        from core.agent_sdk.context_builder import ContextBuilder
        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = builder.build(goal="Fix auth bug")
        self.assertNotIn("codebase_overview", ctx)

    def test_prompt_renders_codebase_understanding(self):
        from core.agent_sdk.context_builder import ContextBuilder
        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = {
            "recommended_skills": [],
            "codebase_overview": {
                "summary": "200 files in 4 clusters",
                "total_files": 200,
            },
            "relevant_symbols": [
                {"name": "auth_func", "path": "core/auth.py", "intent_summary": "Handles auth"}
            ],
        }
        prompt = builder.build_system_prompt(goal="Fix bug", goal_type="bug_fix", context=ctx)
        self.assertIn("Codebase Understanding", prompt)
        self.assertIn("200 files", prompt)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_context_builder.py::TestContextBuilderSemantic -v`

- [ ] **Step 3: Update context_builder.py**

1. Add `semantic_querier` as 4th kwarg in `__init__`:

```python
    def __init__(
        self,
        project_root: Path,
        brain: Any = None,
        vector_store: Any = None,
        semantic_querier: Any = None,
    ) -> None:
        self._project_root = project_root
        self._brain = brain
        self._vector_store = vector_store
        self._semantic_querier = semantic_querier
```

2. Add `_get_codebase_context` method after `_get_available_mcp_categories`:

```python
    def _get_codebase_context(self, goal: str) -> Dict[str, Any]:
        """Query semantic index for goal-relevant codebase context."""
        if self._semantic_querier is None:
            return {}
        try:
            overview = self._semantic_querier.architecture_overview()
            relevant = self._semantic_querier.find_similar(goal, limit=5)
            goal_files = [s["path"] for s in relevant if "path" in s]
            impact = []
            for f in goal_files[:3]:
                deps = self._semantic_querier.what_depends_on(f)
                impact.extend(deps)
            return {
                "codebase_overview": overview,
                "relevant_symbols": relevant,
                "impact_radius": impact,
            }
        except Exception:
            return {}
```

3. Update `build()` to merge codebase context:

```python
    def build(self, goal: str) -> Dict[str, Any]:
        goal_type = self.classify_goal(goal)
        ctx = {
            "goal": goal,
            "goal_type": goal_type,
            "project_root": str(self._project_root),
            "recommended_skills": self._get_recommended_skills(goal_type),
            "memory_hints": self._get_memory_hints(goal),
            "available_mcp_categories": self._get_available_mcp_categories(),
        }
        ctx.update(self._get_codebase_context(goal))
        return ctx
```

4. Add codebase rendering to `build_system_prompt()` after the existing sections:

```python
            if context.get("codebase_overview"):
                overview = context["codebase_overview"]
                summary = overview.get("summary", "")
                parts.append(f"### Codebase Understanding\n**Architecture:** {summary}")
            if context.get("relevant_symbols"):
                lines = []
                for s in context["relevant_symbols"][:5]:
                    desc = s.get("intent_summary", s.get("docstring", ""))
                    lines.append(f"- {s.get('path', '?')}: {s['name']} — {desc}")
                if lines:
                    parts.append("**Relevant Code:**\n" + "\n".join(lines))
            if context.get("impact_radius"):
                paths = list(set(d["path"] for d in context["impact_radius"][:10]))
                if paths:
                    parts.append(f"**Impact Radius:** {len(paths)} files affected: {', '.join(paths[:5])}")
```

- [ ] **Step 4: Run ALL context builder tests**

Run: `python3 -m pytest tests/test_agent_sdk_context_builder.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/context_builder.py tests/test_agent_sdk_context_builder.py
git commit -m "feat: integrate semantic querier into context builder"
```

---

## Task 6: Add query_codebase Tool + Wire into Controller

**Files:**
- Modify: `core/agent_sdk/tool_registry.py`
- Modify: `core/agent_sdk/controller.py`
- Test: `tests/test_agent_sdk_tool_registry.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_sdk_tool_registry.py`:

```python
class TestQueryCodebaseTool(unittest.TestCase):
    def test_query_codebase_in_tool_list(self):
        from core.agent_sdk.tool_registry import create_aura_tools
        tools = create_aura_tools(project_root=Path("/tmp/test"))
        names = [t.name for t in tools]
        self.assertIn("query_codebase", names)

    def test_query_codebase_without_querier(self):
        from core.agent_sdk.tool_registry import _handle_query_codebase
        result = _handle_query_codebase({"query_type": "what_calls", "target": "foo"})
        self.assertIn("error", result)
        self.assertIn("not available", result["error"])

    def test_query_codebase_dispatches(self):
        from core.agent_sdk.tool_registry import _handle_query_codebase
        mock_querier = MagicMock()
        mock_querier.what_calls.return_value = [{"caller": "bar"}]
        result = _handle_query_codebase(
            {"query_type": "what_calls", "target": "foo"},
            semantic_querier=mock_querier,
        )
        mock_querier.what_calls.assert_called_once_with("foo")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_sdk_tool_registry.py::TestQueryCodebaseTool -v`

- [ ] **Step 3: Add handler and tool def to tool_registry.py**

Add `_handle_query_codebase` function (use the exact code from the spec, Section 5 "Tool Handler + Dependency Threading").

Add to `_TOOL_DEFS` list:
```python
    {
        "name": "query_codebase",
        "description": "Query the semantic codebase index for deep code understanding. "
                       "Use this to understand call chains, dependencies, impact of changes, "
                       "and to find relevant code by description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["what_calls", "what_depends_on", "what_changes_break",
                             "summarize", "find_similar", "architecture_overview",
                             "recent_changes"],
                },
                "target": {"type": "string", "description": "File path, symbol name, or description"},
                "depth": {"type": "integer", "default": 2},
            },
            "required": ["query_type"],
        },
        "handler": _handle_query_codebase,
    },
```

Add `semantic_querier` to `create_aura_tools` signature and `deps` dict.

- [ ] **Step 4: Update controller.py**

1. In `__init__`, create querier and pass to context_builder:
```python
        querier = None
        if hasattr(config, 'semantic_index_path') and config.semantic_index_path.exists():
            try:
                from core.agent_sdk.semantic_querier import SemanticQuerier
                querier = SemanticQuerier(db_path=config.semantic_index_path)
            except Exception:
                pass
        self._semantic_querier = querier
        self.context_builder = ContextBuilder(
            project_root=project_root, brain=brain,
            semantic_querier=querier,
        )
```

2. Update `_build_prompt` to use `self.context_builder`:
```python
    def _build_prompt(self, goal: str) -> str:
        context = self.context_builder.build(goal=goal)
        return self.context_builder.build_system_prompt(
            goal=goal, goal_type=context["goal_type"], context=context,
        )
```

3. Update `_build_mcp_server` to pass querier:
```python
        tools = create_aura_tools(
            ...,
            semantic_querier=self._semantic_querier,
        )
```

- [ ] **Step 5: Run ALL tool_registry and controller tests**

Run: `python3 -m pytest tests/test_agent_sdk_tool_registry.py tests/test_agent_sdk_controller.py tests/test_agent_sdk_controller_v2.py -v`

- [ ] **Step 6: Commit**

```bash
git add core/agent_sdk/tool_registry.py core/agent_sdk/controller.py tests/test_agent_sdk_tool_registry.py
git commit -m "feat: add query_codebase tool and wire semantic querier into controller"
```

---

## Task 7: CLI Scan Command

**Files:**
- Modify: `core/agent_sdk/cli_integration.py`
- Test: `tests/test_agent_sdk_cli_integration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_sdk_cli_integration.py`:

```python
class TestAgentScanCLI(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        (Path(self.tmpdir) / "test.py").write_text("def hello():\n    pass\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_handle_agent_scan(self):
        from core.agent_sdk.cli_integration import handle_agent_scan
        result = handle_agent_scan(
            project_root=Path(self.tmpdir),
            db_path=Path(self.tmpdir) / "index.db",
            exclude_patterns=["__pycache__"],
            no_llm=True,
        )
        self.assertIn("files_scanned", result)
        self.assertGreater(result["files_scanned"], 0)

    def test_handle_agent_scan_stats(self):
        from core.agent_sdk.cli_integration import handle_agent_scan, format_scan_stats
        handle_agent_scan(
            project_root=Path(self.tmpdir),
            db_path=Path(self.tmpdir) / "index.db",
            no_llm=True,
        )
        stats = format_scan_stats(Path(self.tmpdir) / "index.db")
        self.assertIn("files", stats.lower())
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Add scan handlers to cli_integration.py**

```python
def handle_agent_scan(
    project_root: Path,
    db_path: Path,
    exclude_patterns: Optional[List[str]] = None,
    model_adapter: Any = None,
    no_llm: bool = False,
) -> Dict[str, Any]:
    """Run a semantic scan of the codebase."""
    from core.agent_sdk.semantic_scanner import SemanticScanner
    scanner = SemanticScanner(
        project_root=project_root,
        db_path=db_path,
        exclude_patterns=exclude_patterns or [".git", "__pycache__", "node_modules"],
        model_adapter=None if no_llm else model_adapter,
    )
    return scanner.scan_full()


def format_scan_stats(db_path: Path) -> str:
    """Format scan statistics for CLI output."""
    from core.agent_sdk.semantic_schema import SemanticDB
    db = SemanticDB(db_path)
    meta = db.get_last_scan()
    if not meta:
        return "No scan data available. Run 'agent scan' first."
    files = db.get_all_files()
    return (
        f"Last scan: {meta['scan_time']}\n"
        f"Type: {meta['scan_type']}, SHA: {meta['scan_sha'][:8]}\n"
        f"Files: {len(files)}, Symbols: {meta['symbols_found']}\n"
        f"LLM calls: {meta['llm_calls_made']}, Cost: ${meta['llm_cost_usd']:.3f}"
    )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py -v`

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/cli_integration.py tests/test_agent_sdk_cli_integration.py
git commit -m "feat: add agent scan CLI command"
```

---

## Task 8: Integration Test

**Files:**
- Create: `tests/test_semantic_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_semantic_integration.py
"""Integration test: full scan → query → context injection."""
import tempfile
import unittest
from pathlib import Path


class TestSemanticEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()
        self.db_path = Path(self.tmpdir) / "index.db"

        # Create a small test project
        (self.project / "auth.py").write_text(
            'def authenticate(user, pwd):\n    """Validate credentials."""\n    return True\n\n'
            'def create_token(user):\n    """Create JWT token."""\n    return "token"\n'
        )
        (self.project / "server.py").write_text(
            'from auth import authenticate\n\n'
            'def handle(req):\n    auth = authenticate(req.user, req.pwd)\n    return auth\n'
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scan_then_query(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        from core.agent_sdk.semantic_querier import SemanticQuerier

        scanner = SemanticScanner(
            project_root=self.project, db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        result = scanner.scan_full()
        self.assertGreater(result["files_scanned"], 0)

        querier = SemanticQuerier(self.db_path)

        # Test all query types
        callers = querier.what_calls("authenticate")
        self.assertGreater(len(callers), 0)

        deps = querier.what_depends_on("auth.py")
        self.assertGreater(len(deps), 0)

        overview = querier.architecture_overview()
        self.assertGreater(overview["total_files"], 0)

        summary = querier.summarize("authenticate")
        self.assertIsInstance(summary, str)

    def test_context_builder_with_semantic_index(self):
        from core.agent_sdk.semantic_scanner import SemanticScanner
        from core.agent_sdk.semantic_querier import SemanticQuerier
        from core.agent_sdk.context_builder import ContextBuilder

        scanner = SemanticScanner(
            project_root=self.project, db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        scanner.scan_full()
        querier = SemanticQuerier(self.db_path)

        builder = ContextBuilder(
            project_root=self.project,
            semantic_querier=querier,
        )
        ctx = builder.build(goal="Fix authentication bug")
        self.assertIn("codebase_overview", ctx)
        self.assertIn("relevant_symbols", ctx)

        prompt = builder.build_system_prompt(
            goal="Fix auth bug", goal_type="bug_fix", context=ctx,
        )
        self.assertIn("Codebase Understanding", prompt)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run integration test**

Run: `python3 -m pytest tests/test_semantic_integration.py -v`

- [ ] **Step 3: Run complete test suite**

Run: `python3 -m pytest tests/test_agent_sdk_*.py tests/test_semantic_*.py tests/integration/test_agent_sdk_integration.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_semantic_integration.py
git commit -m "test: add semantic scanner end-to-end integration tests"
```
