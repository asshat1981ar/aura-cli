# tests/test_semantic_schema.py
"""Tests for the SemanticDB SQLite schema and CRUD helpers."""
import shutil
import tempfile
import unittest
from pathlib import Path


class TestSemanticDB(unittest.TestCase):
    """CRUD and FTS5 tests for SemanticDB."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_semantic.db"
        from core.agent_sdk.semantic_schema import SemanticDB
        self.db = SemanticDB(self.db_path)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. Schema initialisation
    # ------------------------------------------------------------------

    def test_init_creates_tables(self):
        tables = self.db.list_tables()
        for expected in ("files", "symbols", "imports", "call_sites", "relationships", "scan_meta"):
            self.assertIn(expected, tables, f"Expected table '{expected}' not found in {tables}")

    # ------------------------------------------------------------------
    # 2. upsert_file
    # ------------------------------------------------------------------

    def test_upsert_file(self):
        file_id = self.db.upsert_file(
            path="mypackage/foo.py",
            module_name="mypackage.foo",
            cluster="core",
            line_count=100,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha="abc123",
        )
        self.assertIsInstance(file_id, int)
        self.assertGreater(file_id, 0)

        # Upsert again — same path must return same ID
        file_id2 = self.db.upsert_file(
            path="mypackage/foo.py",
            module_name="mypackage.foo",
            cluster="core",
            line_count=120,
            last_modified="2026-01-02T00:00:00",
            last_scan_sha="def456",
        )
        self.assertEqual(file_id, file_id2)

    # ------------------------------------------------------------------
    # 3. insert_symbol
    # ------------------------------------------------------------------

    def test_insert_symbol(self):
        file_id = self.db.upsert_file(
            path="pkg/bar.py",
            module_name="pkg.bar",
            cluster="agents",
            line_count=50,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        symbol_id = self.db.insert_symbol(
            file_id=file_id,
            name="MyClass",
            kind="class",
            line_start=1,
            line_end=40,
            signature="class MyClass:",
            docstring="A useful class.",
            decorators="",
        )
        self.assertIsInstance(symbol_id, int)
        self.assertGreater(symbol_id, 0)

    # ------------------------------------------------------------------
    # 4. insert_import / get_imports_for_file
    # ------------------------------------------------------------------

    def test_insert_import(self):
        file_id = self.db.upsert_file(
            path="pkg/baz.py",
            module_name="pkg.baz",
            cluster="tools",
            line_count=20,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        import_id = self.db.insert_import(
            file_id=file_id,
            imported_module="os.path",
            imported_name="join",
            is_from_import=True,
        )
        self.assertIsInstance(import_id, int)
        self.assertGreater(import_id, 0)

        imports = self.db.get_imports_for_file(file_id)
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0]["imported_module"], "os.path")
        self.assertEqual(imports[0]["imported_name"], "join")
        # stored as 1 for True
        self.assertEqual(imports[0]["is_from_import"], 1)

    # ------------------------------------------------------------------
    # 5. insert_call_site / get_call_sites_for_symbol
    # ------------------------------------------------------------------

    def test_insert_call_site(self):
        file_id = self.db.upsert_file(
            path="pkg/caller.py",
            module_name="pkg.caller",
            cluster="core",
            line_count=30,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        symbol_id = self.db.insert_symbol(
            file_id=file_id,
            name="do_work",
            kind="function",
            line_start=5,
            line_end=20,
            signature="def do_work():",
            docstring=None,
            decorators="",
        )
        call_id = self.db.insert_call_site(
            caller_symbol_id=symbol_id,
            callee_name="helper_fn",
            line=12,
        )
        self.assertIsInstance(call_id, int)
        self.assertGreater(call_id, 0)

        call_sites = self.db.get_call_sites_for_symbol(symbol_id)
        self.assertEqual(len(call_sites), 1)
        self.assertEqual(call_sites[0]["callee_name"], "helper_fn")
        self.assertEqual(call_sites[0]["line"], 12)

    # ------------------------------------------------------------------
    # 6. upsert_relationship / get_relationships_from
    # ------------------------------------------------------------------

    def test_upsert_relationship(self):
        file_a = self.db.upsert_file(
            path="pkg/a.py",
            module_name="pkg.a",
            cluster="core",
            line_count=10,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        file_b = self.db.upsert_file(
            path="pkg/b.py",
            module_name="pkg.b",
            cluster="core",
            line_count=10,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        self.db.upsert_relationship(
            from_file_id=file_a,
            to_file_id=file_b,
            rel_type="imports",
            strength=0.8,
        )
        rels = self.db.get_relationships_from(file_a, rel_type="imports")
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["to_file_id"], file_b)
        self.assertAlmostEqual(rels[0]["strength"], 0.8)

        # Upsert again — strength updated, still one row
        self.db.upsert_relationship(
            from_file_id=file_a,
            to_file_id=file_b,
            rel_type="imports",
            strength=0.9,
        )
        rels2 = self.db.get_relationships_from(file_a, rel_type="imports")
        self.assertEqual(len(rels2), 1)
        self.assertAlmostEqual(rels2[0]["strength"], 0.9)

    # ------------------------------------------------------------------
    # 7. delete_file cascades
    # ------------------------------------------------------------------

    def test_delete_file_cascades(self):
        file_id = self.db.upsert_file(
            path="pkg/transient.py",
            module_name="pkg.transient",
            cluster="core",
            line_count=15,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        self.db.insert_symbol(
            file_id=file_id,
            name="transient_fn",
            kind="function",
            line_start=1,
            line_end=10,
            signature="def transient_fn():",
            docstring=None,
            decorators="",
        )
        self.db.insert_import(
            file_id=file_id,
            imported_module="sys",
            imported_name=None,
            is_from_import=False,
        )

        self.db.delete_file(file_id)

        self.assertIsNone(self.db.get_file_by_path("pkg/transient.py"))
        # symbols and imports must be gone via FK cascade
        symbols = self.db.get_symbols_for_file(file_id)
        self.assertEqual(len(symbols), 0)
        imports = self.db.get_imports_for_file(file_id)
        self.assertEqual(len(imports), 0)

    # ------------------------------------------------------------------
    # 8. record_scan / get_last_scan
    # ------------------------------------------------------------------

    def test_record_scan_meta(self):
        self.db.record_scan(
            scan_sha="sha_abc",
            files_scanned=42,
            symbols_found=300,
            llm_calls_made=5,
            llm_cost_usd=0.12,
            scan_type="full",
            scan_time="2026-01-01T12:00:00",
        )
        last = self.db.get_last_scan()
        self.assertIsNotNone(last)
        self.assertEqual(last["scan_sha"], "sha_abc")
        self.assertEqual(last["files_scanned"], 42)
        self.assertEqual(last["symbols_found"], 300)
        self.assertEqual(last["llm_calls_made"], 5)
        self.assertAlmostEqual(last["llm_cost_usd"], 0.12)
        self.assertEqual(last["scan_type"], "full")
        self.assertEqual(last["scan_time"], "2026-01-01T12:00:00")

    # ------------------------------------------------------------------
    # 9. fts_search
    # ------------------------------------------------------------------

    def test_fts_search(self):
        if not self.db.fts_enabled:
            self.skipTest("FTS5 not available in this SQLite build")

        file_id = self.db.upsert_file(
            path="pkg/searchable.py",
            module_name="pkg.searchable",
            cluster="core",
            line_count=50,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        symbol_id = self.db.insert_symbol(
            file_id=file_id,
            name="compute_embedding",
            kind="function",
            line_start=10,
            line_end=30,
            signature="def compute_embedding(text: str) -> List[float]:",
            docstring="Computes a dense vector embedding for the given text.",
            decorators="",
            intent_summary="Generates embeddings using a local model.",
        )

        results = self.db.fts_search("embedding", limit=5)
        self.assertGreater(len(results), 0)
        names = [r["name"] for r in results]
        self.assertIn("compute_embedding", names)

    # ------------------------------------------------------------------
    # 10. get_all_files
    # ------------------------------------------------------------------

    def test_get_all_files(self):
        self.db.upsert_file(
            path="pkg/alpha.py",
            module_name="pkg.alpha",
            cluster="core",
            line_count=10,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        self.db.upsert_file(
            path="pkg/beta.py",
            module_name="pkg.beta",
            cluster="core",
            line_count=20,
            last_modified="2026-01-01T00:00:00",
            last_scan_sha=None,
        )
        files = self.db.get_all_files()
        self.assertGreaterEqual(len(files), 2)
        paths = [f["path"] for f in files]
        self.assertIn("pkg/alpha.py", paths)
        self.assertIn("pkg/beta.py", paths)


if __name__ == "__main__":
    unittest.main()
