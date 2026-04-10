"""Tests for the three-layer semantic scanner pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent_sdk.semantic_schema import SemanticDB
from core.agent_sdk.semantic_scanner import (
    SemanticScanner,
    build_relationships,
    compute_coupling_scores,
    extract_call_sites,
    extract_imports,
    extract_symbols,
    generate_function_intents,
    generate_module_summary,
)


def _write(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(code))


class TestASTExtraction(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_extract_functions(self):
        src = '''\
            def greet(name: str) -> str:
                """Return a greeting."""
                return f"Hello, {name}"

            def add(a: int, b: int) -> int:
                return a + b
        '''
        f = self.tmpdir / "funcs.py"
        _write(f, src)
        syms = extract_symbols(f)
        names = {s["name"] for s in syms}
        self.assertIn("greet", names)
        self.assertIn("add", names)
        greet = next(s for s in syms if s["name"] == "greet")
        self.assertEqual(greet["kind"], "function")
        self.assertIn("name", greet["signature"])
        self.assertIn("Return a greeting", greet["docstring"])
        self.assertGreater(greet["line_end"], greet["line_start"])

    def test_extract_classes_and_methods(self):
        src = '''\
            class Foo:
                """A foo class."""

                def regular(self):
                    pass

                @staticmethod
                def static_one():
                    pass

                @classmethod
                def class_one(cls):
                    pass

                @property
                def prop(self):
                    return 42
        '''
        f = self.tmpdir / "cls.py"
        _write(f, src)
        syms = extract_symbols(f)
        kinds = {s["name"]: s["kind"] for s in syms}
        self.assertEqual(kinds["Foo"], "class")
        self.assertEqual(kinds["regular"], "method")
        self.assertEqual(kinds["static_one"], "staticmethod")
        self.assertEqual(kinds["class_one"], "classmethod")
        self.assertEqual(kinds["prop"], "property")

    def test_extract_imports(self):
        src = """\
            import os
            from pathlib import Path
            from typing import Any, Dict
        """
        f = self.tmpdir / "imps.py"
        _write(f, src)
        imps = extract_imports(f)
        modules = {i["imported_module"] for i in imps}
        self.assertIn("os", modules)
        self.assertIn("pathlib", modules)
        self.assertIn("typing", modules)
        from_imps = [i for i in imps if i["is_from_import"]]
        names = {i["imported_name"] for i in from_imps}
        self.assertIn("Path", names)
        self.assertIn("Any", names)
        self.assertIn("Dict", names)

    def test_extract_call_sites(self):
        src = """\
            def caller():
                result = len([1, 2, 3])
                print(result)
                return result
        """
        f = self.tmpdir / "calls.py"
        _write(f, src)
        syms = extract_symbols(f)
        self.assertEqual(len(syms), 1)
        calls = extract_call_sites(f, syms[0])
        callee_names = {c["callee_name"] for c in calls}
        self.assertIn("len", callee_names)
        self.assertIn("print", callee_names)

    def test_extract_base_classes(self):
        src = """\
            class Base:
                pass

            class Child(Base):
                pass

            class MultiChild(Base, object):
                pass
        """
        f = self.tmpdir / "inherit.py"
        _write(f, src)
        syms = extract_symbols(f)
        by_name = {s["name"]: s for s in syms}
        self.assertEqual(by_name["Base"].get("bases", []), [])
        self.assertIn("Base", by_name["Child"]["bases"])
        multi_bases = by_name["MultiChild"]["bases"]
        self.assertIn("Base", multi_bases)
        self.assertIn("object", multi_bases)

    def test_extract_handles_syntax_error(self):
        f = self.tmpdir / "bad.py"
        f.write_text("def broken(:\n    pass\n")
        self.assertEqual(extract_symbols(f), [])
        self.assertEqual(extract_imports(f), [])

    def test_extract_handles_unicode_error(self):
        f = self.tmpdir / "binary.py"
        f.write_bytes(b"\xff\xfe invalid unicode \x00\x01")
        self.assertEqual(extract_symbols(f), [])
        self.assertEqual(extract_imports(f), [])


class TestRelationshipAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = self.tmpdir / "test.db"
        self.db = SemanticDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def _make_file(self, rel: str, code: str) -> int:
        abs_path = self.tmpdir / rel
        _write(abs_path, code)
        line_count = len(code.splitlines())
        module_name = rel.replace("/", ".").removesuffix(".py")
        cluster = str(Path(rel).parent)
        return self.db.upsert_file(
            path=rel,
            module_name=module_name,
            cluster=cluster,
            line_count=line_count,
            last_modified="2024-01-01",
            last_scan_sha=None,
        )

    def test_build_import_relationships(self):
        # file b.py has no imports
        b_id = self._make_file("b.py", "x = 1\n")
        # file a.py imports b
        a_id = self._make_file("a.py", "from b import x\n")
        # Record the import in DB
        self.db.insert_import(a_id, "b", "x", True)

        build_relationships(self.db)

        rels = self.db.get_relationships_from(a_id, rel_type="imports")
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["to_file_id"], b_id)

    def test_compute_coupling_scores(self):
        # hub.py imports both a.py and b.py
        a_id = self._make_file("a.py", "x = 1\n")
        b_id = self._make_file("b.py", "y = 2\n")
        hub_id = self._make_file("hub.py", "from a import x\nfrom b import y\n")
        self.db.upsert_relationship(hub_id, a_id, "imports", 1.0)
        self.db.upsert_relationship(hub_id, b_id, "imports", 1.0)

        compute_coupling_scores(self.db)

        files = {f["path"]: f for f in self.db.get_all_files()}
        self.assertGreater(files["hub.py"]["coupling_score"], 0.0)


class TestFullScan(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = self.tmpdir / "scan.db"

    def _make_py(self, rel: str, code: str) -> None:
        abs_path = self.tmpdir / rel
        _write(abs_path, textwrap.dedent(code))

    def test_full_scan_without_llm(self):
        self._make_py(
            "pkg/alpha.py",
            """\
            def alpha_func(x: int) -> int:
                \"\"\"Alpha function.\"\"\"
                return x + 1
        """,
        )
        self._make_py(
            "pkg/beta.py",
            """\
            from pkg.alpha import alpha_func

            def beta_func():
                return alpha_func(10)
        """,
        )

        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            model_adapter=None,
        )
        result = scanner.scan_full(git_sha="abc123")

        self.assertGreaterEqual(result["files_scanned"], 2)
        self.assertGreaterEqual(result["symbols_found"], 2)
        self.assertEqual(result["llm_calls"], 0)

        db = SemanticDB(self.db_path)
        files = db.get_all_files()
        paths = {f["path"] for f in files}
        self.assertTrue(any("alpha.py" in p for p in paths))
        self.assertTrue(any("beta.py" in p for p in paths))

        scan_meta = db.get_last_scan()
        self.assertIsNotNone(scan_meta)
        self.assertEqual(scan_meta["scan_sha"], "abc123")
        db.close()

    def test_scan_respects_exclude_patterns(self):
        self._make_py(
            "good.py",
            """\
            def good():
                pass
        """,
        )
        # This file should be excluded
        bad_dir = self.tmpdir / "__pycache__"
        bad_dir.mkdir(exist_ok=True)
        (bad_dir / "bad.py").write_text("def bad(): pass\n")

        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            exclude_patterns=["__pycache__"],
            model_adapter=None,
        )
        result = scanner.scan_full()

        db = SemanticDB(self.db_path)
        paths = {f["path"] for f in db.get_all_files()}
        db.close()

        self.assertFalse(any("bad.py" in p for p in paths))
        self.assertTrue(any("good.py" in p for p in paths))


class TestIncrementalScan(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = self.tmpdir / "scan.db"

    def _make_py(self, rel: str, code: str) -> Path:
        abs_path = self.tmpdir / rel
        _write(abs_path, textwrap.dedent(code))
        return abs_path

    def test_scan_incremental_processes_only_listed_files(self):
        self._make_py("one.py", "def one(): pass\n")
        self._make_py("two.py", "def two(): pass\n")

        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            model_adapter=None,
        )
        scanner.scan_full(git_sha="sha1")

        # Add a third file and do incremental scan
        self._make_py("three.py", "def three(): pass\n")
        result = scanner.scan_incremental(["three.py"], git_sha="sha2")

        self.assertGreaterEqual(result["files_scanned"], 1)

        db = SemanticDB(self.db_path)
        paths = {f["path"] for f in db.get_all_files()}
        db.close()
        self.assertTrue(any("three.py" in p for p in paths))

    def test_refresh_if_needed_noop_on_matching_sha(self):
        self._make_py("alpha.py", "def alpha(): pass\n")

        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            model_adapter=None,
        )

        with patch("subprocess.check_output", return_value=b"same-sha\n"):
            scanner.scan_full(git_sha="same-sha")
            result = scanner.refresh_if_needed()

        # Should be None (no-op) when SHA matches
        self.assertIsNone(result)

    def test_incremental_deletes_removed_files(self):
        self._make_py("keep.py", "def keep(): pass\n")
        gone_path = self._make_py("gone.py", "def gone(): pass\n")

        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            model_adapter=None,
        )
        scanner.scan_full(git_sha="sha1")

        # Delete the file from disk
        gone_path.unlink()

        # Incremental scan on the gone file
        result = scanner.scan_incremental(["gone.py"], git_sha="sha2")

        db = SemanticDB(self.db_path)
        paths = {f["path"] for f in db.get_all_files()}
        db.close()
        self.assertFalse(any("gone.py" in p for p in paths))


class TestLayer3LLM(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = self.tmpdir / "test.db"
        self.db = SemanticDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def test_generate_module_summary_returns_string(self):
        f = self.tmpdir / "mod.py"
        _write(f, "def foo(): pass\n")
        file_id = self.db.upsert_file(
            path="mod.py",
            module_name="mod",
            cluster=".",
            line_count=1,
            last_modified="2024-01-01",
            last_scan_sha=None,
        )
        adapter = MagicMock()
        adapter.respond.return_value = "This module does foo things."

        result = generate_module_summary(self.db, file_id, f, adapter)

        self.assertIsNotNone(result)
        self.assertIn("foo", result.lower() + "foo")  # result is from mock

    def test_generate_module_summary_returns_none_on_failure(self):
        f = self.tmpdir / "mod2.py"
        _write(f, "def bar(): pass\n")
        file_id = self.db.upsert_file(
            path="mod2.py",
            module_name="mod2",
            cluster=".",
            line_count=1,
            last_modified="2024-01-01",
            last_scan_sha=None,
        )
        adapter = MagicMock()
        adapter.respond.side_effect = RuntimeError("LLM offline")

        result = generate_module_summary(self.db, file_id, f, adapter)
        self.assertIsNone(result)

    def test_generate_function_intents_batches(self):
        f = self.tmpdir / "funcs.py"
        _write(f, "def alpha(): pass\ndef beta(): pass\n")
        file_id = self.db.upsert_file(
            path="funcs.py",
            module_name="funcs",
            cluster=".",
            line_count=2,
            last_modified="2024-01-01",
            last_scan_sha=None,
        )
        sym_id1 = self.db.insert_symbol(file_id, "alpha", "function", 1, 1, "alpha()", None, "")
        sym_id2 = self.db.insert_symbol(file_id, "beta", "function", 2, 2, "beta()", None, "")

        symbols = [
            {"_db_id": sym_id1, "name": "alpha", "kind": "function", "line_start": 1, "line_end": 1, "signature": "alpha()", "docstring": None},
            {"_db_id": sym_id2, "name": "beta", "kind": "function", "line_start": 2, "line_end": 2, "signature": "beta()", "docstring": None},
        ]
        adapter = MagicMock()
        adapter.respond.return_value = "alpha: does alpha.\nbeta: does beta."

        result = generate_function_intents(symbols, f, adapter, batch_size=10)

        self.assertIsInstance(result, dict)
        # Both symbol IDs should be present
        self.assertIn(sym_id1, result)
        self.assertIn(sym_id2, result)


if __name__ == "__main__":
    unittest.main()
