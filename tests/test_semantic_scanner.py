"""Tests for the three-layer semantic scanner pipeline."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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
    _arg_annotation,
    _base_name,
    _call_name,
    _decorator_name,
    _parse_tree,
)


def _write(path: Path, code: str) -> None:
    """Helper to write dedented code to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(code))


class TestASTHelpers(unittest.TestCase):
    """Test internal AST helper functions."""

    def test_decorator_name_simple_name(self):
        code = "@decorator\ndef f(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        self.assertEqual(_decorator_name(func.decorator_list[0]), "decorator")

    def test_decorator_name_attribute(self):
        code = "@module.decorator\ndef f(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        self.assertEqual(_decorator_name(func.decorator_list[0]), "module.decorator")

    def test_decorator_name_nested_attribute(self):
        code = "@pkg.mod.dec\ndef f(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        self.assertEqual(_decorator_name(func.decorator_list[0]), "pkg.mod.dec")

    def test_decorator_name_call(self):
        code = "@decorator()\ndef f(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        self.assertEqual(_decorator_name(func.decorator_list[0]), "decorator")

    def test_decorator_name_call_with_attr(self):
        code = "@module.decorator()\ndef f(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        self.assertEqual(_decorator_name(func.decorator_list[0]), "module.decorator")

    def test_decorator_name_unknown(self):
        code = "@dec[T]\ndef f(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        result = _decorator_name(func.decorator_list[0])
        self.assertEqual(result, "<decorator>")

    def test_base_name_simple(self):
        code = "class Child(Parent): pass"
        tree = ast.parse(code)
        cls = tree.body[0]
        self.assertEqual(_base_name(cls.bases[0]), "Parent")

    def test_base_name_attribute(self):
        code = "class Child(pkg.Parent): pass"
        tree = ast.parse(code)
        cls = tree.body[0]
        self.assertEqual(_base_name(cls.bases[0]), "pkg.Parent")

    def test_base_name_nested_attribute(self):
        code = "class Child(pkg.mod.Parent): pass"
        tree = ast.parse(code)
        cls = tree.body[0]
        self.assertEqual(_base_name(cls.bases[0]), "pkg.mod.Parent")

    def test_base_name_unknown(self):
        code = "class Child(Parent[T]): pass"
        tree = ast.parse(code)
        cls = tree.body[0]
        result = _base_name(cls.bases[0])
        self.assertEqual(result, "<base>")

    def test_call_name_simple(self):
        code = "result = foo()"
        tree = ast.parse(code)
        call = tree.body[0].value
        self.assertEqual(_call_name(call.func), "foo")

    def test_call_name_attribute(self):
        code = "result = obj.method()"
        tree = ast.parse(code)
        call = tree.body[0].value
        self.assertEqual(_call_name(call.func), "method")

    def test_call_name_subscript(self):
        code = "result = obj[0]()"
        tree = ast.parse(code)
        call = tree.body[0].value
        self.assertIsNone(_call_name(call.func))

    def test_arg_annotation_no_annotation(self):
        code = "def f(x): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        result = _arg_annotation(func.args.args[0])
        self.assertEqual(result, "")

    def test_arg_annotation_simple_type(self):
        code = "def f(x: int): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        result = _arg_annotation(func.args.args[0])
        self.assertEqual(result, "int")

    def test_arg_annotation_complex_type(self):
        code = "def f(x: List[str]): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        result = _arg_annotation(func.args.args[0])
        self.assertIn("List", result)
        self.assertIn("str", result)


class TestParseTree(unittest.TestCase):
    """Test _parse_tree function."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_parse_tree_valid_code(self):
        f = self.tmpdir / "valid.py"
        f.write_text("def foo(): pass")
        result = _parse_tree(f)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ast.Module)

    def test_parse_tree_with_unicode_decode_error(self):
        f = self.tmpdir / "bad_unicode.py"
        f.write_bytes(b"\x80\x81invalid")
        result = _parse_tree(f)
        self.assertIsNone(result)

    def test_parse_tree_with_syntax_error(self):
        f = self.tmpdir / "syntax.py"
        f.write_text("def broken(:\n  pass")
        result = _parse_tree(f)
        self.assertIsNone(result)

    def test_parse_tree_file_not_found(self):
        f = self.tmpdir / "nonexistent.py"
        result = _parse_tree(f)
        self.assertIsNone(result)


class TestExtractSymbols(unittest.TestCase):
    """Test extract_symbols function."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_extract_functions(self):
        src = '''
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

    def test_extract_classes_and_methods(self):
        src = '''
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

    def test_extract_decorators(self):
        src = """
            def outer(f):
                return f

            @outer
            @staticmethod
            def decorated():
                pass
        """
        f = self.tmpdir / "deco.py"
        _write(f, src)
        syms = extract_symbols(f)
        decorated = next((s for s in syms if s["name"] == "decorated"), None)
        self.assertIsNotNone(decorated)
        self.assertIn("outer", decorated["decorators"])

    def test_extract_async_function(self):
        src = """
            async def async_func():
                await something()
        """
        f = self.tmpdir / "async_func.py"
        _write(f, src)
        syms = extract_symbols(f)
        self.assertEqual(len(syms), 1)
        self.assertEqual(syms[0]["name"], "async_func")
        self.assertEqual(syms[0]["kind"], "function")

    def test_extract_async_method(self):
        src = """
            class MyClass:
                async def async_method(self):
                    pass
        """
        f = self.tmpdir / "async_method.py"
        _write(f, src)
        syms = extract_symbols(f)
        async_method = next((s for s in syms if s["name"] == "async_method"), None)
        self.assertIsNotNone(async_method)
        self.assertEqual(async_method["kind"], "method")

    def test_extract_function_with_defaults(self):
        src = """
            def func(a, b=10, c=None):
                pass
        """
        f = self.tmpdir / "defaults.py"
        _write(f, src)
        syms = extract_symbols(f)
        self.assertEqual(len(syms), 1)
        sig = syms[0]["signature"]
        self.assertIn("b", sig) and self.assertIn("10", sig)

    def test_extract_function_with_varargs(self):
        src = """
            def func(*args, **kwargs):
                pass
        """
        f = self.tmpdir / "varargs.py"
        _write(f, src)
        syms = extract_symbols(f)
        self.assertEqual(len(syms), 1)
        sig = syms[0]["signature"]
        self.assertIn("*args", sig)
        self.assertIn("**kwargs", sig)

    def test_extract_function_with_kwonly_args(self):
        src = """
            def func(a, *, b, c=10):
                pass
        """
        f = self.tmpdir / "kwonly.py"
        _write(f, src)
        syms = extract_symbols(f)
        self.assertEqual(len(syms), 1)
        sig = syms[0]["signature"]
        self.assertIn("b", sig)

    def test_extract_function_with_return_type(self):
        src = """
            def func() -> int:
                return 42
        """
        f = self.tmpdir / "return_type.py"
        _write(f, src)
        syms = extract_symbols(f)
        self.assertEqual(len(syms), 1)
        sig = syms[0]["signature"]
        self.assertIn("->", sig)
        self.assertIn("int", sig)

    def test_extract_posonly_args(self):
        src = """
            def func(a, /, b):
                pass
        """
        f = self.tmpdir / "posonly.py"
        _write(f, src)
        syms = extract_symbols(f)
        self.assertEqual(len(syms), 1)

    def test_extract_base_classes(self):
        src = """
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

    def test_extract_nested_classes(self):
        src = """
            class Outer:
                class Inner:
                    def method(self):
                        pass
        """
        f = self.tmpdir / "nested.py"
        _write(f, src)
        syms = extract_symbols(f)
        names = {s["name"] for s in syms}
        self.assertIn("Outer", names)
        self.assertIn("Inner", names)
        self.assertIn("method", names)

    def test_extract_multiple_decorators(self):
        src = """
            @dec1
            @dec2
            @dec3
            def func():
                pass
        """
        f = self.tmpdir / "multi_dec.py"
        _write(f, src)
        syms = extract_symbols(f)
        self.assertEqual(len(syms), 1)
        self.assertEqual(len(syms[0]["decorators"]), 3)

    def test_extract_handles_syntax_error(self):
        f = self.tmpdir / "bad.py"
        f.write_text("def broken(:\n    pass\n")
        self.assertEqual(extract_symbols(f), [])

    def test_extract_handles_unicode_error(self):
        f = self.tmpdir / "binary.py"
        f.write_bytes(b"\xff\xfe invalid unicode \x00\x01")
        self.assertEqual(extract_symbols(f), [])


class TestExtractImports(unittest.TestCase):
    """Test extract_imports function."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_extract_imports(self):
        src = """
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

    def test_extract_imports_with_aliases(self):
        src = """
            import os as operating_system
            from pathlib import Path as PathlibPath
        """
        f = self.tmpdir / "alias_imports.py"
        _write(f, src)
        imps = extract_imports(f)
        self.assertEqual(len(imps), 2)
        alias_imp = next((i for i in imps if i["imported_module"] == "os"), None)
        self.assertEqual(alias_imp["imported_name"], "operating_system")

    def test_extract_imports_from_without_module(self):
        src = """
            from . import sibling
            from .. import parent_mod
        """
        f = self.tmpdir / "relative_imports.py"
        _write(f, src)
        imps = extract_imports(f)
        self.assertEqual(len(imps), 2)

    def test_extract_imports_star(self):
        src = """
            from module import *
        """
        f = self.tmpdir / "star_import.py"
        _write(f, src)
        imps = extract_imports(f)
        self.assertEqual(len(imps), 1)
        self.assertEqual(imps[0]["imported_name"], "*")


class TestExtractCallSites(unittest.TestCase):
    """Test extract_call_sites function."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_extract_call_sites(self):
        src = """
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

    def test_extract_call_sites_nested_calls(self):
        src = """
            def caller():
                result = len(list(range(10)))
                return result
        """
        f = self.tmpdir / "nested_calls.py"
        _write(f, src)
        syms = extract_symbols(f)
        calls = extract_call_sites(f, syms[0])
        self.assertGreater(len(calls), 0)


class TestBuildRelationships(unittest.TestCase):
    """Test build_relationships function."""

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
        b_id = self._make_file("b.py", "x = 1\n")
        a_id = self._make_file("a.py", "from b import x\n")
        self.db.insert_import(a_id, "b", "x", True)

        build_relationships(self.db)

        rels = self.db.get_relationships_from(a_id, rel_type="imports")
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["to_file_id"], b_id)

    def test_build_relationships_circular(self):
        a_id = self.db.upsert_file("a.py", "a", "", 1, "2024-01-01", None)
        b_id = self.db.upsert_file("b.py", "b", "", 1, "2024-01-01", None)

        self.db.insert_import(a_id, "b", None, False)
        self.db.insert_import(b_id, "a", None, False)

        build_relationships(self.db)

        rels_a = self.db.get_relationships_from(a_id, rel_type="imports")
        rels_b = self.db.get_relationships_from(b_id, rel_type="imports")

        self.assertGreater(len(rels_a), 0)
        self.assertGreater(len(rels_b), 0)

    def test_build_relationships_no_duplicates(self):
        a_id = self.db.upsert_file("a.py", "a", "", 1, "2024-01-01", None)
        b_id = self.db.upsert_file("b.py", "b", "", 1, "2024-01-01", None)

        self.db.insert_import(a_id, "b", None, False)
        self.db.insert_import(a_id, "b", None, False)

        build_relationships(self.db)

        rels = self.db.get_relationships_from(a_id, rel_type="imports")
        self.assertEqual(len(rels), 1)


class TestComputeCouplingScores(unittest.TestCase):
    """Test compute_coupling_scores function."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = self.tmpdir / "test.db"
        self.db = SemanticDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def test_compute_coupling_scores(self):
        a_id = self.db.upsert_file("a.py", "a", "", 1, "2024-01-01", None)
        b_id = self.db.upsert_file("b.py", "b", "", 1, "2024-01-01", None)
        hub_id = self.db.upsert_file("hub.py", "hub", "", 1, "2024-01-01", None)
        self.db.upsert_relationship(hub_id, a_id, "imports", 1.0)
        self.db.upsert_relationship(hub_id, b_id, "imports", 1.0)

        compute_coupling_scores(self.db)

        files = {f["path"]: f for f in self.db.get_all_files()}
        self.assertGreater(files["hub.py"]["coupling_score"], 0.0)

    def test_compute_coupling_scores_empty_db(self):
        compute_coupling_scores(self.db)
        files = self.db.get_all_files()
        self.assertEqual(len(files), 0)

    def test_compute_coupling_scores_single_file(self):
        f_id = self.db.upsert_file("f.py", "f", "", 1, "2024-01-01", None)

        compute_coupling_scores(self.db)

        files = {f["id"]: f for f in self.db.get_all_files()}
        self.assertEqual(files[f_id]["coupling_score"], 0.0)

    def test_compute_coupling_scores_capped_at_one(self):
        ids = [self.db.upsert_file(f"f{i}.py", f"f{i}", "", 1, "2024-01-01", None) for i in range(3)]

        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j:
                    self.db.upsert_relationship(ids[i], ids[j], "imports", 1.0)

        compute_coupling_scores(self.db)

        files = {f["id"]: f for f in self.db.get_all_files()}
        for f_id in ids:
            self.assertLessEqual(files[f_id]["coupling_score"], 1.0)


class TestGenerateModuleSummary(unittest.TestCase):
    """Test generate_module_summary function."""

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

    def test_generate_module_summary_connection_error(self):
        f = self.tmpdir / "mod.py"
        _write(f, "def foo(): pass")
        file_id = self.db.upsert_file(
            path="mod.py",
            module_name="mod",
            cluster=".",
            line_count=1,
            last_modified="2024-01-01",
            last_scan_sha=None,
        )

        adapter = MagicMock()
        adapter.respond.side_effect = ConnectionError("Connection failed")

        result = generate_module_summary(self.db, file_id, f, adapter)
        self.assertIsNone(result)

    def test_generate_module_summary_timeout_error(self):
        f = self.tmpdir / "mod.py"
        _write(f, "def foo(): pass")
        file_id = self.db.upsert_file(
            path="mod.py",
            module_name="mod",
            cluster=".",
            line_count=1,
            last_modified="2024-01-01",
            last_scan_sha=None,
        )

        adapter = MagicMock()
        adapter.respond.side_effect = TimeoutError("Timeout")

        result = generate_module_summary(self.db, file_id, f, adapter)
        self.assertIsNone(result)

    def test_generate_module_summary_large_file_truncation(self):
        f = self.tmpdir / "large.py"
        large_code = "x = 1\n" * 2000
        f.write_text(large_code)
        file_id = self.db.upsert_file(
            path="large.py",
            module_name="large",
            cluster=".",
            line_count=2000,
            last_modified="2024-01-01",
            last_scan_sha=None,
        )

        adapter = MagicMock()
        adapter.respond.return_value = "Summary of truncated code."

        result = generate_module_summary(self.db, file_id, f, adapter)
        self.assertIsNotNone(result)
        call_args = adapter.respond.call_args
        self.assertIn("truncated", call_args[0][0].lower())


class TestGenerateFunctionIntents(unittest.TestCase):
    """Test generate_function_intents function."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_generate_function_intents_batches(self):
        f = self.tmpdir / "funcs.py"
        _write(f, "def alpha(): pass\ndef beta(): pass\n")

        symbols = [
            {"_db_id": 1, "name": "alpha", "kind": "function", "line_start": 1, "line_end": 1, "signature": "alpha()", "docstring": None},
            {"_db_id": 2, "name": "beta", "kind": "function", "line_start": 2, "line_end": 2, "signature": "beta()", "docstring": None},
        ]
        adapter = MagicMock()
        adapter.respond.return_value = "alpha: does alpha.\nbeta: does beta."

        result = generate_function_intents(symbols, f, adapter, batch_size=10)

        self.assertIsInstance(result, dict)
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_generate_function_intents_connection_error(self):
        f = self.tmpdir / "func.py"
        _write(f, "def func(): pass")

        symbols = [
            {
                "_db_id": 1,
                "name": "func",
                "kind": "function",
                "line_start": 1,
                "line_end": 1,
                "signature": "func()",
                "docstring": None,
            }
        ]

        adapter = MagicMock()
        adapter.respond.side_effect = ConnectionError("Failed")

        result = generate_function_intents(symbols, f, adapter)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get(1), "")

    def test_generate_function_intents_empty_symbols(self):
        f = self.tmpdir / "empty.py"
        _write(f, "")

        adapter = MagicMock()
        result = generate_function_intents([], f, adapter)
        self.assertEqual(result, {})
        adapter.respond.assert_not_called()

    def test_generate_function_intents_parse_response_colon(self):
        f = self.tmpdir / "funcs.py"
        _write(f, "def alpha(): pass\ndef beta(): pass")

        symbols = [
            {
                "_db_id": 1,
                "name": "alpha",
                "kind": "function",
                "line_start": 1,
                "line_end": 1,
                "signature": "alpha()",
                "docstring": None,
            },
            {
                "_db_id": 2,
                "name": "beta",
                "kind": "function",
                "line_start": 2,
                "line_end": 2,
                "signature": "beta()",
                "docstring": None,
            },
        ]

        adapter = MagicMock()
        adapter.respond.return_value = "alpha: does something.\nbeta: does other things."

        result = generate_function_intents(symbols, f, adapter, batch_size=10)

        self.assertEqual(result[1], "does something.")
        self.assertEqual(result[2], "does other things.")

    def test_generate_function_intents_response_without_colons(self):
        f = self.tmpdir / "funcs.py"
        _write(f, "def func(): pass")

        symbols = [
            {
                "_db_id": 1,
                "name": "func",
                "kind": "function",
                "line_start": 1,
                "line_end": 1,
                "signature": "func()",
                "docstring": None,
            }
        ]

        adapter = MagicMock()
        adapter.respond.return_value = "This is raw unparsed response."

        result = generate_function_intents(symbols, f, adapter, batch_size=10)

        self.assertEqual(result[1], "This is raw unparsed response.")


class TestScannerInternalHelpers(unittest.TestCase):
    """Test SemanticScanner internal helper methods."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = self.tmpdir / "test.db"

    def test_is_excluded_exact_match(self):
        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            exclude_patterns=["__pycache__", ".git"],
        )
        self.assertTrue(scanner._is_excluded("__pycache__"))
        self.assertTrue(scanner._is_excluded(".git"))
        self.assertFalse(scanner._is_excluded("src"))

    def test_is_excluded_glob_pattern(self):
        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            exclude_patterns=["*.egg-info", "*.tmp"],
        )
        self.assertTrue(scanner._is_excluded("package.egg-info"))
        self.assertTrue(scanner._is_excluded("file.tmp"))
        self.assertFalse(scanner._is_excluded("module.py"))

    def test_relative_path_conversion(self):
        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
        )
        abs_path = self.tmpdir / "src" / "main.py"
        rel = scanner._relative(abs_path)
        self.assertEqual(rel, "src/main.py")

    def test_relative_path_outside_root(self):
        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
        )
        other_root = Path(tempfile.mkdtemp())
        outside_path = other_root / "file.py"
        rel = scanner._relative(outside_path)
        self.assertEqual(rel, str(outside_path.as_posix()))

    def test_module_name_conversion(self):
        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
        )
        self.assertEqual(scanner._module_name("pkg/mod.py"), "pkg.mod")
        self.assertEqual(scanner._module_name("core/lib/util.py"), "core.lib.util")
        self.assertEqual(scanner._module_name("single.py"), "single")

    def test_cluster_extraction(self):
        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
        )
        self.assertEqual(scanner._cluster("pkg/mod.py"), "pkg")
        self.assertEqual(scanner._cluster("core/lib/util.py"), "core/lib")
        self.assertEqual(scanner._cluster("single.py"), "")

    def test_find_py_files(self):
        (self.tmpdir / "root.py").write_text("x = 1")
        (self.tmpdir / "pkg").mkdir()
        (self.tmpdir / "pkg" / "mod.py").write_text("y = 2")
        (self.tmpdir / "__pycache__").mkdir()
        (self.tmpdir / "__pycache__" / "cache.py").write_text("z = 3")

        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            exclude_patterns=["__pycache__"],
        )
        files = scanner._find_py_files()
        paths = {f.name for f in files}
        self.assertIn("root.py", paths)
        self.assertIn("mod.py", paths)
        self.assertNotIn("cache.py", paths)


class TestFullScan(unittest.TestCase):
    """Test SemanticScanner.scan_full method."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = self.tmpdir / "scan.db"

    def _make_py(self, rel: str, code: str) -> None:
        abs_path = self.tmpdir / rel
        _write(abs_path, textwrap.dedent(code))

    def test_full_scan_without_llm(self):
        self._make_py(
            "pkg/alpha.py",
            '''
            def alpha_func(x: int) -> int:
                """Alpha function."""
                return x + 1
        ''',
        )
        self._make_py(
            "pkg/beta.py",
            """
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
            """
            def good():
                pass
        """,
        )
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
    """Test SemanticScanner incremental scanning methods."""

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

        gone_path.unlink()

        result = scanner.scan_incremental(["gone.py"], git_sha="sha2")

        db = SemanticDB(self.db_path)
        paths = {f["path"] for f in db.get_all_files()}
        db.close()
        self.assertFalse(any("gone.py" in p for p in paths))

    def test_refresh_if_needed_git_unavailable(self):
        (self.tmpdir / "test.py").write_text("x = 1")

        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            model_adapter=None,
        )

        with patch("subprocess.check_output", side_effect=OSError("git not found")):
            result = scanner.refresh_if_needed()

        self.assertIsNotNone(result)
        self.assertIn("files_scanned", result)

    def test_refresh_if_needed_no_previous_scan(self):
        (self.tmpdir / "test.py").write_text("x = 1")

        scanner = SemanticScanner(
            project_root=self.tmpdir,
            db_path=self.db_path,
            model_adapter=None,
        )

        with patch("subprocess.check_output", return_value=b"abc123\n"):
            result = scanner.refresh_if_needed()

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
