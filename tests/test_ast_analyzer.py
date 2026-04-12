"""Tests for the AST analyzer skill."""

from __future__ import annotations

import ast
import os
import tempfile
import textwrap
from pathlib import Path
from unittest import TestCase

from agents.skills.ast_analyzer import ASTAnalyzerSkill, ASTMetrics


class TestASTAnalyzerSkill(TestCase):
    """Test suite for ASTAnalyzerSkill."""

    def setUp(self):
        self.skill = ASTAnalyzerSkill()

    # ------------------------------------------------------------------
    # Helper to write a temp Python file and analyze it
    # ------------------------------------------------------------------
    def _analyze_source(self, source: str) -> ASTMetrics:
        """Parse source and return ASTMetrics via _analyze_file."""
        tree = ast.parse(textwrap.dedent(source))
        return self.skill._analyze_file(tree, "test_file.py")

    def _run_on_tempdir(self, files: dict[str, str]) -> dict:
        """Write files to a temp dir and run the skill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, content in files.items():
                p = Path(tmpdir) / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(textwrap.dedent(content))
            return self.skill.run({"project_root": tmpdir})

    # ------------------------------------------------------------------
    # Test: Analyze a simple Python file
    # ------------------------------------------------------------------
    def test_analyze_simple_file(self):
        result = self._run_on_tempdir(
            {
                "example.py": """\
                    def hello():
                        return "hi"

                    class Greeter:
                        def greet(self):
                            pass
                """
            }
        )
        self.assertEqual(result["files_analyzed"], 1)
        self.assertEqual(result["total_functions"], 2)  # hello + greet
        self.assertEqual(result["total_classes"], 1)
        self.assertNotIn("error", result)

    # ------------------------------------------------------------------
    # Test: Detect too_many_args smell
    # ------------------------------------------------------------------
    def test_detect_too_many_args(self):
        metrics = self._analyze_source(
            """\
            def bloated(a, b, c, d, e, f, g):
                pass
            """
        )
        smell_types = [s["type"] for s in metrics.smells]
        self.assertIn("too_many_args", smell_types)
        smell = next(s for s in metrics.smells if s["type"] == "too_many_args")
        self.assertEqual(smell["count"], 7)
        self.assertEqual(smell["function"], "bloated")
        self.assertEqual(smell["severity"], "medium")

    def test_no_smell_for_six_args(self):
        metrics = self._analyze_source(
            """\
            def ok(a, b, c, d, e, f):
                pass
            """
        )
        smell_types = [s["type"] for s in metrics.smells]
        self.assertNotIn("too_many_args", smell_types)

    # ------------------------------------------------------------------
    # Test: Detect long_function smell
    # ------------------------------------------------------------------
    def test_detect_long_function(self):
        # Generate a function with 55 lines
        body_lines = "\n".join(f"    x{i} = {i}" for i in range(55))
        source = f"def long_func():\n{body_lines}\n"
        metrics = self._analyze_source(source)
        smell_types = [s["type"] for s in metrics.smells]
        self.assertIn("long_function", smell_types)
        smell = next(s for s in metrics.smells if s["type"] == "long_function")
        self.assertEqual(smell["function"], "long_func")
        self.assertGreater(smell["length"], 50)
        self.assertEqual(smell["severity"], "low")

    # ------------------------------------------------------------------
    # Test: Detect god_class smell
    # ------------------------------------------------------------------
    def test_detect_god_class(self):
        methods = "\n".join(f"    def method_{i}(self):\n        pass" for i in range(22))
        source = f"class GodClass:\n{methods}\n"
        metrics = self._analyze_source(source)
        smell_types = [s["type"] for s in metrics.smells]
        self.assertIn("god_class", smell_types)
        smell = next(s for s in metrics.smells if s["type"] == "god_class")
        self.assertEqual(smell["class"], "GodClass")
        self.assertEqual(smell["methods"], 22)
        self.assertEqual(smell["severity"], "high")

    def test_no_god_class_for_20_methods(self):
        methods = "\n".join(f"    def method_{i}(self):\n        pass" for i in range(20))
        source = f"class NormalClass:\n{methods}\n"
        metrics = self._analyze_source(source)
        smell_types = [s["type"] for s in metrics.smells]
        self.assertNotIn("god_class", smell_types)

    # ------------------------------------------------------------------
    # Test: Detect dead code after return
    # ------------------------------------------------------------------
    def test_detect_dead_code_after_return(self):
        metrics = self._analyze_source(
            """\
            def func():
                return 42
                print("unreachable")
            """
        )
        self.assertEqual(len(metrics.dead_code), 1)
        dc = metrics.dead_code[0]
        self.assertEqual(dc["function"], "func")
        self.assertEqual(dc["type"], "unreachable_after_return")

    def test_detect_dead_code_after_raise(self):
        metrics = self._analyze_source(
            """\
            def func():
                raise ValueError("oops")
                print("unreachable")
            """
        )
        self.assertEqual(len(metrics.dead_code), 1)
        self.assertEqual(metrics.dead_code[0]["type"], "unreachable_after_return")

    def test_no_dead_code_when_return_is_last(self):
        metrics = self._analyze_source(
            """\
            def func():
                x = 1
                return x
            """
        )
        self.assertEqual(len(metrics.dead_code), 0)

    # ------------------------------------------------------------------
    # Test: Calculate type coverage
    # ------------------------------------------------------------------
    def test_type_coverage_all_annotated(self):
        metrics = self._analyze_source(
            """\
            def a() -> int:
                return 1

            def b() -> str:
                return "hi"
            """
        )
        self.assertAlmostEqual(metrics.type_annotation_pct, 100.0)

    def test_type_coverage_none_annotated(self):
        metrics = self._analyze_source(
            """\
            def a():
                return 1

            def b():
                return "hi"
            """
        )
        self.assertAlmostEqual(metrics.type_annotation_pct, 0.0)

    def test_type_coverage_half_annotated(self):
        metrics = self._analyze_source(
            """\
            def a() -> int:
                return 1

            def b():
                return "hi"
            """
        )
        self.assertAlmostEqual(metrics.type_annotation_pct, 50.0)

    # ------------------------------------------------------------------
    # Test: Nesting depth calculation
    # ------------------------------------------------------------------
    def test_nesting_depth_flat(self):
        metrics = self._analyze_source(
            """\
            x = 1
            y = 2
            """
        )
        self.assertEqual(metrics.max_nesting, 0)

    def test_nesting_depth_single(self):
        metrics = self._analyze_source(
            """\
            if True:
                x = 1
            """
        )
        self.assertEqual(metrics.max_nesting, 1)

    def test_nesting_depth_deep(self):
        metrics = self._analyze_source(
            """\
            if True:
                for i in range(10):
                    while True:
                        if i:
                            pass
            """
        )
        self.assertEqual(metrics.max_nesting, 4)

    # ------------------------------------------------------------------
    # Test: Import graph collection
    # ------------------------------------------------------------------
    def test_import_graph_from_import(self):
        metrics = self._analyze_source(
            """\
            from os.path import join
            import sys
            import json
            """
        )
        self.assertIn("os.path", metrics.imports)
        self.assertIn("sys", metrics.imports)
        self.assertIn("json", metrics.imports)

    def test_import_graph_in_results(self):
        result = self._run_on_tempdir(
            {
                "app.py": """\
                    import os
                    from pathlib import Path
                """
            }
        )
        self.assertIn("app.py", result["import_graph"])
        self.assertIn("os", result["import_graph"]["app.py"])
        self.assertIn("pathlib", result["import_graph"]["app.py"])

    # ------------------------------------------------------------------
    # Test: Handle syntax errors gracefully
    # ------------------------------------------------------------------
    def test_syntax_error_skipped(self):
        result = self._run_on_tempdir(
            {
                "good.py": "x = 1\n",
                "bad.py": "def foo(\n",  # syntax error
            }
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["files_analyzed"], 1)  # only good.py

    # ------------------------------------------------------------------
    # Test: Elapsed time is present
    # ------------------------------------------------------------------
    def test_elapsed_ms_present(self):
        result = self._run_on_tempdir({"a.py": "x = 1\n"})
        self.assertIn("elapsed_ms", result)
        self.assertIsInstance(result["elapsed_ms"], float)

    # ------------------------------------------------------------------
    # Test: Skips excluded directories
    # ------------------------------------------------------------------
    def test_skips_excluded_dirs(self):
        result = self._run_on_tempdir(
            {
                "src/main.py": "x = 1\n",
                "__pycache__/cached.py": "y = 2\n",
                ".venv/lib/dep.py": "z = 3\n",
            }
        )
        self.assertEqual(result["files_analyzed"], 1)

    # ------------------------------------------------------------------
    # Test: SkillBase error guard
    # ------------------------------------------------------------------
    def test_run_returns_error_on_bad_root(self):
        result = self.skill.run({"project_root": "/nonexistent/path/xyz"})
        # Should either return empty results or an error, not raise
        # The skill's run() wraps _run() in try/except
        self.assertIsInstance(result, dict)
