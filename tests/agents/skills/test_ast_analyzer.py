"""Unit tests for AST Analyzer Skill."""

from __future__ import annotations

import ast
import tempfile
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock

from agents.skills.ast_analyzer import ASTAnalyzerSkill, ASTMetrics


class TestASTAnalyzerSkill(TestCase):
    """Test cases for ASTAnalyzerSkill."""

    def setUp(self):
        """Set up test fixtures."""
        self.skill = ASTAnalyzerSkill()

    # =========================================================================
    # Initialization Tests
    # =========================================================================
    def test_skill_initialization(self):
        """Test that skill initializes correctly."""
        self.assertEqual(self.skill.name, "ast_analyzer")
        self.assertIsNone(self.skill.brain)
        self.assertIsNone(self.skill.model)

    def test_skill_run_with_exception_handling(self):
        """Test that run() catches exceptions and returns error dict."""
        with patch.object(self.skill, "_run", side_effect=Exception("Test error")):
            result = self.skill.run({"project_root": "."})
            self.assertIn("error", result)
            self.assertEqual(result["skill"], "ast_analyzer")

    # =========================================================================
    # Basic Analysis Tests
    # =========================================================================
    def test_analyze_simple_function(self):
        """Test analyzing a simple function."""
        code = "def hello(): pass"
        tree = ast.parse(code)
        metrics = self.skill._analyze_file(tree, "test.py")

        self.assertEqual(metrics.file_path, "test.py")
        self.assertEqual(metrics.functions, 1)
        self.assertEqual(metrics.classes, 0)

    def test_analyze_class_with_methods(self):
        """Test analyzing a class with methods."""
        code = """
class MyClass:
    def method1(self):
        pass
    def method2(self):
        pass
"""
        tree = ast.parse(code)
        metrics = self.skill._analyze_file(tree, "test.py")

        self.assertEqual(metrics.functions, 2)
        self.assertEqual(metrics.classes, 1)

    def test_analyze_empty_file(self):
        """Test analyzing an empty file."""
        code = ""
        tree = ast.parse(code)
        metrics = self.skill._analyze_file(tree, "test.py")

        self.assertEqual(metrics.functions, 0)
        self.assertEqual(metrics.classes, 0)
        self.assertEqual(metrics.imports, [])

    # =========================================================================
    # Import Collection Tests
    # =========================================================================
    def test_collect_regular_imports(self):
        """Test collecting regular import statements."""
        code = "import os\nimport sys\nfrom collections import defaultdict"
        tree = ast.parse(code)
        metrics = self.skill._analyze_file(tree, "test.py")

        self.assertIn("os", metrics.imports)
        self.assertIn("sys", metrics.imports)
        self.assertIn("collections", metrics.imports)

    def test_collect_import_from(self):
        """Test collecting from X import Y statements."""
        code = "from pathlib import Path\nfrom typing import Dict, List"
        tree = ast.parse(code)
        metrics = self.skill._analyze_file(tree, "test.py")

        self.assertIn("pathlib", metrics.imports)
        self.assertIn("typing", metrics.imports)

    # =========================================================================
    # Code Smell Detection Tests
    # =========================================================================
    def test_detect_too_many_args(self):
        """Test detection of function with too many arguments."""
        code = "def func(a, b, c, d, e, f, g): pass"
        tree = ast.parse(code)
        metrics = ASTMetrics(file_path="test.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.skill._check_function_smells(node, "test.py", metrics)

        self.assertEqual(len(metrics.smells), 1)
        self.assertEqual(metrics.smells[0]["type"], "too_many_args")
        self.assertEqual(metrics.smells[0]["count"], 7)

    def test_detect_long_function(self):
        """Test detection of long function."""
        code = "def long_func():\n" + "\n".join([f"    x = {i}" for i in range(60)])
        tree = ast.parse(code)
        metrics = ASTMetrics(file_path="test.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.skill._check_function_smells(node, "test.py", metrics)

        self.assertEqual(len(metrics.smells), 1)
        self.assertEqual(metrics.smells[0]["type"], "long_function")

    def test_detect_god_class(self):
        """Test detection of god class (too many methods)."""
        methods = "\n".join([f"    def method{i}(self): pass" for i in range(25)])
        code = f"class GodClass:\n{methods}"
        tree = ast.parse(code)
        metrics = ASTMetrics(file_path="test.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.skill._check_class_smells(node, "test.py", metrics)

        self.assertEqual(len(metrics.smells), 1)
        self.assertEqual(metrics.smells[0]["type"], "god_class")
        self.assertEqual(metrics.smells[0]["severity"], "high")

    # =========================================================================
    # Dead Code Detection Tests
    # =========================================================================
    def test_detect_dead_code_after_return(self):
        """Test detection of unreachable code after return."""
        code = """def func():
    return 1
    x = 2  # unreachable
"""
        tree = ast.parse(code)
        metrics = ASTMetrics(file_path="test.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.skill._check_function_smells(node, "test.py", metrics)

        self.assertEqual(len(metrics.dead_code), 1)
        self.assertEqual(metrics.dead_code[0]["type"], "unreachable_after_return")

    def test_detect_dead_code_after_raise(self):
        """Test detection of unreachable code after raise."""
        code = """def func():
    raise ValueError()
    x = 2  # unreachable
"""
        tree = ast.parse(code)
        metrics = ASTMetrics(file_path="test.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.skill._check_function_smells(node, "test.py", metrics)

        self.assertEqual(len(metrics.dead_code), 1)

    # =========================================================================
    # Nesting Depth Tests
    # =========================================================================
    def test_max_nesting_depth(self):
        """Test calculation of maximum nesting depth."""
        code = """
def func():
    if True:
        for i in range(10):
            if True:
                while True:
                    pass
"""
        tree = ast.parse(code)
        depth = self.skill._max_nesting_depth(tree)
        self.assertGreaterEqual(depth, 4)

    def test_nesting_depth_no_nesting(self):
        """Test nesting depth for flat code."""
        code = "def func(): pass"
        tree = ast.parse(code)
        depth = self.skill._max_nesting_depth(tree)
        self.assertEqual(depth, 0)

    # =========================================================================
    # Type Coverage Tests
    # =========================================================================
    def test_type_coverage_fully_annotated(self):
        """Test type coverage calculation for fully annotated functions."""
        code = """
def func1() -> int:
    return 1
    
def func2() -> str:
    return "hello"
"""
        tree = ast.parse(code)
        coverage = self.skill._type_coverage(tree)
        self.assertEqual(coverage, 100.0)

    def test_type_coverage_no_annotations(self):
        """Test type coverage calculation for unannotated functions."""
        code = """
def func1():
    return 1
    
def func2():
    return "hello"
"""
        tree = ast.parse(code)
        coverage = self.skill._type_coverage(tree)
        self.assertEqual(coverage, 0.0)

    def test_type_coverage_partial(self):
        """Test type coverage calculation for partially annotated functions."""
        code = """
def func1() -> int:
    return 1
    
def func2():
    return "hello"
"""
        tree = ast.parse(code)
        coverage = self.skill._type_coverage(tree)
        self.assertEqual(coverage, 50.0)

    def test_type_coverage_empty(self):
        """Test type coverage for file with no functions."""
        code = "x = 1"
        tree = ast.parse(code)
        coverage = self.skill._type_coverage(tree)
        self.assertEqual(coverage, 0.0)

    # =========================================================================
    # Full Run Tests
    # =========================================================================
    def test_run_with_project_root(self):
        """Test full run with project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test_module.py"
            test_file.write_text("def hello(): pass")

            result = self.skill._run({"project_root": tmpdir})

            self.assertIn("files_analyzed", result)
            self.assertIn("total_functions", result)
            self.assertIn("smells", result)
            self.assertIn("elapsed_ms", result)
            self.assertGreaterEqual(result["files_analyzed"], 1)

    def test_run_skips_venv_and_cache(self):
        """Test that run skips venv and cache directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create normal file
            normal_file = Path(tmpdir) / "normal.py"
            normal_file.write_text("def hello(): pass")

            # Create file in __pycache__
            pycache = Path(tmpdir) / "__pycache__"
            pycache.mkdir()
            (pycache / "cached.py").write_text("def cached(): pass")

            # Create file in .venv
            venv = Path(tmpdir) / ".venv"
            venv.mkdir()
            (venv / "venv_file.py").write_text("def venv_func(): pass")

            result = self.skill._run({"project_root": tmpdir})

            # Should only analyze normal.py
            self.assertEqual(result["files_analyzed"], 1)

    def test_run_handles_syntax_error(self):
        """Test that run handles syntax errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bad_syntax.py"
            test_file.write_text("def hello(\n")  # Syntax error

            result = self.skill._run({"project_root": tmpdir})

            # Should not crash, should report 0 files analyzed
            self.assertIn("files_analyzed", result)
            self.assertEqual(result["files_analyzed"], 0)

    def test_run_limits_output_size(self):
        """Test that run limits output sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with many functions that would generate many smells
            code = "\n".join([f"def func_{i}(a, b, c, d, e, f, g): pass" for i in range(30)])
            test_file = Path(tmpdir) / "many_smells.py"
            test_file.write_text(code)

            result = self.skill._run({"project_root": tmpdir})

            # Smells should be limited to 20
            self.assertLessEqual(len(result["smells"]), 20)
            # Hotspots should be limited to 10
            self.assertLessEqual(len(result["complexity_hotspots"]), 10)

    def test_run_with_complexity_hotspots(self):
        """Test detection of complexity hotspots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create function with deep nesting
            code = """
def complex_func():
    if True:
        for i in range(10):
            if True:
                if True:
                    if True:
                        pass
"""
            test_file = Path(tmpdir) / "complex.py"
            test_file.write_text(code)

            result = self.skill._run({"project_root": tmpdir})

            self.assertGreaterEqual(len(result["complexity_hotspots"]), 1)

    def test_run_default_project_root(self):
        """Test run with default project root."""
        result = self.skill._run({})

        self.assertIn("files_analyzed", result)
        self.assertIn("smells", result)


class TestASTMetrics(TestCase):
    """Test cases for ASTMetrics dataclass."""

    def test_default_values(self):
        """Test default values of ASTMetrics."""
        metrics = ASTMetrics()
        self.assertEqual(metrics.file_path, "")
        self.assertEqual(metrics.functions, 0)
        self.assertEqual(metrics.classes, 0)
        self.assertEqual(metrics.max_nesting, 0)
        self.assertEqual(metrics.avg_complexity, 0.0)
        self.assertEqual(metrics.type_annotation_pct, 0.0)
        self.assertEqual(metrics.smells, [])
        self.assertEqual(metrics.imports, [])
        self.assertEqual(metrics.dead_code, [])

    def test_custom_values(self):
        """Test setting custom values."""
        metrics = ASTMetrics(file_path="test.py", functions=5, classes=2, max_nesting=3, avg_complexity=5.5, type_annotation_pct=80.0, smells=[{"type": "test"}], imports=["os"], dead_code=[{"line": 10}])
        self.assertEqual(metrics.file_path, "test.py")
        self.assertEqual(metrics.functions, 5)
        self.assertEqual(metrics.smells, [{"type": "test"}])
