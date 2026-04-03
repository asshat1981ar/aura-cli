"""Unit tests for core/code_analysis.py."""
from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

import pytest

from core.code_analysis import CodeAnalyzer, DuplicationMetrics, scan_directory


SIMPLE_PY = """\
def hello():
    return "hello"

def world():
    return "world"
"""

DUPLICATE_PY = """\
def foo():
    x = 1
    return x

def bar():
    x = 1
    return x
"""

DUPLICATE_BLOCKS_PY = """\
def process(items):
    for item in items:
        if item > 0:
            print(item)
            print("positive")
    for item in items:
        if item > 0:
            print(item)
            print("positive")
"""


class TestCodeAnalyzerInit:
    def test_initial_state(self):
        analyzer = CodeAnalyzer()
        assert analyzer.seen_snippets == set()


class TestAnalyzeDuplication:
    def _write_temp(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
        f.write(content)
        f.close()
        return f.name

    def test_no_duplication_in_simple_file(self):
        path = self._write_temp(SIMPLE_PY)
        try:
            analyzer = CodeAnalyzer()
            metrics = analyzer.analyze_duplication(path)
            assert isinstance(metrics, DuplicationMetrics)
            assert metrics.duplicated_lines == 0
            assert metrics.total_lines > 0
            assert metrics.duplicate_ratio == 0.0
        finally:
            os.unlink(path)

    def test_duplicate_functions_detected(self):
        path = self._write_temp(DUPLICATE_PY)
        try:
            # Analyze the same file twice so the second pass sees duplicates
            analyzer = CodeAnalyzer()
            analyzer.analyze_duplication(path)  # first pass seeds seen_snippets
            metrics = analyzer.analyze_duplication(path)  # second pass detects dupes
            assert metrics.duplicated_lines > 0
        finally:
            os.unlink(path)

    def test_returns_duplication_metrics_type(self):
        path = self._write_temp(SIMPLE_PY)
        try:
            analyzer = CodeAnalyzer()
            metrics = analyzer.analyze_duplication(path)
            assert hasattr(metrics, "duplicated_lines")
            assert hasattr(metrics, "total_lines")
            assert hasattr(metrics, "duplicate_ratio")
            assert hasattr(metrics, "hotspots")
        finally:
            os.unlink(path)

    def test_duplicate_ratio_between_0_and_1(self):
        path = self._write_temp(SIMPLE_PY)
        try:
            analyzer = CodeAnalyzer()
            metrics = analyzer.analyze_duplication(path)
            assert 0.0 <= metrics.duplicate_ratio <= 1.0
        finally:
            os.unlink(path)

    def test_hotspots_is_dict(self):
        path = self._write_temp(SIMPLE_PY)
        try:
            analyzer = CodeAnalyzer()
            metrics = analyzer.analyze_duplication(path)
            assert isinstance(metrics.hotspots, dict)
        finally:
            os.unlink(path)


class TestScanDirectory:
    def test_scan_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scan_directory(tmpdir)
            assert results == {}

    def test_scan_finds_py_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "module.py"
            py_file.write_text(SIMPLE_PY)
            results = scan_directory(tmpdir)
            assert str(py_file) in results
            assert isinstance(results[str(py_file)], DuplicationMetrics)

    def test_scan_ignores_non_py_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = Path(tmpdir) / "notes.txt"
            txt_file.write_text("not python")
            results = scan_directory(tmpdir)
            assert str(txt_file) not in results

    def test_scan_recurses_into_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "sub"
            subdir.mkdir()
            py_file = subdir / "mod.py"
            py_file.write_text(SIMPLE_PY)
            results = scan_directory(tmpdir)
            assert str(py_file) in results

    def test_scan_handles_syntax_error_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.py"
            bad_file.write_text("def (broken:")
            # Should not raise; just skip the bad file
            results = scan_directory(tmpdir)
            assert str(bad_file) not in results
