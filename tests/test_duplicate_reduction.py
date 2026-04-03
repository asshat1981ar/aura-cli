"""Unit tests for core/duplicate_reduction.py — DuplicateCodeReducer."""
import os
import tempfile
from collections import defaultdict

from core.duplicate_reduction import DuplicateCodeReducer


class TestDuplicateCodeReducerInit:
    def test_init_stores_codebase_path(self):
        reducer = DuplicateCodeReducer("/some/path")
        assert reducer.codebase_path == "/some/path"

    def test_init_empty_duplicate_blocks(self):
        reducer = DuplicateCodeReducer(".")
        assert isinstance(reducer.duplicate_blocks, defaultdict)
        assert len(reducer.duplicate_blocks) == 0

    def test_init_empty_visited_nodes(self):
        reducer = DuplicateCodeReducer(".")
        assert isinstance(reducer.visited_nodes, set)
        assert len(reducer.visited_nodes) == 0


class TestAnalyzeDuplicates:
    def test_analyze_empty_directory_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reducer = DuplicateCodeReducer(tmpdir)
            result = reducer.analyze_duplicates()
            assert isinstance(result, dict)

    def test_analyze_directory_with_python_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "sample.py")
            with open(py_file, "w") as f:
                f.write("def hello():\n    return 'hello'\n")
            reducer = DuplicateCodeReducer(tmpdir)
            result = reducer.analyze_duplicates()
            assert isinstance(result, dict)

    def test_analyze_ignores_non_python_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = os.path.join(tmpdir, "notes.txt")
            with open(txt_file, "w") as f:
                f.write("not python\n")
            reducer = DuplicateCodeReducer(tmpdir)
            result = reducer.analyze_duplicates()
            assert result == {}


class TestGenerateRefactorPlan:
    def test_refactor_plan_empty_when_no_duplicates(self):
        reducer = DuplicateCodeReducer(".")
        # No duplicates populated
        plan = reducer.generate_refactor_plan()
        assert plan == {}

    def test_refactor_plan_only_includes_multi_location_blocks(self):
        reducer = DuplicateCodeReducer(".")
        reducer.duplicate_blocks[111] = ["file_a.py:foo"]          # single — not a duplicate
        reducer.duplicate_blocks[222] = ["file_a.py:bar", "file_b.py:bar"]  # actual duplicate
        plan = reducer.generate_refactor_plan()
        assert 111 not in plan
        assert 222 in plan
        assert len(plan[222]) == 2

    def test_refactor_plan_returns_dict(self):
        reducer = DuplicateCodeReducer(".")
        assert isinstance(reducer.generate_refactor_plan(), dict)
