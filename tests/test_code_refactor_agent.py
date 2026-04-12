"""Unit tests for agents/code_refactor_agent.py."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path

from agents.code_refactor_agent import DuplicateCodeReducer


def _write_py(directory: str, filename: str, source: str) -> str:
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        f.write(source)
    return path


class TestDuplicateCodeReducerInit(unittest.TestCase):
    def test_default_base_path(self):
        reducer = DuplicateCodeReducer()
        self.assertEqual(reducer.base_path, ".")

    def test_custom_base_path(self):
        reducer = DuplicateCodeReducer("/tmp/project")
        self.assertEqual(reducer.base_path, "/tmp/project")

    def test_initial_state(self):
        reducer = DuplicateCodeReducer()
        self.assertEqual(reducer.duplicate_patterns, [])
        self.assertEqual(reducer.refactored_files, set())


class TestAnalyzeCodebase(unittest.TestCase):
    def test_returns_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "a.py", "x = 1\n")
            reducer = DuplicateCodeReducer(tmpdir)
            result = reducer.analyze_codebase()
        self.assertIsInstance(result, list)

    def test_duplicate_patterns_populated(self):
        # Write two structurally identical files
        source = "x = 1\ny = 2\nz = x + y\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "a.py", source)
            _write_py(tmpdir, "b.py", source)
            reducer = DuplicateCodeReducer(tmpdir)
            result = reducer.analyze_codebase()

        # Both files have same AST structure — expect a duplicate
        self.assertGreater(len(result), 0)
        self.assertEqual(result, reducer.duplicate_patterns)

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reducer = DuplicateCodeReducer(tmpdir)
            result = reducer.analyze_codebase()
        self.assertEqual(result, [])

    def test_duplicate_entry_contains_files_key(self):
        source = "a = 1\nb = 2\nc = a + b\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "x.py", source)
            _write_py(tmpdir, "y.py", source)
            reducer = DuplicateCodeReducer(tmpdir)
            dupes = reducer.analyze_codebase()

        for d in dupes:
            self.assertIn("files", d)
            self.assertIn("type", d)


class TestProposeAbstractions(unittest.TestCase):
    def test_returns_list_for_no_patterns(self):
        reducer = DuplicateCodeReducer()
        result = reducer.propose_abstractions()
        self.assertEqual(result, [])

    def test_returns_one_abstraction_per_pattern(self):
        reducer = DuplicateCodeReducer()
        reducer.duplicate_patterns = [
            {"files": ["core/foo.py", "core/bar.py"], "type": "structural_duplicate"},
        ]
        result = reducer.propose_abstractions()
        self.assertEqual(len(result), 1)

    def test_abstraction_has_required_keys(self):
        reducer = DuplicateCodeReducer()
        reducer.duplicate_patterns = [
            {"files": ["a/alpha.py", "b/beta.py"], "type": "structural_duplicate"},
        ]
        abstraction = reducer.propose_abstractions()[0]
        for key in ("name", "files", "type", "suggested_location"):
            self.assertIn(key, abstraction)

    def test_suggested_location_is_string(self):
        reducer = DuplicateCodeReducer()
        reducer.duplicate_patterns = [
            {"files": ["x.py", "y.py"], "type": "structural_duplicate"},
        ]
        abstraction = reducer.propose_abstractions()[0]
        self.assertIsInstance(abstraction["suggested_location"], str)


class TestRefactorForReuse(unittest.TestCase):
    def test_returns_true_on_success(self):
        reducer = DuplicateCodeReducer()
        result = reducer.refactor_for_reuse({"files": ["a.py", "b.py"]})
        self.assertTrue(result)

    def test_updates_refactored_files(self):
        reducer = DuplicateCodeReducer()
        reducer.refactor_for_reuse({"files": ["core/a.py", "core/b.py"]})
        self.assertIn("core/a.py", reducer.refactored_files)
        self.assertIn("core/b.py", reducer.refactored_files)


class TestValidateChanges(unittest.TestCase):
    def test_no_refactors_returns_success_false(self):
        reducer = DuplicateCodeReducer()
        result = reducer.validate_changes()
        self.assertFalse(result["success"])
        self.assertFalse(result["rollback_required"])

    def test_with_refactored_files_returns_success_true(self):
        reducer = DuplicateCodeReducer()
        reducer.refactored_files = {"a.py", "b.py"}
        result = reducer.validate_changes()
        self.assertTrue(result["success"])
        self.assertIsInstance(result["refactored_files"], list)


if __name__ == "__main__":
    unittest.main()
