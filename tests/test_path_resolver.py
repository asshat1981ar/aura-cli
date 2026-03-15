import json
import tempfile
import unittest
from pathlib import Path

from core.path_resolver import (
    allow_new_test_file_target,
    find_candidate_existing_files,
    _candidate_files_from_symbol_index_cache
)


class TestPathResolver(unittest.TestCase):
    def test_candidate_existing_files_prefers_exact_basename_match(self):
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            (project_root / "run_aura.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (project_root / "tests").mkdir(parents=True, exist_ok=True)
            (project_root / "tests" / "test_run_aura_wrapper.py").write_text("# test\n", encoding="utf-8")

            candidates = find_candidate_existing_files(
                project_root,
                "scripts/run_aura.sh",
                "Update run_aura.sh wrapper help",
                limit=4,
            )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0], "run_aura.sh")

    def test_candidate_existing_files_prefers_exact_python_basename_match(self):
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            (project_root / "agents" / "skills").mkdir(parents=True, exist_ok=True)
            (project_root / "agents" / "skills" / "structural_analyzer.py").write_text(
                "# analyzer\n",
                encoding="utf-8",
            )
            (project_root / "agents" / "skills" / "dependency_analyzer.py").write_text(
                "# dependency\n",
                encoding="utf-8",
            )

            candidates = find_candidate_existing_files(
                project_root,
                "analysis/structural_analyzer.py",
                "Run architecture validation after refactor",
                limit=4,
            )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0], "agents/skills/structural_analyzer.py")

    def test_allow_new_test_file_target_for_regression_goal(self):
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            (project_root / "tests").mkdir(parents=True, exist_ok=True)

            result = allow_new_test_file_target(
                project_root,
                "tests/test_cli_loop.py",
                "Add interactive CLI loop regression tests",
                "",
                False,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_cli_loop.py")

    def test_symbol_index_cache_candidates_filter_invalid_and_stale_paths(self):
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            (project_root / "memory").mkdir(parents=True, exist_ok=True)
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            (project_root / "core" / "good_target.py").write_text("# ok\n", encoding="utf-8")

            payload = {
                "name_index": {
                    "target": [
                        {"file": "core/good_target.py", "line": 1, "type": "function"},
                        {"file": "../outside.py", "line": 1, "type": "function"},
                        {"file": "core/missing_target.py", "line": 1, "type": "function"},
                    ]
                }
            }
            (project_root / "memory" / "symbol_index.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            result = _candidate_files_from_symbol_index_cache(project_root, ["target"], limit=6)

        self.assertEqual(result, ["core/good_target.py"])

if __name__ == "__main__":
    unittest.main()
