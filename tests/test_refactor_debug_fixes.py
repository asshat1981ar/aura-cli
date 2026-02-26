"""
Tests for the bugs fixed in the refactor-and-debug pass:

1. core/goal_archive.py  — dead/unreachable duplicate code removed from _load_archive
2. cli/cli_main.py       — missing imports for _handle_doctor and _handle_clear
3. core/hybrid_loop.py   — incorrect _safe_apply_change call and missing exception imports
"""

import sys
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Bug 1 – GoalArchive._load_archive: dead code removed
# ---------------------------------------------------------------------------

class TestGoalArchiveLoadArchive(unittest.TestCase):
    """Verify _load_archive works correctly after dead-code removal."""

    def _make_archive(self, archive_path):
        from core.goal_archive import GoalArchive
        return GoalArchive(archive_path=archive_path)

    def test_load_returns_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            archive = self._make_archive(str(Path(d) / "goal_archive.json"))
            self.assertEqual(archive.completed, [])

    def test_load_returns_data_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "goal_archive.json"
            p.write_text(json.dumps([["goal1", 9.5], ["goal2", 7.0]]))
            archive = self._make_archive(str(p))
            self.assertEqual(len(archive.completed), 2)

    def test_load_returns_empty_on_corrupted_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "goal_archive.json"
            p.write_text("not valid json{{")
            archive = self._make_archive(str(p))
            self.assertEqual(archive.completed, [])

    def test_record_and_reload_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path_str = str(Path(d) / "goal_archive.json")
            from core.goal_archive import GoalArchive
            a = GoalArchive(archive_path=path_str)
            a.record("my goal", 8.5)
            # Reload from disk — verifies persistence works
            b = GoalArchive(archive_path=path_str)
            self.assertEqual(len(b.completed), 1)
            self.assertEqual(b.completed[0][0], "my goal")
            self.assertAlmostEqual(b.completed[0][1], 8.5)


# ---------------------------------------------------------------------------
# Bug 2 – cli/cli_main.py: missing imports for _handle_doctor / _handle_clear
# ---------------------------------------------------------------------------

class TestCliMainImports(unittest.TestCase):
    """_handle_doctor and _handle_clear must be importable from cli.commands
    and must be available in the cli.cli_main module's namespace."""

    def test_handle_doctor_importable_from_commands(self):
        from cli.commands import _handle_doctor  # noqa: F401

    def test_handle_clear_importable_from_commands(self):
        from cli.commands import _handle_clear  # noqa: F401

    def test_cli_main_imports_handle_doctor(self):
        import cli.cli_main as m
        self.assertTrue(
            hasattr(m, "_handle_doctor"),
            "_handle_doctor must be imported into cli.cli_main",
        )

    def test_cli_main_imports_handle_clear(self):
        import cli.cli_main as m
        self.assertTrue(
            hasattr(m, "_handle_clear"),
            "_handle_clear must be imported into cli.cli_main",
        )


# ---------------------------------------------------------------------------
# Bug 3 & 4 – hybrid_loop.py: OldCodeNotFoundError/FileToolsError imports
#             and correct _safe_apply_change call signature
# ---------------------------------------------------------------------------

class TestHybridLoopImports(unittest.TestCase):
    """OldCodeNotFoundError and FileToolsError must be importable from
    core.hybrid_loop (they are used there and must not cause a NameError)."""

    def test_exception_classes_importable(self):
        from core.file_tools import OldCodeNotFoundError, FileToolsError  # noqa: F401

    def test_hybrid_loop_can_be_imported_without_name_error(self):
        # Verify that the exception classes used in except-clauses of
        # _apply_change_with_debug are the same objects exported by file_tools,
        # i.e. they were imported correctly and won't cause NameError at runtime.
        from core import hybrid_loop
        from core.file_tools import OldCodeNotFoundError, FileToolsError
        self.assertIs(hybrid_loop.OldCodeNotFoundError, OldCodeNotFoundError)
        self.assertIs(hybrid_loop.FileToolsError, FileToolsError)


class TestHybridLoopApplyChangeCallSignature(unittest.TestCase):
    """_apply_change_with_debug must call _safe_apply_change with the correct
    argument order: (project_root, file_path, old_code, new_code, overwrite_file)."""

    def _make_loop(self):
        brain = MagicMock()
        model = MagicMock()
        git = MagicMock()
        from core.hybrid_loop import HybridClosedLoop
        return HybridClosedLoop(model, brain, git)

    def test_apply_change_calls_safe_apply_with_correct_args(self):
        loop = self._make_loop()
        project_root = Path(tempfile.mkdtemp())
        file_path = "subdir/test_file.py"
        old_code = "old content"
        new_code = "new content"

        # Create the file so _safe_apply_change has something to work with
        target = project_root / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(old_code)

        captured = {}

        def fake_safe_apply(root, fp, oc, nc, owf=False):
            captured["root"] = root
            captured["file_path"] = fp
            captured["old_code"] = oc
            captured["new_code"] = nc
            captured["overwrite_file"] = owf

        with patch("core.hybrid_loop._safe_apply_change", side_effect=fake_safe_apply):
            result = loop._apply_change_with_debug(
                project_root=project_root,
                sanitized_file_path=file_path,
                old_code=old_code,
                new_code=new_code,
                overwrite_file=False,
                current_goal="test goal",
                change_idx=0,
                result_json={},
                change={},
            )

        self.assertTrue(result, "Expected successful apply")
        self.assertEqual(captured["root"], project_root, "project_root must be passed as first arg")
        self.assertEqual(captured["file_path"], file_path, "file_path must be passed as second arg")
        self.assertEqual(captured["old_code"], old_code)
        self.assertEqual(captured["new_code"], new_code)
        self.assertFalse(captured["overwrite_file"])

    def test_apply_change_handles_old_code_not_found(self):
        from core.file_tools import OldCodeNotFoundError
        brain = MagicMock()
        model = MagicMock()
        model.respond.return_value = '{"summary":"err","diagnosis":"d","fix_strategy":"f","severity":"HIGH"}'
        git = MagicMock()
        from core.hybrid_loop import HybridClosedLoop
        loop = HybridClosedLoop(model, brain, git)
        project_root = Path(tempfile.mkdtemp())

        with patch("core.hybrid_loop._safe_apply_change", side_effect=OldCodeNotFoundError("not found")):
            result = loop._apply_change_with_debug(
                project_root=project_root,
                sanitized_file_path="file.py",
                old_code="missing",
                new_code="new",
                overwrite_file=False,
                current_goal="test goal",
                change_idx=0,
                result_json={},
                change={},
            )
        self.assertFalse(result, "OldCodeNotFoundError should return False")

    def test_apply_change_handles_file_tools_error(self):
        from core.file_tools import FileToolsError
        brain = MagicMock()
        model = MagicMock()
        model.respond.return_value = '{"summary":"err","diagnosis":"d","fix_strategy":"f","severity":"HIGH"}'
        git = MagicMock()
        from core.hybrid_loop import HybridClosedLoop
        loop = HybridClosedLoop(model, brain, git)
        project_root = Path(tempfile.mkdtemp())

        with patch("core.hybrid_loop._safe_apply_change", side_effect=FileToolsError("fs error")):
            result = loop._apply_change_with_debug(
                project_root=project_root,
                sanitized_file_path="file.py",
                old_code="x",
                new_code="y",
                overwrite_file=False,
                current_goal="test goal",
                change_idx=0,
                result_json={},
                change={},
            )
        self.assertFalse(result, "FileToolsError should return False")


if __name__ == "__main__":
    unittest.main()
