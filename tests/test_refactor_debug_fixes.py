"""
Tests for the bugs fixed in the refactor-and-debug pass:

1. core/goal_archive.py  — dead/unreachable duplicate code removed from _load_archive
2. aura_cli/cli_main.py  — _handle_doctor and _handle_clear available in canonical CLI
3. core/orchestrator.py  — correct apply_change_with_explicit_overwrite_policy call
                           signature and exception imports (migrated from HybridClosedLoop)
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
# Bug 2 – aura_cli/cli_main.py: _handle_doctor / _handle_clear available
# ---------------------------------------------------------------------------

class TestCliMainImports(unittest.TestCase):
    """_handle_doctor and _handle_clear must be importable from aura_cli.commands
    and must be available in the aura_cli.cli_main module's namespace."""

    def test_handle_doctor_importable_from_commands(self):
        from aura_cli.commands import _handle_doctor  # noqa: F401

    def test_handle_clear_importable_from_commands(self):
        from aura_cli.commands import _handle_clear  # noqa: F401

    def test_cli_main_imports_handle_doctor(self):
        import aura_cli.cli_main as m
        self.assertTrue(
            hasattr(m, "_handle_doctor"),
            "_handle_doctor must be imported into aura_cli.cli_main",
        )

    def test_cli_main_imports_handle_clear(self):
        import aura_cli.cli_main as m
        self.assertTrue(
            hasattr(m, "_handle_clear"),
            "_handle_clear must be imported into aura_cli.cli_main",
        )


# ---------------------------------------------------------------------------
# Bug 3 & 4 – orchestrator.py: OldCodeNotFoundError/FileToolsError imports
#             and correct apply_change_with_explicit_overwrite_policy call signature
#             (migrated from HybridClosedLoop to LoopOrchestrator)
# ---------------------------------------------------------------------------

class TestOrchestratorImports(unittest.TestCase):
    """OldCodeNotFoundError and MismatchOverwriteBlockedError must be importable from
    core.orchestrator (they are used there and must not cause a NameError)."""

    def test_exception_classes_importable(self):
        from core.file_tools import OldCodeNotFoundError, FileToolsError  # noqa: F401

    def test_orchestrator_can_be_imported_without_name_error(self):
        # Verify that the exception classes used in except-clauses of
        # _apply_change_set are the same objects exported by file_tools,
        # i.e. they were imported correctly and won't cause NameError at runtime.
        from core import orchestrator
        from core.file_tools import OldCodeNotFoundError, MismatchOverwriteBlockedError
        self.assertIs(orchestrator.OldCodeNotFoundError, OldCodeNotFoundError)
        self.assertIs(orchestrator.MismatchOverwriteBlockedError, MismatchOverwriteBlockedError)


class TestLoopOrchestratorApplyChangeSet(unittest.TestCase):
    """LoopOrchestrator._apply_change_set must call
    apply_change_with_explicit_overwrite_policy with the correct
    argument order: (project_root, file_path, old_code, new_code, overwrite_file)."""

    def _make_orchestrator(self, project_root):
        from core.orchestrator import LoopOrchestrator
        return LoopOrchestrator(agents={}, project_root=project_root)

    def test_apply_change_calls_safe_apply_with_correct_args(self):
        project_root = Path(tempfile.mkdtemp())
        file_path = "subdir/test_file.py"
        old_code = "old content"
        new_code = "new content"

        # Create the file so _snapshot_file_state has something to read
        target = project_root / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(old_code)

        captured = {}

        def fake_safe_apply(root, fp, oc, nc, overwrite_file=False):
            captured["root"] = root
            captured["file_path"] = fp
            captured["old_code"] = oc
            captured["new_code"] = nc
            captured["overwrite_file"] = overwrite_file

        orchestrator = self._make_orchestrator(project_root)
        change_set = {
            "changes": [
                {"file_path": file_path, "old_code": old_code, "new_code": new_code}
            ]
        }
        with patch("core.orchestrator.apply_change_with_explicit_overwrite_policy",
                   side_effect=fake_safe_apply):
            result = orchestrator._apply_change_set(change_set, dry_run=False)

        self.assertIn(file_path, result["applied"], "Expected successful apply")
        self.assertEqual(captured["root"], project_root, "project_root must be passed as first arg")
        self.assertEqual(captured["file_path"], file_path, "file_path must be passed as second arg")
        self.assertEqual(captured["old_code"], old_code)
        self.assertEqual(captured["new_code"], new_code)
        self.assertFalse(captured["overwrite_file"])

    def test_apply_change_handles_old_code_not_found(self):
        from core.file_tools import OldCodeNotFoundError
        project_root = Path(tempfile.mkdtemp())
        orchestrator = self._make_orchestrator(project_root)
        change_set = {
            "changes": [
                {"file_path": "file.py", "old_code": "missing", "new_code": "new"}
            ]
        }

        with patch("core.orchestrator.apply_change_with_explicit_overwrite_policy",
                   side_effect=OldCodeNotFoundError("not found")):
            result = orchestrator._apply_change_set(change_set, dry_run=False)

        self.assertIn("file.py", [f["file"] for f in result["failed"]],
                      "OldCodeNotFoundError should result in a failed entry")

    def test_apply_change_handles_mismatch_overwrite_blocked(self):
        from core.file_tools import MismatchOverwriteBlockedError
        project_root = Path(tempfile.mkdtemp())
        orchestrator = self._make_orchestrator(project_root)
        change_set = {
            "changes": [
                {"file_path": "file.py", "old_code": "missing", "new_code": "new"}
            ]
        }

        with patch("core.orchestrator.apply_change_with_explicit_overwrite_policy",
                   side_effect=MismatchOverwriteBlockedError("blocked")):
            result = orchestrator._apply_change_set(change_set, dry_run=False)

        failed_files = [f["file"] for f in result["failed"]]
        self.assertIn("file.py", failed_files,
                      "MismatchOverwriteBlockedError should result in a failed entry")

    def test_apply_change_handles_file_tools_error(self):
        from core.file_tools import FileToolsError
        project_root = Path(tempfile.mkdtemp())
        orchestrator = self._make_orchestrator(project_root)
        change_set = {
            "changes": [
                {"file_path": "file.py", "old_code": "x", "new_code": "y"}
            ]
        }

        with patch("core.orchestrator.apply_change_with_explicit_overwrite_policy",
                   side_effect=FileToolsError("fs error")):
            result = orchestrator._apply_change_set(change_set, dry_run=False)

        self.assertIn("file.py", [f["file"] for f in result["failed"]],
                      "FileToolsError should result in a failed entry")


if __name__ == "__main__":
    unittest.main()
