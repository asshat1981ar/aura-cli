"""Integration test: atomic rollback on multi-file change-sets.

Verifies that :class:`~core.file_tools.AtomicChangeSet` (exposed via
``apply_atomic``) restores ALL previously written files when an error is
injected on the second write, leaving the working tree identical to its
pre-change state.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from core.file_tools import AtomicChangeSet, apply_atomic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_py_files(tmp_path: Path) -> list[Path]:
    """Create three simple Python files and return their paths."""
    files = [
        tmp_path / "alpha.py",
        tmp_path / "beta.py",
        tmp_path / "gamma.py",
    ]
    for i, fp in enumerate(files):
        fp.write_text(f"# original content {i}\n", encoding="utf-8")
    return files


def _read_contents(files: list[Path]) -> dict[str, str]:
    """Snapshot current content of *files* keyed by filename stem."""
    return {fp.name: fp.read_text(encoding="utf-8") for fp in files}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAtomicRollback:
    """AtomicChangeSet must restore ALL files if any single write fails."""

    def test_all_files_restored_when_second_write_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject an OSError on the second overwrite; verify full rollback."""
        files = _write_py_files(tmp_path)
        original_contents = _read_contents(files)

        # Build a 3-change set: overwrite every file with new content.
        changes = [
            {
                "file_path": str(fp.relative_to(tmp_path)),
                "old_code": "",
                "new_code": f"# modified content {i}\n",
                "overwrite_file": True,
            }
            for i, fp in enumerate(files)
        ]

        # Track how many times the real `replace_code` write path is entered.
        call_count = 0
        original_replace_code = None

        import core.file_tools as _ft

        original_replace_code = _ft.replace_code

        def _failing_replace_code(file_path, old_code, new_code, *, dry_run=False, overwrite_file=False, project_root=None):  # noqa: ANN001
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("Simulated write failure on file 2")
            return original_replace_code(
                file_path,
                old_code,
                new_code,
                dry_run=dry_run,
                overwrite_file=overwrite_file,
                project_root=project_root,
            )

        monkeypatch.setattr(_ft, "replace_code", _failing_replace_code)

        # The AtomicChangeSet must raise (re-raising the injected error).
        with pytest.raises(OSError, match="Simulated write failure"):
            AtomicChangeSet(changes, project_root=tmp_path).apply()

        # ── Verify rollback ──────────────────────────────────────────────────
        restored_contents = _read_contents(files)

        for fname, original in original_contents.items():
            assert restored_contents[fname] == original, f"File '{fname}' was not restored to its original content.\nExpected: {original!r}\nGot:      {restored_contents[fname]!r}"

    def test_no_partial_changes_remain(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """After a failed atomic apply, no modified content must exist on disk."""
        files = _write_py_files(tmp_path)

        changes = [
            {
                "file_path": str(fp.relative_to(tmp_path)),
                "old_code": "",
                "new_code": "# SHOULD_NOT_EXIST\n",
                "overwrite_file": True,
            }
            for fp in files
        ]

        call_count = 0
        import core.file_tools as _ft

        orig = _ft.replace_code

        def _fail_on_second(file_path, old_code, new_code, *, dry_run=False, overwrite_file=False, project_root=None):  # noqa: ANN001
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("Simulated failure")
            return orig(file_path, old_code, new_code, dry_run=dry_run, overwrite_file=overwrite_file, project_root=project_root)

        monkeypatch.setattr(_ft, "replace_code", _fail_on_second)

        with pytest.raises(OSError):
            AtomicChangeSet(changes, project_root=tmp_path).apply()

        for fp in files:
            content = fp.read_text(encoding="utf-8")
            assert "SHOULD_NOT_EXIST" not in content, f"Partial change found in '{fp.name}': {content!r}"

    def test_successful_apply_returns_applied_paths(self, tmp_path: Path) -> None:
        """Sanity check: a clean apply returns all file paths and writes new content."""
        files = _write_py_files(tmp_path)

        changes = [
            {
                "file_path": str(fp.relative_to(tmp_path)),
                "old_code": "",
                "new_code": f"# updated {i}\n",
                "overwrite_file": True,
            }
            for i, fp in enumerate(files)
        ]

        applied = AtomicChangeSet(changes, project_root=tmp_path).apply()

        assert len(applied) == len(files), f"Expected {len(files)} applied paths, got {len(applied)}"

        for i, fp in enumerate(files):
            assert fp.read_text(encoding="utf-8") == f"# updated {i}\n"

    def test_apply_atomic_helper_rolls_back_on_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``apply_atomic`` convenience wrapper also triggers rollback on error."""
        files = _write_py_files(tmp_path)
        original_contents = _read_contents(files)

        changes = [
            {
                "file_path": str(fp.relative_to(tmp_path)),
                "old_code": "",
                "new_code": "# overwritten\n",
                "overwrite_file": True,
            }
            for fp in files
        ]

        call_count = 0
        import core.file_tools as _ft

        orig = _ft.replace_code

        def _fail_third(file_path, old_code, new_code, *, dry_run=False, overwrite_file=False, project_root=None):  # noqa: ANN001
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise OSError("Simulated failure on file 3")
            return orig(file_path, old_code, new_code, dry_run=dry_run, overwrite_file=overwrite_file, project_root=project_root)

        monkeypatch.setattr(_ft, "replace_code", _fail_third)

        with pytest.raises(OSError, match="Simulated failure on file 3"):
            apply_atomic(changes, project_root=tmp_path)

        for fname, original in original_contents.items():
            restored = (tmp_path / fname).read_text(encoding="utf-8")
            assert restored == original, f"File '{fname}' not restored via apply_atomic"
