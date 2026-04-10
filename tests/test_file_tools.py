import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch  # Import patch

# Ensure the project root is on the path for imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.file_tools import (
    FileToolsError,
    MismatchOverwriteBlockedError,
    OldCodeNotFoundError,
    _safe_apply_change,
    apply_change_with_explicit_overwrite_policy,
    allow_mismatch_overwrite_for_change,
    replace_code,
)


class TestFileTools(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file_path = self.test_dir / "test_file.txt"
        self.initial_content = """Line 1
Line 2 - original
Line 3"""
        self.test_file_path.write_text(self.initial_content)

    def tearDown(self):
        # Clean up the temporary directory and its contents
        for item in self.test_dir.iterdir():
            item.unlink()
        self.test_dir.rmdir()

    def test_normal_replacement(self):
        old_code = "Line 2 - original"
        new_code = "Line 2 - replaced"
        replace_code(str(self.test_file_path), old_code, new_code)
        self.assertEqual(
            self.test_file_path.read_text(),
            """Line 1
Line 2 - replaced
Line 3""",
        )

    def test_dry_run_replacement(self):
        old_code = "Line 2 - original"
        new_code = "Line 2 - replaced"
        # Capture stdout to check dry-run output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        replace_code(str(self.test_file_path), old_code, new_code, dry_run=True)

        sys.stdout = sys.__stdout__  # Restore stdout
        output = captured_output.getvalue()

        self.assertIn("DRY RUN: Changes for", output)
        self.assertIn("--- OLD CODE ---\nLine 2 - original", output)
        self.assertIn("--- NEW CODE ---\nLine 2 - replaced", output)
        self.assertEqual(self.test_file_path.read_text(), self.initial_content)  # File should not change

    def test_old_code_not_found_raises_exception(self):
        old_code = "Non-existent line"
        new_code = "Some new code"
        with self.assertRaisesRegex(OldCodeNotFoundError, f"'{old_code}' not found in '{self.test_file_path}'."):
            replace_code(str(self.test_file_path), old_code, new_code)
        self.assertEqual(self.test_file_path.read_text(), self.initial_content)  # File should remain unchanged

    def test_overwrite_file_flag(self):
        new_content = "Completely new file content."
        replace_code(str(self.test_file_path), old_code="", new_code=new_content, overwrite_file=True)
        self.assertEqual(self.test_file_path.read_text(), new_content)

    def test_overwrite_file_flag_with_non_empty_old_code_raises_error(self):
        with self.assertRaisesRegex(ValueError, "When overwrite_file is True, old_code must be an empty string."):
            replace_code(str(self.test_file_path), old_code="something", new_code="new", overwrite_file=True)
        self.assertEqual(self.test_file_path.read_text(), self.initial_content)  # File should remain unchanged

    def test_file_not_found_raises_exception(self):
        non_existent_file = self.test_dir / "non_existent.txt"
        with self.assertRaisesRegex(FileToolsError, "File not found"):
            replace_code(str(non_existent_file), "old", "new")

    def test_atomic_write_integrity_on_success(self):
        old_code = "Line 2 - original"
        new_code = "Line 2 - replaced atomically"
        original_inode = os.stat(self.test_file_path).st_ino

        replace_code(str(self.test_file_path), old_code, new_code)

        self.assertEqual(
            self.test_file_path.read_text(),
            """Line 1
Line 2 - replaced atomically
Line 3""",
        )
        # Inode might change on os.replace, but the content should be correct.
        # This primarily tests successful atomic write, not specifically inode preservation.

    def test_atomic_write_failure_preserves_original(self):
        old_code = "Line 2 - original"
        new_code = "Line 2 - replaced, but failed atomically"

        # Ensure the file exists with initial content
        self.assertEqual(self.test_file_path.read_text(), self.initial_content)

        # Mock os.replace to raise an exception, simulating atomic write failure
        with patch("os.replace") as mock_os_replace:
            mock_os_replace.side_effect = OSError("Simulated atomic write failure")

            # Call replace_code; it should catch the OSError and re-raise as FileToolsError or similar
            with self.assertRaises(FileToolsError):  # replace_code wraps exceptions in FileToolsError
                replace_code(str(self.test_file_path), old_code, new_code)

        # Assert that the original file content remains unchanged
        self.assertEqual(self.test_file_path.read_text(), self.initial_content)

        # Verify that the temporary file used for atomic write is cleaned up.
        # This is implicitly handled by tempfile.NamedTemporaryFile's context manager,
        # but we can't directly assert its absence here without exposing internal tempfile logic.
        # The key assertion is the original file's integrity.

    def test_safe_apply_change_blocks_mismatch_overwrite_when_disabled(self):
        with patch("core.file_tools.recover_old_code_from_git", return_value=None), patch("core.file_tools.log_json") as mock_log:
            with self.assertRaisesRegex(MismatchOverwriteBlockedError, "mismatch overwrite fallback is disabled"):
                _safe_apply_change(
                    self.test_dir,
                    "test_file.txt",
                    old_code="missing marker",
                    new_code="replacement content",
                    allow_mismatch_overwrite=False,
                )

        self.assertEqual(self.test_file_path.read_text(), self.initial_content)
        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("file_tools_old_code_mismatch_blocked", events)
        self.assertNotIn("file_tools_old_code_mismatch_overwrite", events)

    def test_safe_apply_change_mismatch_overwrite_default_remains_enabled(self):
        with patch("core.file_tools.recover_old_code_from_git", return_value=None), patch("core.file_tools.log_json") as mock_log:
            _safe_apply_change(
                self.test_dir,
                "test_file.txt",
                old_code="missing marker",
                new_code="replacement content",
            )

        self.assertEqual(self.test_file_path.read_text(), "replacement content")
        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("file_tools_old_code_mismatch_overwrite", events)
        self.assertNotIn("file_tools_old_code_mismatch_blocked", events)

    def test_allow_mismatch_overwrite_for_change_requires_explicit_full_file_form(self):
        self.assertFalse(allow_mismatch_overwrite_for_change("stale", True))
        self.assertFalse(allow_mismatch_overwrite_for_change("", False))
        self.assertFalse(allow_mismatch_overwrite_for_change(None, True))
        self.assertTrue(allow_mismatch_overwrite_for_change("", True))

    def test_apply_change_with_explicit_overwrite_policy_passes_expected_flag(self):
        with patch("core.file_tools._safe_apply_change") as mock_safe:
            apply_change_with_explicit_overwrite_policy(
                self.test_dir,
                "test_file.txt",
                old_code="stale",
                new_code="replacement",
                overwrite_file=True,
            )

        self.assertFalse(mock_safe.call_args.kwargs["allow_mismatch_overwrite"])

        with patch("core.file_tools._safe_apply_change") as mock_safe:
            apply_change_with_explicit_overwrite_policy(
                self.test_dir,
                "test_file.txt",
                old_code="",
                new_code="replacement",
                overwrite_file=True,
            )

        self.assertTrue(mock_safe.call_args.kwargs["allow_mismatch_overwrite"])


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Supplemental pytest-style tests — uncovered branches
# ===========================================================================

import pytest
from unittest.mock import patch as _patch

from core.file_tools import (
    _aura_safe_loads,
)
import json as _json


# ---------------------------------------------------------------------------
# _safe_apply_change() — happy-path: old_code found → replacement applied
# ---------------------------------------------------------------------------


class TestSafeApplyChangeHappyPath:
    """_safe_apply_change replaces found code correctly."""

    def test_old_code_found_and_replaced(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("def hello():\n    pass\n")

        _safe_apply_change(
            tmp_path,
            "src.py",
            old_code="def hello():\n    pass\n",
            new_code="def hello():\n    return 42\n",
        )

        assert f.read_text() == "def hello():\n    return 42\n"

    def test_partial_old_code_replaced(self, tmp_path):
        f = tmp_path / "config.txt"
        f.write_text("version=1\nname=app\n")

        _safe_apply_change(
            tmp_path,
            "config.txt",
            old_code="version=1",
            new_code="version=2",
        )

        assert "version=2" in f.read_text()
        assert "name=app" in f.read_text()

    def test_full_overwrite_with_empty_old_code(self, tmp_path):
        """overwrite_file=True + old_code='' replaces the entire file."""
        f = tmp_path / "full.txt"
        f.write_text("old content")

        _safe_apply_change(
            tmp_path,
            "full.txt",
            old_code="",
            new_code="brand new content",
            overwrite_file=True,
        )

        assert f.read_text() == "brand new content"

    def test_creates_new_file_when_missing(self, tmp_path):
        """If the file doesn't exist it should be created with new_code."""
        _safe_apply_change(
            tmp_path,
            "new_file.py",
            old_code="",
            new_code="# created",
        )
        assert (tmp_path / "new_file.py").read_text() == "# created"

    def test_overwrites_empty_existing_file(self, tmp_path):
        """An existing but empty file is treated as a blank slate."""
        f = tmp_path / "empty.py"
        f.write_text("")

        _safe_apply_change(
            tmp_path,
            "empty.py",
            old_code="",
            new_code="# filled",
        )

        assert f.read_text() == "# filled"


# ---------------------------------------------------------------------------
# _safe_apply_change() — old_code not found, no new_code → OldCodeNotFoundError
# ---------------------------------------------------------------------------


class TestSafeApplyChangeErrors:
    def test_old_code_not_found_no_new_code_raises(self, tmp_path):
        f = tmp_path / "file.py"
        f.write_text("actual content here")

        with _patch("core.file_tools.recover_old_code_from_git", return_value=None):
            with pytest.raises(OldCodeNotFoundError):
                _safe_apply_change(
                    tmp_path,
                    "file.py",
                    old_code="nonexistent snippet",
                    new_code="",
                )

        # Original content must be untouched
        assert f.read_text() == "actual content here"

    def test_old_code_not_found_allow_mismatch_false_raises(self, tmp_path):
        f = tmp_path / "file.py"
        f.write_text("actual content here")

        with _patch("core.file_tools.recover_old_code_from_git", return_value=None):
            with pytest.raises(MismatchOverwriteBlockedError):
                _safe_apply_change(
                    tmp_path,
                    "file.py",
                    old_code="nonexistent snippet",
                    new_code="replacement",
                    allow_mismatch_overwrite=False,
                )

        assert f.read_text() == "actual content here"


# ---------------------------------------------------------------------------
# apply_change_with_explicit_overwrite_policy() — integration tests
# ---------------------------------------------------------------------------


class TestApplyChangePolicyIntegration:
    def test_blocks_mismatch_when_old_code_nonempty_overwrite_true(self, tmp_path):
        """Non-empty old_code + overwrite_file=True → allow_mismatch=False
        → raises MismatchOverwriteBlockedError when code not found."""
        f = tmp_path / "target.py"
        f.write_text("current content")

        with _patch("core.file_tools.recover_old_code_from_git", return_value=None), _patch("core.file_tools.log_json") as mock_log:
            with pytest.raises(MismatchOverwriteBlockedError):
                apply_change_with_explicit_overwrite_policy(
                    tmp_path,
                    "target.py",
                    old_code="stale snippet",
                    new_code="new content",
                    overwrite_file=True,
                )

        # Verify the mismatch-blocked event was logged
        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        assert "file_tools_old_code_mismatch_blocked" in events

        # File unchanged
        assert f.read_text() == "current content"

    def test_allows_intentional_overwrite_when_old_code_empty(self, tmp_path):
        """old_code='' + overwrite_file=True → allow_mismatch=True
        → file is completely replaced."""
        f = tmp_path / "target.py"
        f.write_text("old stuff")

        apply_change_with_explicit_overwrite_policy(
            tmp_path,
            "target.py",
            old_code="",
            new_code="entirely new content",
            overwrite_file=True,
        )

        assert f.read_text() == "entirely new content"

    def test_normal_replacement_via_policy(self, tmp_path):
        """When old_code is found the normal replacement path is used."""
        f = tmp_path / "module.py"
        f.write_text("x = 1\n")

        apply_change_with_explicit_overwrite_policy(
            tmp_path,
            "module.py",
            old_code="x = 1",
            new_code="x = 99",
        )

        assert "x = 99" in f.read_text()


# ---------------------------------------------------------------------------
# _aura_safe_loads() — JSON parsing with fence cleaning
# ---------------------------------------------------------------------------


class TestAuraSafeLoads:
    def test_valid_json_parsed(self):
        result = _aura_safe_loads('{"key": "value", "n": 42}')
        assert result == {"key": "value", "n": 42}

    def test_valid_json_list_parsed(self):
        result = _aura_safe_loads("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_fenced_json_cleaned_and_parsed(self):
        """Markdown code-fence wrapper must be stripped before parsing."""
        raw = '```json\n{"answer": true}\n```'
        result = _aura_safe_loads(raw)
        assert result == {"answer": True}

    def test_fenced_json_no_language_tag(self):
        raw = '```\n{"x": 1}\n```'
        result = _aura_safe_loads(raw)
        assert result == {"x": 1}

    def test_truly_invalid_json_raises(self):
        with pytest.raises(_json.JSONDecodeError):
            _aura_safe_loads("{not valid json at all!!!}")

    def test_none_input_raises(self):
        """None has no .encode() method — must propagate AttributeError."""
        with pytest.raises((AttributeError, TypeError)):
            _aura_safe_loads(None)  # type: ignore[arg-type]

    def test_ctx_parameter_accepted(self):
        """ctx kwarg must be accepted without error (future logging hook)."""
        result = _aura_safe_loads('{"a": 1}', ctx="test_caller")
        assert result["a"] == 1

    def test_utf8_roundtrip_preserves_content(self):
        """Values with multi-byte UTF-8 must survive the re-encode pass."""
        raw = '{"greeting": "héllo wörld"}'
        result = _aura_safe_loads(raw)
        assert result["greeting"] == "héllo wörld"
