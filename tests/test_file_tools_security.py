"""Security and error-handling tests for core/file_tools.py."""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from core.file_tools import FileToolsError, _validate_file_path, replace_code


class TestValidateFilePath:
    def test_rejects_path_traversal(self):
        with pytest.raises(FileToolsError, match="Path traversal blocked"):
            _validate_file_path("../etc/passwd")

    def test_rejects_nested_path_traversal(self):
        with pytest.raises(FileToolsError, match="Path traversal blocked"):
            _validate_file_path("some/../../etc/passwd")

    def test_rejects_null_bytes(self):
        with pytest.raises(FileToolsError, match="contains null bytes"):
            _validate_file_path("file\x00name.py")

    def test_rejects_null_byte_mid_path(self):
        with pytest.raises(FileToolsError, match="contains null bytes"):
            _validate_file_path("/tmp/foo\x00bar")

    def test_rejects_path_outside_project_root(self, tmp_path):
        outside = tmp_path.parent / "other_project" / "secret.py"
        with pytest.raises(FileToolsError, match="Path outside project root"):
            _validate_file_path(str(outside), project_root=tmp_path)

    def test_allows_valid_relative_path(self, tmp_path):
        # Should not raise
        _validate_file_path("subdir/file.py", project_root=tmp_path)

    def test_allows_simple_filename(self):
        # Should not raise
        _validate_file_path("file.py")

    def test_allows_nested_valid_path(self, tmp_path):
        # Should not raise
        _validate_file_path("a/b/c.py", project_root=tmp_path)


class TestReplaceCodeErrors:
    def test_nonexistent_file_raises_file_tools_error(self, tmp_path):
        missing = tmp_path / "nonexistent.py"
        with pytest.raises(FileToolsError, match="File not found"):
            replace_code(str(missing), "old", "new", project_root=tmp_path)

    def test_permission_denied_raises_file_tools_error(self, tmp_path):
        target = tmp_path / "locked.py"
        target.write_text("some content\n", encoding="utf-8")

        # Make the file unreadable
        readonly_dir = tmp_path / "readonly_dir"
        readonly_dir.mkdir()
        locked_file = readonly_dir / "locked.py"
        locked_file.write_text("some content\n", encoding="utf-8")
        locked_file.chmod(0o000)

        try:
            with pytest.raises(FileToolsError, match="Permission denied"):
                replace_code(str(locked_file), "some content", "new content", project_root=tmp_path)
        finally:
            # Restore permissions so pytest cleanup can remove the file
            locked_file.chmod(0o644)

    def test_path_traversal_in_replace_code_is_rejected(self, tmp_path):
        with pytest.raises(FileToolsError, match="Path traversal blocked"):
            replace_code("../outside.py", "old", "new", project_root=tmp_path)

    def test_null_byte_in_replace_code_is_rejected(self, tmp_path):
        with pytest.raises(FileToolsError, match="contains null bytes"):
            replace_code("file\x00.py", "old", "new", project_root=tmp_path)
