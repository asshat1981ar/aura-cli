"""Security tests for file_tools module.

Tests path traversal prevention, injection blocking, and error handling.
Addresses issues #310, #311, #316.
"""
import pytest
from pathlib import Path
from core.file_tools import replace_code, FileToolsError, _validate_file_path


class TestPathTraversalPrevention:
    """Test path traversal blocking (#316)."""
    
    def test_rejects_path_traversal_dotdot(self, tmp_path):
        """Block ../ in file paths."""
        malicious = str(tmp_path / ".." / ".." / "etc" / "passwd")
        with pytest.raises(FileToolsError, match="traversal"):
            replace_code(malicious, old_code="", new_code="pwned", overwrite_file=True, project_root=tmp_path)

    def test_rejects_absolute_path_outside_project(self, tmp_path):
        """Block absolute paths outside project root."""
        with pytest.raises(FileToolsError, match="outside"):
            replace_code("/etc/passwd", old_code="", new_code="pwned", overwrite_file=True, project_root=tmp_path)

    def test_rejects_null_bytes_in_path(self, tmp_path):
        """Null byte injection in file paths (#311)."""
        with pytest.raises(FileToolsError, match="null"):
            replace_code(str(tmp_path / "file\x00.py"), old_code="", new_code="x", overwrite_file=True, project_root=tmp_path)

    def test_allows_valid_relative_path(self, tmp_path):
        """Valid relative paths within project should work."""
        test_file = tmp_path / "test.py"
        test_file.write_text("old content")
        # Should not raise
        replace_code(str(test_file), old_code="old", new_code="new", project_root=tmp_path)
        assert "new content" in test_file.read_text()


class TestFileIOErrorHandling:
    """Test specific error handling (#310)."""
    
    def test_read_nonexistent_file_raises_specific_error(self, tmp_path):
        """Specific exception for missing files."""
        nonexistent = tmp_path / "nonexistent.py"
        with pytest.raises(FileToolsError, match="not found"):
            replace_code(str(nonexistent), old_code="x", new_code="y", project_root=tmp_path)

    def test_validates_path_before_file_operations(self, tmp_path):
        """Path validation happens before any file operations."""
        malicious = str(tmp_path / ".." / "secret.txt")
        with pytest.raises(FileToolsError, match="traversal"):
            replace_code(malicious, old_code="x", new_code="y", project_root=tmp_path)


class TestValidateFilePath:
    """Test _validate_file_path function directly."""
    
    def test_null_bytes_rejected(self, tmp_path):
        """Null bytes in path should raise."""
        with pytest.raises(FileToolsError, match="null"):
            _validate_file_path(str(tmp_path / "file\x00.py"), tmp_path)

    def test_traversal_rejected(self, tmp_path):
        """Path traversal should raise."""
        with pytest.raises(FileToolsError, match="traversal"):
            _validate_file_path("../etc/passwd", tmp_path)

    def test_outside_project_rejected(self, tmp_path):
        """Paths outside project should raise."""
        with pytest.raises(FileToolsError, match="outside"):
            _validate_file_path("/etc/passwd", tmp_path)

    def test_valid_path_accepted(self, tmp_path):
        """Valid paths should not raise."""
        # Should not raise
        _validate_file_path("src/main.py", tmp_path)

    def test_no_project_root_allows_any_path(self):
        """Without project_root, only traversal/null checks apply."""
        # Should not raise (no project_root means no jail)
        _validate_file_path("/any/absolute/path", None)
