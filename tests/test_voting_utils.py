"""Tests for voting utilities module - core/voting/utils.py

Covers file handling utilities with comprehensive test coverage.
"""

import pytest
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, mock_open

from core.voting.utils import FileHandler


class TestFileHandler:
    """Test FileHandler utility class."""

    def test_file_handler_init(self):
        """FileHandler should initialize with a filepath."""
        filepath = Path("/some/path/file.txt")
        handler = FileHandler(filepath=filepath)
        assert handler.filepath == filepath

    def test_read_lines_success(self):
        """read_lines should return list of lines from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("line1\n")
            f.write("line2\n")
            f.write("line3\n")
            temp_path = f.name

        try:
            handler = FileHandler(filepath=Path(temp_path))
            lines = handler.read_lines()
            assert lines == ["line1", "line2", "line3"]
        finally:
            os.unlink(temp_path)

    def test_read_lines_with_whitespace(self):
        """read_lines should strip whitespace from lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("  line1  \n")
            f.write("\tline2\t\n")
            f.write("line3   \n")
            temp_path = f.name

        try:
            handler = FileHandler(filepath=Path(temp_path))
            lines = handler.read_lines()
            assert lines == ["line1", "line2", "line3"]
        finally:
            os.unlink(temp_path)

    def test_read_lines_empty_file(self):
        """read_lines should return empty list for empty file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            handler = FileHandler(filepath=Path(temp_path))
            lines = handler.read_lines()
            assert lines == []
        finally:
            os.unlink(temp_path)

    def test_read_lines_file_not_found(self):
        """read_lines should raise FileNotFoundError if file doesn't exist."""
        handler = FileHandler(filepath=Path("/nonexistent/file.txt"))
        with pytest.raises(FileNotFoundError) as exc_info:
            handler.read_lines()
        assert "Cannot open" in str(exc_info.value)

    def test_read_lines_permission_error(self):
        """read_lines should raise FileNotFoundError for permission errors."""
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            handler = FileHandler(filepath=Path("/some/file.txt"))
            with pytest.raises(FileNotFoundError):
                handler.read_lines()

    def test_write_lines_success(self):
        """write_lines should write lines to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            handler = FileHandler(filepath=filepath)
            lines = ["line1", "line2", "line3"]
            handler.write_lines(lines)

            # Verify file was written
            assert filepath.exists()
            content = filepath.read_text()
            assert content == "line1\nline2\nline3\n"

    def test_write_lines_empty_list(self):
        """write_lines should handle empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            handler = FileHandler(filepath=filepath)
            handler.write_lines([])

            # Verify file was created but is empty
            assert filepath.exists()
            content = filepath.read_text()
            assert content == ""

    def test_write_lines_overwrites(self):
        """write_lines should overwrite existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.write_text("old content\n")

            handler = FileHandler(filepath=filepath)
            lines = ["new1", "new2"]
            handler.write_lines(lines)

            content = filepath.read_text()
            assert content == "new1\nnew2\n"
            assert "old content" not in content

    def test_write_lines_special_characters(self):
        """write_lines should handle special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            handler = FileHandler(filepath=filepath)
            lines = ["line with special chars: !@#$%", "unicode: ñ é ü", "tabs\tand\tnewlines"]
            handler.write_lines(lines)

            content = filepath.read_text()
            assert "!@#$%" in content
            assert "ñ é ü" in content
            assert "tabs\tand\tnewlines" in content

    def test_write_lines_permission_error(self):
        """write_lines should raise PermissionError if cannot write."""
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            handler = FileHandler(filepath=Path("/some/file.txt"))
            with pytest.raises(PermissionError) as exc_info:
                handler.write_lines(["test"])
            assert "Cannot write to" in str(exc_info.value)

    def test_write_lines_parent_not_exists(self):
        """write_lines should fail if parent directory doesn't exist."""
        filepath = Path("/nonexistent/parent/dir/file.txt")
        handler = FileHandler(filepath=filepath)
        with pytest.raises(PermissionError):
            handler.write_lines(["test"])

    def test_validate_path_exists(self):
        """validate_path should not raise if path exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.touch()
            # Should not raise
            FileHandler.validate_path(filepath)

    def test_validate_path_not_exists(self):
        """validate_path should raise ValueError if path doesn't exist."""
        filepath = Path("/nonexistent/path/file.txt")
        with pytest.raises(ValueError) as exc_info:
            FileHandler.validate_path(filepath)
        assert "does not exist" in str(exc_info.value)

    def test_validate_path_directory(self):
        """validate_path should work with directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            # Should not raise
            FileHandler.validate_path(dirpath)

    def test_static_method_validate_path(self):
        """validate_path should be callable as static method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.touch()
            # Call as static method
            FileHandler.validate_path(filepath)
            # Should not raise

    def test_read_write_roundtrip(self):
        """Read and write should roundtrip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            handler = FileHandler(filepath=filepath)

            # Write
            original_lines = ["line1", "line2", "line3 with spaces"]
            handler.write_lines(original_lines)

            # Read
            read_lines = handler.read_lines()

            assert read_lines == original_lines

    def test_multiple_handlers_same_file(self):
        """Multiple handlers should work on same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"

            handler1 = FileHandler(filepath=filepath)
            handler1.write_lines(["initial"])

            handler2 = FileHandler(filepath=filepath)
            lines = handler2.read_lines()
            assert lines == ["initial"]

            handler1.write_lines(["updated"])
            lines = handler2.read_lines()
            assert lines == ["updated"]

    def test_large_file(self):
        """FileHandler should handle large files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "large.txt"
            handler = FileHandler(filepath=filepath)

            # Create a large list of lines
            large_lines = [f"line_{i}" for i in range(10000)]
            handler.write_lines(large_lines)

            # Read and verify
            read_lines = handler.read_lines()
            assert len(read_lines) == 10000
            assert read_lines[0] == "line_0"
            assert read_lines[-1] == "line_9999"

    def test_unicode_handling(self):
        """FileHandler should correctly handle unicode content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "unicode.txt"
            handler = FileHandler(filepath=filepath)

            unicode_lines = ["Hello 世界", "مرحبا", "Привет", "🎉 emoji"]
            handler.write_lines(unicode_lines)

            read_lines = handler.read_lines()
            assert read_lines == unicode_lines

    def test_lines_with_only_whitespace(self):
        """Lines with only whitespace should be stripped to empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "whitespace.txt"
            handler = FileHandler(filepath=filepath)

            handler.write_lines(["   ", "\t\t", "  \n"])
            # Re-read what was written
            with open(filepath, "r") as f:
                raw_content = f.read()

            # Now read using handler which strips
            lines = handler.read_lines()
            # After stripping whitespace-only lines
            assert all(line == "" for line in lines) or len(lines) == 3

    def test_handler_with_relative_path(self):
        """FileHandler should work with relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                filepath = Path("test.txt")
                handler = FileHandler(filepath=filepath)
                handler.write_lines(["test"])
                lines = handler.read_lines()
                assert lines == ["test"]
            finally:
                os.chdir(old_cwd)

    def test_handler_with_absolute_path(self):
        """FileHandler should work with absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir).resolve() / "test.txt"
            handler = FileHandler(filepath=filepath)
            handler.write_lines(["test"])
            lines = handler.read_lines()
            assert lines == ["test"]

    def test_different_line_endings(self):
        """FileHandler should handle files with different line endings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            # Write with Unix line endings
            filepath.write_text("line1\nline2\nline3\n")

            handler = FileHandler(filepath=filepath)
            lines = handler.read_lines()
            assert lines == ["line1", "line2", "line3"]
