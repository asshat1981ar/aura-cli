"""Tests for streaming_utils module."""

import pytest
from pathlib import Path
from core.streaming_utils import (
    safe_read_text,
    stream_lines,
    stream_chunks,
    get_file_size,
    is_large_file,
    STREAMING_THRESHOLD,
    DEFAULT_CHUNK_SIZE,
)


class TestSafeReadText:
    """Test safe_read_text function."""

    def test_reads_small_file(self, tmp_path):
        test_file = tmp_path / "small.txt"
        test_file.write_text("Hello, World!")
        result = safe_read_text(test_file)
        assert result == "Hello, World!"

    def test_reads_large_file(self, tmp_path):
        test_file = tmp_path / "large.txt"
        content = "x" * (STREAMING_THRESHOLD + 1000)
        test_file.write_text(content)
        result = safe_read_text(test_file)
        assert result == content

    def test_handles_encoding_errors(self, tmp_path):
        test_file = tmp_path / "binary.txt"
        test_file.write_bytes(b"Hello \xff\xfe World")
        result = safe_read_text(test_file)
        assert "Hello" in result
        assert "World" in result

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            safe_read_text(tmp_path / "nonexistent.txt")


class TestStreamLines:
    """Test stream_lines function."""

    def test_streams_lines(self, tmp_path):
        test_file = tmp_path / "lines.txt"
        test_file.write_text("line1\nline2\nline3\n")
        lines = list(stream_lines(test_file))
        assert len(lines) == 3
        assert lines[0] == "line1\n"
        assert lines[1] == "line2\n"
        assert lines[2] == "line3\n"

    def test_memory_efficient(self, tmp_path):
        # Create large file
        test_file = tmp_path / "huge.txt"
        test_file.write_text("x\n" * 10000)
        # Should not load entire file into memory at once
        line_count = sum(1 for _ in stream_lines(test_file))
        assert line_count == 10000


class TestStreamChunks:
    """Test stream_chunks function."""

    def test_stream_text_chunks(self, tmp_path):
        test_file = tmp_path / "chunks.txt"
        content = "abcdefghij"
        test_file.write_text(content)
        chunks = list(stream_chunks(test_file, chunk_size=3))
        assert chunks == ["abc", "def", "ghi", "j"]

    def test_stream_binary_chunks(self, tmp_path):
        test_file = tmp_path / "binary.bin"
        content = b"\x00\x01\x02\x03\x04\x05"
        test_file.write_bytes(content)
        chunks = list(stream_chunks(test_file, chunk_size=2, mode="rb"))
        assert chunks == [b"\x00\x01", b"\x02\x03", b"\x04\x05"]


class TestGetFileSize:
    """Test get_file_size function."""

    def test_returns_correct_size(self, tmp_path):
        test_file = tmp_path / "sized.txt"
        test_file.write_text("12345")  # 5 bytes
        assert get_file_size(test_file) == 5

    def test_returns_zero_for_missing_file(self, tmp_path):
        assert get_file_size(tmp_path / "nonexistent.txt") == 0


class TestIsLargeFile:
    """Test is_large_file function."""

    def test_detects_large_file(self, tmp_path):
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * (STREAMING_THRESHOLD + 1))
        assert is_large_file(test_file) is True

    def test_detects_small_file(self, tmp_path):
        test_file = tmp_path / "small.txt"
        test_file.write_text("small")
        assert is_large_file(test_file) is False

    def test_custom_threshold(self, tmp_path):
        test_file = tmp_path / "medium.txt"
        test_file.write_text("x" * 100)
        assert is_large_file(test_file, threshold=50) is True
        assert is_large_file(test_file, threshold=200) is False
