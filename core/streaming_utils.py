"""Streaming utilities for handling large files safely.

Prevents memory exhaustion when processing files larger than 1MB.
"""
from pathlib import Path
from typing import Iterator, Union


# Default chunk size: 8KB (good balance between memory and I/O)
DEFAULT_CHUNK_SIZE: int = 8192

# File size threshold for streaming (1MB)
STREAMING_THRESHOLD: int = 1024 * 1024  # 1MB


def safe_read_text(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    errors: str = "replace",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Read a text file safely, using streaming for large files.

    Args:
        file_path: Path to the file to read
        encoding: Text encoding (default: utf-8)
        errors: Error handling for encoding issues (default: replace)
        chunk_size: Chunk size for streaming reads

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
        OSError: For other I/O errors
    """
    path = Path(file_path)

    # Get file size
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    # For small files, read normally
    if size < STREAMING_THRESHOLD:
        return path.read_text(encoding=encoding, errors=errors)

    # For large files, stream in chunks
    chunks = []
    with open(path, "r", encoding=encoding, errors=errors) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)

    return "".join(chunks)


def stream_lines(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    errors: str = "replace",
) -> Iterator[str]:
    """Stream a file line by line, memory-efficient for large files.

    Args:
        file_path: Path to the file to read
        encoding: Text encoding (default: utf-8)
        errors: Error handling for encoding issues (default: replace)

    Yields:
        Each line from the file (including newline)

    Example:
        >>> for line in stream_lines("/path/to/large_file.log"):
        ...     if "ERROR" in line:
        ...         print(line.strip())
    """
    path = Path(file_path)
    with open(path, "r", encoding=encoding, errors=errors) as f:
        for line in f:
            yield line


def stream_chunks(
    file_path: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    mode: str = "rt",
    encoding: str = "utf-8",
    errors: str = "replace",
) -> Iterator[str | bytes]:
    """Stream a file in fixed-size chunks.

    Args:
        file_path: Path to the file to read
        chunk_size: Size of each chunk in bytes
        mode: File mode ('rt' for text, 'rb' for binary)
        encoding: Text encoding (for text mode)
        errors: Error handling for encoding issues

    Yields:
        Chunks of the file (str for text mode, bytes for binary)

    Example:
        >>> for chunk in stream_chunks("/path/to/file.bin", mode="rb"):
        ...     process_binary_chunk(chunk)
    """
    path = Path(file_path)

    if "b" in mode:
        # Binary mode
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    else:
        # Text mode
        with open(path, "r", encoding=encoding, errors=errors) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get the size of a file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def is_large_file(file_path: Union[str, Path], threshold: int = STREAMING_THRESHOLD) -> bool:
    """Check if a file is large enough to warrant streaming.

    Args:
        file_path: Path to the file
        threshold: Size threshold in bytes (default: 1MB)

    Returns:
        True if file size >= threshold, False otherwise
    """
    return get_file_size(file_path) >= threshold
