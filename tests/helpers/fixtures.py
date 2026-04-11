"""Test fixtures for isolated test environments."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional


class TestFixture:
    """Test fixture for isolated test environments.

    Provides temporary directories, configuration files, and cleanup
    for test isolation.

    Example:
        >>> fixture = TestFixture()
        >>> config_path = fixture.create_config({"log_level": "DEBUG"})
        >>> # Run tests...
        >>> fixture.cleanup()
    """

    def __init__(self, prefix: str = "aura-test-") -> None:
        """Initialize test fixture with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.config_path = self.temp_dir / ".aura" / "config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._original_home: Optional[str] = None
        self._original_cwd: Optional[str] = None

    def create_config(self, overrides: dict | None = None) -> Path:
        """Create a test configuration file.

        Args:
            overrides: Configuration values to override defaults.

        Returns:
            Path to created config file.
        """
        config = {
            "version": "1.0.0",
            "model_name": "google/gemini-2.0-flash-exp:free",
            "api_url": "https://test-api.aura.dev",
            "timeout": 5,
            "log_level": "DEBUG",
            "dry_run": True,
            "max_iterations": 3,
            "max_cycles": 2,
            "memory_persistence_path": str(self.temp_dir / "memory" / "task_hierarchy_v2.json"),
            "memory_store_path": str(self.temp_dir / "memory" / "store"),
            "goal_queue_path": str(self.temp_dir / "memory" / "goal_queue.json"),
            "goal_archive_path": str(self.temp_dir / "memory" / "goal_archive_v2.json"),
            "brain_db_path": str(self.temp_dir / "memory" / "brain_v2.db"),
            **(overrides or {}),
        }
        self.config_path.write_text(json.dumps(config, indent=2))
        return self.config_path

    def create_file(self, path: str, content: str | bytes) -> Path:
        """Create a file in the temp directory.

        Args:
            path: Relative path within temp directory.
            content: File content (string or bytes).

        Returns:
            Path to created file.
        """
        full_path = self.temp_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            full_path.write_bytes(content)
        else:
            full_path.write_text(content)
        return full_path

    def create_json_file(self, path: str, data: dict) -> Path:
        """Create a JSON file in the temp directory.

        Args:
            path: Relative path within temp directory.
            data: Data to serialize as JSON.

        Returns:
            Path to created file.
        """
        return self.create_file(path, json.dumps(data, indent=2))

    def create_pyproject(self, overrides: dict | None = None) -> Path:
        """Create a minimal pyproject.toml for testing.

        Args:
            overrides: Configuration values to override defaults.

        Returns:
            Path to created pyproject.toml.
        """
        content = f"""[project]
name = "test-project"
version = "0.1.0"
description = "Test project"

[tool.aura]
model_name = "{overrides.get("model_name", "test-model")}"\n"""
        return self.create_file("pyproject.toml", content)

    def mock_home(self) -> None:
        """Set HOME environment variable to temp directory."""
        self._original_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.temp_dir)

    def restore_home(self) -> None:
        """Restore original HOME environment variable."""
        if self._original_home is not None:
            os.environ["HOME"] = self._original_home
        elif "HOME" in os.environ:
            del os.environ["HOME"]

    def mock_cwd(self) -> None:
        """Change current working directory to temp directory."""
        self._original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def restore_cwd(self) -> None:
        """Restore original working directory."""
        if self._original_cwd:
            os.chdir(self._original_cwd)

    def get_path(self, *parts: str) -> Path:
        """Get path within temp directory.

        Args:
            *parts: Path components.

        Returns:
            Absolute path within temp directory.
        """
        return self.temp_dir.joinpath(*parts)

    def cleanup(self) -> None:
        """Clean up temporary files and restore environment."""
        self.restore_home()
        self.restore_cwd()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __enter__(self) -> "TestFixture":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()


@contextmanager
def temp_fixture(prefix: str = "aura-test-") -> Generator[TestFixture, None, None]:
    """Context manager for test fixtures.

    Args:
        prefix: Prefix for temp directory name.

    Example:
        >>> with temp_fixture() as fixture:
        ...     config_path = fixture.create_config()
        ...     # Test with isolated environment
    """
    fixture = TestFixture(prefix=prefix)
    try:
        yield fixture
    finally:
        fixture.cleanup()


class AsyncTestFixture(TestFixture):
    """Extended fixture with async support."""

    def __init__(self, prefix: str = "aura-async-test-") -> None:
        super().__init__(prefix=prefix)
        self._async_tasks: list[Any] = []

    async def cleanup_async(self) -> None:
        """Async cleanup for pending tasks."""
        for task in self._async_tasks:
            if hasattr(task, "cancel") and not task.done():
                task.cancel()
        self.cleanup()
