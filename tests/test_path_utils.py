"""Tests for path_utils module."""

import pytest
from pathlib import Path
from core.path_utils import (
    should_skip_path,
    filter_paths,
    get_project_files,
    DEFAULT_SKIP_PATTERNS,
)


class TestShouldSkipPath:
    """Test should_skip_path function."""

    def test_skips_git_directory(self):
        assert should_skip_path("/project/.git/config") is True
        assert should_skip_path("/project/.git") is True

    def test_skips_pycache(self):
        assert should_skip_path("/project/__pycache__/module.cpython-311.pyc") is True
        assert should_skip_path("__pycache__") is True

    def test_skips_node_modules(self):
        assert should_skip_path("/project/node_modules/lodash/index.js") is True

    def test_skips_venv(self):
        assert should_skip_path("/project/.venv/lib/python3.11/site-packages") is True
        assert should_skip_path("/project/venv/bin/python") is True

    def test_skips_egg_info(self):
        assert should_skip_path("/project/mypackage.egg-info/PKG-INFO") is True

    def test_allows_source_files(self):
        assert should_skip_path("/project/src/main.py") is False
        assert should_skip_path("/project/lib/utils.js") is False

    def test_custom_skip_patterns(self):
        assert should_skip_path("/project/custom/temp.txt", skip_patterns={"custom"}) is True

    def test_additional_patterns(self):
        assert should_skip_path("/project/temp/cache.txt", additional_patterns={"temp"}) is True


class TestFilterPaths:
    """Test filter_paths function."""

    def test_filters_multiple_paths(self):
        paths = [
            "/project/src/main.py",
            "/project/.git/config",
            "/project/__pycache__/cache.pyc",
            "/project/src/utils.py",
        ]
        filtered = filter_paths(paths)
        assert len(filtered) == 2
        assert all(".git" not in str(p) for p in filtered)
        assert all("__pycache__" not in str(p) for p in filtered)


class TestGetProjectFiles:
    """Test get_project_files function."""

    def test_gets_python_files(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.pyc").write_text("")
        (tmp_path / "readme.md").write_text("# README")

        files = get_project_files(tmp_path, extensions={".py"})
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_respects_skip_patterns(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("")

        files = get_project_files(tmp_path)
        assert len(files) == 1
        assert ".git" not in str(files[0])


class TestDefaultSkipPatterns:
    """Test default skip patterns."""

    def test_contains_common_patterns(self):
        assert ".git" in DEFAULT_SKIP_PATTERNS
        assert "__pycache__" in DEFAULT_SKIP_PATTERNS
        assert "node_modules" in DEFAULT_SKIP_PATTERNS
        assert ".venv" in DEFAULT_SKIP_PATTERNS
