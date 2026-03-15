from pathlib import Path

import pytest

from core.sanitizer import SecurityError, sanitize_path


def test_sanitize_path_rejects_sibling_prefix(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    sibling = tmp_path / "repo_evil"
    sibling.mkdir()

    with pytest.raises(SecurityError):
        sanitize_path(sibling / "secret.txt", root)


def test_sanitize_path_resolves_relative_to_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    target = sanitize_path("subdir/file.txt", root)

    assert target == (root / "subdir" / "file.txt").resolve()
    assert str(target).startswith(str(root.resolve()))
