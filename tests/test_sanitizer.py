import pytest

from core.sanitizer import SecurityError, sanitize_path, sanitize_command, get_allowed_commands


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


def test_sanitize_path_absolute_escape(tmp_path):
    """Absolute path outside root raises SecurityError (lines 34-35)."""
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "other" / "secret.txt"

    with pytest.raises(SecurityError):
        sanitize_path(outside, root)


def test_sanitize_command_empty():
    """Empty command list returns None without error (line 44)."""
    result = sanitize_command([])
    assert result is None


def test_sanitize_command_allowed_python():
    """python3 with a plain script arg is allowed (line 48)."""
    sanitize_command(["python3", "script.py"])  # must not raise


def test_sanitize_command_disallowed():
    """Non-allowlisted command raises SecurityError (lines 55-58)."""
    with pytest.raises(SecurityError):
        sanitize_command(["curl", "http://x"])


def test_sanitize_command_disallowed_atypical_python():
    """python-like name that doesn't match the pattern is denied (lines 52-55)."""
    with pytest.raises(SecurityError):
        sanitize_command(["python_evil"])


def test_sanitize_command_python_dangerous_arg_no_m():
    """python3 with -c and no -m raises SecurityError (lines 66-67)."""
    with pytest.raises(SecurityError):
        sanitize_command(["python3", "-c", "import os; os.system('rm -rf /')"])


def test_sanitize_command_python_dangerous_arg_with_m():
    """python3 with -m present makes -c safe (lines 66-67 continue branch)."""
    sanitize_command(["python3", "-m", "pytest", "-c", "setup.cfg"])  # must not raise


def test_get_allowed_commands_includes_base():
    """get_allowed_commands() returns at least python3 and git (lines 20-21)."""
    cmds = get_allowed_commands()
    assert "python3" in cmds
    assert "git" in cmds
