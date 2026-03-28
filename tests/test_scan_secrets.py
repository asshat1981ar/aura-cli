"""Tests for scripts/scan_secrets.py — TDD-first.

Run with:
    python3 -m pytest tests/test_scan_secrets.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / "scripts" / "scan_secrets.py"


def _run(directory: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(directory), *extra_args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Tests: clean directories exit 0
# ---------------------------------------------------------------------------


def test_clean_directory_exits_zero(tmp_path: Path) -> None:
    """A directory with no secrets-like content should exit 0."""
    (tmp_path / "main.py").write_text("x = 1\n")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_empty_directory_exits_zero(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Tests: flagged patterns exit 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, desc",
    [
        ('API_KEY = "abc123def456"\n', "bare api_key assignment"),
        ("secret = 'mysupersecret'\n", "bare secret assignment"),
        ("token = 'abcdefghijklmno'\n", "bare token assignment"),
        ("password = 'hunter2'\n", "bare password assignment"),
        ('Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\n', "bearer token"),
        ("aws_access_key = 'AKIAIOSFODNN7EXAMPLE'\n", "AWS AKIA key"),
        ("openai_key = 'sk-abcdef1234567890abcdef1234567890'\n", "OpenAI sk- key"),
        ("gh_token = 'ghp_abcdef1234567890abcdef1234'\n", "GitHub PAT ghp_"),
    ],
)
def test_flagged_pattern_exits_one(tmp_path: Path, content: str, desc: str) -> None:
    (tmp_path / "config.py").write_text(content)
    result = _run(tmp_path)
    assert result.returncode == 1, f"Expected finding for: {desc}\n{result.stdout}"


# ---------------------------------------------------------------------------
# Tests: ignored patterns exit 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, desc",
    [
        ("# API_KEY = 'abc123'\n", "comment line ignored"),
        ("API_KEY = 'PLACEHOLDER'\n", "PLACEHOLDER value ignored"),
        ("api_key = 'YOUR_API_KEY_HERE'\n", "YOUR_API_KEY_HERE placeholder"),
    ],
)
def test_ignored_pattern_exits_zero(tmp_path: Path, content: str, desc: str) -> None:
    (tmp_path / "config.py").write_text(content)
    result = _run(tmp_path)
    assert result.returncode == 0, f"Unexpected finding for: {desc}\n{result.stdout}"


# ---------------------------------------------------------------------------
# Tests: ignored directories
# ---------------------------------------------------------------------------


def test_git_directory_ignored(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("password = 'hunter2'\n")
    result = _run(tmp_path)
    assert result.returncode == 0, ".git dir should be ignored\n" + result.stdout


def test_node_modules_ignored(tmp_path: Path) -> None:
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "index.js").write_text("const api_key = 'abc123def456';\n")
    result = _run(tmp_path)
    assert result.returncode == 0, "node_modules should be ignored\n" + result.stdout


def test_pycache_ignored(tmp_path: Path) -> None:
    pc = tmp_path / "__pycache__"
    pc.mkdir()
    (pc / "foo.pyc").write_text("password = 'hunter2'\n")
    result = _run(tmp_path)
    assert result.returncode == 0, "__pycache__ should be ignored\n" + result.stdout


def test_venv_ignored(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "activate").write_text("password = 'hunter2'\n")
    result = _run(tmp_path)
    assert result.returncode == 0, ".venv should be ignored\n" + result.stdout


# ---------------------------------------------------------------------------
# Tests: file extensions scanned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "settings.py",
        "config.json",
        "ci.yml",
        "build.yaml",
        "pyproject.toml",
        "setup.cfg",
        "deploy.sh",
        ".env",
    ],
)
def test_scanned_extension(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).write_text("api_key = 'realvalue123'\n")
    result = _run(tmp_path)
    assert result.returncode == 1, f"Should scan {filename}\n{result.stdout}"


def test_binary_file_skipped(tmp_path: Path) -> None:
    """A .jpg file should not be scanned even if it somehow has a match."""
    (tmp_path / "image.jpg").write_bytes(b"\xff\xd8\xff" + b"password = 'hunter2'\n")
    result = _run(tmp_path)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Tests: output / reporting
# ---------------------------------------------------------------------------


def test_output_includes_file_path_on_finding(tmp_path: Path) -> None:
    secret_file = tmp_path / "app.py"
    secret_file.write_text("api_key = 'abc123def456'\n")
    result = _run(tmp_path)
    assert "app.py" in result.stdout, "Output should mention the offending file"


def test_output_includes_line_number_on_finding(tmp_path: Path) -> None:
    secret_file = tmp_path / "app.py"
    secret_file.write_text("x = 1\napi_key = 'abc123def456'\n")
    result = _run(tmp_path)
    assert "2" in result.stdout, "Output should include line number"


def test_clean_output_message(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = 1\n")
    result = _run(tmp_path)
    assert "clean" in result.stdout.lower() or result.stdout.strip() == ""


def test_findings_summary_count(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("api_key = 'abc123'\n")
    (tmp_path / "b.py").write_text("password = 'secret'\n")
    result = _run(tmp_path)
    assert result.returncode == 1
