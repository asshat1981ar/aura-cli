"""Unit tests for agents.applicator.ApplicatorAgent and ApplyResult."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.applicator import ApplicatorAgent, ApplyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brain():
    b = MagicMock()
    b.remember.return_value = None
    return b


def _agent(tmp_path: Path) -> ApplicatorAgent:
    return ApplicatorAgent(_brain(), backup_dir=str(tmp_path / "backups"))


def _wrap(code: str) -> str:
    return f"```python\n{code}\n```"


SAMPLE_CODE = "# AURA_TARGET: agents/sample.py\ndef hello():\n    return 'hi'\n"


# ---------------------------------------------------------------------------
# ApplyResult dataclass
# ---------------------------------------------------------------------------


class TestApplyResult:
    def test_str_success(self):
        r = ApplyResult(success=True, target_path="foo.py", backup_path=None, code="x = 1\n")
        assert "OK" in str(r)
        assert "foo.py" in str(r)

    def test_str_failure(self):
        r = ApplyResult(success=False, target_path=None, backup_path=None, code=None, error="oops")
        assert "FAIL" in str(r)
        assert "oops" in str(r)

    def test_metadata_defaults_to_empty_dict(self):
        r = ApplyResult(success=True, target_path="f.py", backup_path=None, code="x")
        assert r.metadata == {}

    def test_error_defaults_to_none(self):
        r = ApplyResult(success=True, target_path="f.py", backup_path=None, code="x")
        assert r.error is None


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_extracts_python_block(self, tmp_path):
        agent = _agent(tmp_path)
        result = agent._extract_code(_wrap("x = 1"))
        assert result == "x = 1"

    def test_returns_none_when_no_block(self, tmp_path):
        agent = _agent(tmp_path)
        assert agent._extract_code("no code here") is None

    def test_strips_whitespace(self, tmp_path):
        agent = _agent(tmp_path)
        result = agent._extract_code("```python\n\n  x = 1  \n\n```")
        assert result == "x = 1"


# ---------------------------------------------------------------------------
# _detect_target
# ---------------------------------------------------------------------------


class TestDetectTarget:
    def test_finds_aura_target_directive(self, tmp_path):
        agent = _agent(tmp_path)
        code = "# AURA_TARGET: path/to/file.py\nx = 1"
        assert agent._detect_target(code) == "path/to/file.py"

    def test_returns_none_when_no_directive(self, tmp_path):
        agent = _agent(tmp_path)
        assert agent._detect_target("x = 1\ny = 2") is None


# ---------------------------------------------------------------------------
# apply() — happy path
# ---------------------------------------------------------------------------


class TestApplyHappyPath:
    def test_writes_file_and_returns_success(self, tmp_path):
        agent = _agent(tmp_path)
        target = tmp_path / "output.py"
        result = agent.apply(_wrap("x = 1"), target_path=str(target))

        assert result.success is True
        assert target.read_text() == "x = 1"
        assert result.backup_path is None  # no pre-existing file

    def test_detects_target_from_directive(self, tmp_path):
        agent = _agent(tmp_path)
        llm_output = f"```python\n{SAMPLE_CODE}```"
        # The directive says agents/sample.py — but resolve relative to cwd,
        # so just check the returned target_path ends with the expected name.
        result = agent.apply(llm_output)
        assert result.success is True
        assert result.target_path == "agents/sample.py"

    def test_metadata_contains_lines_and_timestamp(self, tmp_path):
        agent = _agent(tmp_path)
        result = agent.apply(_wrap("a = 1\nb = 2"), target_path=str(tmp_path / "f.py"))
        assert result.success is True
        assert "lines" in result.metadata
        assert "timestamp" in result.metadata

    def test_overwrites_existing_file_and_creates_backup(self, tmp_path):
        target = tmp_path / "existing.py"
        target.write_text("old content")
        agent = _agent(tmp_path)
        result = agent.apply(_wrap("new_content = 1"), target_path=str(target))

        assert result.success is True
        assert target.read_text() == "new_content = 1"
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()

    def test_brain_remember_called_on_success(self, tmp_path):
        brain = _brain()
        agent = ApplicatorAgent(brain, backup_dir=str(tmp_path / "backups"))
        agent.apply(_wrap("x = 1"), target_path=str(tmp_path / "out.py"))
        brain.remember.assert_called_once()


# ---------------------------------------------------------------------------
# apply() — error paths
# ---------------------------------------------------------------------------


class TestApplyErrors:
    def test_no_code_block_returns_failure(self, tmp_path):
        agent = _agent(tmp_path)
        result = agent.apply("just plain text, no fences")
        assert result.success is False
        assert "No" in result.error

    def test_no_target_path_and_no_directive_returns_failure(self, tmp_path):
        agent = _agent(tmp_path)
        result = agent.apply(_wrap("x = 1"))  # no directive, no explicit path
        assert result.success is False
        assert result.code == "x = 1"

    def test_allow_overwrite_false_blocks_existing_file(self, tmp_path):
        target = tmp_path / "existing.py"
        target.write_text("original")
        agent = _agent(tmp_path)
        result = agent.apply(_wrap("new = 1"), target_path=str(target), allow_overwrite=False)
        assert result.success is False
        assert "allow_overwrite" in result.error
        # Original file must be untouched
        assert target.read_text() == "original"

    def test_creates_parent_directories(self, tmp_path):
        agent = _agent(tmp_path)
        nested = tmp_path / "deep" / "nested" / "file.py"
        result = agent.apply(_wrap("x = 1"), target_path=str(nested))
        assert result.success is True
        assert nested.exists()


# ---------------------------------------------------------------------------
# rollback()
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_restores_original_content(self, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("original")
        agent = _agent(tmp_path)

        apply_result = agent.apply(_wrap("modified = True"), target_path=str(target))
        assert apply_result.success is True
        assert target.read_text() == "modified = True"

        ok = agent.rollback(apply_result)
        assert ok is True
        assert target.read_text() == "original"

    def test_rollback_returns_false_when_no_backup(self, tmp_path):
        agent = _agent(tmp_path)
        fake_result = ApplyResult(
            success=True, target_path=str(tmp_path / "f.py"),
            backup_path=None, code="x = 1",
        )
        assert agent.rollback(fake_result) is False

    def test_rollback_returns_false_when_backup_missing(self, tmp_path):
        agent = _agent(tmp_path)
        fake_result = ApplyResult(
            success=True,
            target_path=str(tmp_path / "f.py"),
            backup_path=str(tmp_path / "nonexistent.bak"),
            code="x = 1",
        )
        assert agent.rollback(fake_result) is False

    def test_rollback_calls_brain_remember(self, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("original")
        brain = _brain()
        agent = ApplicatorAgent(brain, backup_dir=str(tmp_path / "backups"))

        apply_result = agent.apply(_wrap("x = 1"), target_path=str(target))
        brain.remember.reset_mock()

        agent.rollback(apply_result)
        brain.remember.assert_called_once()
