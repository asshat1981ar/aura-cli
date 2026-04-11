"""Tests for agents/applicator.py — ApplicatorAgent, ApplyResult."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agents.applicator import ApplicatorAgent, ApplyResult


@pytest.fixture
def brain():
    return MagicMock()


@pytest.fixture
def agent(brain, tmp_path):
    return ApplicatorAgent(brain=brain, backup_dir=str(tmp_path / "backups"))


def _wrap(code: str) -> str:
    """Wrap code in a python markdown fence."""
    return f"```python\n{code}\n```"


# ---------------------------------------------------------------------------
# ApplyResult.__str__
# ---------------------------------------------------------------------------


class TestApplyResultStr:
    def test_success_str_includes_ok(self):
        r = ApplyResult(success=True, target_path="x.py", backup_path=None, code="pass")
        assert "OK" in str(r)

    def test_failure_str_includes_fail(self):
        r = ApplyResult(success=False, target_path=None, backup_path=None, code=None, error="bad")
        assert "FAIL" in str(r)
        assert "bad" in str(r)


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_extracts_python_block(self, agent):
        code = agent._extract_code("```python\nx = 1\n```")
        assert code == "x = 1"

    def test_extracts_plain_fence(self, agent):
        code = agent._extract_code("```\ny = 2\n```")
        assert code == "y = 2"

    def test_returns_none_no_fence(self, agent):
        assert agent._extract_code("no code here") is None

    def test_strips_whitespace(self, agent):
        code = agent._extract_code("```python\n  x = 1  \n```")
        assert code == "x = 1"

    def test_extracts_first_block_only(self, agent):
        raw = "```python\nfirst\n```\n```python\nsecond\n```"
        assert agent._extract_code(raw) == "first"


# ---------------------------------------------------------------------------
# _detect_target
# ---------------------------------------------------------------------------


class TestDetectTarget:
    def test_finds_aura_target_directive(self, agent):
        code = "# AURA_TARGET: agents/new_agent.py\ndef foo(): pass"
        assert agent._detect_target(code) == "agents/new_agent.py"

    def test_returns_none_no_directive(self, agent):
        assert agent._detect_target("def foo(): pass") is None

    def test_strips_whitespace_around_path(self, agent):
        code = "#  AURA_TARGET:   core/planner.py  "
        assert agent._detect_target(code) == "core/planner.py"


# ---------------------------------------------------------------------------
# apply — no code block
# ---------------------------------------------------------------------------


class TestApplyNoCode:
    def test_no_fence_returns_failure(self, agent):
        result = agent.apply("no code block here")
        assert not result.success
        assert "code block" in result.error

    def test_no_fence_target_path_preserved(self, agent):
        result = agent.apply("no code", target_path="x.py")
        assert result.target_path == "x.py"


# ---------------------------------------------------------------------------
# apply — no target path
# ---------------------------------------------------------------------------


class TestApplyNoTarget:
    def test_no_target_returns_failure(self, agent):
        result = agent.apply(_wrap("def foo(): pass"))
        assert not result.success
        assert "target" in result.error.lower()

    def test_code_captured_even_without_target(self, agent):
        result = agent.apply(_wrap("x = 1"))
        assert result.code == "x = 1"


# ---------------------------------------------------------------------------
# apply — successful write
# ---------------------------------------------------------------------------


class TestApplySuccess:
    def test_writes_to_explicit_target(self, agent, tmp_path):
        target = tmp_path / "out.py"
        result = agent.apply(_wrap("x = 42"), target_path=str(target))
        assert result.success
        assert target.read_text() == "x = 42"

    def test_uses_aura_target_directive(self, agent, tmp_path):
        target = tmp_path / "module.py"
        code = f"# AURA_TARGET: {target}\ndef bar(): pass"
        result = agent.apply(_wrap(code))
        assert result.success
        assert target.exists()

    def test_metadata_has_lines(self, agent, tmp_path):
        target = tmp_path / "f.py"
        result = agent.apply(_wrap("a = 1\nb = 2"), target_path=str(target))
        assert result.metadata["lines"] >= 2

    def test_metadata_has_timestamp(self, agent, tmp_path):
        target = tmp_path / "f.py"
        result = agent.apply(_wrap("pass"), target_path=str(target))
        assert "timestamp" in result.metadata

    def test_brain_remember_called(self, agent, brain, tmp_path):
        target = tmp_path / "f.py"
        agent.apply(_wrap("pass"), target_path=str(target))
        brain.remember.assert_called_once()

    def test_creates_parent_dirs(self, agent, tmp_path):
        target = tmp_path / "sub" / "dir" / "f.py"
        result = agent.apply(_wrap("x = 1"), target_path=str(target))
        assert result.success
        assert target.exists()


# ---------------------------------------------------------------------------
# apply — overwrite policy
# ---------------------------------------------------------------------------


class TestApplyOverwrite:
    def test_allow_overwrite_false_blocks_existing(self, agent, tmp_path):
        target = tmp_path / "existing.py"
        target.write_text("old content")
        result = agent.apply(_wrap("new content"), target_path=str(target), allow_overwrite=False)
        assert not result.success
        assert "allow_overwrite" in result.error

    def test_allow_overwrite_true_replaces_file(self, agent, tmp_path):
        target = tmp_path / "existing.py"
        target.write_text("old")
        result = agent.apply(_wrap("new = 1"), target_path=str(target), allow_overwrite=True)
        assert result.success
        assert "new = 1" in target.read_text()

    def test_backup_created_for_existing_file(self, agent, tmp_path):
        target = tmp_path / "existing.py"
        target.write_text("original")
        result = agent.apply(_wrap("replaced"), target_path=str(target))
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()

    def test_no_backup_for_new_file(self, agent, tmp_path):
        target = tmp_path / "new_file.py"
        result = agent.apply(_wrap("x = 1"), target_path=str(target))
        assert result.backup_path is None


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_restores_backup(self, agent, tmp_path):
        target = tmp_path / "f.py"
        target.write_text("original")
        result = agent.apply(_wrap("replaced"), target_path=str(target))
        agent.rollback(result)
        assert target.read_text() == "original"

    def test_rollback_returns_true_on_success(self, agent, tmp_path):
        target = tmp_path / "f.py"
        target.write_text("original")
        result = agent.apply(_wrap("replaced"), target_path=str(target))
        assert agent.rollback(result) is True

    def test_rollback_false_no_backup(self, agent):
        r = ApplyResult(success=True, target_path="x.py", backup_path=None, code="pass")
        assert agent.rollback(r) is False

    def test_rollback_false_backup_missing(self, agent, tmp_path):
        r = ApplyResult(
            success=True,
            target_path=str(tmp_path / "f.py"),
            backup_path=str(tmp_path / "nonexistent.bak"),
            code="pass",
        )
        assert agent.rollback(r) is False

    def test_rollback_calls_brain_remember(self, agent, brain, tmp_path):
        target = tmp_path / "f.py"
        target.write_text("v1")
        result = agent.apply(_wrap("v2"), target_path=str(target))
        brain.remember.reset_mock()
        agent.rollback(result)
        brain.remember.assert_called_once()
