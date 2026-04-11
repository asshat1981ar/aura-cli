"""Tests for agents/skills/base.py — SkillBase, iter_py_files, SKIP_DIRS."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agents.skills.base import SkillBase, iter_py_files, SKIP_DIRS


# ---------------------------------------------------------------------------
# Concrete stub for abstract class
# ---------------------------------------------------------------------------

class _EchoSkill(SkillBase):
    name = "echo_skill"

    def _run(self, input_data):
        return {"echo": input_data.get("value", "nothing")}


class _BrokenSkill(SkillBase):
    name = "broken_skill"

    def _run(self, input_data):
        raise ValueError("intentional failure")


# ---------------------------------------------------------------------------
# iter_py_files
# ---------------------------------------------------------------------------

class TestIterPyFiles:
    def test_yields_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        files = list(iter_py_files(tmp_path))
        assert len(files) == 2

    def test_excludes_git_dir(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config").write_text("x")
        (tmp_path / "real.py").write_text("y")
        files = list(iter_py_files(tmp_path))
        names = [f.name for f in files]
        assert "config" not in names
        assert "real.py" in names

    def test_excludes_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "cached.pyc").write_text("x")
        (tmp_path / "module.py").write_text("y")
        files = list(iter_py_files(tmp_path))
        assert all("__pycache__" not in str(f) for f in files)

    def test_excludes_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "index.py").write_text("x")
        (tmp_path / "main.py").write_text("y")
        files = list(iter_py_files(tmp_path))
        # Check no yielded file lives inside the node_modules dir
        assert not any(f.parent.name == "node_modules" for f in files)

    def test_excludes_venv(self, tmp_path):
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "activate.py").write_text("x")
        files = list(iter_py_files(tmp_path))
        assert not any("venv" in str(f) for f in files)

    def test_yields_nested_py_files(self, tmp_path):
        sub = tmp_path / "pkg" / "sub"
        sub.mkdir(parents=True)
        (sub / "deep.py").write_text("x")
        files = list(iter_py_files(tmp_path))
        assert any("deep.py" in str(f) for f in files)

    def test_empty_directory_yields_nothing(self, tmp_path):
        assert list(iter_py_files(tmp_path)) == []


# ---------------------------------------------------------------------------
# SKIP_DIRS
# ---------------------------------------------------------------------------

class TestSkipDirs:
    def test_is_frozenset(self):
        assert isinstance(SKIP_DIRS, frozenset)

    def test_contains_expected_entries(self):
        for entry in (".git", "__pycache__", "node_modules", ".venv", "venv"):
            assert entry in SKIP_DIRS

    def test_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            SKIP_DIRS.add("new_dir")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# SkillBase.run — happy path
# ---------------------------------------------------------------------------

class TestSkillBaseRun:
    def test_run_returns_dict(self):
        skill = _EchoSkill()
        result = skill.run({"value": "hello"})
        assert isinstance(result, dict)

    def test_run_passes_input_to_run(self):
        skill = _EchoSkill()
        result = skill.run({"value": "world"})
        assert result["echo"] == "world"

    def test_run_catches_exceptions(self):
        skill = _BrokenSkill()
        result = skill.run({})
        assert "error" in result

    def test_run_error_includes_skill_name(self):
        skill = _BrokenSkill()
        result = skill.run({})
        assert result["skill"] == "broken_skill"

    def test_run_error_message_in_result(self):
        skill = _BrokenSkill()
        result = skill.run({})
        assert "intentional failure" in result["error"]

    def test_run_empty_input(self):
        skill = _EchoSkill()
        result = skill.run({})
        assert result == {"echo": "nothing"}


# ---------------------------------------------------------------------------
# SkillBase init
# ---------------------------------------------------------------------------

class TestSkillBaseInit:
    def test_brain_stored(self):
        brain = MagicMock()
        skill = _EchoSkill(brain=brain)
        assert skill.brain is brain

    def test_model_stored(self):
        model = MagicMock()
        skill = _EchoSkill(model=model)
        assert skill.model is model

    def test_defaults_none(self):
        skill = _EchoSkill()
        assert skill.brain is None
        assert skill.model is None

    def test_name_attribute(self):
        assert _EchoSkill.name == "echo_skill"
