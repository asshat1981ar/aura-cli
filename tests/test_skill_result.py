"""Tests for SkillResult dataclass and SkillBase.run() integration."""
from __future__ import annotations

from typing import Any, Dict

from agents.skills.base import SkillBase, SkillResult


# --- Concrete test skill ---------------------------------------------------

class _EchoSkill(SkillBase):
    name = "echo"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"echoed": input_data.get("msg", "")}


class _BrokenSkill(SkillBase):
    name = "broken"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("intentional failure")


class _FailureReportingSkill(SkillBase):
    """Skill that returns a status in its data (like test_and_observe)."""
    name = "reporter"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "failure", "details": "tests failed"}


# --- SkillResult unit tests -------------------------------------------------

class TestSkillResult:
    def test_ok_on_success(self):
        r = SkillResult(status="success", skill_name="x", data={"a": 1})
        assert r.ok is True

    def test_ok_on_error(self):
        r = SkillResult(status="error", skill_name="x", error="boom")
        assert r.ok is False

    def test_ok_on_partial(self):
        r = SkillResult(status="partial", skill_name="x", data={"a": 1})
        assert r.ok is True

    def test_as_dict_success(self):
        r = SkillResult(status="success", skill_name="x", data={"k": "v"})
        d = r.as_dict()
        assert d["status"] == "success"
        assert d["skill"] == "x"
        assert d["k"] == "v"
        assert "error" not in d

    def test_as_dict_error(self):
        r = SkillResult(status="error", skill_name="x", error="fail")
        d = r.as_dict()
        assert d["error"] == "fail"

    def test_dict_access(self):
        r = SkillResult(status="success", skill_name="x", data={"k": 42})
        assert r["k"] == 42
        assert r["status"] == "success"
        assert "k" in r

    def test_get_method(self):
        r = SkillResult(status="success", skill_name="x", data={})
        assert r.get("missing", "default") == "default"
        assert r.get("status") == "success"

    def test_duration_default(self):
        r = SkillResult(status="success", skill_name="x")
        assert r.duration_ms == 0.0


# --- SkillBase.run() integration -------------------------------------------

class TestSkillBaseRun:
    def test_success_returns_skill_result(self):
        skill = _EchoSkill()
        result = skill.run({"msg": "hello"})
        assert isinstance(result, SkillResult)
        assert result.ok is True
        assert result.status == "success"
        assert result.skill_name == "echo"
        assert result["echoed"] == "hello"
        assert result.duration_ms > 0

    def test_error_returns_skill_result(self):
        skill = _BrokenSkill()
        result = skill.run({})
        assert isinstance(result, SkillResult)
        assert result.ok is False
        assert result.status == "error"
        assert "intentional failure" in result.error
        assert result.duration_ms > 0

    def test_backward_compat_dict_access(self):
        skill = _EchoSkill()
        result = skill.run({"msg": "test"})
        # Old callers might do result["echoed"] or "error" in result
        assert result["echoed"] == "test"
        assert "error" not in result
        assert result.get("echoed") == "test"

    def test_skill_provided_status_is_preserved(self):
        """Skills like test_and_observe set status='failure' in their data.
        SkillResult should adopt that status rather than defaulting to 'success'."""
        skill = _FailureReportingSkill()
        result = skill.run({})
        assert result.status == "failure"
        assert result["status"] == "failure"
        assert result["details"] == "tests failed"
        # Not an error (skill ran fine), just reporting failure
        assert result.ok is True
