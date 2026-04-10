"""Tests for core.config_schema — ConfigValidator, AuraConfig, validate_config."""
import pytest
from core.config_schema import (
    ConfigValidator,
    validate_config,
    pydantic_available,
)


# ── ConfigValidator.validate ──────────────────────────────────────────────────

def test_validate_valid_config():
    v = ConfigValidator()
    is_valid, _, errors = v.validate({})
    assert is_valid
    assert errors == []


def test_validate_invalid_int_field_legacy(monkeypatch):
    monkeypatch.setattr("core.config_schema.pydantic_available", False)
    v = ConfigValidator()
    v.use_pydantic = False
    is_valid, _, errors = v.validate({"max_iterations": "not-an-int"})
    assert not is_valid
    assert any("max_iterations" in e for e in errors)


def test_validate_invalid_bool_field_legacy(monkeypatch):
    v = ConfigValidator()
    v.use_pydantic = False
    is_valid, _, errors = v.validate({"dry_run": "yes"})
    assert not is_valid
    assert any("dry_run" in e for e in errors)


def test_validate_returns_clean_when_pydantic(monkeypatch):
    if not pydantic_available:
        pytest.skip("pydantic not installed")
    v = ConfigValidator()
    is_valid, validated, errors = v.validate({})
    assert is_valid
    assert isinstance(validated, dict)
    assert errors == []


def test_validate_pydantic_invalid_type():
    if not pydantic_available:
        pytest.skip("pydantic not installed")
    v = ConfigValidator()
    # max_iterations expects int — pass a string to trigger pydantic ValidationError
    is_valid, _, errors = v.validate({"max_iterations": "bad"})
    assert not is_valid
    assert len(errors) > 0


# ── ConfigValidator.get_defaults ─────────────────────────────────────────────

def test_get_defaults_returns_dict():
    v = ConfigValidator()
    defaults = v.get_defaults()
    assert isinstance(defaults, dict)


def test_get_defaults_no_pydantic():
    v = ConfigValidator()
    v.use_pydantic = False
    assert v.get_defaults() == {}


# ── validate_config convenience function ─────────────────────────────────────

def test_validate_config_valid():
    is_valid, errors = validate_config({})
    assert is_valid
    assert errors == []


def test_validate_config_invalid_int():
    is_valid, errors = validate_config({"max_iterations": "oops"})
    if pydantic_available:
        assert not is_valid
    # whether pydantic or legacy, result must be consistent bool
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)


# ── Fallback path (pydantic not available) ───────────────────────────────────

def test_legacy_validate_no_errors():
    v = ConfigValidator()
    v.use_pydantic = False
    errors = v._legacy_validate({"max_iterations": 10, "dry_run": True})
    assert errors == []


def test_legacy_validate_multiple_errors():
    v = ConfigValidator()
    v.use_pydantic = False
    errors = v._legacy_validate({"max_iterations": "bad", "dry_run": "bad"})
    assert len(errors) == 2


# ── validate() – unexpected Exception path (line 197) ────────────────────────

@pytest.mark.skipif(not pydantic_available, reason="pydantic required")
def test_validate_unexpected_exception(monkeypatch):
    """validate() catches generic exceptions and returns errors (line 197)."""
    v = ConfigValidator()

    def boom(**kwargs):
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr("core.config_schema.AuraConfig", boom)
    is_valid, _, errors = v.validate({})
    assert not is_valid
    assert "unexpected boom" in errors[0]


# ── is_pydantic_available() ───────────────────────────────────────────────────

def test_is_pydantic_available_returns_bool():
    """is_pydantic_available() returns a bool consistent with module flag."""
    from core.config_schema import is_pydantic_available
    result = is_pydantic_available()
    assert isinstance(result, bool)
    assert result == pydantic_available
