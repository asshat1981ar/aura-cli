"""Tests for core.config_schema — ConfigValidator, AuraConfig, validate_config."""

import pytest
from core.config_schema import (
    ConfigValidator,
    validate_config,
)


# ── ConfigValidator.validate ──────────────────────────────────────────────────


def test_validate_valid_config():
    v = ConfigValidator()
    is_valid, _, errors = v.validate({})
    assert is_valid
    assert errors == []


def test_validate_returns_clean_when_pydantic():
    v = ConfigValidator()
    is_valid, validated, errors = v.validate({})
    assert is_valid
    assert isinstance(validated, dict)
    assert errors == []


def test_validate_pydantic_invalid_type():
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


# ── validate_config convenience function ─────────────────────────────────────


def test_validate_config_valid():
    is_valid, errors = validate_config({})
    assert is_valid
    assert errors == []


def test_validate_config_invalid_int():
    is_valid, errors = validate_config({"max_iterations": "oops"})
    assert not is_valid
    assert isinstance(errors, list)
    assert len(errors) > 0


# ── validate() – unexpected Exception path ───────────────────────────────────


def test_validate_unexpected_exception(monkeypatch):
    """validate() catches generic exceptions and returns errors."""
    v = ConfigValidator()

    def boom(**kwargs):
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr("core.config_schema.AuraConfig", boom)
    is_valid, _, errors = v.validate({})
    assert not is_valid
    assert "unexpected boom" in errors[0]
