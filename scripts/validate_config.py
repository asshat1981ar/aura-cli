#!/usr/bin/env python3
"""
scripts/validate_config.py — Validate AURA CLI configuration files for consistency.

Checks:
  1. aura.config.json against expected schema
  2. settings.json model references exist in config
  3. pyproject.toml has timeout set
  4. .gitignore covers sensitive patterns

Usage:
    python3 scripts/validate_config.py [--strict]
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ValidationError:
    severity: str  # "error", "warning", "info"
    file: str
    message: str

    def __str__(self) -> str:
        icons = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
        return f"{icons.get(self.severity, '?')} [{self.file}] {self.message}"


def validate_aura_config(path: Path) -> List[ValidationError]:
    errors = []
    if not path.exists():
        errors.append(ValidationError("warning", "aura.config.json", "File not found — skipping"))
        return errors

    try:
        with open(path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(ValidationError("error", "aura.config.json", f"Invalid JSON: {e}"))
        return errors

    profiles = config.get("models", config.get("model_profiles", {}))
    if not profiles:
        errors.append(ValidationError("warning", "aura.config.json", "No 'models' or 'model_profiles' key found"))
    elif isinstance(profiles, dict):
        for name, profile in profiles.items():
            if isinstance(profile, dict):
                if "provider" not in profile:
                    errors.append(ValidationError("error", "aura.config.json", f"Model profile '{name}' missing 'provider' field"))
                if "model" not in profile and "model_name" not in profile:
                    errors.append(ValidationError("warning", "aura.config.json", f"Model profile '{name}' missing 'model' field"))
                port = profile.get("port")
                if port is not None and (not isinstance(port, int) or not 1 <= port <= 65535):
                    errors.append(ValidationError("error", "aura.config.json", f"Model profile '{name}' has invalid port: {port}"))
    return errors


def validate_settings(path: Path, config_path: Path) -> List[ValidationError]:
    errors = []
    if not path.exists():
        errors.append(ValidationError("info", "settings.json", "File not found — skipping"))
        return errors

    try:
        with open(path) as f:
            settings = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(ValidationError("error", "settings.json", f"Invalid JSON: {e}"))
        return errors

    config_profiles: set = set()
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            profiles = config.get("models", config.get("model_profiles", {}))
            if isinstance(profiles, dict):
                config_profiles = set(profiles.keys())
        except (json.JSONDecodeError, KeyError):
            pass

    model_refs = settings.get("model_routing", settings.get("models", {}))
    if isinstance(model_refs, dict) and config_profiles:
        for role, model_ref in model_refs.items():
            if isinstance(model_ref, str) and model_ref not in config_profiles:
                errors.append(ValidationError("warning", "settings.json", f"Role '{role}' references model '{model_ref}' not found in aura.config.json"))
    return errors


def validate_pyproject(path: Path) -> List[ValidationError]:
    errors = []
    if not path.exists():
        errors.append(ValidationError("warning", "pyproject.toml", "File not found"))
        return errors

    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                errors.append(ValidationError("info", "pyproject.toml", "Cannot validate: tomllib/tomli not available"))
                return errors

        with open(path, "rb") as f:
            pyproject = tomllib.load(f)

        pytest_config = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {})
        if "timeout" not in pytest_config:
            errors.append(ValidationError("warning", "pyproject.toml", "No global pytest timeout — tests may hang. Add: timeout = 30"))

        fail_under = pyproject.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under", 0)
        if fail_under and fail_under < 15:
            errors.append(ValidationError("warning", "pyproject.toml", f"Coverage fail_under={fail_under}% is very low. Target: ≥15%"))
    except Exception as e:
        errors.append(ValidationError("error", "pyproject.toml", f"Parse error: {e}"))

    return errors


def validate_gitignore(path: Path) -> List[ValidationError]:
    errors = []
    sensitive_patterns = ["aura_auth.db", ".env", "*.pyc", "__pycache__", "aura_auth.db-wal"]
    if not path.exists():
        errors.append(ValidationError("error", ".gitignore", "File not found"))
        return errors

    content = path.read_text()
    for pattern in sensitive_patterns:
        if pattern not in content:
            errors.append(ValidationError("warning", ".gitignore", f"Pattern '{pattern}' not found — sensitive file may be committed"))
    return errors


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate AURA CLI configuration")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    all_errors: List[ValidationError] = []
    print("🔍 Validating AURA CLI configuration...\n")

    all_errors.extend(validate_aura_config(REPO_ROOT / "aura.config.json"))
    all_errors.extend(validate_settings(REPO_ROOT / "settings.json", REPO_ROOT / "aura.config.json"))
    all_errors.extend(validate_pyproject(REPO_ROOT / "pyproject.toml"))
    all_errors.extend(validate_gitignore(REPO_ROOT / ".gitignore"))

    for severity in ["error", "warning", "info"]:
        items = [e for e in all_errors if e.severity == severity]
        if items:
            print(f"\n{'=' * 50}")
            print(f" {severity.upper()}S ({len(items)})")
            print(f"{'=' * 50}")
            for e in items:
                print(f"  {e}")

    error_count = len([e for e in all_errors if e.severity == "error"])
    warning_count = len([e for e in all_errors if e.severity == "warning"])
    print(f"\n{'=' * 50}")
    print(f"Summary: {error_count} errors, {warning_count} warnings")
    print(f"{'=' * 50}")

    if error_count > 0:
        print("\n❌ Validation FAILED")
        return 1
    elif args.strict and warning_count > 0:
        print("\n⚠️  Validation FAILED (strict mode)")
        return 1
    else:
        print("\n✅ Validation PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
