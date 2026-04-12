"""Hypothesis profiles and shared fixtures for security tests."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

# Register Hypothesis profiles for different environments
settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=500,
)
settings.register_profile(
    "thorough",
    max_examples=10000,
)

# Default to CI profile; override via HYPOTHESIS_PROFILE env var
_profile = os.environ.get("HYPOTHESIS_PROFILE", "ci")
settings.load_profile(_profile)
