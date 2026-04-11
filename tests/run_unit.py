"""tests/run_unit.py — Fast unit-test runner for the known-safe subset.

Run with:
    python -m pytest tests/run_unit.py         # not useful — use the list below
    python -m pytest $(python tests/run_unit.py) --no-cov --timeout=30 -q

Or import SAFE_UNIT_TESTS for programmatic use.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Known-safe, fast unit tests (no external I/O, complete within 30 s each).
# These are the files exercised by the CI "fast-unit" gate.
# ---------------------------------------------------------------------------
SAFE_UNIT_TESTS: list[str] = [
    "tests/test_auth.py",
    "tests/test_jwt_hardening.py",
    "tests/test_server_api.py",
    "tests/test_sanitizer.py",
    "tests/test_correlation.py",
    "tests/test_config_schema.py",
    "tests/test_sandbox_unit.py",
    "tests/test_cli_exit_codes.py",
    "tests/test_sandbox_violations.py",
    "tests/test_e2e_sandbox_retry.py",
    "tests/test_redis_cache.py",
]

if __name__ == "__main__":
    # Print the list of test paths, one per line, for shell consumption.
    print(" ".join(SAFE_UNIT_TESTS))
