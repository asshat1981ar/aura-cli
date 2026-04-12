"""Tests for structured CLI exit codes."""

import subprocess
import sys
import os


def run_main(*args, env_extra=None):
    env = {**os.environ, "AURA_SKIP_CHDIR": "1"}
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "main.py"] + list(args),
        capture_output=True,
        text=True,
        env=env,
        cwd="/home/westonaaron675/aura-cli",
    )
    return result


class TestExitCodes:
    def test_help_exits_zero(self):
        r = run_main("--help")
        assert r.returncode == 0

    def test_unknown_command_exits_nonzero(self):
        r = run_main("nonexistent-command-xyz")
        assert r.returncode != 0

    def test_doctor_exits_zero_or_one(self):
        r = run_main("doctor")
        assert r.returncode in (0, 1)

    def test_exit_code_constants_importable(self):
        """Verify the exit_codes module exports all expected constants."""
        from aura_cli.exit_codes import (
            EXIT_SUCCESS,
            EXIT_FAILURE,
            EXIT_SANDBOX_ERROR,
            EXIT_APPLY_ERROR,
            EXIT_CANCELLED,
            EXIT_LLM_ERROR,
        )

        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE == 1
        assert EXIT_SANDBOX_ERROR == 2
        assert EXIT_APPLY_ERROR == 3
        assert EXIT_CANCELLED == 4
        assert EXIT_LLM_ERROR == 5

    def test_exit_code_values_distinct(self):
        """Each exit code must be unique."""
        from aura_cli.exit_codes import (
            EXIT_SUCCESS,
            EXIT_FAILURE,
            EXIT_SANDBOX_ERROR,
            EXIT_APPLY_ERROR,
            EXIT_CANCELLED,
            EXIT_LLM_ERROR,
        )

        codes = [EXIT_SUCCESS, EXIT_FAILURE, EXIT_SANDBOX_ERROR, EXIT_APPLY_ERROR, EXIT_CANCELLED, EXIT_LLM_ERROR]
        assert len(codes) == len(set(codes)), "Exit code values must be unique"

    def test_help_subcommand_exits_zero(self):
        r = run_main("help")
        assert r.returncode == 0

    def test_json_help_exits_zero(self):
        r = run_main("--json-help")
        assert r.returncode == 0
