"""End-to-end CLI tests using fixture project.

These tests exercise the AURA CLI binary via subprocess with:
- AURA_DRY_RUN=1 to avoid real LLM calls
- AURA_SKIP_CHDIR=1 to keep cwd stable
- A copied temp_project fixture so file mutations are isolated
"""

import os
import pytest


class TestCLIBasic:
    """Smoke tests — every command that should work without an API key."""

    def test_help_exits_zero(self, run_aura):
        rc, stdout, stderr = run_aura("--help")
        assert rc == 0, f"--help returned non-zero: rc={rc}\nstdout={stdout}\nstderr={stderr}"
        combined = (stdout + stderr).lower()
        assert "usage" in combined or "aura" in combined, f"Expected 'usage' or 'aura' in output, got:\n{stdout}"

    def test_goal_status_exits_zero(self, run_aura):
        rc, stdout, stderr = run_aura("goal", "status")
        assert rc == 0, f"'goal status' returned non-zero: rc={rc}\nstdout={stdout}\nstderr={stderr}"

    def test_doctor_exits_zero_or_one(self, run_aura):
        """doctor may return 1 if API keys are missing — both are acceptable."""
        rc, stdout, stderr = run_aura("doctor")
        assert rc in (0, 1), f"'doctor' returned unexpected code: rc={rc}\nstdout={stdout}\nstderr={stderr}"

    def test_dry_run_goal_once(self, run_aura):
        """goal once in dry-run mode should not modify files and exit cleanly."""
        rc, stdout, stderr = run_aura("goal", "once", "add docstrings to utils.py", "--dry-run")
        assert rc in (0, 1), f"'goal once --dry-run' returned unexpected code: rc={rc}\nstdout={stdout}\nstderr={stderr}"


class TestCLIDryRunIsolation:
    """Verify that dry-run mode does not mutate the fixture project."""

    def test_dry_run_does_not_modify_utils(self, run_aura, temp_project):
        utils_path = temp_project / "utils.py"
        original = utils_path.read_text()

        run_aura("goal", "once", "add type hints to utils.py", "--dry-run")

        after = utils_path.read_text()
        assert original == after, "dry-run mode unexpectedly modified utils.py"

    def test_dry_run_does_not_modify_calculator(self, run_aura, temp_project):
        calc_path = temp_project / "calculator.py"
        original = calc_path.read_text()

        run_aura("goal", "once", "refactor calculator", "--dry-run")

        after = calc_path.read_text()
        assert original == after, "dry-run mode unexpectedly modified calculator.py"


class TestCLIExitCodes:
    """Verify exit-code contract for various invocations."""

    def test_unknown_subcommand_exits_nonzero(self, run_aura):
        rc, stdout, stderr = run_aura("this-subcommand-does-not-exist")
        assert rc != 0, "Unknown subcommand should exit non-zero"

    def test_help_flag_on_subcommand(self, run_aura):
        rc, stdout, stderr = run_aura("goal", "--help")
        assert rc == 0, f"'goal --help' returned non-zero: rc={rc}\nstdout={stdout}"
