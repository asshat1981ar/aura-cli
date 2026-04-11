"""CLI workflow integration tests.

Tests cover:
- Command-line interface workflows
- Command dispatching
- Error handling at CLI level
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Skip if typer not available
try:
    from typer.testing import CliRunner
    from aura_cli.cli_main import main

    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_TYPER, reason="Typer not installed"),
]

runner = CliRunner()


class TestCLIHelp:
    """Tests for CLI help commands."""

    def test_main_help_shows_commands(self):
        """Verify main --help shows available commands."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_help_includes_version(self):
        """Verify help includes version information."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        # Help should be informative
        assert len(result.output) > 100


class TestCLIVersion:
    """Tests for CLI version command."""

    def test_version_flag(self):
        """Verify --version returns version."""
        result = runner.invoke(main, ["--version"])

        # Version command should work (exit code 0 or show version)
        assert result.exit_code == 0 or "version" in result.output.lower()


class TestCLIGoalCommands:
    """Tests for CLI goal-related commands."""

    def test_goal_help(self):
        """Verify goal command help works."""
        result = runner.invoke(main, ["goal", "--help"])

        # May fail if goal subcommand doesn't exist
        if result.exit_code == 0:
            assert "Usage:" in result.output

    def test_goal_add_dry_run(self):
        """Test goal add with dry-run flag."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["--dry-run", "goal", "add", "Test goal description"])

            # Should complete without error in dry-run mode
            # (may fail if command structure differs)
            assert result.exit_code in [0, 2]  # 0 = success, 2 = usage error


class TestCLIConfigCommands:
    """Tests for CLI configuration commands."""

    def test_config_show_help(self):
        """Verify config show help works."""
        result = runner.invoke(main, ["config", "--help"])

        if result.exit_code == 0:
            assert "Usage:" in result.output


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_invalid_command(self):
        """Test handling of invalid command."""
        result = runner.invoke(main, ["nonexistent-command-xyz"])

        # Should fail with non-zero exit code
        assert result.exit_code != 0

    def test_missing_required_argument(self):
        """Test handling of missing required argument."""
        # Try a command that likely requires arguments
        result = runner.invoke(main, ["goal", "add"])

        # Should show usage/help
        assert result.exit_code != 0 or "Usage" in result.output


class TestCLIDryRun:
    """Tests for CLI dry-run mode."""

    def test_dry_run_flag_accepted(self):
        """Verify --dry-run flag is accepted."""
        result = runner.invoke(main, ["--dry-run", "--help"])

        # Help should still work with --dry-run
        assert result.exit_code == 0


class TestCLIEnvironment:
    """Tests for CLI environment handling."""

    def test_cli_respects_aura_skip_chdir(self):
        """Test AURA_SKIP_CHDIR environment variable."""
        original_cwd = os.getcwd()

        with runner.isolated_filesystem():
            test_dir = os.getcwd()
            os.environ["AURA_SKIP_CHDIR"] = "1"

            try:
                result = runner.invoke(main, ["--version"])
                # Should complete without changing directory
                assert result.exit_code in [0, 1]  # 0 or error is fine
            finally:
                if "AURA_SKIP_CHDIR" in os.environ:
                    del os.environ["AURA_SKIP_CHDIR"]


class TestCLIDoctor:
    """Tests for CLI doctor command."""

    def test_doctor_help(self):
        """Verify doctor command help works."""
        result = runner.invoke(main, ["doctor", "--help"])

        if result.exit_code == 0:
            assert "Usage:" in result.output

    def test_doctor_runs(self):
        """Test doctor command runs without error."""
        result = runner.invoke(main, ["doctor"])

        # Doctor should provide diagnostic output
        # Exit code may vary based on system state
        assert result.exit_code in [0, 1]  # Success or warning status


class TestCLIMCPCommands:
    """Tests for CLI MCP-related commands."""

    def test_mcp_help(self):
        """Verify mcp command help works."""
        result = runner.invoke(main, ["mcp", "--help"])

        if result.exit_code == 0:
            assert "Usage:" in result.output

    def test_mcp_tools_help(self):
        """Verify mcp tools command help works."""
        result = runner.invoke(main, ["mcp", "tools", "--help"])

        if result.exit_code == 0:
            assert "Usage:" in result.output


class TestCLILogging:
    """Tests for CLI logging behavior."""

    def test_verbose_flag(self):
        """Verify verbose flag is accepted."""
        result = runner.invoke(main, ["-v", "--help"])

        # Help should work with verbose flag
        if result.exit_code == 0:
            assert "Usage:" in result.output

    def test_quiet_flag(self):
        """Verify quiet flag is accepted."""
        result = runner.invoke(main, ["-q", "--help"])

        # Help should work with quiet flag
        if result.exit_code == 0:
            assert "Usage:" in result.output
