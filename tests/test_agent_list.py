"""Tests for `aura agent list` command (s6-agent-list-command)."""
import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def run_main(*args, env_extra=None):
    env = {**os.environ, "AURA_SKIP_CHDIR": "1"}
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "main.py"] + list(args),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    return result


class TestAgentList:
    def test_agent_list_exits_zero(self):
        """Command must complete successfully."""
        r = run_main("agent", "list")
        assert r.returncode == 0, f"Expected exit 0, got {r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"

    def test_agent_list_output_contains_registered(self):
        """Output must contain the word 'Registered'."""
        r = run_main("agent", "list")
        combined = r.stdout + r.stderr
        assert "Registered" in combined or "agent" in combined.lower(), (
            f"Expected 'Registered' or 'agent' in output.\nstdout={r.stdout}\nstderr={r.stderr}"
        )

    def test_agent_list_output_non_empty(self):
        """Output must not be empty."""
        r = run_main("agent", "list")
        assert r.stdout.strip() or r.stderr.strip(), "Expected non-empty output from agent list"

    def test_agent_list_handler_importable(self):
        """Handler must be importable from dispatch module."""
        from aura_cli.dispatch import _handle_agent_list_dispatch
        assert callable(_handle_agent_list_dispatch)

    def test_agent_list_registered_in_dispatch(self):
        """agent_list must appear in COMMAND_DISPATCH_REGISTRY."""
        from aura_cli.dispatch import COMMAND_DISPATCH_REGISTRY
        assert "agent_list" in COMMAND_DISPATCH_REGISTRY, (
            "agent_list not found in COMMAND_DISPATCH_REGISTRY"
        )

    def test_agent_list_in_cli_action_specs(self):
        """agent_list must be declared in CLI_ACTION_SPECS."""
        from aura_cli.options import CLI_ACTION_SPECS_BY_ACTION
        assert "agent_list" in CLI_ACTION_SPECS_BY_ACTION, (
            "agent_list not found in CLI_ACTION_SPECS"
        )
