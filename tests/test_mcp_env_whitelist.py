"""Tests for MCP env_snapshot whitelist security fix (issue #332)."""
from __future__ import annotations

import os
from unittest.mock import patch


class TestEnvSnapshotWhitelist:
    """env_snapshot must not leak arbitrary environment variables."""

    def test_whitelist_constant_exists(self):
        """ENV_WHITELIST must be defined at module level in mcp_server."""
        import tools.mcp_server as mcp_server

        assert hasattr(mcp_server, "ENV_WHITELIST"), (
            "tools/mcp_server.py must define ENV_WHITELIST at module level"
        )
        whitelist = mcp_server.ENV_WHITELIST
        assert isinstance(whitelist, (set, frozenset))
        for key in ("PYTHONPATH", "PATH", "HOME", "USER",
                    "AURA_SKIP_CHDIR", "AURA_ENABLE_SWARM"):
            assert key in whitelist, f"ENV_WHITELIST must contain {key!r}"

    def test_helper_exists(self):
        """_env_snapshot helper must be importable from tools.mcp_server."""
        import tools.mcp_server as mcp_server

        assert hasattr(mcp_server, "_env_snapshot"), (
            "tools/mcp_server.py must define a _env_snapshot(args) helper"
        )

    def test_no_keys_returns_only_whitelisted_vars(self):
        """Without explicit keys, only whitelisted env vars are returned."""
        import tools.mcp_server as mcp_server

        fake_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "USER": "testuser",
            "PYTHONPATH": "",
            "AURA_SKIP_CHDIR": "1",
            "AURA_ENABLE_SWARM": "0",
            "SECRET_API_KEY": "super-secret-value",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "OPENAI_API_KEY": "openai-key",
        }

        with patch.dict(os.environ, fake_env, clear=True):
            result = mcp_server._env_snapshot({})

        snapshot = result["data"]["env"]
        assert "SECRET_API_KEY" not in snapshot, "SECRET_API_KEY must not be in snapshot"
        assert "AWS_SECRET_ACCESS_KEY" not in snapshot
        assert "OPENAI_API_KEY" not in snapshot
        assert snapshot.get("PATH") == "/usr/bin"
        assert snapshot.get("HOME") == "/home/user"
        assert snapshot.get("USER") == "testuser"

    def test_explicit_keys_still_allowed(self):
        """Explicit key request (caller-specified) should still be honored."""
        import tools.mcp_server as mcp_server

        fake_env = {
            "PATH": "/usr/bin",
            "MY_CUSTOM_VAR": "hello",
        }
        with patch.dict(os.environ, fake_env, clear=True):
            result = mcp_server._env_snapshot({"keys": ["PATH", "MY_CUSTOM_VAR"]})

        snapshot = result["data"]["env"]
        assert snapshot.get("PATH") == "/usr/bin"
        assert snapshot.get("MY_CUSTOM_VAR") == "hello"

    def test_whitelist_excludes_secrets_even_if_set(self):
        """Non-whitelisted vars are always excluded when no explicit keys given."""
        import tools.mcp_server as mcp_server

        malicious_env = {k: "value" for k in (
            "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN",
            "DATABASE_URL", "REDIS_PASSWORD"
        )}
        malicious_env["PATH"] = "/usr/bin"

        with patch.dict(os.environ, malicious_env, clear=True):
            result = mcp_server._env_snapshot({})

        snapshot = result["data"]["env"]
        for key in malicious_env:
            if key != "PATH":
                assert key not in snapshot, f"{key} must not appear in snapshot"
        assert "PATH" in snapshot

    def test_missing_whitelisted_key_returns_empty_string(self):
        """Whitelisted vars absent from the real env return empty string."""
        import tools.mcp_server as mcp_server

        with patch.dict(os.environ, {}, clear=True):
            result = mcp_server._env_snapshot({})

        snapshot = result["data"]["env"]
        # All whitelisted keys must be present; absent ones map to ""
        for key in mcp_server.ENV_WHITELIST:
            assert key in snapshot
            assert snapshot[key] == ""
