"""Security boundary tests for sanitizer, server auth, and path validation.

Tests command injection vectors, flag bypass attempts, path traversal,
and authentication edge cases.
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.sanitizer import SecurityError, sanitize_command, sanitize_path, get_allowed_commands


# ---------------------------------------------------------------------------
# sanitize_command — allowlist enforcement
# ---------------------------------------------------------------------------

class TestSanitizeCommandAllowlist:
    def test_allows_python3(self):
        sanitize_command(["python3", "-m", "pytest"])

    def test_allows_git(self):
        sanitize_command(["git", "status"])

    def test_blocks_unknown_command(self):
        with pytest.raises(SecurityError, match="not in the allowlist"):
            sanitize_command(["curl", "http://evil.com"])

    def test_blocks_rm(self):
        """rm was removed from the base allowlist for safety."""
        with pytest.raises(SecurityError, match="not in the allowlist"):
            sanitize_command(["rm", "-rf", "/"])

    def test_blocks_bash(self):
        with pytest.raises(SecurityError, match="not in the allowlist"):
            sanitize_command(["bash", "-c", "echo pwned"])

    def test_blocks_sh(self):
        with pytest.raises(SecurityError, match="not in the allowlist"):
            sanitize_command(["sh", "-c", "echo pwned"])

    def test_blocks_wget(self):
        with pytest.raises(SecurityError, match="not in the allowlist"):
            sanitize_command(["wget", "http://evil.com"])

    def test_blocks_nc(self):
        with pytest.raises(SecurityError, match="not in the allowlist"):
            sanitize_command(["nc", "-l", "4444"])

    def test_empty_command_is_noop(self):
        sanitize_command([])

    def test_config_can_add_commands(self):
        """Config security.allowed_commands can extend the allowlist."""
        allowed = get_allowed_commands()
        assert "python3" in allowed
        assert "rm" not in allowed


# ---------------------------------------------------------------------------
# sanitize_command — dangerous flag detection
# ---------------------------------------------------------------------------

class TestSanitizeCommandDangerousFlags:
    def test_blocks_python_dash_c(self):
        with pytest.raises(SecurityError, match="Dangerous argument"):
            sanitize_command(["python3", "-c", "import os; os.system('rm -rf /')"])

    def test_blocks_python_dash_e(self):
        with pytest.raises(SecurityError, match="Dangerous argument"):
            sanitize_command(["python3", "-e", "malicious"])

    def test_blocks_python_exec(self):
        with pytest.raises(SecurityError, match="Dangerous argument"):
            sanitize_command(["python3", "--exec", "malicious"])

    def test_blocks_python_eval(self):
        with pytest.raises(SecurityError, match="Dangerous argument"):
            sanitize_command(["python3", "--eval", "malicious"])

    def test_allows_python_m_pytest(self):
        """python3 -m pytest is the primary test runner and must be allowed."""
        sanitize_command(["python3", "-m", "pytest", "-q"])

    def test_allows_python_m_unittest(self):
        sanitize_command(["python3", "-m", "unittest", "discover"])

    def test_m_must_be_first_arg_to_bypass(self):
        """The -m flag only bypasses dangerous arg check when it's cmd[1]."""
        with pytest.raises(SecurityError, match="Dangerous argument"):
            sanitize_command(["python3", "-c", "evil", "-m", "unittest"])

    def test_m_later_in_args_does_not_bypass(self):
        """Regression test: -m anywhere in list should NOT disable checks."""
        with pytest.raises(SecurityError, match="Dangerous argument"):
            sanitize_command(["python3", "--eval", "evil", "-m"])

    def test_non_python_commands_skip_flag_check(self):
        """Only python commands check for dangerous flags."""
        sanitize_command(["git", "-c", "user.name=test", "commit"])


# ---------------------------------------------------------------------------
# sanitize_command — python variant detection
# ---------------------------------------------------------------------------

class TestSanitizeCommandPythonVariants:
    def test_python_plain(self):
        sanitize_command(["python", "-m", "pytest"])

    def test_python3_10(self):
        sanitize_command(["python3.10", "-m", "pytest"])

    def test_python3_12(self):
        sanitize_command(["python3.12", "-m", "pytest"])

    def test_rejects_python_like_but_different(self):
        """pythonista, python-debug etc. are not standard python names."""
        with pytest.raises(SecurityError, match="not in the allowlist"):
            sanitize_command(["pythonista", "script.py"])


# ---------------------------------------------------------------------------
# sanitize_path — traversal attacks
# ---------------------------------------------------------------------------

class TestSanitizePathTraversal:
    def test_blocks_parent_traversal(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        with pytest.raises(SecurityError, match="escapes project root"):
            sanitize_path("../../etc/passwd", root)

    def test_blocks_absolute_path_outside_root(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        with pytest.raises(SecurityError, match="escapes project root"):
            sanitize_path("/etc/passwd", root)

    def test_blocks_sibling_directory(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        sibling = tmp_path / "secrets"
        sibling.mkdir()
        with pytest.raises(SecurityError, match="escapes project root"):
            sanitize_path(str(sibling / "key.pem"), root)

    def test_blocks_symlink_escape(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        link = root / "escape"
        link.symlink_to(outside)
        with pytest.raises(SecurityError, match="escapes project root"):
            sanitize_path("escape/secret.txt", root)

    def test_allows_valid_relative_path(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        result = sanitize_path("src/main.py", root)
        assert str(result).startswith(str(root.resolve()))

    def test_allows_nested_path(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        result = sanitize_path("src/deep/nested/file.py", root)
        assert str(result).startswith(str(root.resolve()))

    def test_blocks_null_byte_injection(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        # Null bytes in paths should fail (OS-level rejection)
        with pytest.raises((SecurityError, ValueError, OSError)):
            sanitize_path("src/\x00/../../../etc/passwd", root)


# ---------------------------------------------------------------------------
# Server auth — require_auth behavior
# ---------------------------------------------------------------------------

class TestServerAuth:
    """Test the require_auth function from aura_cli/server.py."""

    def test_rejects_when_no_token_configured(self):
        """Server must deny access by default when AGENT_API_TOKEN is unset."""
        from fastapi import HTTPException
        # Import inline to avoid triggering server startup side effects
        with patch.dict(os.environ, {}, clear=False):
            # Ensure both vars are unset
            env = {k: v for k, v in os.environ.items()
                   if k not in ("AGENT_API_TOKEN", "AGENT_API_ALLOW_UNAUTHENTICATED")}
            with patch.dict(os.environ, env, clear=True):
                from aura_cli.server import require_auth
                with pytest.raises(HTTPException) as exc_info:
                    require_auth(authorization=None)
                assert exc_info.value.status_code == 500

    def test_allows_when_unauthenticated_opt_in(self):
        """Explicit opt-in for unauthenticated access."""
        env_patch = {"AGENT_API_ALLOW_UNAUTHENTICATED": "1"}
        with patch.dict(os.environ, env_patch, clear=False):
            # Remove AGENT_API_TOKEN if set
            os.environ.pop("AGENT_API_TOKEN", None)
            from importlib import reload
            import aura_cli.server as srv
            reload(srv)
            result = srv.require_auth(authorization=None)
            assert result is None

    def test_rejects_missing_header_with_token(self):
        from fastapi import HTTPException
        with patch.dict(os.environ, {"AGENT_API_TOKEN": "test-secret-123"}):
            from importlib import reload
            import aura_cli.server as srv
            reload(srv)
            with pytest.raises(HTTPException) as exc_info:
                srv.require_auth(authorization=None)
            assert exc_info.value.status_code == 401

    def test_rejects_wrong_token(self):
        from fastapi import HTTPException
        with patch.dict(os.environ, {"AGENT_API_TOKEN": "test-secret-123"}):
            from importlib import reload
            import aura_cli.server as srv
            reload(srv)
            with pytest.raises(HTTPException) as exc_info:
                srv.require_auth(authorization="Bearer wrong-token")
            assert exc_info.value.status_code == 403

    def test_accepts_correct_token(self):
        with patch.dict(os.environ, {"AGENT_API_TOKEN": "test-secret-123"}):
            from importlib import reload
            import aura_cli.server as srv
            reload(srv)
            result = srv.require_auth(authorization="Bearer test-secret-123")
            assert result is None
