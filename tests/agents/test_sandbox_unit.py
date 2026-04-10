"""Unit tests for agents/sandbox.py module.

Sprint 4: Unit tests for sandbox module — filesystem restrictions,
resource limits, network blocking, and violation detection.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.sandbox import (
    SandboxAgent,
    SandboxResult,
    _SANDBOX_NETWORK_ENV,
    _SANDBOX_FS_ENV,
    _is_path_allowed,
    _wrap_code_with_fs_restrictions,
    _set_resource_limits,
    _VIOLATION_PATTERNS,
)


class TestSandboxNetworkEnv:
    """Tests for network blocking environment variables."""

    def test_network_env_blocks_http(self):
        """Should route HTTP traffic to non-existent proxy."""
        assert _SANDBOX_NETWORK_ENV["http_proxy"] == "http://127.0.0.1:1"
        assert _SANDBOX_NETWORK_ENV["HTTP_PROXY"] == "http://127.0.0.1:1"

    def test_network_env_blocks_https(self):
        """Should route HTTPS traffic to non-existent proxy."""
        assert _SANDBOX_NETWORK_ENV["https_proxy"] == "http://127.0.0.1:1"
        assert _SANDBOX_NETWORK_ENV["HTTPS_PROXY"] == "http://127.0.0.1:1"

    def test_network_env_clears_no_proxy(self):
        """Should clear no_proxy to ensure all traffic is routed."""
        assert _SANDBOX_NETWORK_ENV["no_proxy"] == ""
        assert _SANDBOX_NETWORK_ENV["NO_PROXY"] == ""

    def test_network_env_disables_ssl_verify(self):
        """Should prevent SSL verification fallbacks."""
        assert _SANDBOX_NETWORK_ENV["REQUESTS_CA_BUNDLE"] == ""


class TestSandboxFsEnv:
    """Tests for filesystem restriction environment variables."""

    def test_fs_env_disables_user_site(self):
        """Should disable user site-packages."""
        assert _SANDBOX_FS_ENV["PYTHONNOUSERSITE"] == "1"

    def test_fs_env_disables_bytecode(self):
        """Should disable bytecode writing."""
        assert _SANDBOX_FS_ENV["PYTHONDONTWRITEBYTECODE"] == "1"

    def test_fs_env_clears_pythonpath(self):
        """Should clear PYTHONPATH to prevent injection."""
        assert _SANDBOX_FS_ENV["PYTHONPATH"] == ""


class TestIsPathAllowed:
    """Tests for path allowlist checking."""

    def test_allows_temp_directory(self):
        """Should allow paths in temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _is_path_allowed(f"{tmpdir}/file.py", tmpdir) is True

    def test_allows_nested_temp_path(self):
        """Should allow nested paths in temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = f"{tmpdir}/subdir/nested/file.py"
            assert _is_path_allowed(nested, tmpdir) is True

    def test_rejects_outside_temp(self):
        """Should reject paths outside temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outside = "/etc/passwd"
            assert _is_path_allowed(outside, tmpdir) is False

    def test_rejects_parent_traversal(self):
        """Should reject paths with parent directory traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = f"{tmpdir}/../../../etc/passwd"
            assert _is_path_allowed(traversal, tmpdir) is False

    def test_allows_system_tmp(self):
        """Should allow paths in /tmp."""
        assert _is_path_allowed("/tmp/test.py", "/some/cwd") is True

    def test_allows_var_tmp(self):
        """Should allow paths in /var/tmp."""
        assert _is_path_allowed("/var/tmp/test.py", "/some/cwd") is True


class TestWrapCodeWithFsRestrictions:
    """Tests for code wrapping with filesystem restrictions."""

    def test_wrap_adds_import_guard(self):
        """Should add import guard at start of code."""
        code = "print('hello')"
        wrapped = _wrap_code_with_fs_restrictions(code, "/tmp")
        
        assert "import builtins" in wrapped
        assert "import os" in wrapped
        assert "import sys" in wrapped

    def test_wrap_adds_sandbox_open(self):
        """Should add sandbox_open function."""
        code = "print('hello')"
        wrapped = _wrap_code_with_fs_restrictions(code, "/tmp")
        
        assert "def _sandbox_open" in wrapped
        assert "_original_open" in wrapped

    def test_wrap_preserves_original_code(self):
        """Should preserve original code after guard."""
        code = "x = 1 + 2\nprint(x)"
        wrapped = _wrap_code_with_fs_restrictions(code, "/tmp")
        
        assert code in wrapped

    def test_wrap_sets_allowed_prefixes(self):
        """Should set allowed prefixes including cwd."""
        code = "pass"
        wrapped = _wrap_code_with_fs_restrictions(code, "/custom/cwd")
        
        assert "'/custom/cwd'" in wrapped
        assert '"/tmp"' in wrapped

    def test_wrap_filters_sys_path(self):
        """Should filter sys.path to remove unsafe entries."""
        code = "pass"
        wrapped = _wrap_code_with_fs_restrictions(code, "/tmp")
        
        assert "sys.path = [p for p in sys.path" in wrapped


class TestSetResourceLimits:
    """Tests for resource limit setting."""

    def test_sets_cpu_limit(self):
        """Should set CPU time limit to 30 seconds."""
        # This test can only run on Unix systems with resource module
        pytest.importorskip("resource", reason="resource module only on Unix")
        
        # Just verify the function doesn't crash
        _set_resource_limits()

    def test_handles_import_error_gracefully(self):
        """Should handle missing resource module on Windows."""
        # On Windows, this should just return without error
        _set_resource_limits()
        # No assertion needed - just checking it doesn't raise


class TestViolationPatterns:
    """Tests for violation detection patterns."""

    def test_detects_permission_error(self):
        """Should detect PermissionError in output."""
        pattern = next(p for p, _ in _VIOLATION_PATTERNS if "PermissionError" in p.pattern)
        assert pattern.search("PermissionError: [Errno 13]") is not None

    def test_detects_import_error(self):
        """Should detect ImportError in output."""
        pattern = next(p for p, _ in _VIOLATION_PATTERNS if "ImportError" in p.pattern)
        assert pattern.search("ImportError: No module named 'x'") is not None

    def test_detects_module_not_found_error(self):
        """Should detect ModuleNotFoundError in output."""
        pattern = next(p for p, _ in _VIOLATION_PATTERNS if "ModuleNotFoundError" in p.pattern)
        assert pattern.search("ModuleNotFoundError: No module named 'y'") is not None

    def test_detects_security_error(self):
        """Should detect SecurityError in output."""
        pattern = next(p for p, _ in _VIOLATION_PATTERNS if "SecurityError" in p.pattern)
        assert pattern.search("SecurityError: Access denied") is not None

    def test_detects_access_denied(self):
        """Should detect 'Access denied' in output."""
        pattern = next(p for p, _ in _VIOLATION_PATTERNS if "Access denied" in p.pattern)
        assert pattern.search("Access denied to resource") is not None


class TestSandboxResult:
    """Tests for SandboxResult dataclass."""

    def test_passed_property_success(self):
        """passed should be True for success with no timeout."""
        result = SandboxResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
        )
        assert result.passed is True

    def test_passed_property_failure(self):
        """passed should be False for failed execution."""
        result = SandboxResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="error",
            timed_out=False,
        )
        assert result.passed is False

    def test_passed_property_timeout(self):
        """passed should be False for timed out execution."""
        result = SandboxResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="timeout",
            timed_out=True,
        )
        assert result.passed is False

    def test_summary_format(self):
        """summary() should return formatted string."""
        result = SandboxResult(
            success=True,
            exit_code=0,
            stdout="output",
            stderr="",
        )
        summary = result.summary()
        assert "[PASS]" in summary
        assert "exit=0" in summary
        assert "stdout=6c" in summary

    def test_summary_timeout(self):
        """summary() should indicate timeout."""
        result = SandboxResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="timeout",
            timed_out=True,
        )
        summary = result.summary()
        assert "[TIMEOUT]" in summary


class TestSandboxAgentInitialization:
    """Tests for SandboxAgent initialization."""

    def test_default_timeout(self):
        """Should use default timeout of 30."""
        brain = MagicMock()
        agent = SandboxAgent(brain=brain)
        assert agent.timeout == 30

    def test_custom_timeout(self):
        """Should accept custom timeout."""
        brain = MagicMock()
        agent = SandboxAgent(brain=brain, timeout=60)
        assert agent.timeout == 60

    def test_default_python_exec(self):
        """Should use sys.executable by default."""
        brain = MagicMock()
        agent = SandboxAgent(brain=brain)
        import sys
        assert agent.python_exec == sys.executable

    def test_custom_python_exec(self):
        """Should accept custom python executable path."""
        brain = MagicMock()
        agent = SandboxAgent(brain=brain, python_exec="/usr/bin/python3.11")
        assert agent.python_exec == "/usr/bin/python3.11"


class TestSandboxAgentRunCode:
    """Tests for SandboxAgent.run_code method."""

    def test_run_code_creates_temp_file(self):
        """Should create temp file with wrapped code."""
        brain = MagicMock()
        agent = SandboxAgent(brain=brain, timeout=10)
        
        with patch.object(agent, '_run') as mock_run:
            mock_run.return_value = SandboxResult(
                success=True,
                exit_code=0,
                stdout="hello",
                stderr="",
            )
            
            result = agent.run_code("print('hello')")
            
            assert result.success is True
            mock_run.assert_called_once()
            # Check that path ends with aura_exec.py
            call_args = mock_run.call_args
            assert "aura_exec.py" in call_args[0][0]

    def test_run_code_wraps_with_fs_restrictions(self):
        """Should wrap code with filesystem restrictions."""
        brain = MagicMock()
        agent = SandboxAgent(brain=brain, timeout=10)
        
        with patch.object(agent, '_run') as mock_run:
            mock_run.return_value = SandboxResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
            )
            
            agent.run_code("x = 1")
            
            # The temp file should contain wrapped code
            # We can't directly check, but the _run should be called
            mock_run.assert_called_once()

    def test_run_code_records_to_brain(self):
        """Should record result to brain memory."""
        brain = MagicMock()
        agent = SandboxAgent(brain=brain, timeout=10)
        
        with patch.object(agent, '_run') as mock_run:
            mock_run.return_value = SandboxResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
            )
            
            agent.run_code("pass")
            
            brain.remember.assert_called_once()
            call_args = brain.remember.call_args[0][0]
            assert "SandboxAgent" in call_args
