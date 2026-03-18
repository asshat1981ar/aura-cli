"""Tests for MCP common infrastructure and unified server (E3)."""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from tools.mcp_common import (
    CallRequest,
    RateLimiter,
    ToolRegistry,
    ToolResult,
    make_auth_dependency,
)


# ---------------------------------------------------------------------------
# CallRequest / ToolResult
# ---------------------------------------------------------------------------

class TestCallRequest:
    def test_minimal(self):
        req = CallRequest(tool_name="foo")
        assert req.tool_name == "foo"
        assert req.args == {}

    def test_with_args(self):
        req = CallRequest(tool_name="bar", args={"x": 1})
        assert req.args == {"x": 1}


class TestToolResult:
    def test_success(self):
        r = ToolResult(tool_name="t", result={"ok": True}, elapsed_ms=1.5)
        assert r.status == "success"
        assert r.error is None

    def test_error(self):
        r = ToolResult(tool_name="t", status="error", error="boom")
        assert r.status == "error"
        assert r.error == "boom"


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_disabled_by_default(self):
        rl = RateLimiter(limit_per_min=0)
        for _ in range(100):
            rl.check("tok")  # should never raise

    def test_enforces_limit(self):
        rl = RateLimiter(limit_per_min=3)
        rl.check("tok")
        rl.check("tok")
        rl.check("tok")
        with pytest.raises(HTTPException) as exc_info:
            rl.check("tok")
        assert exc_info.value.status_code == 429

    def test_per_token_isolation(self):
        rl = RateLimiter(limit_per_min=2)
        rl.check("a")
        rl.check("a")
        rl.check("b")  # different token, should succeed

    def test_window_expires(self):
        rl = RateLimiter(limit_per_min=1)
        rl.check("tok")
        # Manually age the timestamp
        rl._state["tok"][0] -= 61
        rl.check("tok")  # should succeed after window expired


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

class TestAuthDependency:
    def test_no_token_configured_returns_anon(self):
        auth = make_auth_dependency(token_env_var="MCP_TEST_TOKEN_NONEXISTENT")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_TEST_TOKEN_NONEXISTENT", None)
            result = auth(authorization=None)
            assert result == "anon"

    def test_valid_token(self):
        auth = make_auth_dependency(token_env_var="MCP_TEST_TOKEN")
        with patch.dict(os.environ, {"MCP_TEST_TOKEN": "secret123"}):
            result = auth(authorization="Bearer secret123")
            assert result == "secret123"

    def test_missing_header_raises_401(self):
        auth = make_auth_dependency(token_env_var="MCP_TEST_TOKEN")
        with patch.dict(os.environ, {"MCP_TEST_TOKEN": "secret123"}):
            with pytest.raises(HTTPException) as exc_info:
                auth(authorization=None)
            assert exc_info.value.status_code == 401

    def test_invalid_token_raises_403(self):
        auth = make_auth_dependency(token_env_var="MCP_TEST_TOKEN")
        with patch.dict(os.environ, {"MCP_TEST_TOKEN": "secret123"}):
            with pytest.raises(HTTPException) as exc_info:
                auth(authorization="Bearer wrong")
            assert exc_info.value.status_code == 403

    def test_rate_limiter_integration(self):
        rl = RateLimiter(limit_per_min=1)
        auth = make_auth_dependency(token_env_var="MCP_TEST_TOKEN", rate_limiter=rl)
        with patch.dict(os.environ, {"MCP_TEST_TOKEN": "tok"}):
            auth(authorization="Bearer tok")  # first call ok
            with pytest.raises(HTTPException) as exc_info:
                auth(authorization="Bearer tok")  # second call rate-limited
            assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_dispatch(self):
        reg = ToolRegistry(namespace="test")
        reg.register(
            name="echo",
            description="Echo args",
            input_schema={"type": "object"},
            handler=lambda args: {"echo": args},
        )
        assert "echo" in reg.tool_names
        assert len(reg.descriptors) == 1
        result = reg.dispatch("echo", {"msg": "hi"})
        assert result.status == "success"
        assert result.result == {"echo": {"msg": "hi"}}
        assert result.elapsed_ms >= 0

    def test_dispatch_unknown_raises_404(self):
        reg = ToolRegistry(namespace="test")
        with pytest.raises(HTTPException) as exc_info:
            reg.dispatch("nonexistent", {})
        assert exc_info.value.status_code == 404

    def test_dispatch_handler_exception(self):
        reg = ToolRegistry(namespace="test")
        reg.register("fail", "Fail", {}, handler=lambda args: 1 / 0)
        result = reg.dispatch("fail", {})
        assert result.status == "error"
        assert "division by zero" in result.error

    def test_register_batch(self):
        reg = ToolRegistry(namespace="test")
        descriptors = [
            {"name": "a", "description": "A", "inputSchema": {}},
            {"name": "b", "description": "B", "inputSchema": {}},
        ]
        handlers = {
            "a": lambda args: "a_result",
            "b": lambda args: "b_result",
        }
        reg.register_batch(descriptors, handlers)
        assert set(reg.tool_names) == {"a", "b"}
        assert reg.dispatch("a", {}).result == "a_result"
        assert reg.dispatch("b", {}).result == "b_result"

    def test_namespace_in_log(self):
        reg = ToolRegistry(namespace="myns")
        reg.register("t", "", {}, handler=lambda a: None)
        with patch("tools.mcp_common.log_json") as mock_log:
            reg.dispatch("t", {})
            mock_log.assert_called()
            call_args = mock_log.call_args
            assert call_args[1]["details"]["namespace"] == "myns"
