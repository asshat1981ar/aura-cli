"""Unit tests for core/model_adapter.py — ModelAdapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call
import pytest

from core.model_adapter import ModelAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config_get(key, default=None):
    """Return safe defaults so ModelAdapter.__init__ doesn't hit the FS."""
    mapping = {
        "gemini_cli_path": None,
        "codex_cli_path": None,
        "copilot_cli_path": None,
        "mcp_server_url": None,
        "llm_timeout": 60,
        "semantic_memory": {},
        "model_routing": {},
        "primary_provider": "openrouter",
        "local_model_profiles": {},
        "local_model_routing": {},
    }
    return mapping.get(key, default)


def _make_adapter(**kwargs):
    """Construct a ModelAdapter with all I/O side-effects patched."""
    with (
        patch("core.model_adapter.config.get", side_effect=_minimal_config_get),
        patch("core.model_adapter.log_json"),
        patch("core.model_adapter.resolve_openai_api_key", return_value=None),
        patch("memory.embedding_provider.LocalEmbeddingProvider", MagicMock),
    ):
        return ModelAdapter(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestModelAdapterInit:
    def test_adapter_instantiates(self):
        adapter = _make_adapter()
        assert isinstance(adapter, ModelAdapter)

    def test_router_none_by_default(self):
        adapter = _make_adapter()
        assert adapter.router is None

    def test_cache_db_none_by_default(self):
        adapter = _make_adapter()
        assert adapter.cache_db is None

    def test_allowed_tools_non_empty(self):
        adapter = _make_adapter()
        assert len(adapter.ALLOWED_TOOLS) > 0


# ---------------------------------------------------------------------------
# respond() — basic flow
# ---------------------------------------------------------------------------

class TestRespond:
    def test_respond_returns_string(self):
        adapter = _make_adapter()
        with (
            patch.object(adapter, "_get_cached_response", return_value=None),
            patch.object(adapter, "_save_to_cache"),
            patch.object(adapter, "_call_with_timeout", return_value="hello"),
            patch("core.model_adapter.config.get", side_effect=_minimal_config_get),
        ):
            result = adapter.respond("hi")
        assert isinstance(result, str)
        assert result == "hello"

    def test_respond_returns_cache_hit(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_get_cached_response", return_value="cached result"):
            result = adapter.respond("prompt")
        assert result == "cached result"

    def test_respond_saves_to_cache_on_miss(self):
        adapter = _make_adapter()
        with (
            patch.object(adapter, "_get_cached_response", return_value=None),
            patch.object(adapter, "_save_to_cache") as mock_save,
            patch.object(adapter, "_call_with_timeout", return_value="fresh"),
            patch("core.model_adapter.config.get", side_effect=_minimal_config_get),
        ):
            adapter.respond("prompt")
        mock_save.assert_called_once_with("prompt", "fresh")

    def test_respond_falls_back_through_providers(self):
        """When openrouter/openai/anthropic fail, respond() tries call_local last."""
        adapter = _make_adapter()

        call_order = []

        def timeout_side_effect(fn, *args, **kwargs):
            name = fn.__name__
            call_order.append(name)
            if name == "call_local":
                return "local result"  # last resort succeeds
            raise Exception("provider down")

        with (
            patch.object(adapter, "_get_cached_response", return_value=None),
            patch.object(adapter, "_save_to_cache"),
            patch.object(adapter, "_call_with_timeout", side_effect=timeout_side_effect),
            patch("core.model_adapter.config.get", side_effect=_minimal_config_get),
        ):
            result = adapter.respond("prompt")

        assert "call_openrouter" in call_order
        assert "call_openai" in call_order
        assert "call_local" in call_order
        assert result == "local result"

    def test_respond_no_model_returns_error_sentinel(self):
        """If all providers succeed but return None, respond() returns error sentinel."""
        adapter = _make_adapter()

        def timeout_side_effect(fn, *args, **kwargs):
            if fn.__name__ == "call_local":
                return None  # call_local returns None → triggers error sentinel
            raise Exception("fail")

        with (
            patch.object(adapter, "_get_cached_response", return_value=None),
            patch.object(adapter, "_save_to_cache"),
            patch.object(adapter, "_call_with_timeout", side_effect=timeout_side_effect),
            patch("core.model_adapter.config.get", side_effect=_minimal_config_get),
        ):
            result = adapter.respond("prompt")

        assert "Error" in result

    def test_respond_uses_router_when_attached(self):
        adapter = _make_adapter()
        mock_router = MagicMock()
        mock_router.route.return_value = "routed response"
        adapter.router = mock_router

        with (
            patch.object(adapter, "_get_cached_response", return_value=None),
            patch.object(adapter, "_save_to_cache"),
            patch.object(adapter, "_call_with_timeout", wraps=lambda fn, *a, **kw: fn(*a) if hasattr(fn, "__call__") else fn(*a)),
        ):
            result = adapter.respond("hi")

        assert result == "routed response"

    def test_respond_tool_call_blocked_if_not_in_allowlist(self):
        adapter = _make_adapter()
        tool_response = json.dumps({"tool_code": {"name": "rm_rf", "args": {"path": "/"}}})
        with (
            patch.object(adapter, "_get_cached_response", return_value=None),
            patch.object(adapter, "_save_to_cache"),
            patch.object(adapter, "_call_with_timeout", return_value=tool_response),
            patch("core.model_adapter.config.get", side_effect=_minimal_config_get),
        ):
            result = adapter.respond("run tool")
        assert "not allowed" in result


# ---------------------------------------------------------------------------
# respond_for_role()
# ---------------------------------------------------------------------------

class TestRespondForRole:
    def test_respond_for_role_returns_string(self):
        adapter = _make_adapter()
        with (
            patch.object(adapter, "_resolve_local_profile_name", return_value=None),
            patch.object(adapter, "respond", return_value="role response"),
            patch("core.model_adapter.config.get", side_effect=_minimal_config_get),
        ):
            result = adapter.respond_for_role("plan", "make a plan")
        assert result == "role response"

    def test_respond_for_role_uses_openrouter_when_routing_configured(self):
        adapter = _make_adapter()

        def config_with_routing(key, default=None):
            if key == "model_routing":
                return {"plan": "anthropic/claude-3"}
            if key == "primary_provider":
                return "openrouter"
            return _minimal_config_get(key, default)

        with (
            patch.object(adapter, "_resolve_local_profile_name", return_value=None),
            patch.object(adapter, "_get_cached_response", return_value=None),
            patch.object(adapter, "_save_to_cache"),
            patch.object(adapter, "_call_with_timeout", return_value="openrouter result"),
            patch("core.model_adapter.config.get", side_effect=config_with_routing),
        ):
            result = adapter.respond_for_role("plan", "make a plan")
        assert result == "openrouter result"

    def test_respond_for_role_cache_hit_skips_provider(self):
        adapter = _make_adapter()

        def config_with_routing(key, default=None):
            if key == "model_routing":
                return {"summarize": "openai/gpt-4"}
            if key == "primary_provider":
                return "openrouter"
            return _minimal_config_get(key, default)

        with (
            patch.object(adapter, "_resolve_local_profile_name", return_value=None),
            patch.object(adapter, "_get_cached_response", return_value="cached role resp"),
            patch("core.model_adapter.config.get", side_effect=config_with_routing),
        ):
            result = adapter.respond_for_role("summarize", "summarize this")
        assert result == "cached role resp"

    def test_respond_for_role_falls_back_to_respond_on_error(self):
        adapter = _make_adapter()

        def config_with_routing(key, default=None):
            if key == "model_routing":
                return {"plan": "some/model"}
            if key == "primary_provider":
                return "openrouter"
            return _minimal_config_get(key, default)

        with (
            patch.object(adapter, "_resolve_local_profile_name", return_value=None),
            patch.object(adapter, "_get_cached_response", return_value=None),
            patch.object(adapter, "_save_to_cache"),
            patch.object(adapter, "_call_with_timeout", side_effect=Exception("fail")),
            patch.object(adapter, "respond", return_value="fallback") as mock_respond,
            patch("core.model_adapter.config.get", side_effect=config_with_routing),
            patch("core.model_adapter.log_json"),
        ):
            result = adapter.respond_for_role("plan", "do stuff")
        assert result == "fallback"
        mock_respond.assert_called_once_with("do stuff")


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

class TestContextHelpers:
    def test_estimate_context_budget_doc_goal(self):
        adapter = _make_adapter()
        budget = adapter.estimate_context_budget("write docs", goal_type="docs")
        assert budget == 2000 + len("write docs") // 10

    def test_estimate_context_budget_default(self):
        adapter = _make_adapter()
        budget = adapter.estimate_context_budget("x", goal_type="unknown_type")
        assert budget >= 4000

    def test_compress_context_truncates_long_text(self):
        adapter = _make_adapter()
        text = "a" * 10000
        compressed = adapter.compress_context(text, max_tokens=100)
        assert len(compressed) == 400

    def test_compress_context_no_change_if_short(self):
        adapter = _make_adapter()
        text = "short"
        assert adapter.compress_context(text, max_tokens=1000) == text


# ---------------------------------------------------------------------------
# Telemetry / router attachment
# ---------------------------------------------------------------------------

class TestAttachments:
    def test_set_router_stores_router(self):
        adapter = _make_adapter()
        mock_router = MagicMock()
        with patch("core.model_adapter.log_json"):
            adapter.set_router(mock_router)
        assert adapter.router is mock_router

    def test_set_telemetry_agent_stores_agent(self):
        adapter = _make_adapter()
        agent = MagicMock()
        with patch("core.model_adapter.log_json"):
            adapter.set_telemetry_agent(agent)
        assert adapter.telemetry_agent is agent
