from __future__ import annotations

from tools.mcp_server_support import (
    auth_dependency,
    auth_mode_label,
    build_basic_metrics_payload,
    build_tool_metrics,
    resolve_server_port,
)


def test_resolve_server_port_prefers_env(monkeypatch):
    monkeypatch.setenv("MCP_SKILLS_PORT", "9123")

    assert resolve_server_port("skills") == 9123


def test_resolve_server_port_uses_config_when_env_missing(monkeypatch):
    monkeypatch.delenv("MCP_SKILLS_PORT", raising=False)

    with monkeypatch.context() as m:
        from core.config_manager import config as _cfg

        m.setattr(_cfg, "get_mcp_server_port", lambda server_name: 8123)
        assert resolve_server_port("skills") == 8123


def test_auth_dependency_uses_canonical_env(monkeypatch):
    validator = auth_dependency("sadd")
    monkeypatch.setenv("SADD_MCP_TOKEN", "secret")

    assert validator(None, "Bearer secret") == "secret"


def test_auth_mode_label_tracks_shared_auth_state(monkeypatch):
    monkeypatch.setenv("AGENT_API_TOKEN", "secret")

    assert auth_mode_label("dev_tools") == "enabled"


def test_build_tool_metrics_and_basic_payload():
    call_counts = {"a": 3}
    call_errors = {"a": 1, "b": 2}

    assert build_tool_metrics(["a", "b"], call_counts, call_errors) == {
        "a": {"calls": 3, "errors": 1},
        "b": {"calls": 0, "errors": 2},
    }

    payload = build_basic_metrics_payload(
        server_start=0.0,
        tool_names=["a", "b"],
        call_counts=call_counts,
        call_errors=call_errors,
        queue_size=4,
    )
    assert payload["total_calls"] == 3
    assert payload["total_errors"] == 3
    assert payload["queue_size"] == 4
    assert payload["tools"]["b"]["errors"] == 2
