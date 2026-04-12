"""Tests for core/agent_sdk/hooks.py — ToolCallRecord, MetricsCollector, create_hooks."""

import pytest
from core.agent_sdk.hooks import (
    ToolCallRecord,
    MetricsCollector,
    create_hooks,
    get_session_metrics,
    reset_session_metrics,
    _DESTRUCTIVE_TOOLS,
    _REQUIRES_CONTEXT,
)


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------

class TestToolCallRecord:
    def test_fields_stored(self):
        rec = ToolCallRecord(tool_name="Read", elapsed_s=0.1, success=True)
        assert rec.tool_name == "Read"
        assert rec.elapsed_s == 0.1
        assert rec.success is True

    def test_timestamp_auto_set(self):
        rec = ToolCallRecord(tool_name="X", elapsed_s=0.0, success=False)
        assert rec.timestamp > 0


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def test_empty_stats(self):
        mc = MetricsCollector()
        stats = mc.get_stats()
        assert stats == {"tool_calls": {}}

    def test_record_single_tool(self):
        mc = MetricsCollector()
        mc.record_tool_call("Read", 0.5, True)
        stats = mc.get_stats()
        assert "Read" in stats["tool_calls"]
        assert stats["tool_calls"]["Read"]["count"] == 1
        assert stats["tool_calls"]["Read"]["success_rate"] == 1.0

    def test_record_multiple_calls_same_tool(self):
        mc = MetricsCollector()
        mc.record_tool_call("Write", 1.0, True)
        mc.record_tool_call("Write", 0.5, False)
        stats = mc.get_stats()
        call_stats = stats["tool_calls"]["Write"]
        assert call_stats["count"] == 2
        assert call_stats["success_rate"] == 0.5
        assert call_stats["avg_elapsed_s"] == 0.75

    def test_record_multiple_tools(self):
        mc = MetricsCollector()
        mc.record_tool_call("Read", 0.1, True)
        mc.record_tool_call("Bash", 1.2, True)
        stats = mc.get_stats()
        assert "Read" in stats["tool_calls"]
        assert "Bash" in stats["tool_calls"]

    def test_empty_summary(self):
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert summary["total_calls"] == 0
        assert summary["success_rate"] == 0

    def test_summary_with_calls(self):
        mc = MetricsCollector()
        mc.record_tool_call("X", 1.0, True)
        mc.record_tool_call("Y", 0.5, False)
        summary = mc.get_summary()
        assert summary["total_calls"] == 2
        assert summary["total_successes"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["total_elapsed_s"] == 1.5

    def test_all_failures_success_rate_zero(self):
        mc = MetricsCollector()
        mc.record_tool_call("T", 0.1, False)
        mc.record_tool_call("T", 0.2, False)
        stats = mc.get_stats()
        assert stats["tool_calls"]["T"]["success_rate"] == 0.0


# ---------------------------------------------------------------------------
# Session metrics singleton
# ---------------------------------------------------------------------------

class TestSessionMetrics:
    def test_get_session_metrics_returns_collector(self):
        mc = get_session_metrics()
        assert isinstance(mc, MetricsCollector)

    def test_reset_returns_fresh_collector(self):
        mc = reset_session_metrics()
        mc.record_tool_call("X", 0.1, True)
        fresh = reset_session_metrics()
        assert fresh.get_summary()["total_calls"] == 0

    def test_reset_changes_singleton(self):
        mc1 = get_session_metrics()
        reset_session_metrics()
        mc2 = get_session_metrics()
        assert mc1 is not mc2


# ---------------------------------------------------------------------------
# create_hooks
# ---------------------------------------------------------------------------

class TestCreateHooks:
    def test_default_creates_all_hook_sections(self):
        hooks = create_hooks()
        assert "PreToolUse" in hooks
        assert "PostToolUse" in hooks
        assert "PostToolUseFailure" in hooks
        assert "Stop" in hooks

    def test_disable_validation_removes_pre_tool_use(self):
        hooks = create_hooks(enable_validation=False)
        assert "PreToolUse" not in hooks

    def test_disable_metrics_removes_post_hooks(self):
        hooks = create_hooks(enable_metrics=False)
        assert "PostToolUse" not in hooks
        assert "PostToolUseFailure" not in hooks

    def test_stop_hook_always_present(self):
        hooks = create_hooks(enable_validation=False, enable_metrics=False)
        assert "Stop" in hooks

    def test_hooks_are_lists(self):
        hooks = create_hooks()
        for key in hooks:
            assert isinstance(hooks[key], list)

    def test_each_hook_entry_has_matcher_and_hooks(self):
        hooks = create_hooks()
        for entries in hooks.values():
            for entry in entries:
                assert "matcher" in entry
                assert "hooks" in entry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_destructive_tools_contains_bash(self):
        assert "Bash" in _DESTRUCTIVE_TOOLS

    def test_destructive_tools_contains_write(self):
        assert "Write" in _DESTRUCTIVE_TOOLS

    def test_requires_context_is_non_empty(self):
        assert len(_REQUIRES_CONTEXT) > 0
