"""Tests for core.correlation — CorrelationManager, TraceContext, CorrelationScope."""
import pytest
from core.correlation import (
    CorrelationManager,
    CorrelationScope,
    TraceContext,
    get_correlation_id,
    set_correlation_id,
)


# ── TraceContext ──────────────────────────────────────────────────────────────

def test_trace_context_to_dict():
    ctx = TraceContext(trace_id="abc123", parent_id="par", span_id="sp1", baggage={"k": "v"})
    d = ctx.to_dict()
    assert d == {"trace_id": "abc123", "parent_id": "par", "span_id": "sp1", "baggage": {"k": "v"}}


def test_trace_context_from_dict_round_trip():
    data = {"trace_id": "t1", "parent_id": "p1", "span_id": "s1", "baggage": {"env": "test"}}
    ctx = TraceContext.from_dict(data)
    assert ctx.trace_id == "t1"
    assert ctx.parent_id == "p1"
    assert ctx.span_id == "s1"
    assert ctx.baggage == {"env": "test"}


def test_trace_context_from_dict_missing_keys():
    ctx = TraceContext.from_dict({})
    assert len(ctx.trace_id) == 16
    assert ctx.parent_id is None


def test_trace_context_child_inherits_trace_id():
    parent = TraceContext(trace_id="root", span_id="span0", baggage={"a": "1"})
    child = parent.child(b="2")
    assert child.trace_id == "root"
    assert child.parent_id == "span0"
    assert child.baggage == {"a": "1", "b": "2"}


# ── CorrelationManager ────────────────────────────────────────────────────────

def test_correlation_manager_set_get():
    token = CorrelationManager.set("test-trace-1")
    try:
        assert CorrelationManager.get() == "test-trace-1"
    finally:
        CorrelationManager.reset(token)


def test_correlation_manager_reset_restores_none():
    original = CorrelationManager.get()
    token = CorrelationManager.set("temp-id")
    CorrelationManager.reset(token)
    assert CorrelationManager.get() == original


def test_correlation_manager_scope_cm():
    with CorrelationManager.scope("scope-abc"):
        assert CorrelationManager.get() == "scope-abc"
    assert CorrelationManager.get() is None


def test_correlation_manager_scope_auto_generates_id():
    with CorrelationManager.scope() as scope:
        assert CorrelationManager.get() is not None
        assert len(CorrelationManager.get()) == 16
    assert CorrelationManager.get() is None


def test_correlation_manager_generate():
    gid = CorrelationManager.generate()
    assert isinstance(gid, str)
    assert len(gid) == 16


def test_correlation_manager_get_current_context_when_none():
    token = CorrelationManager.set(None)
    try:
        ctx = CorrelationManager.get_current_context()
        assert isinstance(ctx, TraceContext)
        assert ctx.trace_id is not None
    finally:
        CorrelationManager.reset(token)


def test_correlation_manager_get_current_context_with_id():
    token = CorrelationManager.set("xyz-trace")
    try:
        ctx = CorrelationManager.get_current_context()
        assert ctx.trace_id == "xyz-trace"
    finally:
        CorrelationManager.reset(token)


# ── CorrelationScope ──────────────────────────────────────────────────────────

def test_correlation_scope_enter_exit():
    scope = CorrelationScope("manual-trace")
    assert CorrelationManager.get() is None
    scope.__enter__()
    assert CorrelationManager.get() == "manual-trace"
    scope.__exit__(None, None, None)
    assert CorrelationManager.get() is None


# ── Convenience functions ─────────────────────────────────────────────────────

def test_convenience_get_set():
    token = set_correlation_id("conv-trace")
    try:
        assert get_correlation_id() == "conv-trace"
    finally:
        CorrelationManager.reset(token)
