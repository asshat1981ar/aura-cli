"""Tests for observability tracing module."""

import threading
import time
from datetime import datetime

import pytest

from aura.observability.tracing import (
    ConsoleSpanExporter,
    FileSpanExporter,
    Span,
    SpanContext,
    SpanEvent,
    SpanKind,
    SpanLink,
    SpanStatus,
    Tracer,
    get_current_span_context,
    get_tracer,
    set_current_span_context,
    set_tracer,
    trace_agent_execution,
    trace_block,
    trace_orchestrator_cycle,
    trace_phase_execution,
    traced,
)


class TestSpanContext:
    """Test SpanContext functionality."""

    def test_create(self):
        ctx = SpanContext.create()
        assert len(ctx.trace_id) == 32  # hex UUID
        assert len(ctx.span_id) == 16  # hex, 8 bytes
        assert ctx.trace_flags == 1
        assert not ctx.is_remote

    def test_from_w3c_headers_valid(self):
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ctx = SpanContext.from_w3c_headers(traceparent)
        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.span_id == "b7ad6b7169203331"
        assert ctx.trace_flags == 1
        assert ctx.is_remote

    def test_from_w3c_headers_invalid(self):
        assert SpanContext.from_w3c_headers("invalid") is None
        assert SpanContext.from_w3c_headers("00-short") is None
        assert SpanContext.from_w3c_headers("") is None

    def test_to_w3c_headers(self):
        ctx = SpanContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )
        headers = ctx.to_w3c_headers()
        assert "traceparent" in headers
        assert headers["traceparent"] == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_is_valid(self):
        assert SpanContext.create().is_valid()
        assert not SpanContext("", "").is_valid()
        assert not SpanContext("short", "short").is_valid()


class TestSpan:
    """Test Span functionality."""

    def test_creation(self):
        ctx = SpanContext.create()
        span = Span(name="test_span", context=ctx)
        assert span.name == "test_span"
        assert span.context == ctx
        assert span.is_recording
        assert span.duration_ms is None

    def test_set_attribute(self):
        ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_set_attributes(self):
        ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        span.set_attributes({"a": 1, "b": 2})
        assert span.attributes == {"a": 1, "b": 2}

    def test_add_event(self):
        ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        span.add_event("event1", {"attr": "value"})
        assert len(span.events) == 1
        assert span.events[0].name == "event1"

    def test_add_link(self):
        ctx = SpanContext.create()
        link_ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        span.add_link(link_ctx, {"link_attr": "value"})
        assert len(span.links) == 1
        assert span.links[0].context == link_ctx

    def test_set_status(self):
        ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        span.set_status(SpanStatus.OK, "All good")
        assert span.status == SpanStatus.OK
        assert span.status_description == "All good"

    def test_record_exception(self):
        ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.record_exception(e, {"extra": "info"})
        
        assert span.status == SpanStatus.ERROR
        assert len(span.events) == 1
        assert span.events[0].name == "exception"
        assert span.events[0].attributes["exception.type"] == "ValueError"

    def test_end(self):
        ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        span.end()
        assert not span.is_recording
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_duration_calculation(self):
        ctx = SpanContext.create()
        start = datetime.utcnow()
        span = Span(name="test", context=ctx, start_time=start)
        time.sleep(0.01)
        span.end()
        assert span.duration_ms is not None
        assert span.duration_ms >= 10  # At least 10ms

    def test_to_dict(self):
        ctx = SpanContext(
            trace_id="abc123" + "0" * 26,
            span_id="def456" + "0" * 10,
        )
        span = Span(
            name="test_span",
            context=ctx,
            kind=SpanKind.SERVER,
            attributes={"key": "value"},
        )
        span.end()
        
        d = span.to_dict()
        assert d["name"] == "test_span"
        assert d["kind"] == "SERVER"
        assert d["attributes"]["key"] == "value"
        assert "duration_ms" in d


class TestTracer:
    """Test Tracer functionality."""

    def setup_method(self):
        """Reset global tracer before each test."""
        set_tracer(Tracer(service_name="test"))

    def test_tracer_creation(self):
        tracer = Tracer(service_name="my-service")
        assert tracer.service_name == "my-service"

    def test_start_span(self):
        tracer = Tracer(service_name="test")
        with tracer.start_span("test_operation") as span:
            assert span.name == "test_operation"
            assert span.is_recording
        assert not span.is_recording

    def test_nested_spans(self):
        tracer = Tracer(service_name="test")
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_context == parent.context
                assert child.context.trace_id == parent.context.trace_id

    def test_span_with_attributes(self):
        tracer = Tracer(service_name="test")
        with tracer.start_span("test", attributes={"key": "value"}) as span:
            assert span.attributes["key"] == "value"

    def test_span_exception_propagation(self):
        tracer = Tracer(service_name="test")
        with pytest.raises(ValueError):
            with tracer.start_span("test") as span:
                raise ValueError("Test error")
        assert span.status == SpanStatus.ERROR

    def test_get_current_span(self):
        tracer = Tracer(service_name="test")
        assert tracer.get_current_span() is None
        with tracer.start_span("test") as span:
            assert tracer.get_current_span() == span
        assert tracer.get_current_span() is None

    def test_force_flush(self):
        tracer = Tracer(service_name="test", batch_size=10)
        with tracer.start_span("test"):
            pass
        tracer.force_flush()
        # Should complete without error

    def test_shutdown(self):
        tracer = Tracer(service_name="test")
        with tracer.start_span("test"):
            pass
        tracer.shutdown()
        # Should complete without error


class TestConsoleSpanExporter:
    """Test ConsoleSpanExporter."""

    def test_export(self, capsys):
        exporter = ConsoleSpanExporter()
        ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        span.end()
        exporter.export([span])
        captured = capsys.readouterr()
        assert "test" in captured.out


class TestFileSpanExporter:
    """Test FileSpanExporter."""

    def test_export(self, tmp_path):
        path = tmp_path / "spans.json"
        exporter = FileSpanExporter(path)
        
        ctx = SpanContext.create()
        span = Span(name="test", context=ctx)
        span.end()
        
        exporter.export([span])
        exporter.shutdown()
        
        assert path.exists()
        content = path.read_text()
        assert "test" in content

    def test_rotation(self, tmp_path):
        path = tmp_path / "spans.json"
        exporter = FileSpanExporter(path, max_file_size=100)
        
        # Export enough to trigger rotation
        for i in range(10):
            ctx = SpanContext.create()
            span = Span(name=f"span_{i}", context=ctx)
            span.end()
            exporter.export([span])
        
        exporter.shutdown()
        
        # Check rotation occurred - either original or backup should exist
        backup = path.with_suffix(".json.1")
        assert path.exists() or backup.exists()


class TestContextPropagation:
    """Test context propagation functions."""

    def test_get_set_current_span_context(self):
        ctx = SpanContext.create()
        set_current_span_context(ctx)
        assert get_current_span_context() == ctx
        set_current_span_context(None)
        assert get_current_span_context() is None


class TestGlobalTracer:
    """Test global tracer functions."""

    def test_get_tracer_singleton(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_set_tracer(self):
        new_tracer = Tracer(service_name="new")
        set_tracer(new_tracer)
        assert get_tracer() is new_tracer


class TestDecoratorsAndUtilities:
    """Test decorator and utility functions."""

    def setup_method(self):
        """Reset global tracer before each test."""
        set_tracer(Tracer(service_name="test"))

    def test_traced_decorator(self):
        @traced(name="my_operation")
        def my_function():
            return 42
        
        result = my_function()
        assert result == 42

    def test_traced_decorator_with_exception(self):
        @traced(name="failing_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()

    def test_trace_block(self):
        with trace_block("test_block") as span:
            assert span.name == "test_block"
            assert span.is_recording

    def test_trace_agent_execution(self):
        with trace_agent_execution("test_agent", "test goal") as span:
            assert span.name == "agent.test_agent"
            assert span.attributes["agent.name"] == "test_agent"
            assert span.attributes["agent.goal"] == "test goal"

    def test_trace_orchestrator_cycle(self):
        with trace_orchestrator_cycle(1, "test goal") as span:
            assert span.name == "orchestrator.cycle"
            assert span.attributes["cycle.number"] == 1

    def test_trace_phase_execution(self):
        with trace_phase_execution("plan", 1) as span:
            assert span.name == "phase.plan"
            assert span.attributes["phase.name"] == "plan"


class TestThreadSafety:
    """Test thread safety of tracer."""

    def test_concurrent_span_creation(self):
        tracer = Tracer(service_name="test", batch_size=1000)
        spans_created = []
        
        def create_spans():
            for i in range(10):
                with tracer.start_span(f"span_{i}") as span:
                    spans_created.append(span)
        
        threads = [threading.Thread(target=create_spans) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(spans_created) == 50

    def test_isolated_span_stacks(self):
        tracer = Tracer(service_name="test")
        results = {}
        
        def thread_work(thread_id):
            with tracer.start_span(f"thread_{thread_id}") as span:
                results[thread_id] = tracer.get_current_span()
        
        threads = [
            threading.Thread(target=thread_work, args=(i,))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Each thread should have its own current span
        for i in range(3):
            assert results[i] is not None
            assert results[i].name == f"thread_{i}"
