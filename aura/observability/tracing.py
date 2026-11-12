"""Distributed tracing for AURA.

Provides OpenTelemetry-compatible tracing with span context propagation,
allowing tracking of operations across async boundaries and service calls.
"""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Protocol


class SpanStatus(Enum):
    """Span execution status."""
    OK = auto()
    ERROR = auto()
    CANCELLED = auto()
    UNKNOWN = auto()


class SpanKind(Enum):
    """Type of span based on its role in the trace."""
    INTERNAL = auto()    # Internal operation
    SERVER = auto()      # Incoming request handler
    CLIENT = auto()      # Outgoing request
    PRODUCER = auto()    # Message producer
    CONSUMER = auto()    # Message consumer


@dataclass
class SpanContext:
    """Context for trace propagation across boundaries.
    
    Follows W3C Trace Context standard for compatibility.
    """
    trace_id: str
    span_id: str
    trace_flags: int = 1  # 1 = sampled
    trace_state: str = ""
    is_remote: bool = False
    
    @classmethod
    def create(cls) -> SpanContext:
        """Create a new root span context."""
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
        )
    
    @classmethod
    def from_w3c_headers(cls, traceparent: str, tracestate: str = "") -> Optional[SpanContext]:
        """Parse W3C traceparent header.
        
        Format: 00-<trace-id>-<span-id>-<flags>
        """
        try:
            parts = traceparent.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None
            
            return cls(
                trace_id=parts[1],
                span_id=parts[2],
                trace_flags=int(parts[3], 16),
                trace_state=tracestate,
                is_remote=True,
            )
        except (ValueError, IndexError):
            return None
    
    def to_w3c_headers(self) -> Dict[str, str]:
        """Convert to W3C header format."""
        traceparent = f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"
        headers = {"traceparent": traceparent}
        if self.trace_state:
            headers["tracestate"] = self.trace_state
        return headers
    
    def is_valid(self) -> bool:
        """Check if context has valid IDs."""
        return bool(self.trace_id and self.span_id and len(self.trace_id) == 32)


# Context variable for current span context
_current_span_context: ContextVar[Optional[SpanContext]] = ContextVar(
    "current_span_context", default=None
)


def get_current_span_context() -> Optional[SpanContext]:
    """Get the current span context from context vars."""
    return _current_span_context.get()


def set_current_span_context(ctx: Optional[SpanContext]) -> None:
    """Set the current span context."""
    _current_span_context.set(ctx)


@dataclass
class SpanEvent:
    """An event that occurred during a span."""
    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class SpanLink:
    """Link to another span (for batch operations, etc.)."""
    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single operation within a trace.
    
    Spans form a tree structure where each span has a parent
    (except root spans). They track timing, status, and events.
    """
    name: str
    context: SpanContext
    parent_context: Optional[SpanContext] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNKNOWN
    status_description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    
    def __post_init__(self):
        self._lock = threading.Lock()
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate span duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    @property
    def is_recording(self) -> bool:
        """Check if span is still recording."""
        return self.end_time is None
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        with self._lock:
            if self.is_recording:
                self.attributes[key] = value
    
    def set_attributes(self, attrs: Dict[str, Any]) -> None:
        """Set multiple span attributes."""
        with self._lock:
            if self.is_recording:
                self.attributes.update(attrs)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        with self._lock:
            if self.is_recording:
                self.events.append(SpanEvent(name=name, attributes=attributes or {}))
    
    def add_link(self, context: SpanContext, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a link to another span."""
        with self._lock:
            if self.is_recording:
                self.links.append(SpanLink(context=context, attributes=attributes or {}))
    
    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set span status."""
        with self._lock:
            if self.is_recording:
                self.status = status
                self.status_description = description
    
    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an exception as a span event."""
        attrs = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
        }
        if attributes:
            attrs.update(attributes)
        self.add_event("exception", attrs)
        self.set_status(SpanStatus.ERROR, str(exception))
    
    def end(self, end_time: Optional[datetime] = None) -> None:
        """End the span."""
        with self._lock:
            if self.is_recording:
                self.end_time = end_time or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "name": self.name,
            "context": {
                "trace_id": self.context.trace_id,
                "span_id": self.context.span_id,
            },
            "parent_span_id": self.parent_context.span_id if self.parent_context else None,
            "kind": self.kind.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.name,
            "status_description": self.status_description,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
        }


class SpanExporter(Protocol):
    """Protocol for span exporters."""
    
    def export(self, spans: List[Span]) -> None:
        """Export spans to backend."""
        ...
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        ...


class ConsoleSpanExporter:
    """Export spans to console (for debugging)."""
    
    def export(self, spans: List[Span]) -> None:
        for span in spans:
            duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "incomplete"
            status_icon = "✓" if span.status == SpanStatus.OK else "✗" if span.status == SpanStatus.ERROR else "?"
            indent = "  " * self._get_depth(span)
            print(f"{indent}[{status_icon}] {span.name} ({duration})")
    
    def _get_depth(self, span: Span) -> int:
        """Calculate span depth in trace tree."""
        # Simplified - in real implementation would track from root
        return 0
    
    def shutdown(self) -> None:
        pass


class FileSpanExporter:
    """Export spans to JSON file."""
    
    def __init__(self, path: Path, max_file_size: int = 10 * 1024 * 1024):
        self.path = path
        self.max_file_size = max_file_size
        self._lock = threading.Lock()
    
    def export(self, spans: List[Span]) -> None:
        with self._lock:
            try:
                # Append to file
                mode = "a" if self.path.exists() else "w"
                with open(self.path, mode) as f:
                    for span in spans:
                        f.write(json.dumps(span.to_dict()) + "\n")
                
                # Rotate if too large
                if self.path.stat().st_size > self.max_file_size:
                    self._rotate()
            except IOError:
                pass
    
    def _rotate(self) -> None:
        """Rotate log file."""
        backup = self.path.with_suffix(".json.1")
        if backup.exists():
            backup.unlink()
        self.path.rename(backup)
    
    def shutdown(self) -> None:
        pass


class Tracer:
    """Main tracer for creating and managing spans.
    
    Thread-safe tracer that supports nested spans and context propagation.
    
    Example:
        >>> tracer = Tracer(service_name="aura-agent")
        >>> with tracer.start_span("process_goal") as span:
        ...     span.set_attribute("goal.id", "123")
        ...     with tracer.start_span("plan") as child_span:
        ...         do_planning()
    """
    
    def __init__(
        self,
        service_name: str = "aura",
        exporter: Optional[SpanExporter] = None,
        batch_size: int = 100,
        max_queue_size: int = 1000,
    ):
        self.service_name = service_name
        self.exporter = exporter or ConsoleSpanExporter()
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        
        self._spans: List[Span] = []
        self._span_stack: threading.local = threading.local()
        self._lock = threading.RLock()
        self._stopped = False
    
    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        context: Optional[SpanContext] = None,
    ) -> Generator[Span, None, None]:
        """Start a new span (context manager).
        
        Args:
            name: Span name (operation being performed)
            kind: Type of span
            attributes: Initial span attributes
            context: Parent context (None = use current context)
        
        Yields:
            The created span
        """
        # Get parent context
        if context is None:
            context = get_current_span_context()
        
        # Create new span context
        new_context = SpanContext(
            trace_id=context.trace_id if context else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
        )
        
        # Create span
        span = Span(
            name=name,
            context=new_context,
            parent_context=context,
            kind=kind,
            attributes=attributes or {},
        )
        
        # Set as current context
        token = _current_span_context.set(new_context)
        
        # Track in stack for this thread
        if not hasattr(self._span_stack, 'stack'):
            self._span_stack.stack = []
        self._span_stack.stack.append(span)
        
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._span_stack.stack.pop()
            _current_span_context.reset(token)
            self._export_span(span)
    
    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Start span and automatically set as current context."""
        with self.start_span(name, kind, attributes) as span:
            yield span
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current span for this thread."""
        if hasattr(self._span_stack, 'stack') and self._span_stack.stack:
            return self._span_stack.stack[-1]
        return None
    
    def _export_span(self, span: Span) -> None:
        """Queue span for export."""
        with self._lock:
            if self._stopped:
                return
            
            self._spans.append(span)
            
            if len(self._spans) >= self.batch_size:
                self._flush()
    
    def _flush(self) -> None:
        """Export queued spans."""
        spans_to_export = self._spans[:self.batch_size]
        self._spans = self._spans[self.batch_size:]
        
        try:
            self.exporter.export(spans_to_export)
        except Exception:
            # Don't crash on export failure
            pass
    
    def force_flush(self) -> None:
        """Immediately export all queued spans."""
        with self._lock:
            if self._spans:
                self.exporter.export(self._spans[:])
                self._spans.clear()
    
    def shutdown(self) -> None:
        """Shutdown tracer and flush remaining spans."""
        with self._lock:
            self._stopped = True
            self.force_flush()
            self.exporter.shutdown()


# Global tracer instance
_global_tracer: Optional[Tracer] = None
_tracer_lock = threading.Lock()


def get_tracer(service_name: str = "aura") -> Tracer:
    """Get or create global tracer."""
    global _global_tracer
    
    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = Tracer(service_name=service_name)
    
    return _global_tracer


def set_tracer(tracer: Tracer) -> None:
    """Set the global tracer."""
    global _global_tracer
    with _tracer_lock:
        _global_tracer = tracer


# Convenience decorators and context managers
def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to trace function execution.
    
    Example:
        >>> @traced(name="my_operation")
        ... def my_function():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_span(span_name, kind, attributes) as span:
                span.set_attribute("function.args_count", len(args))
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def trace_block(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Context manager for tracing a block of code.
    
    Example:
        >>> with trace_block("database_query"):
        ...     db.execute(query)
    """
    tracer = get_tracer()
    with tracer.start_span(name, kind, attributes) as span:
        yield span


# AURA-specific tracing utilities
def trace_agent_execution(agent_name: str, goal: str):
    """Context manager for tracing agent execution."""
    return trace_block(
        name=f"agent.{agent_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            "agent.name": agent_name,
            "agent.goal": goal[:100],  # Truncate long goals
        },
    )


def trace_orchestrator_cycle(cycle_number: int, goal: str):
    """Context manager for tracing orchestrator cycle."""
    return trace_block(
        name="orchestrator.cycle",
        kind=SpanKind.SERVER,
        attributes={
            "cycle.number": cycle_number,
            "cycle.goal": goal[:100],
        },
    )


def trace_phase_execution(phase_name: str, cycle_number: int):
    """Context manager for tracing pipeline phase."""
    return trace_block(
        name=f"phase.{phase_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            "phase.name": phase_name,
            "phase.cycle": cycle_number,
        },
    )
