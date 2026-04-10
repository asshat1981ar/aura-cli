"""Correlation ID management for distributed tracing.

Provides trace context propagation across the AURA system,
enabling request lifecycle tracking through logs.
"""

from __future__ import annotations

import uuid
import contextvars
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Context variable for current correlation ID
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)


@dataclass
class TraceContext:
    """Trace context for a request or operation.

    Attributes:
        trace_id: Unique identifier for the trace
        parent_id: Parent trace ID for nested operations
        span_id: Current span identifier
        baggage: Key-value pairs propagated with the trace
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_id: Optional[str] = None
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    baggage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "span_id": self.span_id,
            "baggage": self.baggage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        """Create from dictionary."""
        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())[:16]),
            parent_id=data.get("parent_id"),
            span_id=data.get("span_id", str(uuid.uuid4())[:8]),
            baggage=data.get("baggage", {}),
        )

    def child(self, **baggage) -> "TraceContext":
        """Create a child context for nested operations.

        Args:
            **baggage: Additional baggage to merge

        Returns:
            New TraceContext with this as parent
        """
        merged_baggage = {**self.baggage, **baggage}
        return TraceContext(
            trace_id=self.trace_id,  # Same trace
            parent_id=self.span_id,
            baggage=merged_baggage,
        )


class CorrelationManager:
    """Manager for correlation ID lifecycle.

    Usage:
        # Set correlation ID for current context
        CorrelationManager.set("abc-123")

        # Get current correlation ID
        trace_id = CorrelationManager.get()

        # Use as context manager
        with CorrelationManager.scope("new-trace"):
            # All logs in this block have the new trace ID
            process_request()
    """

    @staticmethod
    def get() -> Optional[str]:
        """Get current correlation ID."""
        return _correlation_id.get()

    @staticmethod
    def set(trace_id: str) -> contextvars.Token:
        """Set correlation ID for current context.

        Args:
            trace_id: The trace ID to set

        Returns:
            Token for restoring previous value
        """
        return _correlation_id.set(trace_id)

    @staticmethod
    def reset(token: contextvars.Token) -> None:
        """Reset correlation ID using token from set()."""
        _correlation_id.reset(token)

    @classmethod
    def scope(cls, trace_id: Optional[str] = None):
        """Context manager for correlation ID scope.

        Args:
            trace_id: Trace ID to use (generates new if None)

        Returns:
            CorrelationScope context manager
        """
        return CorrelationScope(trace_id)

    @staticmethod
    def generate() -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())[:16]

    @staticmethod
    def get_current_context() -> TraceContext:
        """Get full trace context from current correlation ID."""
        trace_id = _correlation_id.get()
        if trace_id:
            return TraceContext(trace_id=trace_id)
        return TraceContext()  # New context


class CorrelationScope:
    """Context manager for correlation ID scoping."""

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or CorrelationManager.generate()
        self.token: Optional[contextvars.Token] = None

    def __enter__(self) -> "CorrelationScope":
        self.token = CorrelationManager.set(self.trace_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.token:
            CorrelationManager.reset(self.token)


def get_correlation_id() -> Optional[str]:
    """Convenience function to get current correlation ID."""
    return CorrelationManager.get()


def set_correlation_id(trace_id: str) -> contextvars.Token:
    """Convenience function to set correlation ID."""
    return CorrelationManager.set(trace_id)
