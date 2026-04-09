"""Telemetry and observability integration for AURA.

Provides LangSmith tracing, metrics collection, and performance monitoring.
All features are optional and behind feature flags.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from functools import wraps

from core.logging_utils import log_json
from core.correlation import get_correlation_id

# Optional LangSmith integration
langsmith_available = False
try:
    from langsmith import Client as LangSmithClient
    from langsmith.run_trees import RunTree
    langsmith_available = True
except ImportError:
    LangSmithClient = None
    RunTree = None


@dataclass
class SpanContext:
    """Context for a telemetry span.
    
    Attributes:
        name: Span name
        span_id: Unique span identifier
        trace_id: Parent trace ID
        start_time: Unix timestamp when span started
        metadata: Additional span metadata
    """
    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id or get_correlation_id(),
            "start_time": self.start_time,
            "metadata": self.metadata,
        }


@dataclass
class TelemetryEvent:
    """A telemetry event for tracking operations.
    
    Attributes:
        event_type: Type of event (e.g., "llm_call", "tool_use", "goal_start")
        timestamp: Unix timestamp
        duration_ms: Duration in milliseconds
        success: Whether the operation succeeded
        metadata: Event metadata
    """
    event_type: str
    timestamp: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelemetryManager:
    """Manager for telemetry and observability.
    
    Supports multiple backends:
    - LangSmith (optional, requires langsmith package)
    - Local JSON logging (always available)
    - Custom exporters
    
    Usage:
        telemetry = TelemetryManager()
        
        with telemetry.span("goal_execution", goal="refactor code"):
            # Your code here
            pass
            
        # Or manual event tracking
        telemetry.record_event("llm_call", duration_ms=1500, metadata={"model": "gpt-4"})
    """
    
    def __init__(
        self,
        enable_langsmith: Optional[bool] = None,
        langsmith_api_key: Optional[str] = None,
        project_name: str = "aura-cli",
    ):
        """Initialize telemetry manager.
        
        Args:
            enable_langsmith: Force LangSmith on/off (auto-detect if None)
            langsmith_api_key: LangSmith API key (falls back to env var)
            project_name: Project name for LangSmith
        """
        self.project_name = project_name
        self.events: List[TelemetryEvent] = []
        self.exporters: List[Callable[[TelemetryEvent], None]] = []
        
        # Determine if LangSmith should be used
        if enable_langsmith is None:
            enable_langsmith = os.environ.get("AURA_ENABLE_LANGSMITH", "false").lower() == "true"
        
        self.langsmith_client: Optional[Any] = None
        if enable_langsmith and langsmith_available:
            try:
                api_key = langsmith_api_key or os.environ.get("LANGSMITH_API_KEY")
                if api_key:
                    self.langsmith_client = LangSmithClient(api_key=api_key)
                    log_json("INFO", "langsmith_initialized", {"project": project_name})
                else:
                    log_json("WARN", "langsmith_no_api_key")
            except Exception as e:
                log_json("ERROR", "langsmith_init_failed", {"error": str(e)})
        elif enable_langsmith and not langsmith_available:
            log_json("WARN", "langsmith_requested_but_not_available")
    
    @contextmanager
    def span(self, name: str, **metadata):
        """Create a telemetry span context.
        
        Args:
            name: Span name
            **metadata: Additional metadata
            
        Yields:
            SpanContext
        """
        span_ctx = SpanContext(name=name, metadata=metadata)
        trace_id = get_correlation_id()
        if trace_id:
            span_ctx.trace_id = trace_id
        
        # Start LangSmith run if available
        langsmith_run = None
        if self.langsmith_client:
            try:
                langsmith_run = RunTree(
                    name=name,
                    run_type="chain",
                    inputs=metadata,
                    project_name=self.project_name,
                )
                langsmith_run.post()
            except Exception as e:
                log_json("WARN", "langsmith_run_start_failed", {"error": str(e)})
        
        log_json("DEBUG", "telemetry_span_start", {
            "name": name,
            "span_id": span_ctx.span_id,
            "trace_id": span_ctx.trace_id,
        })
        
        try:
            yield span_ctx
            success = True
        except Exception as e:
            success = False
            span_ctx.metadata["error"] = str(e)
            if langsmith_run:
                try:
                    langsmith_run.end(error=str(e))
                    langsmith_run.patch()
                except Exception:
                    pass
            raise
        finally:
            duration_ms = (time.time() - span_ctx.start_time) * 1000
            
            # Record event
            event = TelemetryEvent(
                event_type=f"span:{name}",
                duration_ms=duration_ms,
                success=success,
                metadata={**span_ctx.metadata, "span_id": span_ctx.span_id},
            )
            self._record_event(event)
            
            # End LangSmith run
            if langsmith_run and success:
                try:
                    langsmith_run.end(outputs={"duration_ms": duration_ms})
                    langsmith_run.patch()
                except Exception as e:
                    log_json("WARN", "langsmith_run_end_failed", {"error": str(e)})
            
            log_json("DEBUG", "telemetry_span_end", {
                "name": name,
                "span_id": span_ctx.span_id,
                "duration_ms": duration_ms,
                "success": success,
            })
    
    def record_event(
        self,
        event_type: str,
        duration_ms: Optional[float] = None,
        success: bool = True,
        **metadata
    ) -> None:
        """Record a telemetry event.
        
        Args:
            event_type: Type of event
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            **metadata: Additional metadata
        """
        event = TelemetryEvent(
            event_type=event_type,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata,
        )
        self._record_event(event)
    
    def _record_event(self, event: TelemetryEvent) -> None:
        """Internal method to record event to all exporters."""
        self.events.append(event)
        
        # Call registered exporters
        for exporter in self.exporters:
            try:
                exporter(event)
            except Exception as e:
                log_json("WARN", "telemetry_exporter_failed", {
                    "exporter": exporter.__name__,
                    "error": str(e),
                })
        
        # Log to JSON
        log_json("DEBUG", "telemetry_event", {
            "event_type": event.event_type,
            "duration_ms": event.duration_ms,
            "success": event.success,
            "metadata": event.metadata,
        })
    
    def add_exporter(self, exporter: Callable[[TelemetryEvent], None]) -> None:
        """Add a custom event exporter.
        
        Args:
            exporter: Function that takes a TelemetryEvent
        """
        self.exporters.append(exporter)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics.
        
        Returns:
            Statistics about recorded events
        """
        if not self.events:
            return {"total_events": 0}
        
        total_duration = sum(
            e.duration_ms for e in self.events if e.duration_ms is not None
        )
        success_count = sum(1 for e in self.events if e.success)
        
        # Group by event type
        by_type: Dict[str, int] = {}
        for e in self.events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
        
        return {
            "total_events": len(self.events),
            "success_count": success_count,
            "failure_count": len(self.events) - success_count,
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / len(self.events) if self.events else 0,
            "by_type": by_type,
        }
    
    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()


def traced(name: Optional[str] = None, **default_metadata):
    """Decorator for tracing function execution.
    
    Args:
        name: Span name (defaults to function name)
        **default_metadata: Default metadata for the span
        
    Usage:
        telemetry = TelemetryManager()
        
        @traced("process_goal", telemetry=telemetry)
        def process_goal(goal: str):
            # Your code
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get telemetry instance from kwargs or args
            telemetry = kwargs.get('telemetry') or (
                args[0] if args and isinstance(args[0], TelemetryManager) else None
            )
            
            if telemetry is None:
                # No telemetry, just run the function
                return func(*args, **kwargs)
            
            span_name = name or func.__name__
            metadata = {**default_metadata, "function": func.__name__}
            
            with telemetry.span(span_name, **metadata):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global telemetry instance for convenience
_global_telemetry: Optional[TelemetryManager] = None


def get_telemetry() -> TelemetryManager:
    """Get or create global telemetry manager."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = TelemetryManager()
    return _global_telemetry


def configure_telemetry(
    enable_langsmith: Optional[bool] = None,
    langsmith_api_key: Optional[str] = None,
) -> TelemetryManager:
    """Configure global telemetry manager.
    
    Args:
        enable_langsmith: Enable LangSmith integration
        langsmith_api_key: LangSmith API key
        
    Returns:
        Configured TelemetryManager
    """
    global _global_telemetry
    _global_telemetry = TelemetryManager(
        enable_langsmith=enable_langsmith,
        langsmith_api_key=langsmith_api_key,
    )
    return _global_telemetry
