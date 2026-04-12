"""AURA Observability Module — Metrics, tracing, and monitoring."""

from .metrics import (
    MetricsStore,
    MetricType,
    MetricValue,
    get_metrics_store,
    record_agent_execution,
    record_cycle_metrics,
    record_memory_operation,
)
from .tracing import (
    Tracer,
    Span,
    SpanContext,
    get_tracer,
    set_tracer,
    trace_agent_execution,
    trace_block,
    trace_orchestrator_cycle,
    trace_phase_execution,
    traced,
)

__all__ = [
    "MetricsStore",
    "MetricType",
    "MetricValue",
    "get_metrics_store",
    "record_agent_execution",
    "record_cycle_metrics",
    "record_memory_operation",
    "Tracer",
    "Span",
    "SpanContext",
    "get_tracer",
    "set_tracer",
    "trace_agent_execution",
    "trace_block",
    "trace_orchestrator_cycle",
    "trace_phase_execution",
    "traced",
]
