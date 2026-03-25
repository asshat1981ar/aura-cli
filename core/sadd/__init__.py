"""Sub-Agent Driven Development (SADD) system.

Decomposes design specs into parallel workstreams executed by sub-agents,
each running the full LoopOrchestrator pipeline.
"""

from core.sadd.types import (
    DesignSpec,
    SessionConfig,
    SessionReport,
    WorkstreamOutcome,
    WorkstreamResult,
    WorkstreamSpec,
)

__all__ = [
    "DesignSpec",
    "SessionConfig",
    "SessionReport",
    "WorkstreamOutcome",
    "WorkstreamResult",
    "WorkstreamSpec",
]
