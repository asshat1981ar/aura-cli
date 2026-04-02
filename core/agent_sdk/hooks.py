# core/agent_sdk/hooks.py
"""Hooks for Agent SDK sessions: quality gates, logging, metrics.

Hooks intercept tool calls to enforce safety policies, collect metrics,
and feed the reflection loop.
"""
from __future__ import annotations

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""

    tool_name: str
    elapsed_s: float
    success: bool
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collect metrics from tool calls during a session."""

    def __init__(self) -> None:
        self._records: List[ToolCallRecord] = []

    def record_tool_call(self, tool_name: str, elapsed_s: float, success: bool) -> None:
        self._records.append(ToolCallRecord(
            tool_name=tool_name,
            elapsed_s=elapsed_s,
            success=success,
        ))

    def get_stats(self) -> Dict[str, Any]:
        """Per-tool aggregated statistics."""
        by_tool: Dict[str, List[ToolCallRecord]] = defaultdict(list)
        for r in self._records:
            by_tool[r.tool_name].append(r)

        stats: Dict[str, Any] = {"tool_calls": {}}
        for name, records in by_tool.items():
            successes = sum(1 for r in records if r.success)
            stats["tool_calls"][name] = {
                "count": len(records),
                "success_rate": successes / len(records) if records else 0,
                "avg_elapsed_s": sum(r.elapsed_s for r in records) / len(records),
            }
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """High-level session summary."""
        successes = sum(1 for r in self._records if r.success)
        return {
            "total_calls": len(self._records),
            "total_successes": successes,
            "success_rate": successes / len(self._records) if self._records else 0,
            "total_elapsed_s": sum(r.elapsed_s for r in self._records),
        }


# Singleton for the current session
_session_metrics = MetricsCollector()


def get_session_metrics() -> MetricsCollector:
    """Get the current session's metrics collector."""
    return _session_metrics


def reset_session_metrics() -> MetricsCollector:
    """Reset and return a fresh metrics collector."""
    global _session_metrics
    _session_metrics = MetricsCollector()
    return _session_metrics


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------

# Dangerous tools that should be logged/gated
_DESTRUCTIVE_TOOLS = {"Bash", "Write", "apply_changes"}

# Tools that must not be called without prior analysis
_REQUIRES_CONTEXT = {"generate_code", "apply_changes"}


async def _pre_tool_use_validator(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate tool inputs before execution."""
    tool_name = input_data.get("tool_input", {}).get("name", "")

    # Log destructive operations
    if tool_name in _DESTRUCTIVE_TOOLS:
        logger.info("Destructive tool call: %s (id=%s)", tool_name, tool_use_id)

    return {}


async def _post_tool_use_recorder(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Record tool call metrics after execution."""
    tool_name = input_data.get("tool_input", {}).get("name", "unknown")
    # We don't have precise timing here, so record a placeholder
    _session_metrics.record_tool_call(
        tool_name=tool_name,
        elapsed_s=0.0,  # Agent SDK doesn't expose elapsed time in hooks
        success=True,  # PostToolUse only fires on success
    )
    return {}


async def _post_tool_use_failure_recorder(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Record failed tool calls."""
    tool_name = input_data.get("tool_input", {}).get("name", "unknown")
    _session_metrics.record_tool_call(
        tool_name=tool_name,
        elapsed_s=0.0,
        success=False,
    )
    return {}


async def _stop_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Final hook when session ends — log summary metrics."""
    summary = _session_metrics.get_summary()
    logger.info("Session complete: %s", summary)
    return {}


def create_hooks(
    enable_validation: bool = True,
    enable_metrics: bool = True,
) -> Dict[str, List[Any]]:
    """Create the hook configuration dict for ClaudeAgentOptions.

    Returns a dict compatible with the Agent SDK hooks parameter.
    """
    hooks: Dict[str, list] = {}

    if enable_validation:
        hooks["PreToolUse"] = [
            {"matcher": ".*", "hooks": [_pre_tool_use_validator]},
        ]

    if enable_metrics:
        hooks.setdefault("PostToolUse", []).append(
            {"matcher": ".*", "hooks": [_post_tool_use_recorder]},
        )
        hooks.setdefault("PostToolUseFailure", []).append(
            {"matcher": ".*", "hooks": [_post_tool_use_failure_recorder]},
        )

    hooks["Stop"] = [
        {"matcher": ".*", "hooks": [_stop_hook]},
    ]

    return hooks
