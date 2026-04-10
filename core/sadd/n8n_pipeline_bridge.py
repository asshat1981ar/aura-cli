"""Bridge connecting SADD workstream execution to n8n P1-P5 pipeline."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class N8nPipelineBridge:
    """Routes SADD workstream events through n8n P1-P5 pipeline.

    Integrates with:
    - P1: Goal complexity classification (fast lane vs dual lane)
    - P2: Quality gate for plan review
    - P3: Pipeline coordinator for complex workstreams
    - P4: Feedback loop after cycle completion
    - P5a: Observability trace store
    - WF-7: Session lifecycle events
    - WF-8: Workstream monitoring
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        n8n_cfg = config.get("n8n_connector", {})
        self.enabled = n8n_cfg.get("enabled", False)
        self.base_url = n8n_cfg.get("base_url", "http://localhost:5678")
        self.timeout = n8n_cfg.get("timeout_seconds", 10.0)

        # Pipeline webhooks
        self.goal_route_url = n8n_cfg.get("goal_route_webhook", "")
        self.quality_gate_url = n8n_cfg.get("quality_gate_webhook", "")
        self.quality_gate_enabled = n8n_cfg.get("quality_gate_enabled", False)
        self.pipeline_coordinator_url = n8n_cfg.get("pipeline_coordinator_webhook", "")
        self.route_complex_through_p3 = n8n_cfg.get("route_complex_through_p3", False)
        self.feedback_url = n8n_cfg.get("feedback_loop_webhook", "")
        self.trace_url = n8n_cfg.get("observability_webhook", "")
        self.trace_enabled = n8n_cfg.get("trace_enabled", True)

        # SADD-specific webhooks
        self.session_manager_url = n8n_cfg.get("session_manager_webhook", "")
        self.workstream_monitor_url = n8n_cfg.get("workstream_monitor_webhook", "")

    def _post(self, url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST JSON to an n8n webhook. Returns parsed response or None on failure."""
        if not url or not self.enabled:
            return None
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode()
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return {"status": resp.status, "body": body}
        except Exception as exc:
            logger.debug("n8n webhook failed: %s -> %s", url, exc)
            return None

    # -- P1: Goal Intelligence Router --

    def route_goal(self, ws_id: str, goal_text: str, tags: list[str] | None = None) -> Dict[str, Any]:
        """Classify workstream complexity via P1. Returns routing decision."""
        result = self._post(self.goal_route_url, {
            "source": "sadd",
            "ws_id": ws_id,
            "goal": goal_text,
            "tags": tags or [],
        })
        if result and result.get("complexity"):
            return result
        # Default: local execution
        return {"complexity": "simple", "lane": "fast", "route": "local"}

    # -- P2: Quality Gate --

    def quality_gate(self, ws_id: str, plan: Dict[str, Any], goal_text: str) -> Dict[str, Any]:
        """Run plan through P2 quality gate. Returns verdict."""
        if not self.quality_gate_enabled:
            return {"verdict": "pass", "skipped": True}
        result = self._post(self.quality_gate_url, {
            "source": "sadd",
            "ws_id": ws_id,
            "goal": goal_text,
            "plan": plan,
        })
        if result and result.get("verdict"):
            return result
        return {"verdict": "pass", "skipped": True}

    # -- P3: Pipeline Coordinator --

    def coordinate_pipeline(self, ws_id: str, goal_text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route complex workstreams through P3 for full pipeline orchestration."""
        if not self.route_complex_through_p3:
            return None
        return self._post(self.pipeline_coordinator_url, {
            "source": "sadd",
            "ws_id": ws_id,
            "goal": goal_text,
            "context": context,
        })

    # -- P4: Feedback Loop --

    def send_feedback(self, ws_id: str, cycle_id: str, passed: bool,
                      phase_outputs: Dict[str, Any]) -> None:
        """Send cycle feedback to P4."""
        self._post(self.feedback_url, {
            "source": "sadd",
            "ws_id": ws_id,
            "cycle_id": cycle_id,
            "passed": passed,
            "verification_status": phase_outputs.get("verification", {}).get("status", "skip"),
            "learnings": phase_outputs.get("reflection", {}).get("learnings", []),
            "quality": phase_outputs.get("quality", {}),
        })

    # -- P5a: Trace Store --

    def trace(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit observability trace to P5a."""
        if not self.trace_enabled:
            return
        self._post(self.trace_url, {
            "source": "sadd",
            "event_type": event_type,
            "timestamp": time.time(),
            **payload,
        })

    # -- WF-7: Session Manager --

    def notify_session(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Send session lifecycle event to WF-7."""
        self._post(self.session_manager_url, {
            "event_type": event_type,
            "source": "sadd",
            **payload,
        })

    # -- WF-8: Workstream Monitor --

    def notify_workstream(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Send workstream telemetry to WF-8."""
        self._post(self.workstream_monitor_url, {
            "event_type": event_type,
            "source": "sadd",
            **payload,
        })
