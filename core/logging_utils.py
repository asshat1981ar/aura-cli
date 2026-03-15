import json
import datetime
import os
import sys
from typing import Any, Optional
from core.redaction import mask_secrets

def _maybe_add(log_entry: dict, key: str, value: Any) -> None:
    if value is None:
        return
    log_entry[key] = value

def log_json(
    level: str,
    event: str,
    goal: str = None,
    details: dict = None,
    *,
    corr_id: str = None,
    phase: str = None,
    skill: str = None,
    component: str = None,
    input_fingerprint: str = None,
    latency_ms: Optional[float] = None,
    retries: Optional[int] = None,
    outcome: str = None,
    failure_reason: str = None,
    payload_bytes: Optional[int] = None,
):
    """
    Emits a single-line structured JSON log with masking and optional telemetry fields.
    Optional fields: corr_id, phase, skill, component, input_fingerprint, latency_ms, retries,
    outcome, failure_reason, payload_bytes.
    """
    safe_details = mask_secrets(details) if details else None

    log_entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": level.upper(),
        "event": event,
    }
    if goal:
        log_entry["goal"] = goal

    _maybe_add(log_entry, "corr_id", corr_id)
    _maybe_add(log_entry, "phase", phase)
    _maybe_add(log_entry, "skill", skill)
    _maybe_add(log_entry, "component", component)
    _maybe_add(log_entry, "input_fingerprint", input_fingerprint)
    _maybe_add(log_entry, "latency_ms", latency_ms)
    _maybe_add(log_entry, "retries", retries)
    _maybe_add(log_entry, "outcome", outcome)
    _maybe_add(log_entry, "failure_reason", failure_reason)
    _maybe_add(log_entry, "payload_bytes", payload_bytes)

    if safe_details:
        log_entry["details"] = safe_details

    if details and "error" in details:
        log_entry["error_context"] = {
            "type": type(details["error"]).__name__ if not isinstance(details["error"], str) else "RuntimeError",
            "message": str(details["error"]),
        }
    
    stream_name = os.getenv("AURA_LOG_STREAM", "stderr").lower()
    stream = sys.stdout if stream_name == "stdout" else sys.stderr
    stream.write(json.dumps(log_entry) + "\n")
    stream.flush() # Ensure the log is written immediately
