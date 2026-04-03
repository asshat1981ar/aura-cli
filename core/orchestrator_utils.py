"""Utility functions for LoopOrchestrator.

This module contains helper functions and classes that were extracted
from orchestrator.py to reduce its size and improve maintainability.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.logging_utils import log_json


class BeadsSyncLoop:
    """Triggers beads synchronization (dolt push/pull) periodically."""

    EVERY_N = 5

    def __init__(self, beads_skill):
        self._skill = beads_skill
        self._n = 0

    def on_cycle_complete(self, _entry):
        if isinstance(_entry, dict) and bool(_entry.get("dry_run")):
            return
        self._n += 1
        if self._n % self.EVERY_N == 0:
            log_json("INFO", "beads_sync_loop_starting")
            # Try to pull latest changes from remote
            self._skill.run({"cmd": "dolt", "args": ["pull"]})
            # Push local changes to remote
            self._skill.run({"cmd": "dolt", "args": ["push"]})


def analyze_error(error: str, context: Optional[dict] = None) -> Optional[str]:
    """Analyze error and suggest recovery action.
    
    Args:
        error: Error message string
        context: Optional additional context
        
    Returns:
        Suggested recovery action or None
    """
    error_lower = error.lower()
    
    if "syntax" in error_lower or "indent" in error_lower:
        return "syntax_fix"
    elif "import" in error_lower or "module" in error_lower:
        return "dependency_check"
    elif "permission" in error_lower or "access" in error_lower:
        return "permission_fix"
    elif "timeout" in error_lower or "deadline" in error_lower:
        return "retry_with_timeout"
    elif "memory" in error_lower:
        return "memory_optimization"
    
    return None


def generate_cycle_id() -> str:
    """Generate unique cycle identifier."""
    return f"cycle_{uuid.uuid4().hex[:12]}_{int(time.time())}"


def load_json_config(path: Path) -> dict:
    """Load JSON config file safely.
    
    Args:
        path: Path to JSON config file
        
    Returns:
        Loaded config dict or empty dict on error
    """
    if not path.exists():
        return {}
    
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log_json("WARN", "config_load_error", {"path": str(path), "error": str(e)})
        return {}
    except OSError as e:
        log_json("WARN", "config_read_error", {"path": str(path), "error": str(e)})
        return {}


def snapshot_file_state(file_path: str) -> Dict:
    """Capture file state before modification for rollback.
    
    Args:
        file_path: Path to file
        
    Returns:
        Snapshot dict with path, exists flag, and content
    """
    path = Path(file_path)
    snapshot = {
        "path": file_path,
        "existed": path.exists(),
        "content": None,
        "timestamp": time.time(),
    }
    
    if path.exists():
        try:
            snapshot["content"] = path.read_text()
        except (OSError, IOError) as e:
            log_json("WARN", "snapshot_read_error", {"path": file_path, "error": str(e)})
            snapshot["content"] = None
    
    return snapshot


def restore_file_snapshots(snapshots: List[Dict]) -> None:
    """Restore file states from snapshots.
    
    Args:
        snapshots: List of file snapshots
    """
    for snapshot in snapshots:
        path = Path(snapshot["path"])
        
        try:
            if not snapshot["existed"]:
                # File didn't exist before, remove it
                if path.exists():
                    path.unlink()
                    log_json("INFO", "snapshot_rollback_delete", {"path": str(path)})
            elif snapshot["content"] is not None:
                # Restore original content
                path.write_text(snapshot["content"])
                log_json("INFO", "snapshot_rollback_restore", {"path": str(path)})
        except (OSError, IOError) as e:
            log_json("ERROR", "snapshot_rollback_error", {"path": str(path), "error": str(e)})


def normalize_verification_result(verification: Dict) -> Dict:
    """Normalize verification result to standard format.
    
    Args:
        verification: Raw verification result
        
    Returns:
        Normalized verification dict
    """
    if not isinstance(verification, dict):
        return {
            "success": False,
            "passed": 0,
            "failed": 1,
            "errors": ["Invalid verification result format"],
            "details": {},
        }
    
    # Handle different result formats
    if "status" in verification:
        status = verification["status"]
        success = status in ("pass", "passed", "success", True)
    elif "success" in verification:
        success = bool(verification["success"])
    else:
        success = False
    
    return {
        "success": success,
        "passed": verification.get("passed", verification.get("pass_count", 0)),
        "failed": verification.get("failed", verification.get("fail_count", 0)),
        "errors": verification.get("errors", []),
        "details": verification.get("details", {}),
        "message": verification.get("message", ""),
    }


def route_verification_failure(verification: Dict) -> str:
    """Determine routing action for verification failure.
    
    Args:
        verification: Verification result dict
        
    Returns:
        Routing action: 'retry', 'replan', 'escalate', or 'abort'
    """
    errors = verification.get("errors", [])
    details = verification.get("details", {})
    
    # Check for specific error patterns
    error_text = " ".join(str(e) for e in errors).lower()
    
    # Syntax errors -> retry with fix
    if any(kw in error_text for kw in ["syntax", "indent", "parse"]):
        return "retry"
    
    # Test failures -> replan
    if "test" in error_text or verification.get("failed", 0) > 0:
        return "replan"
    
    # Sandbox errors -> escalate
    if "sandbox" in error_text or "execution" in error_text:
        return "escalate"
    
    # Timeout errors -> retry
    if "timeout" in error_text or "deadline" in error_text:
        return "retry"
    
    # Check detail patterns
    failure_modes = details.get("failure_modes", [])
    if failure_modes:
        return "replan"
    
    # Default: retry once then abort
    return "retry"


def calculate_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def format_execution_summary(
    goal: str,
    cycle_id: str,
    stop_reason: str,
    cycles: int,
    phase_outputs: Dict,
    duration_seconds: float,
) -> Dict:
    """Format execution summary for logging and UI.
    
    Args:
        goal: Original goal
        cycle_id: Cycle identifier
        stop_reason: Why execution stopped
        cycles: Number of cycles executed
        phase_outputs: Outputs from each phase
        duration_seconds: Total execution time
        
    Returns:
        Summary dict
    """
    return {
        "goal": goal[:100] + "..." if len(goal) > 100 else goal,
        "cycle_id": cycle_id,
        "stop_reason": stop_reason,
        "cycles": cycles,
        "duration_seconds": round(duration_seconds, 2),
        "phases_completed": list(phase_outputs.keys()),
        "timestamp": time.time(),
    }


def merge_skill_context(base_context: Dict, skill_results: Dict) -> Dict:
    """Merge skill analysis results into context.
    
    Args:
        base_context: Base execution context
        skill_results: Results from skill dispatch
        
    Returns:
        Merged context
    """
    merged = dict(base_context)
    
    if skill_results:
        merged["skill_analysis"] = skill_results
        
        # Extract specific insights
        if "issues" in skill_results:
            merged["known_issues"] = skill_results["issues"]
        if "suggestions" in skill_results:
            merged["implementation_hints"] = skill_results["suggestions"]
    
    return merged


def should_retry_phase(failure_count: int, max_retries: int, error_type: str) -> bool:
    """Determine if phase should be retried.
    
    Args:
        failure_count: Number of failures so far
        max_retries: Maximum allowed retries
        error_type: Type of error encountered
        
    Returns:
        True if should retry
    """
    if failure_count >= max_retries:
        return False
    
    # Non-retryable errors
    non_retryable = ["permission_denied", "not_found", "invalid_config", "auth_error"]
    if error_type in non_retryable:
        return False
    
    return True


def extract_code_from_response(response: Any) -> Optional[str]:
    """Extract code from agent response.
    
    Args:
        response: Agent response (string or dict)
        
    Returns:
        Extracted code or None
    """
    if isinstance(response, str):
        # Look for code blocks
        import re
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        return response
    
    if isinstance(response, dict):
        # Try common code fields
        for key in ["code", "implementation", "content", "result", "output"]:
            if key in response:
                return str(response[key])
    
    return None


def sanitize_goal_for_logging(goal: str, max_length: int = 200) -> str:
    """Sanitize goal for logging (truncate, remove newlines).
    
    Args:
        goal: Original goal string
        max_length: Maximum length
        
    Returns:
        Sanitized goal
    """
    sanitized = goal.replace('\n', ' ').replace('\r', '')
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 3] + "..."
    return sanitized
