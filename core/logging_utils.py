import json
import datetime
import os
import sys

def log_json(level: str, event: str, goal: str = None, details: dict = None):
    """
    Emits a single-line JSON log to stderr by default.
    Automatically masks sensitive info in details.
    
    Args:
        level (str): Log level (e.g., "INFO", "WARN", "ERROR").
        event (str): Short description of the event.
        goal (str, optional): The current goal being processed. Defaults to None.
        details (dict, optional): A dictionary for additional information. Defaults to None.
    """
    # R3: Unified Control Plane - Automated Secret Masking
    try:
        from core.sanitizer import mask_secrets
        safe_details = mask_secrets(details) if details else None
    except ImportError:
        safe_details = details

    log_entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": level.upper(),
        "event": event,
    }
    if goal:
        log_entry["goal"] = goal
    if safe_details:
        log_entry["details"] = safe_details
    
    stream_name = os.getenv("AURA_LOG_STREAM", "stderr").lower()
    stream = sys.stdout if stream_name == "stdout" else sys.stderr
    stream.write(json.dumps(log_entry) + "\n")
    stream.flush() # Ensure the log is written immediately
