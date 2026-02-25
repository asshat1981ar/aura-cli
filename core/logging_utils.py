import json
import datetime
import sys

def log_json(level: str, event: str, goal: str = None, details: dict = None):
    """
    Emits a single-line JSON log to stdout.
    
    Args:
        level (str): Log level (e.g., "INFO", "WARN", "ERROR").
        event (str): Short description of the event.
        goal (str, optional): The current goal being processed. Defaults to None.
        details (dict, optional): A dictionary for additional information. Defaults to None.
    """
    log_entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": level.upper(),
        "event": event,
    }
    if goal:
        log_entry["goal"] = goal
    if details:
        log_entry["details"] = details
    
    # Emit to stdout
    sys.stdout.write(json.dumps(log_entry) + "\n")
    sys.stdout.flush() # Ensure the log is written immediately

