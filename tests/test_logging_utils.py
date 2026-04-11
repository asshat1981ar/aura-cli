"""Tests for core/logging_utils.py — log_json."""

import io
import json
import pytest
from unittest.mock import patch


def _capture_log(level, event, **kwargs):
    """Call log_json and return the parsed JSON dict written to stderr."""
    buf = io.StringIO()
    with patch("sys.stderr", buf):
        from core.logging_utils import log_json

        log_json(level, event, **kwargs)
    output = buf.getvalue().strip()
    return json.loads(output)


class TestLogJson:
    def test_emits_valid_json(self):
        entry = _capture_log("INFO", "test_event")
        assert isinstance(entry, dict)

    def test_level_uppercased(self):
        entry = _capture_log("info", "evt")
        assert entry["level"] == "INFO"

    def test_event_in_output(self):
        entry = _capture_log("INFO", "my_event")
        assert entry["event"] == "my_event"

    def test_timestamp_present(self):
        entry = _capture_log("INFO", "evt")
        assert "ts" in entry

    def test_goal_included_when_provided(self):
        entry = _capture_log("INFO", "evt", goal="build feature")
        assert entry["goal"] == "build feature"

    def test_goal_absent_when_none(self):
        entry = _capture_log("INFO", "evt")
        assert "goal" not in entry

    def test_details_included(self):
        entry = _capture_log("INFO", "evt", details={"key": "value"})
        assert entry["details"]["key"] == "value"

    def test_details_absent_when_none(self):
        entry = _capture_log("INFO", "evt")
        assert "details" not in entry

    def test_correlation_id_included(self):
        entry = _capture_log("INFO", "evt", correlation_id="trace-123")
        assert entry["trace_id"] == "trace-123"

    def test_writes_to_stdout_when_env_set(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch.dict("os.environ", {"AURA_LOG_STREAM": "stdout"}):
                from core.logging_utils import log_json

                log_json("INFO", "stdout_event")
        output = buf.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["event"] == "stdout_event"

    def test_secrets_masked_in_details(self):
        entry = _capture_log("INFO", "evt", details={"api_key": "sk-supersecret123456789"})
        # The value should be masked — not the raw secret
        details_str = json.dumps(entry.get("details", {}))
        assert "sk-supersecret123456789" not in details_str

    def test_error_level_works(self):
        entry = _capture_log("ERROR", "something_failed")
        assert entry["level"] == "ERROR"

    def test_warn_level_works(self):
        entry = _capture_log("WARN", "caution")
        assert entry["level"] == "WARN"
