import json
import unittest
import os
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.cli_entrypoint_test_utils import run_main_subprocess
from core.logging_utils import log_json

class TestLogStreamingRobustness(unittest.TestCase):
    def test_log_json_to_stdout_with_interleaved_prints(self):
        original_stdout = sys.stdout
        out = io.StringIO()
        sys.stdout = out
        
        with patch.dict(os.environ, {"AURA_LOG_STREAM": "stdout"}):
            print("Non-JSON line 1")
            log_json("INFO", "event1", details={"key": "val1"})
            print("Non-JSON line 2")
            log_json("ERROR", "event2", details={"key": "val2"})
            
        sys.stdout = original_stdout
        output = out.getvalue()
        lines = output.strip().split("\n")
        
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "Non-JSON line 1")
        self.assertEqual(lines[2], "Non-JSON line 2")
        
        # Verify JSON lines are valid
        entry1 = json.loads(lines[1])
        self.assertEqual(entry1["event"], "event1")
        entry2 = json.loads(lines[3])
        self.assertEqual(entry2["event"], "event2")

    def test_log_json_to_stderr_remains_clean(self):
        original_stderr = sys.stderr
        err = io.StringIO()
        sys.stderr = err
        
        # Default stream is stderr
        with patch.dict(os.environ, {"AURA_LOG_STREAM": "stderr"}):
            log_json("INFO", "event1")
            
        sys.stderr = original_stderr
        output = err.getvalue()
        lines = output.strip().split("\n")
        
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["event"], "event1")

    def test_canonical_json_with_interleaved_logs(self):
        # Simulation of aura status --json with AURA_LOG_STREAM=stdout
        proc = run_main_subprocess("goal", "status", "--json", env_overrides={"AURA_LOG_STREAM": "stdout"})
        
        self.assertEqual(proc.returncode, 0)
        lines = proc.stdout.strip().split("\n")
        
        # Verify multiple JSON lines exist
        json_objects = []
        for line in lines:
            try:
                json_objects.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        
        self.assertGreater(len(json_objects), 1)
        
        # The last one should be the canonical status payload
        last_obj = json_objects[-1]
        self.assertIn("schema_version", last_obj)
        self.assertIn("queue", last_obj)
        
        # The earlier ones should be log events
        self.assertTrue(any(obj.get("event") == "config_loaded_from_file" for obj in json_objects[:-1]))

    def test_log_streamer_resilience(self):
        from aura_cli.tui.log_streamer import LogStreamer
        streamer = LogStreamer(level_filter="DEBUG")
        
        # Should not crash on these
        streamer.process_line("")
        streamer.process_line("{")
        streamer.process_line("null")
        streamer.process_line("123")
        streamer.process_line('{"level": "INFO", "event": "ok"}')
        
        assert True

if __name__ == "__main__":
    unittest.main()
