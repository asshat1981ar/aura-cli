"""
tests/test_tui.py — AURA Studio TUI test suite.

Tests data models, panel builders, log streamer, and doctor v2.
Does NOT test live Rich display (no terminal required).
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. AuraStudio — data model and callbacks
# ---------------------------------------------------------------------------

class TestAuraStudioModel(unittest.TestCase):

    def setUp(self):
        from aura_cli.tui.app import AuraStudio
        self.app = AuraStudio(runtime={}, refresh_rate=1.0)

    def test_instantiation(self):
        self.assertIsNotNone(self.app)

    def test_on_phase_start_updates_status(self):
        self.app.on_phase_start("plan")
        self.assertEqual(self.app._phases_status.get("plan"), "⟳")

    def test_on_phase_complete_sets_check(self):
        self.app.on_phase_start("act")
        self.app.on_phase_complete("act", elapsed_ms=120.0, success=True)
        self.assertEqual(self.app._phases_status.get("act"), "✅")

    def test_on_phase_complete_failure_sets_x(self):
        self.app.on_phase_start("verify")
        self.app.on_phase_complete("verify", elapsed_ms=50.0, success=False)
        self.assertEqual(self.app._phases_status.get("verify"), "❌")

    def test_on_cycle_start_sets_goal(self):
        self.app.on_cycle_start("Fix broken tests")
        self.assertEqual(self.app._current_goal, "Fix broken tests")

    def test_on_cycle_start_resets_phases(self):
        self.app._phases_status = {"plan": "✅"}
        self.app.on_cycle_start("new goal")
        self.assertEqual(self.app._phases_status, {})

    def test_on_cycle_complete_appends_log(self):
        self.app.on_cycle_complete({"goal": "test", "success": True, "duration_s": 1.5})
        self.assertEqual(len(self.app._cycle_log), 1)
        self.assertEqual(self.app._cycle_log[0]["goal"], "test")

    def test_on_cycle_complete_caps_at_50(self):
        for i in range(55):
            self.app.on_cycle_complete({"goal": f"g{i}", "success": True, "duration_s": 1.0})
        self.assertEqual(len(self.app._cycle_log), 50)

    def test_render_once_does_not_raise(self):
        """render_once() should never raise regardless of runtime state."""
        try:
            self.app.render_once()
        except Exception as e:
            self.fail(f"render_once() raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 2. Panel builders
# ---------------------------------------------------------------------------

class TestCyclePanel(unittest.TestCase):

    def test_build_returns_object(self):
        from aura_cli.tui.panels.cycle_panel import build_cycle_panel
        result = build_cycle_panel(
            current_goal="Write tests",
            phases_status={"plan": "✅", "act": "⟳"},
            current_phase="act",
        )
        self.assertIsNotNone(result)

    def test_build_with_empty_status(self):
        from aura_cli.tui.panels.cycle_panel import build_cycle_panel
        result = build_cycle_panel(current_goal="", phases_status={}, current_phase="")
        self.assertIsNotNone(result)

    def test_build_long_goal_truncated(self):
        from aura_cli.tui.panels.cycle_panel import build_cycle_panel
        long_goal = "A" * 100
        # Should not raise
        result = build_cycle_panel(current_goal=long_goal, phases_status={}, current_phase="")
        self.assertIsNotNone(result)


class TestQueuePanel(unittest.TestCase):

    def test_build_with_none_queue(self):
        from aura_cli.tui.panels.queue_panel import build_queue_panel
        result = build_queue_panel(goal_queue=None)
        self.assertIsNotNone(result)

    def test_build_with_mock_queue(self):
        from collections import deque
        from aura_cli.tui.panels.queue_panel import build_queue_panel
        mock_q = MagicMock()
        mock_q.queue = deque(["goal A", "goal B", "goal C"])
        result = build_queue_panel(goal_queue=mock_q)
        self.assertIsNotNone(result)

    def test_build_with_empty_queue(self):
        from collections import deque
        from aura_cli.tui.panels.queue_panel import build_queue_panel
        mock_q = MagicMock()
        mock_q.queue = deque()
        result = build_queue_panel(goal_queue=mock_q)
        self.assertIsNotNone(result)

    def test_build_with_many_goals(self):
        from collections import deque
        from aura_cli.tui.panels.queue_panel import build_queue_panel
        mock_q = MagicMock()
        mock_q.queue = deque([f"goal {i}" for i in range(20)])
        result = build_queue_panel(goal_queue=mock_q)
        self.assertIsNotNone(result)


class TestMemoryPanel(unittest.TestCase):

    def test_build_with_none_brain(self):
        from aura_cli.tui.panels.memory_panel import build_memory_panel
        result = build_memory_panel(brain=None)
        self.assertIsNotNone(result)

    def test_build_with_mock_brain(self):
        from aura_cli.tui.panels.memory_panel import build_memory_panel
        mock_brain = MagicMock()
        mock_brain.count_memories.return_value = 30419
        mock_brain.recall_recent.return_value = ["memory A", "memory B"]
        result = build_memory_panel(brain=mock_brain)
        self.assertIsNotNone(result)

    def test_build_with_failing_brain(self):
        from aura_cli.tui.panels.memory_panel import build_memory_panel
        mock_brain = MagicMock()
        mock_brain.count_memories.side_effect = Exception("DB error")
        mock_brain.recall_recent.side_effect = Exception("DB error")
        # Should not raise
        try:
            result = build_memory_panel(brain=mock_brain)
        except Exception as e:
            self.fail(f"build_memory_panel raised {type(e).__name__}: {e}")


class TestMetricsPanel(unittest.TestCase):

    def test_build_with_empty_log(self):
        from aura_cli.tui.panels.metrics_panel import build_metrics_panel
        result = build_metrics_panel(cycle_log=[])
        self.assertIsNotNone(result)

    def test_build_with_cycle_data(self):
        from aura_cli.tui.panels.metrics_panel import build_metrics_panel
        log = [
            {"goal": f"g{i}", "success": i % 3 != 0, "duration_s": 1.5 + i * 0.2}
            for i in range(10)
        ]
        result = build_metrics_panel(cycle_log=log)
        self.assertIsNotNone(result)

    def test_build_with_single_entry(self):
        from aura_cli.tui.panels.metrics_panel import build_metrics_panel
        result = build_metrics_panel(cycle_log=[{"success": True, "duration_s": 2.5}])
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# 3. Sparkline helper
# ---------------------------------------------------------------------------

class TestSparkline(unittest.TestCase):

    def test_empty_returns_dashes(self):
        from aura_cli.tui.panels.metrics_panel import _sparkline
        result = _sparkline([])
        self.assertIsInstance(result, str)
        self.assertIn("─", result)

    def test_uniform_values(self):
        from aura_cli.tui.panels.metrics_panel import _sparkline
        result = _sparkline([5.0, 5.0, 5.0])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 3)

    def test_varying_values(self):
        from aura_cli.tui.panels.metrics_panel import _sparkline
        result = _sparkline([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 5)

    def test_width_respected(self):
        from aura_cli.tui.panels.metrics_panel import _sparkline
        result = _sparkline([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], width=8)
        self.assertLessEqual(len(result), 8)


# ---------------------------------------------------------------------------
# 4. Log Streamer
# ---------------------------------------------------------------------------

class TestLogStreamer(unittest.TestCase):

    def _make_streamer(self, level="DEBUG"):
        from aura_cli.tui.log_streamer import LogStreamer
        return LogStreamer(level_filter=level)

    def test_instantiation(self):
        streamer = self._make_streamer()
        self.assertIsNotNone(streamer)

    def test_process_valid_json_line(self):
        streamer = self._make_streamer()
        line = json.dumps({"level": "INFO", "event": "test_event", "ts": "2026-01-01T00:00:00"})
        result = streamer.process_line(line)
        self.assertTrue(result)

    def test_process_empty_line_returns_false(self):
        streamer = self._make_streamer()
        result = streamer.process_line("")
        self.assertFalse(result)

    def test_process_plain_text_line(self):
        streamer = self._make_streamer()
        result = streamer.process_line("plain text log line")
        self.assertTrue(result)

    def test_level_filter_blocks_debug(self):
        streamer = self._make_streamer(level="WARN")
        line = json.dumps({"level": "DEBUG", "event": "debug_event", "ts": ""})
        result = streamer.process_line(line)
        self.assertFalse(result)

    def test_level_filter_passes_error(self):
        streamer = self._make_streamer(level="WARN")
        line = json.dumps({"level": "ERROR", "event": "err_event", "ts": ""})
        result = streamer.process_line(line)
        self.assertTrue(result)

    def test_stream_fd_processes_lines(self):
        from aura_cli.tui.log_streamer import LogStreamer
        streamer = LogStreamer(level_filter="DEBUG")
        lines = [
            json.dumps({"level": "INFO", "event": f"evt{i}", "ts": ""}) + "\n"
            for i in range(5)
        ]
        fd = io.StringIO("".join(lines))
        # Should not raise
        streamer.stream_fd(fd)

    def test_stream_fd_tail_mode(self):
        from aura_cli.tui.log_streamer import LogStreamer
        streamer = LogStreamer(level_filter="DEBUG")
        lines = [
            json.dumps({"level": "INFO", "event": f"evt{i}", "ts": ""}) + "\n"
            for i in range(20)
        ]
        fd = io.StringIO("".join(lines))
        # Should not raise, tail=5 shows only last 5
        streamer.stream_fd(fd, tail=5)

    def test_parse_line_helper(self):
        from aura_cli.tui.log_streamer import _parse_line
        record = _parse_line('{"level": "INFO", "event": "test"}')
        self.assertIsNotNone(record)
        self.assertEqual(record["level"], "INFO")

    def test_parse_line_invalid_json(self):
        from aura_cli.tui.log_streamer import _parse_line
        record = _parse_line("not json")
        self.assertIsNotNone(record)
        self.assertIn("event", record)

    def test_passes_level_filter(self):
        from aura_cli.tui.log_streamer import _passes_level_filter
        self.assertTrue(_passes_level_filter({"level": "ERROR"}, "WARN"))
        self.assertFalse(_passes_level_filter({"level": "DEBUG"}, "WARN"))
        self.assertTrue(_passes_level_filter({"level": "WARN"}, "WARN"))


# ---------------------------------------------------------------------------
# 5. Doctor v2
# ---------------------------------------------------------------------------

class TestDoctorV2(unittest.TestCase):

    def test_run_returns_list(self):
        from aura_cli.doctor import run_doctor_v2
        results = run_doctor_v2(rich_output=False)
        self.assertIsInstance(results, list)

    def test_run_returns_nonempty(self):
        from aura_cli.doctor import run_doctor_v2
        results = run_doctor_v2(rich_output=False)
        self.assertGreater(len(results), 5)

    def test_each_result_has_required_keys(self):
        from aura_cli.doctor import run_doctor_v2
        results = run_doctor_v2(rich_output=False)
        for r in results:
            self.assertIn("check", r)
            self.assertIn("status", r)
            self.assertIn("detail", r)

    def test_status_values_are_valid(self):
        from aura_cli.doctor import run_doctor_v2
        results = run_doctor_v2(rich_output=False)
        valid = {"PASS", "WARN", "FAIL"}
        for r in results:
            self.assertIn(r["status"], valid,
                          f"Invalid status '{r['status']}' for check '{r['check']}'")

    def test_python_check_passes(self):
        from aura_cli.doctor import run_doctor_v2
        results = run_doctor_v2(rich_output=False)
        python_check = next((r for r in results if "Python" in r["check"]), None)
        self.assertIsNotNone(python_check)
        self.assertEqual(python_check["status"], "PASS")

    def test_rich_tui_check_present(self):
        from aura_cli.doctor import run_doctor_v2
        results = run_doctor_v2(rich_output=False)
        tui_check = next((r for r in results if "TUI" in r["check"] or "rich" in r["check"].lower()), None)
        self.assertIsNotNone(tui_check)

    def test_run_with_rich_output_no_crash(self):
        from aura_cli.doctor import run_doctor_v2
        try:
            run_doctor_v2(rich_output=True)
        except Exception as e:
            self.fail(f"run_doctor_v2(rich_output=True) raised {type(e).__name__}: {e}")

    def test_run_with_custom_project_root(self):
        import tempfile
        from aura_cli.doctor import run_doctor_v2
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_doctor_v2(project_root=Path(tmpdir), rich_output=False)
            self.assertIsInstance(results, list)
            # All DB/queue checks should be WARN (tmpdir is empty)
            brain_check = next((r for r in results if "Brain" in r["check"]), None)
            if brain_check:
                self.assertIn(brain_check["status"], ("WARN", "FAIL"))


if __name__ == "__main__":
    unittest.main()
