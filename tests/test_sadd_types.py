"""Unit tests for the SADD type system (core.sadd.types)."""

import json
import unittest

from core.sadd.types import (
    DesignSpec,
    SADDValidationError,
    SessionConfig,
    SessionReport,
    WorkstreamOutcome,
    WorkstreamResult,
    WorkstreamSpec,
    validate_result,
    validate_spec,
)


def _make_ws(**overrides) -> WorkstreamSpec:
    """Helper to create a valid WorkstreamSpec with sensible defaults."""
    defaults = {
        "id": "ws-1",
        "title": "Implement feature",
        "goal_text": "Build the new feature end-to-end",
    }
    defaults.update(overrides)
    return WorkstreamSpec(**defaults)


def _make_design(**overrides) -> DesignSpec:
    """Helper to create a valid DesignSpec with one workstream."""
    defaults = {
        "title": "Test Design",
        "summary": "A test design spec",
        "workstreams": [_make_ws()],
    }
    defaults.update(overrides)
    return DesignSpec(**defaults)


class TestWorkstreamSpec(unittest.TestCase):
    """Tests for WorkstreamSpec dataclass."""

    def test_defaults(self):
        ws = WorkstreamSpec(id="ws-1", title="A task", goal_text="Do something")
        self.assertEqual(ws.id, "ws-1")
        self.assertEqual(ws.title, "A task")
        self.assertEqual(ws.goal_text, "Do something")
        self.assertEqual(ws.priority, 1)
        self.assertEqual(ws.estimated_cycles, 5)
        self.assertEqual(ws.depends_on, [])
        self.assertEqual(ws.tags, [])
        self.assertEqual(ws.acceptance_criteria, [])
        self.assertEqual(ws.execution_mode, "automatic")

    def test_all_fields(self):
        ws = WorkstreamSpec(
            id="ws-2",
            title="Full spec",
            goal_text="Complete all steps",
            priority=3,
            estimated_cycles=10,
            depends_on=["ws-1"],
            tags=["backend", "critical"],
            acceptance_criteria=["Tests pass", "Lint clean"],
            execution_mode="dry_run",
        )
        self.assertEqual(ws.id, "ws-2")
        self.assertEqual(ws.priority, 3)
        self.assertEqual(ws.estimated_cycles, 10)
        self.assertEqual(ws.depends_on, ["ws-1"])
        self.assertEqual(ws.tags, ["backend", "critical"])
        self.assertEqual(ws.acceptance_criteria, ["Tests pass", "Lint clean"])
        self.assertEqual(ws.execution_mode, "dry_run")


class TestWorkstreamResult(unittest.TestCase):
    """Tests for WorkstreamResult dataclass and serialization."""

    def test_to_dict_from_dict_round_trip(self):
        result = WorkstreamResult(
            ws_id="ws-1",
            status="completed",
            cycles_used=3,
            stop_reason="goal_met",
            changed_files=["src/main.py", "tests/test_main.py"],
            verification_summary="All tests pass",
            reflector_output="Good outcome",
            elapsed_s=42.5,
            error=None,
        )
        d = result.to_dict()
        restored = WorkstreamResult.from_dict(d)

        self.assertEqual(restored.ws_id, result.ws_id)
        self.assertEqual(restored.status, result.status)
        self.assertEqual(restored.cycles_used, result.cycles_used)
        self.assertEqual(restored.stop_reason, result.stop_reason)
        self.assertEqual(restored.changed_files, result.changed_files)
        self.assertEqual(restored.verification_summary, result.verification_summary)
        self.assertEqual(restored.reflector_output, result.reflector_output)
        self.assertAlmostEqual(restored.elapsed_s, result.elapsed_s)
        self.assertIsNone(restored.error)

    def test_from_dict_ignores_extra_keys(self):
        d = {"ws_id": "ws-x", "status": "failed", "unexpected_key": 99}
        result = WorkstreamResult.from_dict(d)
        self.assertEqual(result.ws_id, "ws-x")
        self.assertEqual(result.status, "failed")

    def test_defaults(self):
        result = WorkstreamResult(ws_id="ws-1")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.cycles_used, 0)
        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.changed_files, [])
        self.assertEqual(result.verification_summary, "")
        self.assertEqual(result.reflector_output, "")
        self.assertAlmostEqual(result.elapsed_s, 0.0)
        self.assertIsNone(result.error)


class TestDesignSpec(unittest.TestCase):
    """Tests for DesignSpec dataclass and serialization."""

    def test_to_dict_from_dict_round_trip(self):
        ws1 = _make_ws(id="ws-1", title="First")
        ws2 = _make_ws(id="ws-2", title="Second", depends_on=["ws-1"])
        spec = DesignSpec(
            title="My Design",
            summary="Design summary",
            workstreams=[ws1, ws2],
            raw_markdown="# Design\n...",
            parse_confidence=0.95,
        )
        d = spec.to_dict()
        # from_dict mutates the dict (pops workstreams), so pass a copy
        restored = DesignSpec.from_dict(dict(d))

        self.assertEqual(restored.title, spec.title)
        self.assertEqual(restored.summary, spec.summary)
        self.assertEqual(restored.raw_markdown, spec.raw_markdown)
        self.assertAlmostEqual(restored.parse_confidence, spec.parse_confidence)
        self.assertEqual(len(restored.workstreams), 2)
        self.assertEqual(restored.workstreams[0].id, "ws-1")
        self.assertEqual(restored.workstreams[1].depends_on, ["ws-1"])


class TestSessionReport(unittest.TestCase):
    """Tests for SessionReport dataclass and its methods."""

    def _make_report(self) -> SessionReport:
        return SessionReport(
            session_id="test-session-001",
            design_title="Refactor Module",
            total_workstreams=3,
            completed=2,
            failed=1,
            skipped=0,
            outcomes=[
                WorkstreamOutcome(id="ws-1", title="Parse", status="completed", cycles_used=2, elapsed_s=10.0),
                WorkstreamOutcome(id="ws-2", title="Transform", status="completed", cycles_used=3, elapsed_s=15.5),
                WorkstreamOutcome(id="ws-3", title="Emit", status="failed", cycles_used=5, stop_reason="max_cycles", elapsed_s=25.0),
            ],
            elapsed_s=50.5,
            learnings=["Parsing was faster than expected"],
        )

    def test_summary_output_format(self):
        report = self._make_report()
        summary = report.summary()

        self.assertIn("SADD Session: Refactor Module", summary)
        self.assertIn("3 total", summary)
        self.assertIn("2 completed", summary)
        self.assertIn("1 failed", summary)
        self.assertIn("0 skipped", summary)
        self.assertIn("50.5s", summary)
        self.assertIn("[completed] Parse", summary)
        self.assertIn("[failed] Emit", summary)
        self.assertIn("Parsing was faster than expected", summary)

    def test_to_json_produces_valid_json(self):
        report = self._make_report()
        json_str = report.to_json()
        parsed = json.loads(json_str)

        self.assertEqual(parsed["session_id"], "test-session-001")
        self.assertEqual(parsed["design_title"], "Refactor Module")
        self.assertEqual(parsed["total_workstreams"], 3)
        self.assertEqual(len(parsed["outcomes"]), 3)
        self.assertIsInstance(parsed["learnings"], list)

    def test_to_dict(self):
        report = self._make_report()
        d = report.to_dict()
        self.assertEqual(d["session_id"], "test-session-001")
        self.assertIsInstance(d["outcomes"], list)
        self.assertEqual(d["outcomes"][0]["id"], "ws-1")


class TestSessionConfig(unittest.TestCase):
    """Tests for SessionConfig defaults."""

    def test_defaults(self):
        cfg = SessionConfig()
        self.assertEqual(cfg.max_parallel, 3)
        self.assertEqual(cfg.max_cycles_per_workstream, 5)
        self.assertFalse(cfg.dry_run)
        self.assertFalse(cfg.fail_fast)
        self.assertTrue(cfg.retry_failed)


class TestValidateSpec(unittest.TestCase):
    """Tests for validate_spec()."""

    def test_valid_spec_no_errors(self):
        spec = _make_design()
        errors = validate_spec(spec)
        self.assertEqual(errors, [])

    def test_missing_title(self):
        spec = _make_design(title="")
        errors = validate_spec(spec)
        self.assertTrue(any("title is required" in e for e in errors))

    def test_no_workstreams(self):
        spec = _make_design(workstreams=[])
        errors = validate_spec(spec)
        self.assertTrue(any("at least one workstream" in e for e in errors))

    def test_duplicate_workstream_ids(self):
        ws1 = _make_ws(id="dup")
        ws2 = _make_ws(id="dup", title="Another")
        spec = _make_design(workstreams=[ws1, ws2])
        errors = validate_spec(spec)
        self.assertTrue(any("Duplicate workstream ID" in e for e in errors))

    def test_missing_goal_text(self):
        ws = _make_ws(goal_text="")
        spec = _make_design(workstreams=[ws])
        errors = validate_spec(spec)
        self.assertTrue(any("goal_text is required" in e for e in errors))

    def test_invalid_priority(self):
        ws = _make_ws(priority=0)
        spec = _make_design(workstreams=[ws])
        errors = validate_spec(spec)
        self.assertTrue(any("priority must be >= 1" in e for e in errors))

    def test_invalid_estimated_cycles(self):
        ws = _make_ws(estimated_cycles=0)
        spec = _make_design(workstreams=[ws])
        errors = validate_spec(spec)
        self.assertTrue(any("estimated_cycles must be >= 1" in e for e in errors))


class TestValidateResult(unittest.TestCase):
    """Tests for validate_result()."""

    def test_valid_result_no_errors(self):
        result = WorkstreamResult(ws_id="ws-1", status="completed", cycles_used=2)
        errors = validate_result(result)
        self.assertEqual(errors, [])

    def test_missing_ws_id(self):
        result = WorkstreamResult(ws_id="", status="completed")
        errors = validate_result(result)
        self.assertTrue(any("ws_id is required" in e for e in errors))

    def test_invalid_status(self):
        result = WorkstreamResult(ws_id="ws-1", status="unknown")
        errors = validate_result(result)
        self.assertTrue(any("Invalid status" in e for e in errors))

    def test_negative_cycles_used(self):
        result = WorkstreamResult(ws_id="ws-1", cycles_used=-1)
        errors = validate_result(result)
        self.assertTrue(any("cycles_used must be >= 0" in e for e in errors))

    def test_negative_elapsed_s(self):
        result = WorkstreamResult(ws_id="ws-1", elapsed_s=-1.0)
        errors = validate_result(result)
        self.assertTrue(any("elapsed_s must be >= 0" in e for e in errors))


class TestSADDValidationError(unittest.TestCase):
    """Tests for SADDValidationError exception."""

    def test_is_exception(self):
        self.assertTrue(issubclass(SADDValidationError, Exception))

    def test_can_raise_and_catch(self):
        with self.assertRaises(SADDValidationError):
            raise SADDValidationError("spec is invalid")


if __name__ == "__main__":
    unittest.main()
