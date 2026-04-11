"""Tests for the Innovation Catalyst CLI commands.

Tests the innovate command family:
- innovate start
- innovate list
- innovate show
- innovate resume
- innovate export
- innovate techniques
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestInnovateTechniques(unittest.TestCase):
    """Test the innovate techniques command."""

    def test_list_techniques_table_output(self):
        """Verify techniques command returns table output."""
        from aura_cli.commands import _handle_innovate_techniques

        args = MagicMock()
        args.json = False
        args.output = "table"

        # Should not raise
        _handle_innovate_techniques(args)

    def test_list_techniques_json_output(self):
        """Verify techniques command returns valid JSON."""
        from aura_cli.commands import _handle_innovate_techniques

        args = MagicMock()
        args.json = False
        args.output = "json"

        # Capture output by mocking print
        with patch("builtins.print") as mock_print:
            _handle_innovate_techniques(args)

            # Should have printed JSON
            calls = mock_print.call_args_list
            json_output = "".join(str(call[0][0]) for call in calls if "{" in str(call[0][0]))
            data = json.loads(json_output)

            self.assertIn("techniques", data)
            self.assertGreater(len(data["techniques"]), 0)

            # Verify expected techniques exist
            technique_ids = {t["id"] for t in data["techniques"]}
            expected = {"scamper", "six_hats", "mind_map", "reverse", "worst_idea"}
            self.assertTrue(expected.issubset(technique_ids))


class TestInnovateStart(unittest.TestCase):
    """Test the innovate start command."""

    def test_start_session_basic(self):
        """Verify start command creates a session."""
        from aura_cli.commands import _handle_innovate_start, _meta_conductor

        # Reset conductor singleton for clean test
        import aura_cli.commands

        aura_cli.commands._meta_conductor = None

        args = MagicMock()
        args.problem_statement = ["How", "to", "improve", "testing?"]
        args.techniques = "scamper,mind_map"
        args.constraints = ""
        args.execute_phase = None
        args.batch_file = None
        args.json = False
        args.output = "table"

        runtime = {"brain": None}

        with patch("builtins.print") as mock_print:
            _handle_innovate_start(args, runtime)

            # Should have printed session info
            output = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            self.assertIn("Innovation Session", output)
            self.assertIn("Session ID", output)

    def test_start_session_with_constraints(self):
        """Verify start command accepts constraints."""
        from aura_cli.commands import _handle_innovate_start
        import aura_cli.commands

        aura_cli.commands._meta_conductor = None

        args = MagicMock()
        args.problem_statement = ["Test", "problem"]
        args.techniques = "scamper"
        args.constraints = '{"max_ideas": 10}'
        args.execute_phase = None
        args.batch_file = None
        args.json = False
        args.output = "table"

        runtime = {"brain": None}

        # Should not raise with valid JSON constraints
        with patch("builtins.print"):
            _handle_innovate_start(args, runtime)

    def test_start_session_invalid_constraints(self):
        """Verify start command rejects invalid constraints."""
        from aura_cli.commands import _handle_innovate_start
        import aura_cli.commands

        aura_cli.commands._meta_conductor = None

        args = MagicMock()
        args.problem_statement = ["Test", "problem"]
        args.techniques = "scamper"
        args.constraints = "invalid json"
        args.execute_phase = None
        args.batch_file = None
        args.json = False
        args.output = "table"

        runtime = {"brain": None}

        with patch("builtins.print") as mock_print:
            _handle_innovate_start(args, runtime)

            # Should have printed error
            output = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            self.assertIn("Invalid JSON", output)

    def test_start_batch_mode(self):
        """Verify start command with batch file."""
        from aura_cli.commands import _handle_innovate_start
        import aura_cli.commands

        aura_cli.commands._meta_conductor = None

        # Create temp batch file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Problem 1\n")
            f.write("Problem 2\n")
            batch_path = f.name

        try:
            args = MagicMock()
            args.problem_statement = []
            args.techniques = "scamper"
            args.constraints = ""
            args.execute_phase = None
            args.batch_file = batch_path
            args.json = False
            args.output = "table"

            runtime = {"brain": None}

            with patch("builtins.print") as mock_print:
                _handle_innovate_start(args, runtime)

                # Should have printed multiple sessions
                output = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
                self.assertIn("2 Innovation Sessions", output)
        finally:
            Path(batch_path).unlink(missing_ok=True)


class TestInnovateList(unittest.TestCase):
    """Test the innovate list command."""

    def test_list_empty_sessions(self):
        """Verify list command shows empty state."""
        from aura_cli.commands import _handle_innovate_list
        import aura_cli.commands

        aura_cli.commands._meta_conductor = None

        args = MagicMock()
        args.json = False
        args.output = "table"
        args.limit = 20

        runtime = {"brain": None}

        with patch("builtins.print") as mock_print:
            _handle_innovate_list(args, runtime)

            output = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            self.assertIn("No active sessions", output)

    def test_list_json_output(self):
        """Verify list command returns valid JSON."""
        from aura_cli.commands import _handle_innovate_list
        import aura_cli.commands

        aura_cli.commands._meta_conductor = None

        args = MagicMock()
        args.json = True
        args.output = "json"
        args.limit = 20

        runtime = {"brain": None}

        with patch("builtins.print") as mock_print:
            _handle_innovate_list(args, runtime)

            calls = mock_print.call_args_list
            json_output = "".join(str(call[0][0]) for call in calls if "{" in str(call[0][0]))
            data = json.loads(json_output)

            self.assertIn("sessions", data)
            self.assertIn("total", data)


class TestInnovateShow(unittest.TestCase):
    """Test the innovate show command."""

    def test_show_nonexistent_session(self):
        """Verify show command handles missing session."""
        from aura_cli.commands import _handle_innovate_show
        import aura_cli.commands

        aura_cli.commands._meta_conductor = None

        args = MagicMock()
        args.session_id = "nonexistent123"
        args.json = False
        args.output = "table"
        args.show_ideas = False

        runtime = {"brain": None}

        with patch("builtins.print") as mock_print:
            _handle_innovate_show(args, runtime)

            output = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            self.assertIn("not found", output)


class TestInnovateExport(unittest.TestCase):
    """Test the innovate export command."""

    def test_export_nonexistent_session(self):
        """Verify export command handles missing session."""
        from aura_cli.commands import _handle_innovate_export
        import aura_cli.commands

        aura_cli.commands._meta_conductor = None

        args = MagicMock()
        args.session_id = "nonexistent123"
        args.format = "markdown"
        args.output = None

        runtime = {"brain": None}

        with patch("builtins.print") as mock_print:
            _handle_innovate_export(args, runtime)

            output = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
            self.assertIn("not found", output)


class TestBrainPersistence(unittest.TestCase):
    """Test Brain integration for session persistence."""

    def test_save_and_load_session(self):
        """Verify sessions can be saved and loaded from brain."""
        from memory.brain import Brain
        from agents.schemas import InnovationSessionState, InnovationPhase
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_brain.db"
            brain = Brain(db_path=str(db_path))

            # Create test session
            session = InnovationSessionState(
                session_id="test123",
                problem_statement="Test problem",
                status="active",
                current_phase=InnovationPhase.IMMERSION,
                phases_completed=[],
                techniques=["scamper", "mind_map"],
                constraints={"max_ideas": 10},
                ideas_generated=5,
                ideas_selected=2,
            )

            # Save
            brain.save_innovation_session(session)

            # Load
            loaded = brain.get_innovation_session("test123")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["session_id"], "test123")
            self.assertEqual(loaded["problem_statement"], "Test problem")
            self.assertEqual(loaded["ideas_generated"], 5)
            self.assertEqual(loaded["techniques"], ["scamper", "mind_map"])

    def test_list_sessions_with_status_filter(self):
        """Verify session listing with status filter."""
        from memory.brain import Brain
        from agents.schemas import InnovationSessionState, InnovationPhase
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_brain.db"
            brain = Brain(db_path=str(db_path))

            # Create active session
            active = InnovationSessionState(
                session_id="active1",
                problem_statement="Active problem",
                status="active",
                current_phase=InnovationPhase.IMMERSION,
            )
            brain.save_innovation_session(active)

            # Create completed session
            completed = InnovationSessionState(
                session_id="completed1",
                problem_statement="Completed problem",
                status="completed",
                current_phase=InnovationPhase.TRANSFORMATION,
            )
            brain.save_innovation_session(completed)

            # List active only
            active_sessions = brain.list_innovation_sessions(status="active")
            self.assertEqual(len(active_sessions), 1)
            self.assertEqual(active_sessions[0]["session_id"], "active1")

            # List all
            all_sessions = brain.list_innovation_sessions()
            self.assertEqual(len(all_sessions), 2)


if __name__ == "__main__":
    unittest.main()
