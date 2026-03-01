import io
import json
import unittest
from unittest.mock import patch, MagicMock
from aura_cli.cli_main import cli_interaction_loop

class TestCLIInteractiveLoop(unittest.TestCase):
    def setUp(self):
        self.runtime = {
            "project_root": "/tmp/aura-test",
            "goal_queue": MagicMock(),
            "goal_archive": MagicMock(),
            "loop": MagicMock(),
            "debugger": MagicMock(),
            "planner": MagicMock(),
            "memory_persistence_path": "/tmp/aura-test/memory/task_hierarchy.json"
        }
        self.args = MagicMock()

    @patch("aura_cli.cli_main.input")
    @patch("aura_cli.cli_main._handle_add")
    def test_loop_handles_add_command(self, mock_handle_add, mock_input):
        # Sequence of inputs: "add my goal", then "exit" to break the loop
        mock_input.side_effect = ["add my goal", "exit"]
        
        with patch("aura_cli.cli_main._handle_exit") as mock_exit:
            cli_interaction_loop(self.args, self.runtime)
            
        mock_handle_add.assert_called_once_with(self.runtime["goal_queue"], "add my goal")
        mock_exit.assert_called_once()

    @patch("aura_cli.cli_main.input")
    @patch("aura_cli.cli_main._handle_status")
    def test_loop_handles_status_command(self, mock_handle_status, mock_input):
        mock_input.side_effect = ["status", "exit"]
        
        with patch("aura_cli.cli_main._handle_exit"):
            cli_interaction_loop(self.args, self.runtime)
            
        mock_handle_status.assert_called_once()
        # Verify it passed the expected parts of the runtime
        args, kwargs = mock_handle_status.call_args
        self.assertEqual(kwargs["project_root"], "/tmp/aura-test")

    @patch("aura_cli.cli_main.input")
    @patch("aura_cli.cli_main._handle_help")
    def test_loop_handles_help_command(self, mock_handle_help, mock_input):
        mock_input.side_effect = ["help", "exit"]
        
        with patch("aura_cli.cli_main._handle_exit"):
            cli_interaction_loop(self.args, self.runtime)
            
        mock_handle_help.assert_called_once()

    @patch("aura_cli.cli_main.input")
    @patch("aura_cli.cli_main.log_json")
    def test_loop_handles_invalid_command(self, mock_log_json, mock_input):
        mock_input.side_effect = ["bogus", "exit"]
        
        with patch("aura_cli.cli_main._handle_exit"):
            cli_interaction_loop(self.args, self.runtime)
            
        mock_log_json.assert_called_with("WARN", "invalid_cli_command", details={"command": "bogus"})

    @patch("aura_cli.cli_main.input")
    @patch("aura_cli.cli_main.log_json")
    def test_loop_handles_eof_error(self, mock_log_json, mock_input):
        mock_input.side_effect = EOFError()
        
        cli_interaction_loop(self.args, self.runtime)
            
        mock_log_json.assert_called_with("INFO", "aura_cli_eof_received", details={"message": "End of input received, exiting."})

if __name__ == "__main__":
    unittest.main()
