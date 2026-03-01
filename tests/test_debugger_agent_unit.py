"""Unit tests for the DebuggerAgent."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.debugger import DebuggerAgent

class TestDebuggerAgent(unittest.TestCase):
    def setUp(self):
        self.mock_brain = MagicMock()
        self.mock_model = MagicMock()
        self.agent = DebuggerAgent(self.mock_brain, self.mock_model)

    def test_debugger_instantiation(self):
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.brain, self.mock_brain)
        self.assertEqual(self.agent.model, self.mock_model)

    def test_diagnose_success(self):
        mock_response = json.dumps({
            "summary": "Syntax error in foo.py",
            "diagnosis": "Missing colon after if statement",
            "fix_strategy": "Add colon at line 45",
            "severity": "HIGH"
        })
        self.mock_model.respond.return_value = mock_response
        
        result = self.agent.diagnose("SyntaxError: expected ':'", "fix syntax", "if True print(1)")
        
        self.assertEqual(result["summary"], "Syntax error in foo.py")
        self.assertEqual(result["severity"], "HIGH")
        self.mock_model.respond.assert_called_once()

    def test_diagnose_invalid_json(self):
        self.mock_model.respond.return_value = "Not a JSON"
        
        result = self.agent.diagnose("Error")
        
        self.assertEqual(result["summary"], "LLM diagnosis failed.")
        self.assertEqual(result["severity"], "CRITICAL")

    def test_diagnose_missing_keys(self):
        self.mock_model.respond.return_value = json.dumps({"only_one_key": "val"})
        
        result = self.agent.diagnose("Error")
        
        self.assertEqual(result["summary"], "LLM diagnosis failed.")
        self.assertEqual(result["severity"], "CRITICAL")

    def test_diagnose_model_exception(self):
        self.mock_model.respond.side_effect = Exception("API error")
        
        result = self.agent.diagnose("Error")
        
        self.assertEqual(result["summary"], "LLM diagnosis failed.")
        self.assertEqual(result["severity"], "CRITICAL")

if __name__ == "__main__":
    unittest.main()
