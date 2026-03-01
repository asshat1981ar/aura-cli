"""Unit tests for the agent registry adapters."""
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.registry import PlannerAdapter, CriticAdapter, ActAdapter, SandboxAdapter, default_agents

class TestRegistryAdapters(unittest.TestCase):
    def test_planner_adapter(self):
        mock_agent = MagicMock()
        mock_agent.plan.return_value = ["step 1"]
        adapter = PlannerAdapter(mock_agent)
        
        input_data = {"goal": "test goal"}
        result = adapter.run(input_data)
        
        self.assertEqual(result["steps"], ["step 1"])
        mock_agent.plan.assert_called_once()

    def test_critic_adapter(self):
        mock_agent = MagicMock()
        mock_agent.critique_plan.return_value = "looks good"
        adapter = CriticAdapter(mock_agent)
        
        input_data = {"task": "test goal", "plan": []}
        result = adapter.run(input_data)
        
        self.assertEqual(result["issues"], ["looks good"])
        mock_agent.critique_plan.assert_called_once()

    def test_act_adapter_aura_target(self):
        mock_agent = MagicMock()
        mock_agent.implement.return_value = "# AURA_TARGET: foo.py\ncode here"
        mock_agent.AURA_TARGET_DIRECTIVE = "# AURA_TARGET:"
        adapter = ActAdapter(mock_agent)
        
        input_data = {"task": "test goal"}
        result = adapter.run(input_data)
        
        self.assertEqual(result["changes"][0]["file_path"], "foo.py")
        self.assertEqual(result["changes"][0]["new_code"], "code here")

    def test_sandbox_adapter_skip_on_dry_run(self):
        adapter = SandboxAdapter(MagicMock())
        result = adapter.run({"dry_run": True})
        self.assertEqual(result["status"], "skip")
        self.assertTrue(result["passed"])

    def test_sandbox_adapter_pass(self):
        mock_agent = MagicMock()
        mock_res = MagicMock(passed=True, exit_code=0, stdout="", stderr="", timed_out=False)
        mock_res.summary.return_value = "passed"
        mock_agent.run_code.return_value = mock_res
        adapter = SandboxAdapter(mock_agent)
        
        input_data = {
            "act": {
                "changes": [{"new_code": "print(1)"}]
            }
        }
        result = adapter.run(input_data)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["passed"])

    def test_default_agents_wires_everything(self):
        mock_brain = MagicMock()
        mock_model = "gpt-4"
        agents = default_agents(mock_brain, mock_model)
        
        expected_phases = ["ingest", "plan", "critique", "synthesize", "act", "sandbox", "verify", "reflect"]
        for phase in expected_phases:
            self.assertIn(phase, agents)

if __name__ == "__main__":
    unittest.main()
