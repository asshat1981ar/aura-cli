"""Unit tests for ReflectorAgent.

Addresses Technical Debt 16.6: No Unit Tests for Agent Implementations
"""

import unittest
from unittest.mock import Mock

from agents.reflector import ReflectorAgent


class TestReflectorAgent(unittest.TestCase):
    """Test suite for ReflectorAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = ReflectorAgent()

    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "reflect")

    def test_run_basic_success(self):
        """Test basic run with success verification."""
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {},
            "next_actions": ["action1", "action2"],
            "pipeline_run_id": "run_123"
        }
        
        result = self.agent.run(input_data)
        
        self.assertEqual(result["summary"], "Verification status: success.")
        self.assertEqual(result["learnings"], [])
        self.assertEqual(result["next_actions"], ["action1", "action2"])
        self.assertEqual(result["pipeline_run_id"], "run_123")

    def test_run_basic_failure(self):
        """Test basic run with failure verification."""
        input_data = {
            "verification": {"status": "failure", "failures": ["Error 1", "Error 2"]},
            "skill_context": {},
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertIn("Verification status: failure.", result["summary"])
        self.assertEqual(len(result["learnings"]), 1)
        self.assertIn("Failures: Error 1; Error 2", result["learnings"])

    def test_run_with_name_error(self):
        """Test run with NameError in failures."""
        input_data = {
            "verification": {
                "status": "failure",
                "failures": ["NameError: name 'foo' is not defined"]
            },
            "skill_context": {},
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("context_gap:" in learning for learning in result["learnings"]))
        self.assertTrue(any("NameError" in learning for learning in result["learnings"]))

    def test_run_with_import_error(self):
        """Test run with ImportError in failures."""
        input_data = {
            "verification": {
                "status": "failure",
                "failures": ["ImportError: No module named 'missing'"]
            },
            "skill_context": {},
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("ImportError" in learning for learning in result["learnings"]))

    def test_run_with_module_not_found_error(self):
        """Test run with ModuleNotFoundError in failures."""
        input_data = {
            "verification": {
                "status": "failure",
                "failures": ["ModuleNotFoundError: No module named 'nonexistent'"]
            },
            "skill_context": {},
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("ModuleNotFoundError" in learning for learning in result["learnings"]))

    def test_run_with_attribute_error(self):
        """Test run with AttributeError in failures."""
        input_data = {
            "verification": {
                "status": "failure",
                "failures": ["AttributeError: 'object' has no attribute 'bar'"]
            },
            "skill_context": {},
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("AttributeError" in learning for learning in result["learnings"]))

    def test_run_with_not_defined_error(self):
        """Test run with 'not defined' error pattern."""
        input_data = {
            "verification": {
                "status": "failure",
                "failures": ["variable 'x' is not defined"]
            },
            "skill_context": {},
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("not defined" in learning for learning in result["learnings"]))

    def test_run_with_security_scanner_skill(self):
        """Test run with security_scanner skill context."""
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {
                "security_scanner": {"critical_count": 2, "findings": [{}, {}]}
            },
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("security_scanner" in learning for learning in result["learnings"]))
        self.assertEqual(result["skill_summary"]["security_scanner"]["critical"], 2)

    def test_run_with_architecture_validator_skill(self):
        """Test run with architecture_validator skill context."""
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {
                "architecture_validator": {"coupling_score": 1.5, "circular_deps": [["a", "b"]]}
            },
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("coupling" in learning for learning in result["learnings"]))
        self.assertEqual(result["skill_summary"]["architecture_validator"]["coupling"], 1.5)

    def test_run_with_complexity_scorer_skill(self):
        """Test run with complexity_scorer skill context."""
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {
                "complexity_scorer": {"high_risk_count": 3}
            },
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("complexity_scorer" in learning for learning in result["learnings"]))
        self.assertEqual(result["skill_summary"]["complexity_scorer"]["high_risk"], 3)

    def test_run_with_test_coverage_analyzer_skill(self):
        """Test run with test_coverage_analyzer skill context."""
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {
                "test_coverage_analyzer": {"coverage_pct": 45.5, "meets_target": False}
            },
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("coverage" in learning for learning in result["learnings"]))
        self.assertEqual(result["skill_summary"]["test_coverage_analyzer"]["coverage_pct"], 45.5)

    def test_run_with_tech_debt_quantifier_skill(self):
        """Test run with tech_debt_quantifier skill context."""
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {
                "tech_debt_quantifier": {"debt_score": 75}
            },
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertTrue(any("tech_debt" in learning for learning in result["learnings"]))
        self.assertEqual(result["skill_summary"]["tech_debt_quantifier"]["debt_score"], 75)

    def test_run_with_linter_enforcer_skill(self):
        """Test run with linter_enforcer skill context."""
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {
                "linter_enforcer": {"violations": [{}, {}, {}]}
            },
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        self.assertEqual(result["skill_summary"]["linter_enforcer"]["violations"], 3)

    def test_run_empty_skill_context(self):
        """Test run with empty skill context."""
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {},
            "next_actions": ["next"]
        }
        
        result = self.agent.run(input_data)
        
        self.assertEqual(result["learnings"], [])
        self.assertEqual(result["skill_summary"], {})
        self.assertEqual(result["next_actions"], ["next"])

    def test_run_missing_keys(self):
        """Test run with minimal input data."""
        input_data = {}
        
        result = self.agent.run(input_data)
        
        self.assertIn("Verification status: skip.", result["summary"])
        self.assertEqual(result["learnings"], [])
        self.assertEqual(result["next_actions"], [])
        self.assertEqual(result["skill_summary"], {})

    def test_multiple_failure_types(self):
        """Test run with multiple different failure types."""
        input_data = {
            "verification": {
                "status": "failure",
                "failures": [
                    "NameError: name 'x' is not defined",
                    "ImportError: cannot import name 'y'",
                    "Some other error"
                ]
            },
            "skill_context": {},
            "next_actions": []
        }
        
        result = self.agent.run(input_data)
        
        # Should have learnings for failures + context gaps
        self.assertTrue(len(result["learnings"]) >= 3)

    def test_skill_summary_with_empty_data(self):
        """Test skill summary with empty data."""
        # Test with empty/invalid skill context values
        input_data = {
            "verification": {"status": "success"},
            "skill_context": {
                "security_scanner": {},  # Empty dict
                "linter_enforcer": {"violations": []}  # Empty list
            },
            "next_actions": []
        }
        
        # Should not raise an exception
        result = self.agent.run(input_data)
        self.assertIn("skill_summary", result)
        # Empty dict should result in no findings (critical_count default 0)
        self.assertEqual(result["skill_summary"]["security_scanner"]["critical"], 0)
        self.assertEqual(result["skill_summary"]["linter_enforcer"]["violations"], 0)


class TestReflectorAgentIntegration(unittest.TestCase):
    """Integration-style tests for ReflectorAgent."""

    def test_full_workflow_simulation(self):
        """Test a full workflow simulation."""
        agent = ReflectorAgent()
        
        input_data = {
            "verification": {
                "status": "failure",
                "failures": [
                    "NameError: name 'undefined_var' is not defined",
                    "AttributeError: 'NoneType' object has no attribute 'method'"
                ]
            },
            "skill_context": {
                "security_scanner": {"critical_count": 0, "findings": []},
                "complexity_scorer": {"high_risk_count": 2},
                "test_coverage_analyzer": {"coverage_pct": 30.0, "meets_target": False}
            },
            "next_actions": ["fix_name_error", "fix_attribute_error"],
            "pipeline_run_id": "test_run_001"
        }
        
        result = agent.run(input_data)
        
        # Verify complete workflow
        self.assertIn("Verification status: failure.", result["summary"])
        self.assertTrue(len(result["learnings"]) > 0)
        self.assertEqual(len(result["next_actions"]), 2)
        self.assertEqual(result["pipeline_run_id"], "test_run_001")
        
        # Verify skill summary structure
        self.assertIn("security_scanner", result["skill_summary"])
        self.assertIn("complexity_scorer", result["skill_summary"])
        self.assertIn("test_coverage_analyzer", result["skill_summary"])


if __name__ == "__main__":
    unittest.main()
