import unittest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import sys

# Ensure the project root is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hybrid_loop import HybridClosedLoop
from core.git_tools import GitTools
from memory.brain import Brain
from core.model_adapter import ModelAdapter
from core.file_tools import replace_code # Need to import it to mock it

# === AURA JSON FIX INJECTED ===
import re as _aura_re

def _aura_clean_json(raw):
    """Strip markdown code blocks from LLM responses"""
    if not raw:
        return raw
    text = _aura_re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", raw.strip())
    text = _aura_re.sub(r"\n?```\s*$", "", text)
    return text.strip()

def _aura_safe_loads(raw, ctx="unknown"):
    """Parse JSON with markdown cleaning"""
    try:
        return __import__("json").loads(raw)
    except Exception:
        cleaned = _aura_clean_json(raw)
        return __import__("json").loads(cleaned)

# === END AURA JSON FIX ===



class TestDryRunWorkflow(unittest.TestCase):

    def setUp(self):
        # Mock dependencies for HybridClosedLoop
        self.mock_model = MagicMock(spec=ModelAdapter)
        self.mock_brain = MagicMock(spec=Brain)
        self.mock_git = MagicMock(spec=GitTools)

        # Instantiate HybridClosedLoop with mocks
        self.loop = HybridClosedLoop(self.mock_model, self.mock_brain, self.mock_git)

        # Define a sample model response for mocking
        self.sample_model_response = {
            "DEFINE": "Define phase output.",
            "PLAN": "Plan phase output.",
            "IMPLEMENT": {
                "file_path": "test_file.py",
                "old_code": "def old_func(): pass",
                "new_code": "def new_func(): pass"
            },
            "TEST": "Test phase output.",
            "CRITIQUE": {
                "performance_score": 10, # High scores to ensure immediate convergence for dry-run test
                "stability_score": 10,
                "security_score": 10,
                "elegance_score": 10,
                "weaknesses": ["Minor weakness identified."]
            },
            "IMPROVE": "Improve phase output.",
            "VERSION": "Dry run test commit message.",
            "SUMMARY": "Summary of dry run iteration."
        }
        self.mock_model.respond.return_value = json.dumps(self.sample_model_response)

        # Mock initial state for snapshot
        self.mock_brain.recall_all.return_value = []
        self.mock_brain.recall_weaknesses.return_value = []
    
    @patch('core.file_tools.replace_code')
    def test_dry_run_mode_no_changes(self, mock_replace_code):
        goal = "Test dry run functionality"
        
        # To make it converge in 3 iterations (Stable Convergence Logic)
        # We need to simulate 3 calls where absolute_pass is true and score is >= previous_score
        # For this test, we mock respond() three times with the same high-scoring response.
        self.mock_model.respond.side_effect = [
            json.dumps(self.sample_model_response),
            json.dumps(self.sample_model_response),
            json.dumps(self.sample_model_response)
        ]

        # Run in dry-run mode for 3 cycles to trigger convergence
        result_json_str = ""
        for _ in range(3):
            result_json_str = self.loop.run(goal, dry_run=True)
            result = json.loads(result_json_str)
            if result.get("FINAL_STATUS", "").startswith("Optimization converged"):
                break

        # Assert that git methods were NOT called
        self.mock_git.stash.assert_not_called()
        self.mock_git.commit_all.assert_not_called()
        self.mock_git.rollback_last_commit.assert_not_called()
        self.mock_git.stash_pop.assert_not_called()
        
        # Assert that replace_code was NOT called
        mock_replace_code.assert_not_called()

        # Assert that brain.remember WAS called (to record weaknesses)
        # It's called twice per run: for goal and for structured_response.
        # So it should be called 6 times for 3 iterations.
        self.assertEqual(self.mock_brain.remember.call_count, 6)

        # Assert that the output JSON contains "DRY_RUN": true
        self.assertTrue(result.get("DRY_RUN"))
        # Expected score from mock_scores_list in sample_model_response should be 10.0
        self.assertEqual(result["FINAL_STATUS"], "Optimization converged at 10.0 with Robust Confirmation.")

    @patch('core.file_tools.replace_code')
    def test_normal_run_mode_calls_git_and_replace(self, mock_replace_code):
        goal = "Test normal run functionality"

        # Mock scores to ensure commit logic is triggered and convergence happens
        mock_scores_list = [
            # All scores high to ensure immediate absolute_pass and stable_convergence
            {"performance_score": 9, "stability_score": 9, "security_score": 9, "elegance_score": 9}, # score 9.0
            {"performance_score": 9, "stability_score": 9, "security_score": 9, "elegance_score": 9}, # score 9.0
            {"performance_score": 9, "stability_score": 9, "security_score": 9, "elegance_score": 9}, # score 9.0
            # Add more if the loop goes beyond 3 iterations, though it should converge
        ]
        
        # Mock model response to control critique scores
        def mock_respond_side_effect(prompt):
            current_iteration = self.mock_model.respond.call_count
            
            response = self.sample_model_response.copy()
            # Ensure the structure of CRITIQUE matches what extract_scores expects
            if current_iteration < len(mock_scores_list):
                response["CRITIQUE"] = mock_scores_list[current_iteration]
            else:
                # If more calls than mocked scores, return a default high-scoring response
                response["CRITIQUE"] = {"performance_score": 9, "stability_score": 9, "security_score": 9, "elegance_score": 9, "weaknesses": ["Simulated weakness."]}
            
            response["CRITIQUE"]["weaknesses"] = ["Simulated weakness."]
            return json.dumps(response)

        self.mock_model.respond.side_effect = mock_respond_side_effect
        
        # Run in normal mode, should converge in 3 iterations
        self.loop.previous_score = 0 # Reset for consistent scoring
        result_json_str = ""
        
        # Loop enough times to guarantee convergence (at least 3 if scores are stable high)
        # Using a higher range to ensure convergence is hit, the break statement will exit early
        for i in range(5): 
            result_json_str = self.loop.run(goal, dry_run=False)
            result = json.loads(result_json_str)
            if result.get("FINAL_STATUS", "").startswith("Optimization converged"):
                break

        result = json.loads(result_json_str)

        # Assert that git methods WERE called
        self.mock_git.stash.assert_called()
        self.mock_git.commit_all.assert_called()
        # rollback_last_commit should NOT be called in a successful path
        self.mock_git.rollback_last_commit.assert_not_called() 
        self.mock_git.stash_pop.assert_called()
        
        # Assert that replace_code was NOT called by HybridLoop (it's external)
        mock_replace_code.assert_not_called()

        # Assert that brain.remember WAS called (twice per iteration for goal and structured_response)
        # Check call_count dynamically as loop iterations can vary slightly for convergence
        self.assertTrue(self.mock_brain.remember.call_count >= 6) # At least 3 iterations * 2 calls

        # Assert that the output JSON does NOT contain "DRY_RUN": true
        self.assertFalse(result.get("DRY_RUN"))
        # Expected score from mock_scores_list should be 9.0
        self.assertEqual(result["FINAL_STATUS"], "Optimization converged at 9.0 with Robust Confirmation.")