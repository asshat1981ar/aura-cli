import unittest
from unittest.mock import MagicMock, call
from core.sandbox_loop import run_sandbox_loop, MAX_SANDBOX_RETRIES

class TestSandboxLoop(unittest.TestCase):
    def test_sandbox_passes_first_try(self):
        run_phase = MagicMock(return_value={"passed": True})
        notify_ui = MagicMock()
        act = {"changes": []}
        task_bundle = {}
        phase_outputs = {}

        result_act, passed, attempts = run_sandbox_loop(
            run_phase, notify_ui, "/tmp", "goal", act, task_bundle, False, phase_outputs
        )

        self.assertTrue(passed)
        self.assertEqual(attempts, 0)
        self.assertEqual(run_phase.call_count, 1) # 1 sandbox call
        run_phase.assert_called_with("sandbox", {
            "act": act,
            "dry_run": False,
            "project_root": "/tmp"
        })

    def test_sandbox_retries_and_succeeds(self):
        # sandbox fail 1, act fix, sandbox pass 2
        run_phase = MagicMock(side_effect=[
            {"passed": False, "details": {"stderr": "error1"}}, # sandbox 1
            {"changes": ["fix"]}, # act fix 1
            {"passed": True} # sandbox 2
        ])
        notify_ui = MagicMock()
        act = {"changes": []}
        task_bundle = {}
        phase_outputs = {}

        result_act, passed, attempts = run_sandbox_loop(
            run_phase, notify_ui, "/tmp", "goal", act, task_bundle, False, phase_outputs
        )

        self.assertTrue(passed)
        self.assertEqual(attempts, 1)
        self.assertEqual(run_phase.call_count, 3) # sandbox, act, sandbox
        
        # Check fix hints
        self.assertEqual(task_bundle["fix_hints"], ["error1"])
        self.assertEqual(result_act, {"changes": ["fix"]})

    def test_sandbox_fails_after_retries(self):
        # MAX_SANDBOX_RETRIES is 3.
        # 1. Sandbox 1 (fail) -> retry
        # 2. Act 1
        # 3. Sandbox 2 (fail) -> retry
        # 4. Act 2
        # 5. Sandbox 3 (fail) -> Stop
        
        failures = [{"passed": False, "details": {"stderr": "err"}}] * 3
        acts = [{"changes": ["fix"]}] * 2
        
        # side_effect needs interleaved: sand, act, sand, act, sand
        side_effect = [
            failures[0], acts[0],
            failures[1], acts[1],
            failures[2]
        ]
        
        run_phase = MagicMock(side_effect=side_effect)
        notify_ui = MagicMock()
        task_bundle = {}
        
        result_act, passed, attempts = run_sandbox_loop(
            run_phase, notify_ui, "/tmp", "goal", {}, task_bundle, False, {}
        )
        
        self.assertFalse(passed)
        self.assertEqual(attempts, 2)
        self.assertEqual(run_phase.call_count, 5)
        self.assertEqual(task_bundle["fix_hints"], ["err"])

    def test_dry_run_skips_retry(self):
        run_phase = MagicMock(return_value={"passed": False}) # fail but dry run
        notify_ui = MagicMock()
        
        result_act, passed, attempts = run_sandbox_loop(
            run_phase, notify_ui, "/tmp", "goal", {}, {}, True, {}
        )
        
        self.assertFalse(passed) # Passed is strictly from result, but loop breaks
        self.assertEqual(attempts, 0)
        self.assertEqual(run_phase.call_count, 1)

if __name__ == '__main__':
    unittest.main()
