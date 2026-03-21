"""Tests for guaranteed-execution lifecycle hooks."""
import json
import unittest

from core.hooks import (
    HookConfig, HookEngine, HookExecution, HookResult, HookTiming,
)


class TestHookConfig(unittest.TestCase):
    def test_from_string(self):
        h = HookConfig.from_dict("echo hello")
        self.assertEqual(h.command, "echo hello")
        self.assertTrue(h.blocking)

    def test_from_dict(self):
        h = HookConfig.from_dict({
            "command": "python3 check.py",
            "blocking": False,
            "timeout_seconds": 10,
        })
        self.assertEqual(h.command, "python3 check.py")
        self.assertFalse(h.blocking)
        self.assertEqual(h.timeout_seconds, 10)

    def test_from_dict_with_extras(self):
        h = HookConfig.from_dict({
            "command": "test",
            "unknown_field": True,
        })
        self.assertEqual(h.command, "test")


class TestHookEngine(unittest.TestCase):
    def test_init_empty(self):
        engine = HookEngine()
        self.assertEqual(len(engine.hooks), 0)
        self.assertEqual(len(engine.history), 0)

    def test_load_config(self):
        config = {
            "hooks": {
                "pre_apply": [
                    {"command": "echo pre", "blocking": True},
                ],
                "post_verify": [
                    "echo post",
                ],
            }
        }
        engine = HookEngine(config)
        self.assertEqual(len(engine.hooks["pre_apply"]), 1)
        self.assertEqual(len(engine.hooks["post_verify"]), 1)

    def test_get_hooks(self):
        config = {"hooks": {"pre_act": [{"command": "echo 1"}]}}
        engine = HookEngine(config)
        hooks = engine.get_hooks(HookTiming.PRE, "act")
        self.assertEqual(len(hooks), 1)
        # No hooks for unregistered phase
        hooks = engine.get_hooks(HookTiming.POST, "act")
        self.assertEqual(len(hooks), 0)

    def test_run_pre_hooks_pass(self):
        config = {"hooks": {"pre_plan": [{"command": "true", "blocking": True}]}}
        engine = HookEngine(config)
        proceed, result_input = engine.run_pre_hooks("plan", {"goal": "test"})
        self.assertTrue(proceed)
        self.assertEqual(len(engine.history), 1)

    def test_run_pre_hooks_block(self):
        config = {"hooks": {"pre_apply": [
            {"command": "exit 2", "blocking": True},
        ]}}
        engine = HookEngine(config)
        proceed, _ = engine.run_pre_hooks("apply", {"changes": []})
        self.assertFalse(proceed)
        self.assertEqual(engine.history[0].result, HookResult.BLOCK)

    def test_run_pre_hooks_modify(self):
        config = {"hooks": {"pre_plan": [
            {"command": 'echo \'{"extra": true}\'', "blocking": True},
        ]}}
        engine = HookEngine(config)
        proceed, modified = engine.run_pre_hooks("plan", {"goal": "test"})
        self.assertTrue(proceed)
        self.assertTrue(modified.get("extra"))

    def test_run_pre_hooks_error_continues(self):
        config = {"hooks": {"pre_act": [
            {"command": "exit 1", "blocking": True},
        ]}}
        engine = HookEngine(config)
        proceed, _ = engine.run_pre_hooks("act", {})
        self.assertTrue(proceed)  # exit 1 is error, not block
        self.assertEqual(engine.history[0].result, HookResult.ERROR)

    def test_run_post_hooks(self):
        config = {"hooks": {"post_verify": [{"command": "echo done"}]}}
        engine = HookEngine(config)
        engine.run_post_hooks("verify", {"passed": True})
        self.assertEqual(len(engine.history), 1)

    def test_hook_timeout(self):
        config = {"hooks": {"pre_act": [
            {"command": "sleep 10", "timeout_seconds": 1},
        ]}}
        engine = HookEngine(config)
        proceed, _ = engine.run_pre_hooks("act", {})
        self.assertTrue(proceed)
        self.assertEqual(engine.history[0].result, HookResult.TIMEOUT)

    def test_audit_log(self):
        config = {"hooks": {"pre_plan": [{"command": "true"}]}}
        engine = HookEngine(config)
        engine.run_pre_hooks("plan", {})
        log = engine.get_audit_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["phase"], "plan")
        self.assertEqual(log[0]["timing"], "pre")
        self.assertIn("command", log[0])
        self.assertIn("timestamp", log[0])

    def test_no_hooks_for_phase(self):
        engine = HookEngine()
        proceed, _ = engine.run_pre_hooks("act", {"goal": "test"})
        self.assertTrue(proceed)
        self.assertEqual(len(engine.history), 0)

    def test_hook_env_vars(self):
        config = {"hooks": {"pre_act": [
            {"command": "echo $AURA_PHASE", "env": {"CUSTOM": "val"}},
        ]}}
        engine = HookEngine(config)
        proceed, _ = engine.run_pre_hooks("act", {})
        self.assertTrue(proceed)
        self.assertIn("act", engine.history[0].stdout)


if __name__ == "__main__":
    unittest.main()
