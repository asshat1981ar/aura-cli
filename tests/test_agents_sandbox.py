"""
Tests for ApplicatorAgent, SandboxAgent, TesterAgent, and SandboxAdapter.

Run with::

    AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_agents_sandbox.py -v
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("AURA_SKIP_CHDIR", "1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brain():
    brain = MagicMock()
    brain.recall_all.return_value = []
    brain.remember.return_value = None
    return brain


# ---------------------------------------------------------------------------
# SandboxAgent
# ---------------------------------------------------------------------------

class TestSandboxAgentRunCode(unittest.TestCase):

    def setUp(self):
        from agents.sandbox import SandboxAgent
        self.agent = SandboxAgent(_brain(), timeout=10)

    def test_run_code_passes_hello(self):
        result = self.agent.run_code("print('hello')")
        self.assertTrue(result.passed)
        self.assertIn("hello", result.stdout)

    def test_run_code_fails_on_syntax_error(self):
        result = self.agent.run_code("def broken(")
        self.assertFalse(result.passed)

    def test_run_code_fails_on_runtime_error(self):
        result = self.agent.run_code("raise ValueError('boom')")
        self.assertFalse(result.passed)
        self.assertIn("ValueError", result.stderr)

    def test_run_code_captures_stdout(self):
        result = self.agent.run_code("print('aura_output_42')")
        self.assertIn("aura_output_42", result.stdout)

    def test_run_code_extra_files(self):
        code = "from helper import greet\nprint(greet())"
        extra = {"helper.py": "def greet(): return 'hi from helper'"}
        result = self.agent.run_code(code, extra_files=extra)
        self.assertTrue(result.passed)
        self.assertIn("hi from helper", result.stdout)

    def test_run_code_returns_sandbox_result_type(self):
        from agents.sandbox import SandboxResult
        result = self.agent.run_code("x = 1")
        self.assertIsInstance(result, SandboxResult)

    def test_sandbox_result_summary(self):
        result = self.agent.run_code("print('ok')")
        s = result.summary()
        self.assertIn("PASS", s)

    def test_sandbox_result_str(self):
        result = self.agent.run_code("print('ok')")
        s = str(result)
        self.assertIn("SandboxResult", s)


class TestSandboxAgentRunTests(unittest.TestCase):

    def setUp(self):
        from agents.sandbox import SandboxAgent
        self.agent = SandboxAgent(_brain(), timeout=15)

    def test_passing_tests(self):
        code = "def add(a, b): return a + b"
        tests = """
def test_add():
    from source import add
    assert add(1, 2) == 3
"""
        result = self.agent.run_tests(code, tests)
        self.assertTrue(result.passed, f"Expected pass but got: {result.stderr}")

    def test_failing_tests(self):
        code = "def add(a, b): return a - b"  # Wrong implementation
        tests = """
def test_add():
    from source import add
    assert add(1, 2) == 3
"""
        result = self.agent.run_tests(code, tests)
        self.assertFalse(result.passed)

    def test_metadata_has_counts(self):
        code = "def f(): return 1"
        tests = "def test_f():\n    from source import f\n    assert f() == 1"
        result = self.agent.run_tests(code, tests)
        self.assertIn("passed", result.metadata)


# ---------------------------------------------------------------------------
# ApplicatorAgent
# ---------------------------------------------------------------------------

class TestApplicatorAgent(unittest.TestCase):

    def setUp(self):
        from agents.applicator import ApplicatorAgent
        self.tmp = tempfile.mkdtemp()
        self.agent = ApplicatorAgent(_brain(), backup_dir=str(Path(self.tmp) / "backups"))

    def test_apply_creates_file(self):
        target = str(Path(self.tmp) / "out.py")
        result = self.agent.apply("```python\nx = 1\n```", target_path=target)
        self.assertTrue(result.success, result.error)
        self.assertTrue(Path(target).exists())
        self.assertEqual(Path(target).read_text().strip(), "x = 1")

    def test_apply_detects_aura_target_directive(self):
        target = str(Path(self.tmp) / "directive.py")
        code = f"```python\n# AURA_TARGET: {target}\nvalue = 42\n```"
        result = self.agent.apply(code)
        self.assertTrue(result.success, result.error)
        self.assertIn("directive.py", result.target_path)

    def test_apply_extracts_code_from_markdown_block(self):
        target = str(Path(self.tmp) / "extracted.py")
        code_block = "```python\nresult = 'extracted'\n```"
        result = self.agent.apply(code_block, target_path=target)
        self.assertTrue(result.success, result.error)
        content = Path(target).read_text()
        self.assertIn("extracted", content)

    def test_apply_creates_backup(self):
        target = str(Path(self.tmp) / "existing.py")
        Path(target).write_text("original content\n")
        result = self.agent.apply("```python\nnew content\n```", target_path=target)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.backup_path)
        self.assertTrue(Path(result.backup_path).exists())

    def test_rollback_restores_original(self):
        target = str(Path(self.tmp) / "rollback.py")
        Path(target).write_text("original\n")
        result = self.agent.apply("```python\nmodified\n```", target_path=target)
        self.assertTrue(result.success)
        rolled = self.agent.rollback(result)
        self.assertTrue(rolled)
        self.assertEqual(Path(target).read_text(), "original\n")

    def test_apply_metadata_has_lines(self):
        target = str(Path(self.tmp) / "lines.py")
        result = self.agent.apply("```python\na = 1\nb = 2\nc = 3\n```", target_path=target)
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.metadata["lines"], 3)

    def test_apply_no_target_no_directive_returns_error(self):
        result = self.agent.apply("```python\nx = 1\n```")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)


# ---------------------------------------------------------------------------
# SandboxAdapter (registry wrapper)
# ---------------------------------------------------------------------------

class TestSandboxAdapter(unittest.TestCase):

    def setUp(self):
        from agents.sandbox import SandboxAgent
        from agents.registry import SandboxAdapter
        self.agent = SandboxAdapter(SandboxAgent(_brain(), timeout=10))

    def _act(self, code):
        return {"changes": [{"new_code": code, "file_path": "x.py", "old_code": ""}]}

    def test_pass_on_valid_code(self):
        result = self.agent.run({"act": self._act("x = 1 + 1"), "dry_run": False})
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["passed"])

    def test_fail_on_runtime_error(self):
        result = self.agent.run({"act": self._act("raise RuntimeError('test')"), "dry_run": False})
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["passed"])

    def test_skip_on_dry_run(self):
        result = self.agent.run({"act": self._act("bad code!!!"), "dry_run": True})
        self.assertEqual(result["status"], "skip")
        self.assertTrue(result["passed"])

    def test_skip_when_no_code(self):
        result = self.agent.run({"act": {"changes": []}, "dry_run": False})
        self.assertEqual(result["status"], "skip")

    def test_details_included(self):
        result = self.agent.run({"act": self._act("print('hello')"), "dry_run": False})
        self.assertIn("details", result)
        self.assertIn("stdout", result["details"])

    def test_summary_string_present(self):
        result = self.agent.run({"act": self._act("x = 42"), "dry_run": False})
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 0)


# ---------------------------------------------------------------------------
# TesterAgent
# ---------------------------------------------------------------------------

class TestTesterAgent(unittest.TestCase):

    def setUp(self):
        from agents.sandbox import SandboxAgent
        from agents.tester import TesterAgent
        brain = _brain()
        model = MagicMock()
        model.respond.return_value = "def test_placeholder(): assert True"
        sandbox = SandboxAgent(brain, timeout=10)
        self.agent = TesterAgent(brain, model, sandbox)

    def test_generate_tests_returns_string(self):
        result = self.agent.generate_tests("def f(x): return x + 1")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_evaluate_code_passing(self):
        code = "def add(a, b): return a + b"
        tests = "def test_add():\n    from source import add\n    assert add(2, 3) == 5"
        result = self.agent.evaluate_code(code, tests)
        self.assertIn("summary", result)
        self.assertIn("actual_output", result)
        self.assertTrue(result["actual_output"]["passed"])

    def test_evaluate_code_failing(self):
        code = "def add(a, b): return 0"
        tests = "def test_add():\n    from source import add\n    assert add(2, 3) == 5"
        result = self.agent.evaluate_code(code, tests)
        self.assertFalse(result["actual_output"]["passed"])
        self.assertIn("FAIL", result["summary"])


if __name__ == "__main__":
    unittest.main()
