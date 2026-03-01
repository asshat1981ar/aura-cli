import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.file_tools import apply_change_with_explicit_overwrite_policy as real_apply_change
from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore


class _StaticAgent:
    def __init__(self, payload):
        self._payload = payload

    def run(self, _input_data):
        return self._payload


def _base_agents(*, change_set):
    return {
        "ingest": _StaticAgent({
            "goal": "test goal",
            "snapshot": "snapshot",
            "memory_summary": "",
            "constraints": {},
        }),
        "plan": _StaticAgent({"steps": ["step"], "risks": []}),
        "critique": _StaticAgent({"issues": [], "fixes": []}),
        "synthesize": _StaticAgent({
            "tasks": [{"id": "t1", "title": "demo", "intent": "", "files": [], "tests": []}],
        }),
        "act": _StaticAgent(change_set),
        "sandbox": _StaticAgent({"passed": True, "summary": "ok"}),
        "verify": _StaticAgent({"status": "pass", "failures": [], "logs": ""}),
        "reflect": _StaticAgent({"summary": "done", "learnings": [], "next_actions": []}),
    }


class TestDryRunWorkflow(unittest.TestCase):
    def _make_orchestrator(self, project_root: Path, *, change_set):
        return LoopOrchestrator(
            agents=_base_agents(change_set=change_set),
            memory_store=MemoryStore(project_root / "memory_store"),
            policy=Policy(max_cycles=1),
            project_root=project_root,
        )

    def test_dry_run_mode_no_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "test_file.py"
            orchestrator = self._make_orchestrator(
                root,
                change_set={
                    "changes": [{
                        "file_path": "test_file.py",
                        "old_code": "",
                        "new_code": "def new_func():\n    return 1\n",
                        "overwrite_file": True,
                    }],
                },
            )

            with patch("core.orchestrator.apply_change_with_explicit_overwrite_policy") as mock_apply:
                result = orchestrator.run_loop("Test dry run functionality", max_cycles=1, dry_run=True)

            self.assertEqual(result["stop_reason"], "PASS")
            mock_apply.assert_not_called()
            self.assertFalse(target.exists())

            entry = result["history"][0]
            self.assertEqual(entry["phase_outputs"]["apply_result"]["applied"], ["test_file.py"])
            self.assertEqual(entry["phase_outputs"]["verification"]["status"], "pass")

    def test_normal_run_mode_applies_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "test_file.py"
            target.write_text("def old_func():\n    pass\n", encoding="utf-8")

            orchestrator = self._make_orchestrator(
                root,
                change_set={
                    "changes": [{
                        "file_path": "test_file.py",
                        "old_code": "def old_func():\n    pass\n",
                        "new_code": "def new_func():\n    return 1\n",
                        "overwrite_file": False,
                    }],
                },
            )

            with patch(
                "core.orchestrator.apply_change_with_explicit_overwrite_policy",
                wraps=real_apply_change,
            ) as mock_apply:
                result = orchestrator.run_loop("Test normal run functionality", max_cycles=1, dry_run=False)

            self.assertEqual(result["stop_reason"], "PASS")
            mock_apply.assert_called_once()
            self.assertEqual(target.read_text(encoding="utf-8"), "def new_func():\n    return 1\n")

            entry = result["history"][0]
            self.assertEqual(entry["phase_outputs"]["apply_result"]["failed"], [])
            self.assertEqual(entry["phase_outputs"]["verification"]["status"], "pass")
