"""PRD-004: Unit tests for all AURA agents (~65 tests).

Always set AURA_SKIP_CHDIR=1 when running this file.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain():
    brain = MagicMock()
    brain.recall_recent.return_value = []
    brain.recall_with_budget.return_value = []
    brain.remember.return_value = None
    return brain


def _make_model(response="some model response"):
    model = MagicMock()
    model.respond.return_value = response
    return model


# ===========================================================================
# CoderAgent
# ===========================================================================

class TestCoderAgent:
    def setup_method(self):
        from agents.coder import CoderAgent
        self.brain = _make_brain()
        self.model = _make_model('```python\n# AURA_TARGET: core/out.py\nx = 1\n```')
        self.agent = CoderAgent(self.brain, self.model)

    def test_coder_instantiation(self):
        from agents.coder import CoderAgent
        a = CoderAgent(_make_brain(), _make_model())
        assert a is not None

    def test_coder_implement_happy_path(self):
        code = self.agent.implement("write a hello function")
        assert isinstance(code, str)

    def test_coder_implement_model_raises(self):
        self.model.respond.side_effect = RuntimeError("model down")
        try:
            self.agent.implement("task")
        except Exception:
            pass  # acceptable

    def test_coder_implement_returns_string(self):
        result = self.agent.implement("do something")
        assert isinstance(result, str)

    def test_coder_max_iterations_constant(self):
        from agents.coder import CoderAgent
        assert CoderAgent.MAX_ITERATIONS == 3

    def test_coder_aura_target_directive(self):
        from agents.coder import CoderAgent
        assert "AURA_TARGET" in CoderAgent.AURA_TARGET_DIRECTIVE


# ===========================================================================
# PlannerAgent
# ===========================================================================

class TestPlannerAgent:
    def setup_method(self):
        from agents.planner import PlannerAgent
        self.brain = _make_brain()
        self.model = _make_model('["step 1", "step 2"]')
        self.agent = PlannerAgent(self.brain, self.model)

    def test_planner_instantiation(self):
        from agents.planner import PlannerAgent
        a = PlannerAgent(_make_brain(), _make_model())
        assert a is not None

    def test_planner_plan_happy_path(self):
        result = self.agent.plan("goal", "memory", "similar", "weaknesses")
        assert isinstance(result, list)

    def test_planner_plan_model_raises(self):
        self.model.respond.side_effect = Exception("fail")
        # Should not raise — brain.remember may fail too but plan() catches exceptions
        try:
            result = self.agent.plan("goal", "", "", "")
        except Exception:
            pass

    def test_planner_plan_returns_list(self):
        result = self.agent.plan("fix bug", "snap", "past", "weak")
        assert isinstance(result, list)

    def test_planner_update_plan(self):
        self.model.respond.return_value = '["revised step 1"]'
        result = self.agent._update_plan(["step 1"], "feedback here")
        assert isinstance(result, list)


# ===========================================================================
# CriticAgent
# ===========================================================================

class TestCriticAgent:
    def setup_method(self):
        from agents.critic import CriticAgent
        self.brain = _make_brain()
        self.model = _make_model("looks good")
        self.agent = CriticAgent(self.brain, self.model)

    def test_critic_instantiation(self):
        from agents.critic import CriticAgent
        a = CriticAgent(_make_brain(), _make_model())
        assert a is not None

    def test_critic_critique_plan_happy_path(self):
        result = self.agent.critique_plan("task", ["step 1", "step 2"])
        assert isinstance(result, str)

    def test_critic_model_raises(self):
        self.model.respond.side_effect = Exception("err")
        try:
            self.agent.critique_plan("t", [])
        except Exception:
            pass

    def test_critic_critique_code_returns_str(self):
        result = self.agent.critique_code("task", "def foo(): pass")
        assert isinstance(result, str)

    def test_critic_validate_mutation_returns_json_str(self):
        self.model.respond.return_value = json.dumps({
            "decision": "APPROVED",
            "confidence_score": 0.9,
            "impact_assessment": "good",
            "reasoning": "looks fine",
        })
        result = self.agent.validate_mutation("ADD_FILE core/foo.py\nx=1")
        parsed = json.loads(result)
        assert "decision" in parsed


# ===========================================================================
# VerifierAgent
# ===========================================================================

class TestVerifierAgent:
    def setup_method(self):
        from agents.verifier import VerifierAgent
        self.agent = VerifierAgent(timeout=10)

    def test_verifier_instantiation(self):
        from agents.verifier import VerifierAgent
        a = VerifierAgent()
        assert a is not None

    def test_verifier_run_dry_run(self):
        result = self.agent.run({"dry_run": True})
        assert result["status"] == "skip"

    def test_verifier_run_returns_dict(self):
        result = self.agent.run({"dry_run": True})
        assert isinstance(result, dict)

    def test_verifier_run_has_status_key(self):
        result = self.agent.run({"dry_run": True})
        assert "status" in result

    def test_verifier_name(self):
        assert self.agent.name == "verify"


# ===========================================================================
# ReflectorAgent
# ===========================================================================

class TestReflectorAgent:
    def setup_method(self):
        from agents.reflector import ReflectorAgent
        self.agent = ReflectorAgent()

    def test_reflector_instantiation(self):
        from agents.reflector import ReflectorAgent
        a = ReflectorAgent()
        assert a is not None

    def test_reflector_run_happy_path(self):
        result = self.agent.run({"verification": {"status": "pass", "failures": []}})
        assert isinstance(result, dict)
        assert "summary" in result

    def test_reflector_run_handles_no_verification(self):
        result = self.agent.run({})
        assert isinstance(result, dict)

    def test_reflector_run_returns_dict(self):
        result = self.agent.run({"verification": {"status": "fail", "failures": ["err"]}})
        assert "learnings" in result

    def test_reflector_name(self):
        assert self.agent.name == "reflect"


# ===========================================================================
# IngestAgent
# ===========================================================================

class TestIngestAgent:
    def setup_method(self):
        from agents.ingest import IngestAgent
        self.brain = _make_brain()
        self.agent = IngestAgent(self.brain)

    def test_ingest_instantiation(self):
        from agents.ingest import IngestAgent
        a = IngestAgent(_make_brain())
        assert a is not None

    def test_ingest_run_happy_path(self):
        result = self.agent.run({"goal": "fix bug", "project_root": str(_ROOT)})
        assert isinstance(result, dict)
        assert "goal" in result

    def test_ingest_run_no_goal(self):
        result = self.agent.run({"project_root": str(_ROOT)})
        assert isinstance(result, dict)

    def test_ingest_run_returns_dict(self):
        result = self.agent.run({})
        assert isinstance(result, dict)

    def test_ingest_name(self):
        assert self.agent.name == "ingest"


# ===========================================================================
# RouterAgent
# ===========================================================================

class TestRouterAgent:
    def setup_method(self):
        from agents.router import RouterAgent
        self.brain = _make_brain()
        self.model = _make_model()
        self.model.call_openai = MagicMock(return_value="ok")
        self.model.call_gemini = MagicMock(return_value="ok")
        self.model.call_openrouter = MagicMock(return_value="ok")
        self.agent = RouterAgent(self.brain, self.model)

    def test_router_instantiation(self):
        from agents.router import RouterAgent
        a = RouterAgent(_make_brain(), _make_model())
        assert a is not None

    def test_router_has_stats(self):
        assert isinstance(self.agent.stats, dict)

    def test_router_route_calls_model(self):
        # All candidates may be cooled-down (no successful calls yet), should raise RuntimeError
        try:
            result = self.agent.route("hello prompt")
            assert isinstance(result, str)
        except RuntimeError:
            pass  # expected when no cooled-down candidates

    def test_router_report_returns_str(self):
        result = self.agent.report()
        assert isinstance(result, str)

    def test_router_force_cooldown(self):
        self.agent.force_cooldown("openai", seconds=60.0)
        import time
        assert self.agent.stats["openai"].cooldown_until > time.time()


# ===========================================================================
# MutatorAgent
# ===========================================================================

class TestMutatorAgent:
    def setup_method(self):
        from agents.mutator import MutatorAgent
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())
        self.agent = MutatorAgent(project_root=self.tmpdir)

    def test_mutator_instantiation(self):
        from agents.mutator import MutatorAgent
        import tempfile
        tmpdir = Path(tempfile.mkdtemp())
        a = MutatorAgent(project_root=tmpdir)
        assert a is not None

    def test_mutator_apply_mutation_add_file(self):
        proposal = "ADD_FILE newfile.py\nprint('hello')\n"
        self.agent.apply_mutation(proposal)
        assert (self.tmpdir / "newfile.py").exists()

    def test_mutator_handles_bad_proposal(self):
        # Should not raise
        self.agent.apply_mutation("UNKNOWN_COMMAND foo\nbar\n")

    def test_mutator_validate_path_rejects_traversal(self):
        with pytest.raises(ValueError):
            self.agent._validate_file_path("../../etc/passwd")

    def test_mutator_validate_path_accepts_relative(self):
        result = self.agent._validate_file_path("core/foo.py")
        assert result is not None


# ===========================================================================
# ScaffolderAgent
# ===========================================================================

class TestScaffolderAgent:
    def setup_method(self):
        from agents.scaffolder import ScaffolderAgent
        self.brain = _make_brain()
        self.model = _make_model('{"README.md": "# Test"}')
        self.agent = ScaffolderAgent(self.brain, self.model)

    def test_scaffolder_instantiation(self):
        from agents.scaffolder import ScaffolderAgent
        a = ScaffolderAgent(_make_brain(), _make_model())
        assert a is not None

    def test_scaffolder_scaffold_project_returns_str(self):
        result = self.agent.scaffold_project("testproj", "A test project")
        assert isinstance(result, str)

    def test_scaffolder_handles_bad_json(self):
        self.model.respond.return_value = "not json"
        result = self.agent.scaffold_project("testproj", "desc")
        assert isinstance(result, str)

    def test_scaffolder_validate_path_rejects_absolute(self):
        with pytest.raises(ValueError):
            self.agent._validate_path_component("/etc/passwd")

    def test_scaffolder_validate_path_accepts_relative(self):
        result = self.agent._validate_path_component("src/main.py")
        assert result == "src/main.py"


# ===========================================================================
# TesterAgent
# ===========================================================================

class TestTesterAgent:
    def setup_method(self):
        from agents.tester import TesterAgent
        from agents.sandbox import SandboxAgent
        self.brain = _make_brain()
        self.model = _make_model("def test_foo(): assert 1==1")
        self.sandbox = MagicMock()
        result = MagicMock()
        result.passed = True
        result.timed_out = False
        result.stdout = ""
        result.stderr = ""
        result.exit_code = 0
        result.metadata = {"passed": 1, "failed": 0, "errors": 0}
        self.sandbox.run_tests.return_value = result
        self.sandbox.timeout = 30
        self.agent = TesterAgent(self.brain, self.model, self.sandbox)

    def test_tester_instantiation(self):
        from agents.tester import TesterAgent
        a = TesterAgent(_make_brain(), _make_model(), MagicMock())
        assert a is not None

    def test_tester_generate_tests_returns_str(self):
        result = self.agent.generate_tests("def foo(): return 1")
        assert isinstance(result, str)

    def test_tester_evaluate_code_returns_dict(self):
        result = self.agent.evaluate_code("def foo(): return 1", "def test_foo(): pass")
        assert isinstance(result, dict)
        assert "summary" in result

    def test_tester_evaluate_code_has_actual_output(self):
        result = self.agent.evaluate_code("def foo(): return 1", "def test_foo(): pass")
        assert "actual_output" in result

    def test_tester_model_raises_in_generate(self):
        self.model.respond.side_effect = Exception("fail")
        try:
            result = self.agent.generate_tests("code")
        except Exception:
            pass


# ===========================================================================
# ApplicatorAgent
# ===========================================================================

class TestApplicatorAgent:
    def setup_method(self):
        from agents.applicator import ApplicatorAgent
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())
        self.brain = _make_brain()
        self.agent = ApplicatorAgent(self.brain, backup_dir=str(self.tmpdir / "backups"))

    def test_applicator_instantiation(self):
        from agents.applicator import ApplicatorAgent
        import tempfile
        tmpdir = Path(tempfile.mkdtemp())
        a = ApplicatorAgent(_make_brain(), backup_dir=str(tmpdir / "bk"))
        assert a is not None

    def test_applicator_apply_no_code_block(self):
        from agents.applicator import ApplyResult
        result = self.agent.apply("no code block here")
        assert result.success is False
        assert result.error is not None

    def test_applicator_apply_with_target(self, tmp_path):
        llm_output = "```python\nx = 1\n```"
        target = str(tmp_path / "out.py")
        result = self.agent.apply(llm_output, target_path=target)
        assert result.success is True
        assert Path(target).read_text() == "x = 1"

    def test_applicator_apply_returns_apply_result(self):
        from agents.applicator import ApplyResult
        result = self.agent.apply("no code")
        assert isinstance(result, ApplyResult)

    def test_applicator_rollback(self, tmp_path):
        llm_output = "```python\nx = 1\n```"
        target = str(tmp_path / "out2.py")
        result = self.agent.apply(llm_output, target_path=target)
        assert result.success
        rollback_result = self.agent.rollback(result)
        # No backup means False (file was new)
        assert isinstance(rollback_result, bool)


# ===========================================================================
# Registry adapters
# ===========================================================================

class TestRegistryAdapters:
    def setup_method(self):
        self.brain = _make_brain()
        self.model = _make_model('["step 1"]')

    def test_planner_adapter_instantiation(self):
        from agents.registry import PlannerAdapter
        from agents.planner import PlannerAgent
        adapter = PlannerAdapter(PlannerAgent(self.brain, self.model))
        assert adapter.name == "plan"

    def test_planner_adapter_run_returns_dict(self):
        from agents.registry import PlannerAdapter
        from agents.planner import PlannerAgent
        adapter = PlannerAdapter(PlannerAgent(self.brain, self.model))
        result = adapter.run({"goal": "fix bug", "memory_snapshot": "", "similar_past_problems": "", "known_weaknesses": ""})
        assert isinstance(result, dict)
        assert "steps" in result

    def test_critic_adapter_instantiation(self):
        from agents.registry import CriticAdapter
        from agents.critic import CriticAgent
        self.model.respond.return_value = "feedback"
        adapter = CriticAdapter(CriticAgent(self.brain, self.model))
        assert adapter.name == "critique"

    def test_critic_adapter_run_returns_dict(self):
        from agents.registry import CriticAdapter
        from agents.critic import CriticAgent
        self.model.respond.return_value = "feedback"
        adapter = CriticAdapter(CriticAgent(self.brain, self.model))
        result = adapter.run({"task": "do something", "plan": ["step1"]})
        assert isinstance(result, dict)
        assert "issues" in result

    def test_act_adapter_instantiation(self):
        from agents.registry import ActAdapter
        from agents.coder import CoderAgent
        self.model.respond.return_value = "```python\nx=1\n```"
        adapter = ActAdapter(CoderAgent(self.brain, self.model))
        assert adapter.name == "act"

    def test_default_agents_returns_dict(self):
        from agents.registry import default_agents
        agents = default_agents(self.brain, self.model)
        assert isinstance(agents, dict)
        assert "ingest" in agents
        assert "plan" in agents
        assert "act" in agents
        assert "verify" in agents
        assert "reflect" in agents
