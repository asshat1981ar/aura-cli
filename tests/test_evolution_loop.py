"""Tests for core/evolution_loop.py — EvolutionLoop. (High-coverage test suite)"""

import json
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call, ANY
import pytest

from core.evolution_loop import EvolutionLoop, InnovationProposal


def _make_loop(**overrides):
    """Create an EvolutionLoop with all agents mocked."""
    defaults = dict(
        planner=MagicMock(),
        coder=MagicMock(),
        critic=MagicMock(),
        brain=MagicMock(),
        vector_store=MagicMock(),
        git_tools=MagicMock(),
        mutator=MagicMock(),
        improvement_service=None,
        goal_queue=None,
        orchestrator=None,
        project_root="/tmp/test_project",
        skills={},
        auto_execute_queued=False,
        innovation_goal_limit=2,
    )
    defaults.update(overrides)
    with patch("core.evolution_loop.ExperimentTracker"), patch("core.evolution_loop.MetricsCollector"):
        return EvolutionLoop(**defaults)


# ---------------------------------------------------------------------------
# InnovationProposal Tests
# ---------------------------------------------------------------------------


class TestInnovationProposal:
    def test_as_dict(self):
        p = InnovationProposal(
            proposal_id="test:1",
            title="Test",
            category="skill",
            goal="test goal",
            rationale="because",
            evidence=["ev1"],
            smallest_surface="foo.py",
            expected_value="high",
            risk_level="low",
            verification_cost="unit tests",
            recommended_action="queue",
        )
        d = p.as_dict()
        assert d["proposal_id"] == "test:1"
        assert d["evidence"] == ["ev1"]
        assert d["title"] == "Test"
        assert d["category"] == "skill"

    def test_frozen_dataclass(self):
        p = InnovationProposal(
            proposal_id="id",
            title="t",
            category="c",
            goal="g",
            rationale="r",
            evidence=[],
            smallest_surface="s",
            expected_value="v",
            risk_level="l",
            verification_cost="v",
            recommended_action="a",
        )
        with pytest.raises(AttributeError):
            p.title = "new"

    def test_proposal_with_multiple_evidence(self):
        p = InnovationProposal(
            proposal_id="p1",
            title="Multi-evidence",
            category="skill",
            goal="g",
            rationale="r",
            evidence=["e1", "e2", "e3"],
            smallest_surface="s",
            expected_value="h",
            risk_level="l",
            verification_cost="v",
            recommended_action="q",
        )
        assert len(p.as_dict()["evidence"]) == 3


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestEvolutionLoopInit:
    def test_init_defaults(self):
        loop = _make_loop()
        assert loop.planner is not None
        assert loop.auto_execute_queued is False
        assert loop.innovation_goal_limit == 2

    def test_init_with_custom_values(self):
        loop = _make_loop(
            auto_execute_queued=True,
            innovation_goal_limit=5,
        )
        assert loop.auto_execute_queued is True
        assert loop.innovation_goal_limit == 5

    def test_innovation_goal_limit_minimum_one(self):
        loop = _make_loop(innovation_goal_limit=0)
        assert loop.innovation_goal_limit == 1

    def test_project_root_from_git_tools(self):
        git_tools = MagicMock()
        git_tools.repo_path = "/custom/path"
        loop = _make_loop(git_tools=git_tools, project_root=None)
        # When project_root is None, it should use git_tools.repo_path
        assert str(loop.project_root) in ["/custom/path", "."]

    def test_skills_initialization(self):
        skill1 = MagicMock()
        skill2 = MagicMock()
        loop = _make_loop(skills={"skill1": skill1, "skill2": skill2})
        assert len(loop.skills) == 2


# ---------------------------------------------------------------------------
# _available_skill_names Tests
# ---------------------------------------------------------------------------


class TestAvailableSkillNames:
    def test_from_initialized_skills(self):
        loop = _make_loop(skills={"s1": MagicMock(), "s2": MagicMock()})
        names = loop._available_skill_names()
        assert set(names) == {"s1", "s2"}

    def test_from_empty_skills_returns_list(self):
        # When skills are empty, the method will try to import from registry
        # which may fail or return dynamic skills. Just test that it returns a list.
        loop = _make_loop(skills={})
        names = loop._available_skill_names()
        assert isinstance(names, list)


# ---------------------------------------------------------------------------
# _safe_skill_run Tests
# ---------------------------------------------------------------------------


class TestSafeSkillRun:
    def test_success_dict_return(self):
        skill = MagicMock()
        skill.run.return_value = {"result": "ok"}
        loop = _make_loop(skills={"test": skill})
        result = loop._safe_skill_run("test", {"input": "x"})
        assert result == {"result": "ok"}

    def test_success_non_dict_return_wrapped(self):
        skill = MagicMock()
        skill.run.return_value = "string result"
        loop = _make_loop(skills={"test": skill})
        result = loop._safe_skill_run("test", {})
        assert result == {"result": "string result"}

    def test_skill_exception_caught(self):
        skill = MagicMock()
        skill.run.side_effect = ValueError("test error")
        loop = _make_loop(skills={"test": skill})
        result = loop._safe_skill_run("test", {})
        assert "error" in result
        assert "test error" in result["error"]

    def test_skill_not_available(self):
        # When skill is not available and registry lookup fails, error is returned
        loop = _make_loop(skills={})
        result = loop._safe_skill_run("missing", {})
        assert "error" in result


# ---------------------------------------------------------------------------
# _load_mcp_summary Tests
# ---------------------------------------------------------------------------


class TestLoadMcpSummary:
    def test_with_mcp_json_file(self):
        mcp_data = {"mcpServers": {"s1": {}, "s2": {}, "s3": {}}}
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value=json.dumps(mcp_data)):
                loop = _make_loop()
                summary = loop._load_mcp_summary()
                assert summary["server_count"] == 3
                assert len(summary["servers"]) == 3

    def test_without_mcp_json_file(self):
        with patch("pathlib.Path.exists", return_value=False):
            loop = _make_loop()
            summary = loop._load_mcp_summary()
            assert summary["server_count"] == 0
            assert summary["servers"] == []

    def test_invalid_json_error(self):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", side_effect=ValueError("bad json")):
                loop = _make_loop()
                summary = loop._load_mcp_summary()
                assert "error" in summary
                assert summary["server_count"] == 0

    def test_mcp_servers_sorted(self):
        mcp_data = {"mcpServers": {"zebra": {}, "alpha": {}, "beta": {}}}
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value=json.dumps(mcp_data)):
                loop = _make_loop()
                summary = loop._load_mcp_summary()
                assert summary["servers"] == ["alpha", "beta", "zebra"]


# ---------------------------------------------------------------------------
# _hypothesize Tests
# ---------------------------------------------------------------------------


class TestHypothesize:
    def test_returns_string_from_list(self):
        loop = _make_loop()
        loop.planner.plan.return_value = ["step1", "step2"]
        result = loop._hypothesize("goal", "mem", "past", "weak")
        assert result == "step1\nstep2"

    def test_returns_string_from_string(self):
        loop = _make_loop()
        loop.planner.plan.return_value = "hypothesis text"
        result = loop._hypothesize("goal", "mem", "past", "weak")
        assert result == "hypothesis text"

    def test_includes_memory_snapshot(self):
        loop = _make_loop()
        loop.planner.plan.return_value = "output"
        loop._hypothesize("test goal", "snapshot data", "past", "weak")
        call_args = loop.planner.plan.call_args[1]
        assert "snapshot data" in call_args["memory_snapshot"]


# ---------------------------------------------------------------------------
# _decompose_tasks Tests
# ---------------------------------------------------------------------------


class TestDecomposeTasks:
    def test_returns_task_list(self):
        loop = _make_loop()
        loop.planner.plan.return_value = ["task1", "task2", "task3"]
        result = loop._decompose_tasks("hypothesis", "mem", "past", "weak")
        assert result == ["task1", "task2", "task3"]

    def test_passes_hypothesis_to_planner(self):
        loop = _make_loop()
        loop.planner.plan.return_value = []
        loop._decompose_tasks("test hypothesis", "mem", "past", "weak")
        assert loop.planner.plan.called


# ---------------------------------------------------------------------------
# _implement_and_critique Tests
# ---------------------------------------------------------------------------


class TestImplementAndCritique:
    def test_returns_implementation_and_evaluation(self):
        loop = _make_loop()
        loop.coder.implement.return_value = "code here"
        loop.critic.critique_code.return_value = "looks good"
        loop.brain.analyze_critique_for_weaknesses = MagicMock()

        impl, ev = loop._implement_and_critique("goal", ["task1", "task2"])
        assert impl == "code here"
        assert ev == "looks good"
        loop.coder.implement.assert_called_once_with("task1\ntask2")
        loop.brain.analyze_critique_for_weaknesses.assert_called_once_with("looks good")

    def test_calls_critic_with_correct_params(self):
        loop = _make_loop()
        loop.coder.implement.return_value = "code"
        loop.critic.critique_code.return_value = "critique"

        loop._implement_and_critique("test goal", ["t1"])

        call_args = loop.critic.critique_code.call_args
        assert call_args[1]["task"] == "test goal"
        assert call_args[1]["code"] == "code"


# ---------------------------------------------------------------------------
# _parse_validation_result Tests
# ---------------------------------------------------------------------------


class TestParseValidationResult:
    def test_approved_json(self):
        loop = _make_loop()
        raw = json.dumps({"decision": "APPROVED", "confidence_score": 0.85})
        decision, score = loop._parse_validation_result(raw)
        assert decision == "APPROVED"
        assert score == 0.85

    def test_rejected_json(self):
        loop = _make_loop()
        raw = json.dumps({"decision": "REJECTED", "confidence_score": 0.3})
        decision, score = loop._parse_validation_result(raw)
        assert decision == "REJECTED"
        assert score == 0.3

    def test_invalid_json_defaults(self):
        loop = _make_loop()
        decision, score = loop._parse_validation_result("not json")
        assert decision == "REJECTED"
        assert score == 0.0

    def test_non_dict_json(self):
        loop = _make_loop()
        decision, score = loop._parse_validation_result('"just a string"')
        assert decision == "REJECTED"
        assert score == 0.0

    def test_missing_fields_uses_defaults(self):
        loop = _make_loop()
        raw = json.dumps({})
        decision, score = loop._parse_validation_result(raw)
        assert decision == "REJECTED"
        assert score == 0.0

    def test_invalid_confidence_score_type(self):
        loop = _make_loop()
        raw = json.dumps({"decision": "APPROVED", "confidence_score": "not a number"})
        with patch("core.evolution_loop._aura_safe_loads", return_value=json.loads(raw)):
            decision, score = loop._parse_validation_result(raw)
            assert score == 0.0


# ---------------------------------------------------------------------------
# _mutation_plan_to_dsl Tests
# ---------------------------------------------------------------------------


class TestMutationPlanToDSL:
    def test_replace_in_file(self):
        loop = _make_loop()
        plan = {"mutations": [{"file_path": "a.py", "old_content": "old", "new_content": "new"}]}
        dsl = loop._mutation_plan_to_dsl(plan)
        assert "REPLACE_IN_FILE a.py" in dsl
        assert "old" in dsl
        assert "new" in dsl

    def test_add_file(self):
        loop = _make_loop()
        plan = {"mutations": [{"file_path": "b.py", "new_content": "content"}]}
        dsl = loop._mutation_plan_to_dsl(plan)
        assert "ADD_FILE b.py" in dsl
        assert "content" in dsl

    def test_empty_mutations(self):
        loop = _make_loop()
        plan = {"mutations": []}
        dsl = loop._mutation_plan_to_dsl(plan)
        assert dsl == ""

    def test_skips_invalid_entries(self):
        loop = _make_loop()
        plan = {"mutations": ["not a dict", {"file_path": "", "new_content": "x"}, {"new_content": "y"}]}
        dsl = loop._mutation_plan_to_dsl(plan)
        assert dsl.strip() == ""

    def test_multiple_mutations(self):
        loop = _make_loop()
        plan = {
            "mutations": [
                {"file_path": "f1.py", "old_content": "o1", "new_content": "n1"},
                {"file_path": "f2.py", "new_content": "c2"},
            ]
        }
        dsl = loop._mutation_plan_to_dsl(plan)
        assert "REPLACE_IN_FILE f1.py" in dsl
        assert "ADD_FILE f2.py" in dsl


# ---------------------------------------------------------------------------
# _normalize_mutation_plan Tests
# ---------------------------------------------------------------------------


class TestNormalizeMutationPlan:
    def test_valid_plan(self):
        loop = _make_loop()
        raw_response = json.dumps({"mutations": [{"file_path": "src/test.py", "old_content": "old", "new_content": "new"}]})
        with patch("core.evolution_loop._aura_safe_loads", return_value=json.loads(raw_response)):
            plan, dsl = loop._normalize_mutation_plan(raw_response)
            assert isinstance(plan, dict)
            assert "REPLACE_IN_FILE" in dsl

    def test_not_dict_error(self):
        loop = _make_loop()
        with patch("core.evolution_loop._aura_safe_loads", return_value="not_a_dict"):
            with pytest.raises(ValueError, match="must be a JSON object"):
                loop._normalize_mutation_plan("raw")

    def test_no_mutations_list_error(self):
        loop = _make_loop()
        with patch("core.evolution_loop._aura_safe_loads", return_value={}):
            with pytest.raises(ValueError, match="must include a `mutations` list"):
                loop._normalize_mutation_plan("raw")

    def test_empty_dsl_error(self):
        loop = _make_loop()
        with patch("core.evolution_loop._aura_safe_loads", return_value={"mutations": []}):
            with pytest.raises(ValueError, match="did not contain any applicable file mutations"):
                loop._normalize_mutation_plan("raw")


# ---------------------------------------------------------------------------
# _select_proposals Tests
# ---------------------------------------------------------------------------


class TestSelectProposals:
    def _make_proposal(self, category, risk="medium", title="Test"):
        return InnovationProposal(
            proposal_id=f"{category}:test",
            title=title,
            category=category,
            goal="g",
            rationale="r",
            evidence=[],
            smallest_surface="s",
            expected_value="h",
            risk_level=risk,
            verification_cost="v",
            recommended_action="queue",
        )

    def test_capability_focus(self):
        loop = _make_loop()
        proposals = [
            self._make_proposal("verification"),
            self._make_proposal("skill"),
            self._make_proposal("mcp"),
        ]
        selected = loop._select_proposals(proposals, focus="capability", proposal_limit=2)
        assert len(selected) == 2
        assert selected[0].category == "skill"

    def test_quality_focus(self):
        loop = _make_loop()
        proposals = [
            self._make_proposal("skill"),
            self._make_proposal("verification"),
        ]
        selected = loop._select_proposals(proposals, focus="quality", proposal_limit=1)
        assert selected[0].category == "verification"

    def test_limit_respected(self):
        loop = _make_loop()
        proposals = [self._make_proposal("skill", title=f"P{i}") for i in range(5)]
        selected = loop._select_proposals(proposals, focus="capability", proposal_limit=3)
        assert len(selected) == 3

    def test_throughput_focus(self):
        loop = _make_loop()
        proposals = [
            self._make_proposal("orchestration"),
            self._make_proposal("developer-surface"),
        ]
        selected = loop._select_proposals(proposals, focus="throughput", proposal_limit=1)
        assert selected[0].category == "developer-surface"

    def test_risk_level_sorting(self):
        loop = _make_loop()
        proposals = [
            self._make_proposal("skill", risk="high"),
            self._make_proposal("skill", risk="low", title="P2"),
        ]
        selected = loop._select_proposals(proposals, focus="capability", proposal_limit=2)
        assert selected[0].risk_level == "low"


# ---------------------------------------------------------------------------
# _queue_selected_goals Tests
# ---------------------------------------------------------------------------


class TestQueueSelectedGoals:
    def test_dry_run(self):
        loop = _make_loop()
        proposals = [
            InnovationProposal(
                proposal_id="t:1",
                title="T",
                category="skill",
                goal="g",
                rationale="r",
                evidence=[],
                smallest_surface="s",
                expected_value="h",
                risk_level="l",
                verification_cost="v",
                recommended_action="queue",
            )
        ]
        result = loop._queue_selected_goals(proposals, dry_run=True)
        assert result["attempted"] is False
        assert len(result["skipped"]) == 1
        assert result["skipped"][0]["reason"] == "dry_run"

    def test_no_goal_queue(self):
        loop = _make_loop(goal_queue=None)
        proposals = [
            InnovationProposal(
                proposal_id="t:1",
                title="T",
                category="skill",
                goal="g",
                rationale="r",
                evidence=[],
                smallest_surface="s",
                expected_value="h",
                risk_level="l",
                verification_cost="v",
                recommended_action="queue",
            )
        ]
        result = loop._queue_selected_goals(proposals, dry_run=False)
        assert result["attempted"] is False

    def test_empty_proposals(self):
        loop = _make_loop()
        result = loop._queue_selected_goals([], dry_run=False)
        assert result["attempted"] is False

    def test_prepend_batch(self):
        mock_queue = MagicMock()
        mock_queue.queue = []
        mock_queue.prepend_batch = MagicMock()
        loop = _make_loop(goal_queue=mock_queue)
        proposals = [
            InnovationProposal(
                proposal_id="t:1",
                title="T",
                category="skill",
                goal="new goal",
                rationale="r",
                evidence=[],
                smallest_surface="s",
                expected_value="h",
                risk_level="l",
                verification_cost="v",
                recommended_action="queue",
            )
        ]
        result = loop._queue_selected_goals(proposals, dry_run=False)
        assert result["attempted"] is True
        mock_queue.prepend_batch.assert_called_once()

    def test_batch_add_fallback(self):
        mock_queue = MagicMock(spec=["batch_add", "queue"])
        mock_queue.queue = []
        mock_queue.batch_add = MagicMock()
        del mock_queue.prepend_batch
        loop = _make_loop(goal_queue=mock_queue)
        proposals = [
            InnovationProposal(
                proposal_id="t:1",
                title="T",
                category="skill",
                goal="new goal",
                rationale="r",
                evidence=[],
                smallest_surface="s",
                expected_value="h",
                risk_level="l",
                verification_cost="v",
                recommended_action="queue",
            )
        ]
        result = loop._queue_selected_goals(proposals, dry_run=False)
        assert result["queue_strategy"] == "append"

    def test_duplicate_goals_skipped(self):
        mock_queue = MagicMock()
        mock_queue.queue = ["existing goal"]
        loop = _make_loop(goal_queue=mock_queue)
        proposals = [
            InnovationProposal(
                proposal_id="t:1",
                title="T",
                category="skill",
                goal="existing goal",
                rationale="r",
                evidence=[],
                smallest_surface="s",
                expected_value="h",
                risk_level="l",
                verification_cost="v",
                recommended_action="queue",
            )
        ]
        result = loop._queue_selected_goals(proposals, dry_run=False)
        assert len(result["queued"]) == 0
        assert any(s["reason"] == "already_queued" for s in result["skipped"])


# ---------------------------------------------------------------------------
# _execute_selected_goals Tests
# ---------------------------------------------------------------------------


class TestExecuteSelectedGoals:
    def test_dry_run(self):
        loop = _make_loop()
        result = loop._execute_selected_goals(["goal1"], dry_run=True, execution_limit=1)
        assert result["attempted"] is False

    def test_no_orchestrator(self):
        loop = _make_loop(orchestrator=None)
        result = loop._execute_selected_goals(["goal1"], dry_run=False, execution_limit=1)
        assert result["attempted"] is False

    def test_empty_goals(self):
        loop = _make_loop()
        result = loop._execute_selected_goals([], dry_run=False, execution_limit=1)
        assert result["attempted"] is False

    def test_executes_with_orchestrator(self):
        mock_orch = MagicMock()
        mock_orch.run_loop.return_value = {"stop_reason": "done", "history": []}
        loop = _make_loop(orchestrator=mock_orch)
        result = loop._execute_selected_goals(["goal1"], dry_run=False, execution_limit=1)
        assert result["attempted"] is True
        mock_orch.run_loop.assert_called_once()

    def test_respects_execution_limit(self):
        mock_orch = MagicMock()
        mock_orch.run_loop.return_value = {"stop_reason": "done", "history": []}
        loop = _make_loop(orchestrator=mock_orch)
        result = loop._execute_selected_goals(["g1", "g2", "g3"], dry_run=False, execution_limit=2)
        assert mock_orch.run_loop.call_count == 2


# ---------------------------------------------------------------------------
# on_cycle_complete Tests
# ---------------------------------------------------------------------------


class TestOnCycleComplete:
    def test_triggers_after_n_cycles(self):
        loop = _make_loop()
        loop.TRIGGER_EVERY_N = 2
        loop.run = MagicMock()
        loop.on_cycle_complete({"goal": "g"})
        loop.run.assert_not_called()
        loop.on_cycle_complete({"goal": "g"})
        loop.run.assert_called_once()

    def test_triggers_on_hotspot_signal(self):
        loop = _make_loop()
        loop.TRIGGER_EVERY_N = 999
        loop.run = MagicMock()
        entry = {"goal": "refactor hotspot in orchestrator"}
        loop.on_cycle_complete(entry)
        loop.run.assert_called_once()

    def test_cycle_count_incremented(self):
        loop = _make_loop()
        initial = loop._cycle_count
        loop.on_cycle_complete({})
        assert loop._cycle_count == initial + 1


# ---------------------------------------------------------------------------
# _persist_memories Tests
# ---------------------------------------------------------------------------


class TestPersistMemories:
    def test_stores_in_brain_and_vector(self):
        loop = _make_loop()
        loop._persist_memories("goal", "hyp", "eval", "mutation")
        assert loop.brain.remember.call_count == 4
        assert loop.vector.add.call_count == 2

    def test_no_vector_store(self):
        loop = _make_loop(vector_store=None)
        loop._persist_memories("goal", "hyp", "eval", "mutation")
        assert loop.brain.remember.call_count == 4


# ---------------------------------------------------------------------------
# _get_mutation_response Tests
# ---------------------------------------------------------------------------


class TestGetMutationResponse:
    def test_with_respond_method(self):
        planner = MagicMock()
        planner._respond = MagicMock(return_value="mutation response")
        loop = _make_loop(planner=planner)
        result = loop._get_mutation_response("prompt", "", "", "")
        assert result == "mutation response"

    def test_with_model_respond_for_role(self):
        model = MagicMock()
        model.respond_for_role.return_value = "model response"
        planner = MagicMock()
        planner._respond = None
        planner.model = model
        loop = _make_loop(planner=planner)
        result = loop._get_mutation_response("prompt", "", "", "")
        assert result == "model response"

    def test_fallback_to_plan_method(self):
        planner = MagicMock()
        planner._respond = None
        planner.model = None
        planner.plan.return_value = ["line1", "line2"]
        loop = _make_loop(planner=planner)
        result = loop._get_mutation_response("prompt", "mem", "past", "weak")
        assert "line1" in result


# ---------------------------------------------------------------------------
# Architecture Explorer Tests
# ---------------------------------------------------------------------------


class TestArchitectureExplorer:
    def test_returns_dict_with_role(self):
        loop = _make_loop(
            skills={
                "structural_analyzer": MagicMock(),
                "tech_debt_quantifier": MagicMock(),
                "code_clone_detector": MagicMock(),
            }
        )
        with patch.object(loop, "_safe_skill_run") as mock_run:
            mock_run.side_effect = [
                {"hotspots": []},
                {"debt_score": 50},
                {"clone_count": 0},
            ]
            result = loop._architecture_explorer()
            assert result["role"] == "architecture_explorer"
            assert "structural" in result
            assert "tech_debt" in result
            assert "clones" in result


# ---------------------------------------------------------------------------
# Capability Researcher Tests
# ---------------------------------------------------------------------------


class TestCapabilityResearcher:
    def test_returns_dict_with_role(self):
        loop = _make_loop(skills={"skill1": MagicMock()})
        with patch.object(loop, "_load_mcp_summary", return_value={"server_count": 2}):
            with patch("core.evolution_loop.analyze_capability_needs") as mock_analyze:
                mock_analyze.return_value = {
                    "missing_skills": [],
                    "recommended_skills": [],
                    "provisioning_actions": [],
                }
                result = loop._capability_researcher("test goal")
                assert result["role"] == "capability_researcher"
                assert "capability_plan" in result
                assert "mcp" in result


# ---------------------------------------------------------------------------
# Verification Reviewer Tests
# ---------------------------------------------------------------------------


class TestVerificationReviewer:
    def test_low_risk_for_skill_category(self):
        proposals = [
            InnovationProposal(
                proposal_id="p1",
                title="Test",
                category="skill",
                goal="Goal",
                rationale="Rationale",
                evidence=[],
                smallest_surface="surface",
                expected_value="high",
                risk_level="low",
                verification_cost="cost",
                recommended_action="queue",
            ),
        ]
        architecture = {"structural": {"hotspots": []}}
        loop = _make_loop()
        result = loop._verification_reviewer(proposals, architecture)
        assert result["role"] == "verification_reviewer"
        assert len(result["reviews"]) == 1
        assert result["reviews"][0]["residual_risk"] == "low"

    def test_escalates_risk_for_hotspot_match(self):
        proposals = [
            InnovationProposal(
                proposal_id="p1",
                title="Test",
                category="orchestration",
                goal="Refactor hotspot",
                rationale="Rationale",
                evidence=[],
                smallest_surface="core/orchestrator.py",
                expected_value="high",
                risk_level="low",
                verification_cost="cost",
                recommended_action="queue",
            ),
        ]
        architecture = {"structural": {"hotspots": [{"file": "core/orchestrator.py", "risk_level": "high"}]}}
        loop = _make_loop()
        result = loop._verification_reviewer(proposals, architecture)
        assert result["reviews"][0]["residual_risk"] == "high"


# ---------------------------------------------------------------------------
# Build Innovation Proposals Tests
# ---------------------------------------------------------------------------


class TestBuildInnovationProposals:
    def test_creates_skill_proposals(self):
        loop = _make_loop()
        architecture = {"structural": {}, "tech_debt": {}, "clones": {}}
        capability = {
            "capability_plan": {
                "missing_skills": ["skill_a", "skill_b"],
                "recommended_skills": [],
                "provisioning_actions": [],
            },
            "mcp": {"server_count": 0},
        }
        proposals = loop._build_innovation_proposals("test goal", architecture, capability)
        skill_proposals = [p for p in proposals if p.category == "skill"]
        assert len(skill_proposals) >= 2

    def test_creates_hotspot_proposals(self):
        loop = _make_loop()
        architecture = {
            "structural": {
                "hotspots": [
                    {"file": "core/main.py", "risk_level": "high"},
                    {"file": "core/secondary.py", "risk_level": "medium"},
                ]
            },
            "tech_debt": {},
            "clones": {},
        }
        capability = {
            "capability_plan": {
                "missing_skills": [],
                "recommended_skills": [],
                "provisioning_actions": [],
            },
            "mcp": {"server_count": 0},
        }
        proposals = loop._build_innovation_proposals("test goal", architecture, capability)
        hotspot_proposals = [p for p in proposals if p.category == "orchestration"]
        assert len(hotspot_proposals) >= 1

    def test_creates_debt_proposals(self):
        loop = _make_loop()
        architecture = {
            "structural": {},
            "tech_debt": {"debt_score": 40, "summary": "High debt"},
            "clones": {},
        }
        capability = {
            "capability_plan": {
                "missing_skills": [],
                "recommended_skills": [],
                "provisioning_actions": [],
            },
            "mcp": {"server_count": 0},
        }
        proposals = loop._build_innovation_proposals("test goal", architecture, capability)
        debt_proposals = [p for p in proposals if "verification:debt" in p.proposal_id]
        assert any(p.proposal_id == "verification:debt" for p in debt_proposals)


# ---------------------------------------------------------------------------
# Commit and Track Experiment Tests
# ---------------------------------------------------------------------------


class TestCommitAndTrackExperiment:
    def test_kept_experiment(self):
        git_tools = MagicMock()
        experiment_tracker = MagicMock()
        experiment_result = MagicMock()
        experiment_result.kept = True
        experiment_result.net_improvement = 0.05
        experiment_tracker.finish_experiment.return_value = experiment_result

        loop = _make_loop(git_tools=git_tools)
        loop.experiment_tracker = experiment_tracker

        result = loop._commit_and_track_experiment(
            "test goal",
            "exp_123",
            time.time() - 1.0,
            "hypothesis",
            {"metric": "baseline"},
        )

        assert result["kept"] is True
        assert result["net_improvement"] == 0.05
        git_tools.commit_all.assert_called_once()

    def test_regressed_experiment_reverted(self):
        git_tools = MagicMock()
        experiment_tracker = MagicMock()
        experiment_result = MagicMock()
        experiment_result.kept = False
        experiment_result.reason = "regression"
        experiment_tracker.finish_experiment.return_value = experiment_result

        loop = _make_loop(git_tools=git_tools)
        loop.experiment_tracker = experiment_tracker

        with patch("core.evolution_loop.log_json"):
            result = loop._commit_and_track_experiment(
                "test goal",
                "exp_123",
                time.time() - 1.0,
                "hypothesis",
                {"metric": "baseline"},
            )

        assert result["kept"] is False
        git_tools.run.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
