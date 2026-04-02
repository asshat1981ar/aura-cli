"""Unit tests for agents/schemas.py — Pydantic structured output schemas."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from agents.schemas import (
    DebugStrategy,
    PlanStep,
    PlannerOutput,
    CriticIssue,
    CriticOutput,
    CodeChange,
    CoderOutput,
    MutationValidationOutput,
    InnovationPhase,
    BrainstormingTechnique,
    Idea,
    TechniqueResult,
    InnovationOutput,
    InnovationSessionState,
    MetaConductorOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan_step(n: int = 1) -> PlanStep:
    return PlanStep(
        step_number=n,
        description=f"Step {n} description",
        verification="Run tests",
    )


def _make_idea(novelty: float = 0.8, feasibility: float = 0.7, impact: float = 0.9) -> Idea:
    return Idea(
        description="Use caching to speed up queries",
        technique=BrainstormingTechnique.SCAMPER.value,
        novelty=novelty,
        feasibility=feasibility,
        impact=impact,
    )


def _make_technique_result() -> TechniqueResult:
    return TechniqueResult(
        technique=BrainstormingTechnique.SCAMPER.value,
        ideas=[_make_idea()],
        idea_count=1,
    )


# ---------------------------------------------------------------------------
# TestDebugStrategy
# ---------------------------------------------------------------------------

class TestDebugStrategy:
    def test_valid_debug_strategy(self):
        ds = DebugStrategy(
            summary="NullPointerException in ingest",
            diagnosis="The context object is not initialized before access",
            fix_strategy="Initialize context in __init__ before calling ingest()",
            severity="HIGH",
        )
        assert ds.summary == "NullPointerException in ingest"
        assert ds.diagnosis == "The context object is not initialized before access"
        assert ds.fix_strategy == "Initialize context in __init__ before calling ingest()"
        assert ds.severity == "HIGH"

    def test_fix_instructions_property(self):
        ds = DebugStrategy(
            summary="Error summary",
            diagnosis="Root cause",
            fix_strategy="Apply this fix",
            severity="LOW",
        )
        assert ds.fix_instructions == ds.fix_strategy
        assert ds.fix_instructions == "Apply this fix"

    def test_severity_all_valid_values(self):
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            ds = DebugStrategy(
                summary="s", diagnosis="d", fix_strategy="f", severity=sev
            )
            assert ds.severity == sev

    def test_severity_validation_invalid(self):
        with pytest.raises(ValidationError):
            DebugStrategy(
                summary="s",
                diagnosis="d",
                fix_strategy="f",
                severity="UNKNOWN",
            )

    def test_severity_case_sensitive(self):
        # Pydantic Literal is case-sensitive
        with pytest.raises(ValidationError):
            DebugStrategy(
                summary="s", diagnosis="d", fix_strategy="f", severity="high"
            )


# ---------------------------------------------------------------------------
# TestPlanStep
# ---------------------------------------------------------------------------

class TestPlanStep:
    def test_valid_plan_step_with_target_file(self):
        step = PlanStep(
            step_number=1,
            description="Refactor the loader",
            target_file="core/loader.py",
            verification="Run pytest core/tests/",
        )
        assert step.step_number == 1
        assert step.target_file == "core/loader.py"

    def test_target_file_defaults_to_none(self):
        step = PlanStep(step_number=2, description="Write tests", verification="pytest")
        assert step.target_file is None


# ---------------------------------------------------------------------------
# TestPlannerOutput
# ---------------------------------------------------------------------------

class TestPlannerOutput:
    def _valid_kwargs(self):
        return dict(
            analysis="The goal is clear.",
            gap_assessment="Missing error handling.",
            approach="Add try/except blocks.",
            risk_assessment="Low risk.",
            plan=[_make_plan_step(1), _make_plan_step(2)],
            confidence=0.85,
            total_steps=2,
            estimated_complexity="medium",
        )

    def test_valid_planner_output(self):
        po = PlannerOutput(**self._valid_kwargs())
        assert po.confidence == 0.85
        assert po.total_steps == 2
        assert len(po.plan) == 2
        assert po.estimated_complexity == "medium"

    def test_confidence_lower_bound(self):
        kw = self._valid_kwargs()
        kw["confidence"] = 0.0
        po = PlannerOutput(**kw)
        assert po.confidence == 0.0

    def test_confidence_upper_bound(self):
        kw = self._valid_kwargs()
        kw["confidence"] = 1.0
        po = PlannerOutput(**kw)
        assert po.confidence == 1.0

    def test_confidence_above_one_raises(self):
        kw = self._valid_kwargs()
        kw["confidence"] = 1.01
        with pytest.raises(ValidationError):
            PlannerOutput(**kw)

    def test_confidence_below_zero_raises(self):
        kw = self._valid_kwargs()
        kw["confidence"] = -0.01
        with pytest.raises(ValidationError):
            PlannerOutput(**kw)

    def test_complexity_low(self):
        kw = self._valid_kwargs()
        kw["estimated_complexity"] = "low"
        po = PlannerOutput(**kw)
        assert po.estimated_complexity == "low"

    def test_complexity_high(self):
        kw = self._valid_kwargs()
        kw["estimated_complexity"] = "high"
        po = PlannerOutput(**kw)
        assert po.estimated_complexity == "high"

    def test_complexity_invalid_raises(self):
        kw = self._valid_kwargs()
        kw["estimated_complexity"] = "extreme"
        with pytest.raises(ValidationError):
            PlannerOutput(**kw)

    def test_empty_plan_allowed(self):
        kw = self._valid_kwargs()
        kw["plan"] = []
        kw["total_steps"] = 0
        po = PlannerOutput(**kw)
        assert po.plan == []


# ---------------------------------------------------------------------------
# TestCriticIssue
# ---------------------------------------------------------------------------

class TestCriticIssue:
    def test_valid_critic_issue(self):
        issue = CriticIssue(
            severity="major",
            category="feasibility",
            description="Step 3 is unreachable.",
            recommendation="Merge step 2 and 3.",
        )
        assert issue.severity == "major"
        assert issue.category == "feasibility"

    def test_severity_invalid_raises(self):
        with pytest.raises(ValidationError):
            CriticIssue(
                severity="blocker",
                category="other",
                description="desc",
                recommendation="rec",
            )

    def test_category_invalid_raises(self):
        with pytest.raises(ValidationError):
            CriticIssue(
                severity="minor",
                category="performance",
                description="desc",
                recommendation="rec",
            )

    def test_all_severity_values(self):
        for sev in ("critical", "major", "minor", "suggestion"):
            issue = CriticIssue(
                severity=sev, category="other", description="d", recommendation="r"
            )
            assert issue.severity == sev

    def test_all_category_values(self):
        for cat in ("completeness", "clarity", "feasibility", "alignment", "safety", "other"):
            issue = CriticIssue(
                severity="minor", category=cat, description="d", recommendation="r"
            )
            assert issue.category == cat


# ---------------------------------------------------------------------------
# TestCriticOutput
# ---------------------------------------------------------------------------

class TestCriticOutput:
    def _valid_kwargs(self):
        return dict(
            initial_assessment="Looks promising.",
            completeness_check="All requirements addressed.",
            feasibility_analysis="Steps are feasible.",
            risk_identification="Minimal risk.",
            overall_assessment="approve",
            confidence=0.9,
            summary="Plan is solid.",
        )

    def test_valid_critic_output(self):
        issue = CriticIssue(
            severity="minor",
            category="clarity",
            description="Step 1 is vague.",
            recommendation="Add more detail.",
        )
        co = CriticOutput(**self._valid_kwargs(), issues=[issue], positive_aspects=["Clear structure"])
        assert co.overall_assessment == "approve"
        assert len(co.issues) == 1
        assert co.positive_aspects == ["Clear structure"]

    def test_default_empty_lists(self):
        co = CriticOutput(**self._valid_kwargs())
        assert co.issues == []
        assert co.positive_aspects == []

    def test_all_assessment_values(self):
        for assessment in ("approve", "approve_with_changes", "request_changes", "reject"):
            kw = self._valid_kwargs()
            kw["overall_assessment"] = assessment
            co = CriticOutput(**kw)
            assert co.overall_assessment == assessment

    def test_invalid_assessment_raises(self):
        kw = self._valid_kwargs()
        kw["overall_assessment"] = "defer"
        with pytest.raises(ValidationError):
            CriticOutput(**kw)

    def test_confidence_bounds(self):
        kw = self._valid_kwargs()
        kw["confidence"] = 1.5
        with pytest.raises(ValidationError):
            CriticOutput(**kw)


# ---------------------------------------------------------------------------
# TestCodeChange
# ---------------------------------------------------------------------------

class TestCodeChange:
    def test_valid_code_change(self):
        cc = CodeChange(
            file_path="core/loader.py",
            search_block="def old_load():\n    pass",
            replace_block="def new_load():\n    return []",
            reasoning="Improve loader to return empty list instead of None.",
        )
        assert cc.file_path == "core/loader.py"
        assert "new_load" in cc.replace_block


# ---------------------------------------------------------------------------
# TestCoderOutput
# ---------------------------------------------------------------------------

class TestCoderOutput:
    def _valid_kwargs(self):
        return dict(
            problem_analysis="Need to add caching layer.",
            approach_selection="Use functools.lru_cache.",
            design_considerations="Cache invalidation on write.",
            testing_strategy="Mock cache and verify hits.",
            aura_target="core/cache.py",
            code="from functools import lru_cache\n\n@lru_cache\ndef get(key): ...",
            explanation="LRU cache wraps the get method.",
            confidence=0.92,
        )

    def test_valid_coder_output(self):
        co = CoderOutput(
            **self._valid_kwargs(),
            dependencies=["functools"],
            edge_cases_handled=["empty key", "None value"],
        )
        assert co.aura_target == "core/cache.py"
        assert co.dependencies == ["functools"]
        assert len(co.edge_cases_handled) == 2

    def test_defaults_for_lists(self):
        co = CoderOutput(**self._valid_kwargs())
        assert co.dependencies == []
        assert co.edge_cases_handled == []

    def test_confidence_out_of_range_raises(self):
        kw = self._valid_kwargs()
        kw["confidence"] = -0.1
        with pytest.raises(ValidationError):
            CoderOutput(**kw)

    def test_confidence_exactly_one(self):
        kw = self._valid_kwargs()
        kw["confidence"] = 1.0
        co = CoderOutput(**kw)
        assert co.confidence == 1.0


# ---------------------------------------------------------------------------
# TestMutationValidationOutput
# ---------------------------------------------------------------------------

class TestMutationValidationOutput:
    def _valid_kwargs(self):
        return dict(
            impact_analysis="This mutation affects ingest phase.",
            safety_assessment="No safety concerns identified.",
            effectiveness_evaluation="High probability of success.",
            decision="APPROVED",
            confidence_score=0.88,
            impact_assessment="Positive impact on throughput.",
            reasoning="All checks passed.",
        )

    def test_valid_mutation_output(self):
        mvo = MutationValidationOutput(**self._valid_kwargs())
        assert mvo.decision == "APPROVED"
        assert mvo.confidence_score == 0.88
        assert mvo.recommendations is None

    def test_decision_values(self):
        for decision in ("APPROVED", "REJECTED", "NEEDS_REVISION"):
            kw = self._valid_kwargs()
            kw["decision"] = decision
            mvo = MutationValidationOutput(**kw)
            assert mvo.decision == decision

    def test_invalid_decision_raises(self):
        kw = self._valid_kwargs()
        kw["decision"] = "PENDING"
        with pytest.raises(ValidationError):
            MutationValidationOutput(**kw)

    def test_optional_recommendations_none(self):
        mvo = MutationValidationOutput(**self._valid_kwargs())
        assert mvo.recommendations is None

    def test_optional_recommendations_with_value(self):
        kw = self._valid_kwargs()
        kw["recommendations"] = "Revisit step 2 before re-running."
        mvo = MutationValidationOutput(**kw)
        assert mvo.recommendations == "Revisit step 2 before re-running."

    def test_confidence_score_bounds(self):
        kw = self._valid_kwargs()
        kw["confidence_score"] = 1.1
        with pytest.raises(ValidationError):
            MutationValidationOutput(**kw)


# ---------------------------------------------------------------------------
# TestInnovationEnums
# ---------------------------------------------------------------------------

class TestInnovationEnums:
    def test_innovation_phases_count(self):
        assert len(InnovationPhase) == 5

    def test_innovation_phases_values(self):
        expected = {"immersion", "divergence", "convergence", "incubation", "transformation"}
        actual = {phase.value for phase in InnovationPhase}
        assert actual == expected

    def test_innovation_phase_names(self):
        for name in ("IMMERSION", "DIVERGENCE", "CONVERGENCE", "INCUBATION", "TRANSFORMATION"):
            assert hasattr(InnovationPhase, name)

    def test_brainstorming_techniques_count(self):
        assert len(BrainstormingTechnique) == 8

    def test_brainstorming_techniques_values(self):
        expected = {
            "SCAMPER",
            "Six Thinking Hats",
            "Mind Mapping",
            "Reverse Brainstorming",
            "Worst Idea",
            "Lotus Blossom",
            "Star Brainstorming",
            "Bisociative Association",
        }
        actual = {t.value for t in BrainstormingTechnique}
        assert actual == expected

    def test_brainstorming_technique_is_str_enum(self):
        assert isinstance(BrainstormingTechnique.SCAMPER, str)
        assert BrainstormingTechnique.SCAMPER == "SCAMPER"

    def test_innovation_phase_is_str_enum(self):
        assert isinstance(InnovationPhase.DIVERGENCE, str)
        assert InnovationPhase.DIVERGENCE == "divergence"


# ---------------------------------------------------------------------------
# TestIdea
# ---------------------------------------------------------------------------

class TestIdea:
    def test_valid_idea(self):
        idea = _make_idea()
        assert idea.novelty == 0.8
        assert idea.feasibility == 0.7
        assert idea.impact == 0.9
        assert idea.metadata == {}

    def test_idea_with_metadata(self):
        idea = Idea(
            description="Idea with meta",
            technique="SCAMPER",
            novelty=0.5,
            feasibility=0.5,
            impact=0.5,
            metadata={"source": "divergence", "round": 2},
        )
        assert idea.metadata["source"] == "divergence"

    def test_novelty_above_one_raises(self):
        with pytest.raises(ValidationError):
            Idea(
                description="d", technique="SCAMPER",
                novelty=1.01, feasibility=0.5, impact=0.5,
            )

    def test_novelty_below_zero_raises(self):
        with pytest.raises(ValidationError):
            Idea(
                description="d", technique="SCAMPER",
                novelty=-0.1, feasibility=0.5, impact=0.5,
            )

    def test_feasibility_bounds(self):
        with pytest.raises(ValidationError):
            Idea(
                description="d", technique="SCAMPER",
                novelty=0.5, feasibility=1.5, impact=0.5,
            )

    def test_impact_bounds(self):
        with pytest.raises(ValidationError):
            Idea(
                description="d", technique="SCAMPER",
                novelty=0.5, feasibility=0.5, impact=-1.0,
            )

    def test_boundary_values_accepted(self):
        idea = Idea(
            description="d", technique="SCAMPER",
            novelty=0.0, feasibility=1.0, impact=0.0,
        )
        assert idea.novelty == 0.0
        assert idea.feasibility == 1.0


# ---------------------------------------------------------------------------
# TestTechniqueResult
# ---------------------------------------------------------------------------

class TestTechniqueResult:
    def test_valid_technique_result(self):
        tr = _make_technique_result()
        assert tr.idea_count == 1
        assert tr.execution_time_ms is None

    def test_with_execution_time(self):
        tr = TechniqueResult(
            technique="Mind Mapping",
            ideas=[],
            idea_count=0,
            execution_time_ms=250,
        )
        assert tr.execution_time_ms == 250


# ---------------------------------------------------------------------------
# TestInnovationOutput
# ---------------------------------------------------------------------------

class TestInnovationOutput:
    def _make_output(self, **overrides):
        idea = _make_idea()
        tr = _make_technique_result()
        base = dict(
            session_id="session-abc-123",
            problem_statement="How might we reduce CI build times?",
            phase=InnovationPhase.CONVERGENCE,
            techniques_used=[BrainstormingTechnique.SCAMPER.value],
            all_ideas=[idea],
            selected_ideas=[idea],
            technique_results={BrainstormingTechnique.SCAMPER.value: tr},
            diversity_score=0.75,
            novelty_score=0.80,
            feasibility_score=0.65,
            total_ideas_generated=1,
            total_ideas_selected=1,
        )
        base.update(overrides)
        return InnovationOutput(**base)

    def test_valid_innovation_output(self):
        output = self._make_output()
        assert output.session_id == "session-abc-123"
        assert output.phase == InnovationPhase.CONVERGENCE
        assert output.diversity_score == 0.75
        assert output.reasoning == {}
        assert isinstance(output.timestamp, datetime)

    def test_score_constraints_diversity(self):
        with pytest.raises(ValidationError):
            self._make_output(diversity_score=1.1)

    def test_score_constraints_novelty(self):
        with pytest.raises(ValidationError):
            self._make_output(novelty_score=-0.01)

    def test_score_constraints_feasibility(self):
        with pytest.raises(ValidationError):
            self._make_output(feasibility_score=2.0)

    def test_boundary_scores_accepted(self):
        output = self._make_output(diversity_score=0.0, novelty_score=1.0, feasibility_score=0.0)
        assert output.diversity_score == 0.0
        assert output.novelty_score == 1.0

    def test_reasoning_populated(self):
        output = self._make_output(reasoning={"step1": "analyzed problem"})
        assert output.reasoning["step1"] == "analyzed problem"

    def test_empty_selected_ideas(self):
        output = self._make_output(selected_ideas=[], total_ideas_selected=0)
        assert output.total_ideas_selected == 0
        assert output.selected_ideas == []


# ---------------------------------------------------------------------------
# TestInnovationSessionState
# ---------------------------------------------------------------------------

class TestInnovationSessionState:
    def _make_state(self, **overrides):
        base = dict(
            session_id="state-session-001",
            problem_statement="Improve developer onboarding.",
        )
        base.update(overrides)
        return InnovationSessionState(**base)

    def test_default_state(self):
        state = self._make_state()
        assert state.current_phase == InnovationPhase.IMMERSION
        assert state.status == "active"
        assert state.output is None
        assert state.phases_completed == []
        assert state.ideas_generated == 0
        assert state.ideas_selected == 0
        assert state.techniques == []
        assert state.constraints == {}
        assert state.metadata == {}

    def test_status_values(self):
        for status in ("active", "completed", "paused"):
            state = self._make_state(status=status)
            assert state.status == status

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            self._make_state(status="running")

    def test_with_output(self):
        idea = _make_idea()
        tr = _make_technique_result()
        output = InnovationOutput(
            session_id="state-session-001",
            problem_statement="Improve developer onboarding.",
            phase=InnovationPhase.TRANSFORMATION,
            techniques_used=["SCAMPER"],
            all_ideas=[idea],
            selected_ideas=[idea],
            technique_results={"SCAMPER": tr},
            diversity_score=0.8,
            novelty_score=0.9,
            feasibility_score=0.7,
            total_ideas_generated=1,
            total_ideas_selected=1,
        )
        state = self._make_state(status="completed", output=output)
        assert state.output is not None
        assert state.output.session_id == "state-session-001"

    def test_phases_completed_tracking(self):
        state = self._make_state(
            current_phase=InnovationPhase.DIVERGENCE,
            phases_completed=[InnovationPhase.IMMERSION],
        )
        assert InnovationPhase.IMMERSION in state.phases_completed
        assert state.current_phase == InnovationPhase.DIVERGENCE

    def test_timestamps_are_datetime(self):
        state = self._make_state()
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)


# ---------------------------------------------------------------------------
# TestMetaConductorOutput
# ---------------------------------------------------------------------------

class TestMetaConductorOutput:
    def _valid_kwargs(self):
        return dict(
            session_id="meta-session-007",
            problem_statement="Design a new auth system.",
            phases=[InnovationPhase.IMMERSION, InnovationPhase.DIVERGENCE],
            current_phase=InnovationPhase.IMMERSION,
            phase_tasks={"immersion": "Research existing auth systems."},
            convergence_criteria={"novelty_threshold": 0.7, "feasibility_threshold": 0.6},
            confidence=0.78,
        )

    def test_valid_meta_conductor_output(self):
        mco = MetaConductorOutput(**self._valid_kwargs())
        assert mco.session_id == "meta-session-007"
        assert mco.catalyst_methodology is True
        assert mco.confidence == 0.78

    def test_catalyst_methodology_default(self):
        mco = MetaConductorOutput(**self._valid_kwargs())
        assert mco.catalyst_methodology is True

    def test_catalyst_methodology_can_be_false(self):
        kw = self._valid_kwargs()
        kw["catalyst_methodology"] = False
        mco = MetaConductorOutput(**kw)
        assert mco.catalyst_methodology is False

    def test_confidence_bounds(self):
        kw = self._valid_kwargs()
        kw["confidence"] = 1.5
        with pytest.raises(ValidationError):
            MetaConductorOutput(**kw)

    def test_empty_phases(self):
        kw = self._valid_kwargs()
        kw["phases"] = []
        mco = MetaConductorOutput(**kw)
        assert mco.phases == []
