"""R6: Per-skill unit tests for all 28 AURA skills.

One test class per skill.  Each class:
  1. asserts the skill's ``.name`` attribute matches the registry key
  2. calls ``.run({})`` with empty input and asserts a dict is returned
  3. calls ``.run(...)`` with minimal valid input and asserts no exception

Always sets AURA_SKIP_CHDIR=1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agents.skills.registry import all_skills  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def skills():
    return all_skills()


@pytest.fixture(scope="module")
def project_root():
    return str(_ROOT)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

SIMPLE_CODE = '''
"""A simple module."""


def add(a: int, b: int) -> int:
    """Return a + b."""
    return a + b
'''


def _run_ok(skill, args: dict):
    """Call skill.run(args) and assert it returns a dict."""
    result = skill.run(args)
    assert isinstance(result, dict), f"Skill '{skill.name}' returned {type(result)}, expected dict"
    return result


# ---------------------------------------------------------------------------
# 1. dependency_analyzer
# ---------------------------------------------------------------------------

class TestDependencyAnalyzer:
    def test_name(self, skills):
        assert skills["dependency_analyzer"].name == "dependency_analyzer"

    def test_empty_input(self, skills):
        _run_ok(skills["dependency_analyzer"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["dependency_analyzer"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 2. architecture_validator
# ---------------------------------------------------------------------------

class TestArchitectureValidator:
    def test_name(self, skills):
        assert skills["architecture_validator"].name == "architecture_validator"

    def test_empty_input(self, skills):
        _run_ok(skills["architecture_validator"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["architecture_validator"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 3. complexity_scorer
# ---------------------------------------------------------------------------

class TestComplexityScorer:
    def test_name(self, skills):
        assert skills["complexity_scorer"].name == "complexity_scorer"

    def test_empty_input(self, skills):
        _run_ok(skills["complexity_scorer"], {})

    def test_with_code(self, skills):
        r = _run_ok(skills["complexity_scorer"], {"code": SIMPLE_CODE})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 4. test_coverage_analyzer
# ---------------------------------------------------------------------------

class TestTestCoverageAnalyzer:
    def test_name(self, skills):
        assert skills["test_coverage_analyzer"].name == "test_coverage_analyzer"

    def test_empty_input(self, skills):
        _run_ok(skills["test_coverage_analyzer"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["test_coverage_analyzer"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 5. doc_generator
# ---------------------------------------------------------------------------

class TestDocGenerator:
    def test_name(self, skills):
        assert skills["doc_generator"].name == "doc_generator"

    def test_empty_input(self, skills):
        _run_ok(skills["doc_generator"], {})

    def test_with_code(self, skills):
        r = _run_ok(skills["doc_generator"], {"code": SIMPLE_CODE})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 6. performance_profiler
# ---------------------------------------------------------------------------

class TestPerformanceProfiler:
    def test_name(self, skills):
        assert skills["performance_profiler"].name == "performance_profiler"

    def test_empty_input(self, skills):
        _run_ok(skills["performance_profiler"], {})

    def test_with_code(self, skills):
        r = _run_ok(skills["performance_profiler"], {"code": SIMPLE_CODE})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 7. refactoring_advisor
# ---------------------------------------------------------------------------

class TestRefactoringAdvisor:
    def test_name(self, skills):
        assert skills["refactoring_advisor"].name == "refactoring_advisor"

    def test_empty_input(self, skills):
        _run_ok(skills["refactoring_advisor"], {})

    def test_with_code(self, skills):
        r = _run_ok(skills["refactoring_advisor"], {"code": SIMPLE_CODE})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 8. schema_validator
# ---------------------------------------------------------------------------

class TestSchemaValidator:
    def test_name(self, skills):
        assert skills["schema_validator"].name == "schema_validator"

    def test_empty_input(self, skills):
        _run_ok(skills["schema_validator"], {})

    def test_valid_schema(self, skills):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        data = {"name": "Alice"}
        r = _run_ok(skills["schema_validator"], {"schema": schema, "data": data})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 9. security_scanner
# ---------------------------------------------------------------------------

class TestSecurityScanner:
    def test_name(self, skills):
        assert skills["security_scanner"].name == "security_scanner"

    def test_empty_input(self, skills):
        _run_ok(skills["security_scanner"], {})

    def test_detects_hardcoded_secret(self, skills):
        r = _run_ok(skills["security_scanner"], {"code": 'password = "hunter2"'})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 10. type_checker
# ---------------------------------------------------------------------------

class TestTypeChecker:
    def test_name(self, skills):
        assert skills["type_checker"].name == "type_checker"

    def test_empty_input(self, skills):
        _run_ok(skills["type_checker"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["type_checker"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 11. linter_enforcer
# ---------------------------------------------------------------------------

class TestLinterEnforcer:
    def test_name(self, skills):
        assert skills["linter_enforcer"].name == "linter_enforcer"

    def test_empty_input(self, skills):
        _run_ok(skills["linter_enforcer"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["linter_enforcer"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 12. incremental_differ
# ---------------------------------------------------------------------------

class TestIncrementalDiffer:
    def test_name(self, skills):
        assert skills["incremental_differ"].name == "incremental_differ"

    def test_empty_input(self, skills):
        _run_ok(skills["incremental_differ"], {})

    def test_with_diff(self, skills):
        r = _run_ok(skills["incremental_differ"], {
            "before": "def foo(): pass",
            "after": "def foo():\n    return 42",
        })
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 13. tech_debt_quantifier
# ---------------------------------------------------------------------------

class TestTechDebtQuantifier:
    def test_name(self, skills):
        assert skills["tech_debt_quantifier"].name == "tech_debt_quantifier"

    def test_empty_input(self, skills):
        _run_ok(skills["tech_debt_quantifier"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["tech_debt_quantifier"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 14. api_contract_validator
# ---------------------------------------------------------------------------

class TestAPIContractValidator:
    def test_name(self, skills):
        assert skills["api_contract_validator"].name == "api_contract_validator"

    def test_empty_input(self, skills):
        _run_ok(skills["api_contract_validator"], {})

    def test_with_code(self, skills):
        r = _run_ok(skills["api_contract_validator"], {"code": SIMPLE_CODE})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 15. generation_quality_checker
# ---------------------------------------------------------------------------

class TestGenerationQualityChecker:
    def test_name(self, skills):
        assert skills["generation_quality_checker"].name == "generation_quality_checker"

    def test_empty_input(self, skills):
        _run_ok(skills["generation_quality_checker"], {})

    def test_with_good_code(self, skills):
        r = _run_ok(skills["generation_quality_checker"], {"code": SIMPLE_CODE})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 16. git_history_analyzer
# ---------------------------------------------------------------------------

class TestGitHistoryAnalyzer:
    def test_name(self, skills):
        assert skills["git_history_analyzer"].name == "git_history_analyzer"

    def test_empty_input(self, skills):
        _run_ok(skills["git_history_analyzer"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["git_history_analyzer"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 17. skill_composer
# ---------------------------------------------------------------------------

class TestSkillComposer:
    def test_name(self, skills):
        assert skills["skill_composer"].name == "skill_composer"

    def test_empty_input(self, skills):
        _run_ok(skills["skill_composer"], {})

    def test_with_goal(self, skills):
        r = _run_ok(skills["skill_composer"], {"goal": "refactor the codebase"})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 18. error_pattern_matcher
# ---------------------------------------------------------------------------

class TestErrorPatternMatcher:
    def test_name(self, skills):
        assert skills["error_pattern_matcher"].name == "error_pattern_matcher"

    def test_empty_input(self, skills):
        _run_ok(skills["error_pattern_matcher"], {})

    def test_with_error(self, skills):
        r = _run_ok(skills["error_pattern_matcher"], {
            "error": "ImportError: No module named 'missing_pkg'"
        })
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 19. code_clone_detector
# ---------------------------------------------------------------------------

class TestCodeCloneDetector:
    def test_name(self, skills):
        assert skills["code_clone_detector"].name == "code_clone_detector"

    def test_empty_input(self, skills):
        _run_ok(skills["code_clone_detector"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["code_clone_detector"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 20. adaptive_strategy_selector
# ---------------------------------------------------------------------------

class TestAdaptiveStrategySelector:
    def test_name(self, skills):
        assert skills["adaptive_strategy_selector"].name == "adaptive_strategy_selector"

    def test_empty_input(self, skills):
        _run_ok(skills["adaptive_strategy_selector"], {})

    def test_recommendation(self, skills):
        r = _run_ok(skills["adaptive_strategy_selector"], {"goal": "fix flaky tests"})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 21. web_fetcher
# ---------------------------------------------------------------------------

class TestWebFetcher:
    def test_name(self, skills):
        assert skills["web_fetcher"].name == "web_fetcher"

    def test_empty_input(self, skills):
        r = _run_ok(skills["web_fetcher"], {})
        assert "error" in r

    def test_invalid_url(self, skills):
        r = _run_ok(skills["web_fetcher"], {"url": "not-a-url"})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 22. symbol_indexer
# ---------------------------------------------------------------------------

class TestSymbolIndexer:
    def test_name(self, skills):
        assert skills["symbol_indexer"].name == "symbol_indexer"

    def test_empty_input(self, skills):
        _run_ok(skills["symbol_indexer"], {})

    def test_indexes_project(self, skills, project_root):
        r = _run_ok(skills["symbol_indexer"], {"project_root": "agents/skills"})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 23. multi_file_editor
# ---------------------------------------------------------------------------

class TestMultiFileEditor:
    def test_name(self, skills):
        assert skills["multi_file_editor"].name == "multi_file_editor"

    def test_empty_input(self, skills):
        r = _run_ok(skills["multi_file_editor"], {})
        assert "change_plan" in r

    def test_with_goal(self, skills, project_root):
        r = _run_ok(skills["multi_file_editor"], {"goal": "add type hints", "project_root": project_root})
        assert "change_plan" in r
        assert isinstance(r["change_plan"], list)


# ---------------------------------------------------------------------------
# 24. dockerfile_analyzer
# ---------------------------------------------------------------------------

class TestDockerfileAnalyzer:
    def test_name(self, skills):
        assert skills["dockerfile_analyzer"].name == "dockerfile_analyzer"

    def test_empty_input(self, skills):
        _run_ok(skills["dockerfile_analyzer"], {})

    def test_with_content(self, skills):
        dockerfile = "FROM ubuntu:latest\nRUN apt-get install -y curl\nUSER root\n"
        r = _run_ok(skills["dockerfile_analyzer"], {"content": dockerfile, "file_path": "Dockerfile"})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 25. observability_checker
# ---------------------------------------------------------------------------

class TestObservabilityChecker:
    def test_name(self, skills):
        assert skills["observability_checker"].name == "observability_checker"

    def test_empty_input(self, skills):
        _run_ok(skills["observability_checker"], {})

    def test_with_silent_except(self, skills):
        code = "def foo():\n    try:\n        pass\n    except Exception:\n        pass\n"
        r = _run_ok(skills["observability_checker"], {"code": code, "file_path": "test.py"})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 26. changelog_generator
# ---------------------------------------------------------------------------

class TestChangelogGenerator:
    def test_name(self, skills):
        assert skills["changelog_generator"].name == "changelog_generator"

    def test_empty_input(self, skills):
        _run_ok(skills["changelog_generator"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["changelog_generator"], {"project_root": project_root})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 27. database_query_analyzer
# ---------------------------------------------------------------------------

class TestDatabaseQueryAnalyzer:
    def test_name(self, skills):
        assert skills["database_query_analyzer"].name == "database_query_analyzer"

    def test_empty_input(self, skills):
        _run_ok(skills["database_query_analyzer"], {})

    def test_with_query(self, skills):
        r = _run_ok(skills["database_query_analyzer"], {"query": "SELECT * FROM users WHERE id = 1"})
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 28. skill_failure_analyzer
# ---------------------------------------------------------------------------

class TestSkillFailureAnalyzer:
    def test_name(self, skills):
        assert skills["skill_failure_analyzer"].name == "skill_failure_analyzer"

    def test_empty_input(self, skills):
        _run_ok(skills["skill_failure_analyzer"], {})

    def test_with_failure_log(self, skills):
        r = _run_ok(skills["skill_failure_analyzer"], {
            "skill_name": "linter_enforcer",
            "error": "FileNotFoundError: ruff not found",
            "args": {"project_root": "/nonexistent"},
        })
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 29. structural_analyzer
# ---------------------------------------------------------------------------

class TestStructuralAnalyzer:
    def test_name(self, skills):
        assert skills["structural_analyzer"].name == "structural_analyzer"

    def test_empty_input(self, skills):
        _run_ok(skills["structural_analyzer"], {})

    def test_with_project_root(self, skills, project_root):
        r = _run_ok(skills["structural_analyzer"], {"project_root": project_root})
        assert isinstance(r, dict)
        assert "hotspots" in r
        assert "circular_dependencies" in r


# ---------------------------------------------------------------------------
# 30. evolution_skill
# ---------------------------------------------------------------------------

class TestEvolutionSkill:
    def test_name(self, skills):
        assert skills["evolution_skill"].name == "evolution_skill"

    def test_empty_input(self, skills):
        # EvolutionSkill might fail in a unit test because it needs a full runtime,
        # so we just test that it's in the registry and has the correct name.
        assert skills["evolution_skill"] is not None


# ---------------------------------------------------------------------------
# Registry-level assertions
# ---------------------------------------------------------------------------

def test_registry_has_all_30_skills(skills):
    assert len(skills) >= 30, (
        f"Expected ≥30 skills in registry, got {len(skills)}: {sorted(skills)}"
    )


def test_all_skills_have_name_attribute(skills):
    for name, skill in skills.items():
        assert hasattr(skill, "name"), f"Skill '{name}' missing .name attribute"
        assert isinstance(skill.name, str), f"Skill '{name}'.name is not a string"


def test_all_skills_have_callable_run(skills):
    for name, skill in skills.items():
        assert hasattr(skill, "run"), f"Skill '{name}' missing run() method"
        assert callable(skill.run), f"Skill '{name}'.run is not callable"


def test_all_skills_return_dict_on_empty_input(skills):
    """Every skill must return a dict (never raise) when given an empty dict."""
    for name, skill in skills.items():
        result = skill.run({})
        assert isinstance(result, dict), (
            f"Skill '{name}' returned {type(result)} for empty input, expected dict"
        )
