"""PRD-004: Comprehensive tests for all 28 AURA skills (28 × 5 = 140 tests)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_PROJECT_ROOT = str(_ROOT)
_SIMPLE_CODE = "def add(a, b):\n    return a + b\n"


# ---------------------------------------------------------------------------
# Helper mixin — 5 standard tests for each skill
# ---------------------------------------------------------------------------

class _SkillTestMixin:
    """Mixin providing the 5 standard skill tests.

    Subclasses must set:
        skill_cls  — the skill class to test
        minimal_valid_input — dict passed for the "minimal valid input" test
    """
    skill_cls = None
    minimal_valid_input: dict = {}

    def _make(self):
        return self.skill_cls()

    def test_has_name_attribute(self):
        s = self._make()
        self.assertTrue(hasattr(s, "name"))
        self.assertIsInstance(s.name, str)
        self.assertTrue(len(s.name) > 0)

    def test_run_returns_dict(self):
        s = self._make()
        result = s.run({})
        self.assertIsInstance(result, dict)

    def test_never_raises_on_empty_input(self):
        s = self._make()
        try:
            result = s.run({})
        except Exception as e:
            self.fail(f"Skill raised {type(e).__name__}: {e}")

    def test_never_raises_on_garbage_input(self):
        s = self._make()
        try:
            result = s.run({"invalid_key": None, "garbage": "data", "x": 12345})
        except Exception as e:
            self.fail(f"Skill raised {type(e).__name__}: {e}")

    def test_run_minimal_valid_input(self):
        s = self._make()
        try:
            result = s.run(self.minimal_valid_input)
        except Exception as e:
            self.fail(f"Skill raised {type(e).__name__}: {e}")
        self.assertIsInstance(result, dict)


# ===========================================================================
# 1. DependencyAnalyzer
# ===========================================================================

class TestDependencyAnalyzer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.dependency_analyzer import DependencyAnalyzerSkill
    skill_cls = DependencyAnalyzerSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 2. ArchitectureValidator
# ===========================================================================

class TestArchitectureValidator(_SkillTestMixin, unittest.TestCase):
    from agents.skills.architecture_validator import ArchitectureValidatorSkill
    skill_cls = ArchitectureValidatorSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 3. ComplexityScorer
# ===========================================================================

class TestComplexityScorer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.complexity_scorer import ComplexityScorerSkill
    skill_cls = ComplexityScorerSkill
    minimal_valid_input = {"code": _SIMPLE_CODE}


# ===========================================================================
# 4. TestCoverageAnalyzer
# ===========================================================================

class TestTestCoverageAnalyzer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.test_coverage_analyzer import TestCoverageAnalyzerSkill
    skill_cls = TestCoverageAnalyzerSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 5. DocGenerator
# ===========================================================================

class TestDocGenerator(_SkillTestMixin, unittest.TestCase):
    from agents.skills.doc_generator import DocGeneratorSkill
    skill_cls = DocGeneratorSkill
    minimal_valid_input = {"code": _SIMPLE_CODE}


# ===========================================================================
# 6. PerformanceProfiler
# ===========================================================================

class TestPerformanceProfiler(_SkillTestMixin, unittest.TestCase):
    from agents.skills.performance_profiler import PerformanceProfilerSkill
    skill_cls = PerformanceProfilerSkill
    minimal_valid_input = {"code": _SIMPLE_CODE}


# ===========================================================================
# 7. RefactoringAdvisor
# ===========================================================================

class TestRefactoringAdvisor(_SkillTestMixin, unittest.TestCase):
    from agents.skills.refactoring_advisor import RefactoringAdvisorSkill
    skill_cls = RefactoringAdvisorSkill
    minimal_valid_input = {"code": _SIMPLE_CODE}


# ===========================================================================
# 8. SchemaValidator
# ===========================================================================

class TestSchemaValidator(_SkillTestMixin, unittest.TestCase):
    from agents.skills.schema_validator import SchemaValidatorSkill
    skill_cls = SchemaValidatorSkill
    minimal_valid_input = {
        "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        "instance": {"x": 1},
    }


# ===========================================================================
# 9. SecurityScanner
# ===========================================================================

class TestSecurityScanner(_SkillTestMixin, unittest.TestCase):
    from agents.skills.security_scanner import SecurityScannerSkill
    skill_cls = SecurityScannerSkill
    minimal_valid_input = {"code": _SIMPLE_CODE}


# ===========================================================================
# 10. TypeChecker
# ===========================================================================

class TestTypeChecker(_SkillTestMixin, unittest.TestCase):
    from agents.skills.type_checker import TypeCheckerSkill
    skill_cls = TypeCheckerSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 11. LinterEnforcer
# ===========================================================================

class TestLinterEnforcer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.linter_enforcer import LinterEnforcerSkill
    skill_cls = LinterEnforcerSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 12. IncrementalDiffer
# ===========================================================================

class TestIncrementalDiffer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.incremental_differ import IncrementalDifferSkill
    skill_cls = IncrementalDifferSkill
    minimal_valid_input = {
        "old_code": "def foo(): return 1\n",
        "new_code": "def foo(): return 2\n",
    }


# ===========================================================================
# 13. TechDebtQuantifier
# ===========================================================================

class TestTechDebtQuantifier(_SkillTestMixin, unittest.TestCase):
    from agents.skills.tech_debt_quantifier import TechDebtQuantifierSkill
    skill_cls = TechDebtQuantifierSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 14. APIContractValidator
# ===========================================================================

class TestAPIContractValidator(_SkillTestMixin, unittest.TestCase):
    from agents.skills.api_contract_validator import APIContractValidatorSkill
    skill_cls = APIContractValidatorSkill
    minimal_valid_input = {"code": _SIMPLE_CODE}


# ===========================================================================
# 15. GenerationQualityChecker
# ===========================================================================

class TestGenerationQualityChecker(_SkillTestMixin, unittest.TestCase):
    from agents.skills.generation_quality_checker import GenerationQualityCheckerSkill
    skill_cls = GenerationQualityCheckerSkill
    minimal_valid_input = {"task": "add two numbers", "generated_code": _SIMPLE_CODE}


# ===========================================================================
# 16. GitHistoryAnalyzer
# ===========================================================================

class TestGitHistoryAnalyzer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.git_history_analyzer import GitHistoryAnalyzerSkill
    skill_cls = GitHistoryAnalyzerSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 17. SkillComposer
# ===========================================================================

class TestSkillComposer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.skill_composer import SkillComposerSkill
    skill_cls = SkillComposerSkill
    minimal_valid_input = {"goal": "refactor the codebase"}


# ===========================================================================
# 18. ErrorPatternMatcher
# ===========================================================================

class TestErrorPatternMatcher(_SkillTestMixin, unittest.TestCase):
    from agents.skills.error_pattern_matcher import ErrorPatternMatcherSkill
    skill_cls = ErrorPatternMatcherSkill
    minimal_valid_input = {"current_error": "NameError: name 'foo' is not defined"}


# ===========================================================================
# 19. CodeCloneDetector
# ===========================================================================

class TestCodeCloneDetector(_SkillTestMixin, unittest.TestCase):
    from agents.skills.code_clone_detector import CodeCloneDetectorSkill
    skill_cls = CodeCloneDetectorSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 20. AdaptiveStrategySelector
# ===========================================================================

class TestAdaptiveStrategySelector(_SkillTestMixin, unittest.TestCase):
    from agents.skills.adaptive_strategy_selector import AdaptiveStrategySelectorSkill
    skill_cls = AdaptiveStrategySelectorSkill
    minimal_valid_input = {"goal": "improve code quality"}


# ===========================================================================
# 21. WebFetcher
# ===========================================================================

class TestWebFetcher(_SkillTestMixin, unittest.TestCase):
    from agents.skills.web_fetcher import WebFetcherSkill
    skill_cls = WebFetcherSkill
    minimal_valid_input = {"query": "python unittest tutorial"}

    def test_run_with_bad_url_returns_dict(self):
        from agents.skills.web_fetcher import WebFetcherSkill
        s = WebFetcherSkill()
        result = s.run({"url": "http://localhost:9999/nonexistent"})
        self.assertIsInstance(result, dict)


# ===========================================================================
# 22. SymbolIndexer
# ===========================================================================

class TestSymbolIndexer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.symbol_indexer import SymbolIndexerSkill
    skill_cls = SymbolIndexerSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 23. MultiFileEditor
# ===========================================================================

class TestMultiFileEditor(_SkillTestMixin, unittest.TestCase):
    from agents.skills.multi_file_editor import MultiFileEditorSkill
    skill_cls = MultiFileEditorSkill
    minimal_valid_input = {"goal": "rename function foo to bar", "project_root": _PROJECT_ROOT}


# ===========================================================================
# 24. DockerfileAnalyzer
# ===========================================================================

class TestDockerfileAnalyzer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.dockerfile_analyzer import DockerfileAnalyzerSkill
    skill_cls = DockerfileAnalyzerSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 25. ObservabilityChecker
# ===========================================================================

class TestObservabilityChecker(_SkillTestMixin, unittest.TestCase):
    from agents.skills.observability_checker import ObservabilityCheckerSkill
    skill_cls = ObservabilityCheckerSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 26. ChangelogGenerator
# ===========================================================================

class TestChangelogGenerator(_SkillTestMixin, unittest.TestCase):
    from agents.skills.changelog_generator import ChangelogGeneratorSkill
    skill_cls = ChangelogGeneratorSkill
    minimal_valid_input = {"project_root": _PROJECT_ROOT}


# ===========================================================================
# 27. DatabaseQueryAnalyzer
# ===========================================================================

class TestDatabaseQueryAnalyzer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.database_query_analyzer import DatabaseQueryAnalyzerSkill
    skill_cls = DatabaseQueryAnalyzerSkill
    minimal_valid_input = {"code": "SELECT * FROM users WHERE id = 1;"}


# ===========================================================================
# 28. SkillFailureAnalyzer
# ===========================================================================

class TestSkillFailureAnalyzer(_SkillTestMixin, unittest.TestCase):
    from agents.skills.skill_failure_analyzer import SkillFailureAnalyzerSkill
    skill_cls = SkillFailureAnalyzerSkill
    minimal_valid_input = {"error": "KeyError: 'project_root'", "skill_name": "complexity_scorer"}


# ===========================================================================
# Registry integration test
# ===========================================================================

class TestSkillRegistry(unittest.TestCase):
    def test_all_skills_returns_dict(self):
        from agents.skills.registry import all_skills
        skills = all_skills()
        self.assertIsInstance(skills, dict)

    def test_all_skills_count(self):
        from agents.skills.registry import all_skills
        skills = all_skills()
        self.assertGreaterEqual(len(skills), 28)

    def test_all_skills_have_run_method(self):
        from agents.skills.registry import all_skills
        skills = all_skills()
        for name, skill in skills.items():
            self.assertTrue(hasattr(skill, "run"), f"Skill {name} missing run()")

    def test_all_skills_have_name_attribute(self):
        from agents.skills.registry import all_skills
        skills = all_skills()
        for name, skill in skills.items():
            self.assertTrue(hasattr(skill, "name"), f"Skill {name} missing name attr")

    def test_all_skills_run_returns_dict(self):
        from agents.skills.registry import all_skills
        skills = all_skills()
        for name, skill in skills.items():
            result = skill.run({})
            self.assertIsInstance(result, dict, f"Skill {name} run() did not return dict")


if __name__ == "__main__":
    unittest.main()
