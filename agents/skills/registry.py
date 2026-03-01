"""Registry for all AURA skill modules."""
from __future__ import annotations
from typing import Dict

from agents.skills.base import SkillBase


def all_skills(brain=None, model=None) -> Dict[str, SkillBase]:
    """
    Return a dict mapping skill_name -> skill_instance for all 23 skills.

    Args:
        brain: Optional Brain instance (currently unused by skills; reserved for future use).
        model: Optional ModelAdapter instance (currently unused; reserved for future use).

    Returns:
        Dict of {skill_name: skill_instance}
    """
    from agents.skills.dependency_analyzer import DependencyAnalyzerSkill
    from agents.skills.architecture_validator import ArchitectureValidatorSkill
    from agents.skills.complexity_scorer import ComplexityScorerSkill
    from agents.skills.test_coverage_analyzer import TestCoverageAnalyzerSkill
    from agents.skills.doc_generator import DocGeneratorSkill
    from agents.skills.performance_profiler import PerformanceProfilerSkill
    from agents.skills.refactoring_advisor import RefactoringAdvisorSkill
    from agents.skills.schema_validator import SchemaValidatorSkill
    from agents.skills.security_scanner import SecurityScannerSkill
    from agents.skills.type_checker import TypeCheckerSkill
    from agents.skills.linter_enforcer import LinterEnforcerSkill
    from agents.skills.incremental_differ import IncrementalDifferSkill
    from agents.skills.tech_debt_quantifier import TechDebtQuantifierSkill
    from agents.skills.api_contract_validator import APIContractValidatorSkill
    from agents.skills.generation_quality_checker import GenerationQualityCheckerSkill
    from agents.skills.git_history_analyzer import GitHistoryAnalyzerSkill
    from agents.skills.skill_composer import SkillComposerSkill
    from agents.skills.error_pattern_matcher import ErrorPatternMatcherSkill
    from agents.skills.code_clone_detector import CodeCloneDetectorSkill
    from agents.skills.adaptive_strategy_selector import AdaptiveStrategySelectorSkill
    from agents.skills.web_fetcher import WebFetcherSkill
    from agents.skills.symbol_indexer import SymbolIndexerSkill
    from agents.skills.multi_file_editor import MultiFileEditorSkill
    from agents.skills.dockerfile_analyzer import DockerfileAnalyzerSkill
    from agents.skills.observability_checker import ObservabilityCheckerSkill
    from agents.skills.changelog_generator import ChangelogGeneratorSkill
    from agents.skills.database_query_analyzer import DatabaseQueryAnalyzerSkill
    from agents.skills.skill_failure_analyzer import SkillFailureAnalyzerSkill
    from agents.skills.security_hardener import SecurityHardenerSkill
    from agents.skills.structural_analyzer import StructuralAnalyzerSkill

    return {
        "dependency_analyzer": DependencyAnalyzerSkill(),
        "architecture_validator": ArchitectureValidatorSkill(),
        "complexity_scorer": ComplexityScorerSkill(),
        "test_coverage_analyzer": TestCoverageAnalyzerSkill(),
        "doc_generator": DocGeneratorSkill(),
        "performance_profiler": PerformanceProfilerSkill(),
        "refactoring_advisor": RefactoringAdvisorSkill(),
        "schema_validator": SchemaValidatorSkill(),
        "security_scanner": SecurityScannerSkill(),
        "type_checker": TypeCheckerSkill(),
        "linter_enforcer": LinterEnforcerSkill(),
        "incremental_differ": IncrementalDifferSkill(),
        "tech_debt_quantifier": TechDebtQuantifierSkill(),
        "api_contract_validator": APIContractValidatorSkill(),
        "generation_quality_checker": GenerationQualityCheckerSkill(),
        "git_history_analyzer": GitHistoryAnalyzerSkill(),
        "skill_composer": SkillComposerSkill(),
        "error_pattern_matcher": ErrorPatternMatcherSkill(),
        "code_clone_detector": CodeCloneDetectorSkill(),
        "adaptive_strategy_selector": AdaptiveStrategySelectorSkill(),
        "web_fetcher": WebFetcherSkill(),
        "symbol_indexer": SymbolIndexerSkill(),
        "multi_file_editor": MultiFileEditorSkill(),
        "dockerfile_analyzer": DockerfileAnalyzerSkill(),
        "observability_checker": ObservabilityCheckerSkill(),
        "changelog_generator": ChangelogGeneratorSkill(),
        "database_query_analyzer": DatabaseQueryAnalyzerSkill(),
        "skill_failure_analyzer": SkillFailureAnalyzerSkill(),
        "security_hardener": SecurityHardenerSkill(),
        "structural_analyzer": StructuralAnalyzerSkill(),
    }
