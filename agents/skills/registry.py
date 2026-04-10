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
    from agents.skills.beads_skill import BeadsSkill
    from agents.skills.structural_analyzer import StructuralAnalyzerSkill
    from agents.skills.evolution_skill import EvolutionSkill
    from agents.skills.skill_generator import SkillGeneratorSkill
    from agents.skills.ast_analyzer import ASTAnalyzerSkill
    from agents.skills.eval_optimizer import EvalOptimizerSkill
    from agents.prompt_forge import PromptForgeAgent

    return {
        "beads_skill": BeadsSkill(brain=brain, model=model),
        "dependency_analyzer": DependencyAnalyzerSkill(brain=brain, model=model),
        "architecture_validator": ArchitectureValidatorSkill(brain=brain, model=model),
        "complexity_scorer": ComplexityScorerSkill(brain=brain, model=model),
        "test_coverage_analyzer": TestCoverageAnalyzerSkill(brain=brain, model=model),
        "doc_generator": DocGeneratorSkill(brain=brain, model=model),
        "performance_profiler": PerformanceProfilerSkill(brain=brain, model=model),
        "refactoring_advisor": RefactoringAdvisorSkill(brain=brain, model=model),
        "schema_validator": SchemaValidatorSkill(brain=brain, model=model),
        "security_scanner": SecurityScannerSkill(brain=brain, model=model),
        "type_checker": TypeCheckerSkill(brain=brain, model=model),
        "linter_enforcer": LinterEnforcerSkill(brain=brain, model=model),
        "incremental_differ": IncrementalDifferSkill(brain=brain, model=model),
        "tech_debt_quantifier": TechDebtQuantifierSkill(brain=brain, model=model),
        "api_contract_validator": APIContractValidatorSkill(brain=brain, model=model),
        "generation_quality_checker": GenerationQualityCheckerSkill(brain=brain, model=model),
        "git_history_analyzer": GitHistoryAnalyzerSkill(brain=brain, model=model),
        "skill_composer": SkillComposerSkill(brain=brain, model=model),
        "error_pattern_matcher": ErrorPatternMatcherSkill(brain=brain, model=model),
        "code_clone_detector": CodeCloneDetectorSkill(brain=brain, model=model),
        "adaptive_strategy_selector": AdaptiveStrategySelectorSkill(brain=brain, model=model),
        "web_fetcher": WebFetcherSkill(brain=brain, model=model),
        "symbol_indexer": SymbolIndexerSkill(brain=brain, model=model),
        "multi_file_editor": MultiFileEditorSkill(brain=brain, model=model),
        "dockerfile_analyzer": DockerfileAnalyzerSkill(brain=brain, model=model),
        "observability_checker": ObservabilityCheckerSkill(brain=brain, model=model),
        "changelog_generator": ChangelogGeneratorSkill(brain=brain, model=model),
        "database_query_analyzer": DatabaseQueryAnalyzerSkill(brain=brain, model=model),
        "skill_failure_analyzer": SkillFailureAnalyzerSkill(brain=brain, model=model),
        "security_hardener": SecurityHardenerSkill(brain=brain, model=model),
        "structural_analyzer": StructuralAnalyzerSkill(brain=brain, model=model),
        "evolution_skill": EvolutionSkill(brain=brain, model=model),
        "skill_generator": SkillGeneratorSkill(brain=brain, model=model),
        "ast_analyzer": ASTAnalyzerSkill(brain=brain, model=model),
        "eval_optimizer": EvalOptimizerSkill(brain=brain, model=model),
        "prompt_forge": PromptForgeAgent(project_root="."),
    }
