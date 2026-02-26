"""
agents/skills — Pluggable software-engineering skill modules for AURA.

Each skill is a standalone class with a run(input_data: dict) -> dict interface.
Skills are stateless helpers that can be composed into workflows via SkillComposerSkill.

Available skills:
    dependency_analyzer     – Parse requirements files; find CVEs and conflicts
    architecture_validator  – Detect circular imports; measure module coupling
    complexity_scorer       – Cyclomatic complexity + nesting depth per function
    test_coverage_analyzer  – coverage.py integration with heuristic fallback
    doc_generator           – Docstring templates + README sections from AST
    performance_profiler    – cProfile hotspots + AST anti-pattern detection
    refactoring_advisor     – Code smells: god functions, deep nesting, magic numbers
    schema_validator        – JSON schema validation + Pydantic model discovery
    security_scanner        – Secrets, SQL injection, unsafe calls (regex + AST)
    type_checker            – mypy integration with annotation coverage fallback
    linter_enforcer         – flake8 + AST naming convention checks
    incremental_differ      – Unified diff, symbol change detection, impact analysis
    tech_debt_quantifier    – TODO/FIXME/HACK counts + debt score (0-100)
    api_contract_validator  – FastAPI/Flask endpoint extraction + breaking-change detection
    generation_quality_checker – Score AI-generated code quality without LLM
    git_history_analyzer    – Git log analysis: hotspots, authors, change patterns
    skill_composer          – Map natural-language goals to ordered skill workflows
    error_pattern_matcher   – Match runtime errors to known patterns + fix steps
    code_clone_detector     – Exact (AST hash) and near-duplicate (Jaccard) clone detection
    adaptive_strategy_selector – Track strategy success rates; recommend best strategy
    web_fetcher             – HTTP GET and DuckDuckGo search with HTML→text stripping
    symbol_indexer          – Project-wide AST symbol map with callers/callees
    multi_file_editor       – Plan dependency-ordered multi-file changes from a goal
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

__all__ = [
    "DependencyAnalyzerSkill",
    "ArchitectureValidatorSkill",
    "ComplexityScorerSkill",
    "TestCoverageAnalyzerSkill",
    "DocGeneratorSkill",
    "PerformanceProfilerSkill",
    "RefactoringAdvisorSkill",
    "SchemaValidatorSkill",
    "SecurityScannerSkill",
    "TypeCheckerSkill",
    "LinterEnforcerSkill",
    "IncrementalDifferSkill",
    "TechDebtQuantifierSkill",
    "APIContractValidatorSkill",
    "GenerationQualityCheckerSkill",
    "GitHistoryAnalyzerSkill",
    "SkillComposerSkill",
    "ErrorPatternMatcherSkill",
    "CodeCloneDetectorSkill",
    "AdaptiveStrategySelectorSkill",
    "WebFetcherSkill",
    "SymbolIndexerSkill",
    "MultiFileEditorSkill",
]
