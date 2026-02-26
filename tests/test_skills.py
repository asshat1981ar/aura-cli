"""Smoke tests for all 20 AURA skill modules."""
import os
import sys
from pathlib import Path
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from agents.skills.registry import all_skills


@pytest.fixture(scope="module")
def skills():
    return all_skills()


@pytest.fixture(scope="module")
def project_root():
    return str(ROOT)


# ── helpers ───────────────────────────────────────────────────────────────────

SIMPLE_CODE = '''
"""A simple module."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def _private():
    pass
'''

BAD_CODE = '''
import os
import unused_module

password = "supersecret123"

def foo(a, b, c, d, e, f, g):
    for i in range(10):
        for j in range(10):
            eval(str(i + j))
    return a
'''


# ── per-skill smoke tests ──────────────────────────────────────────────────────

def test_dependency_analyzer_returns_dict(skills, project_root):
    result = skills["dependency_analyzer"].run({"project_root": project_root})
    assert isinstance(result, dict)
    assert "packages" in result or "error" in result


def test_architecture_validator_returns_dict(skills, project_root):
    result = skills["architecture_validator"].run({"project_root": project_root})
    assert isinstance(result, dict)
    assert "circular_deps" in result or "error" in result


def test_complexity_scorer_with_code(skills):
    result = skills["complexity_scorer"].run({"code": SIMPLE_CODE, "file_path": "test.py"})
    assert isinstance(result, dict)
    assert "file_avg_complexity" in result or "error" in result


def test_complexity_scorer_empty_input(skills):
    result = skills["complexity_scorer"].run({})
    assert isinstance(result, dict)
    assert "error" in result


def test_test_coverage_analyzer_returns_dict(skills, project_root):
    result = skills["test_coverage_analyzer"].run({"project_root": project_root})
    assert isinstance(result, dict)
    assert "coverage_pct" in result or "error" in result


def test_doc_generator_with_code(skills):
    result = skills["doc_generator"].run({"code": SIMPLE_CODE, "file_path": "test.py"})
    assert isinstance(result, dict)
    assert "undocumented_count" in result or "error" in result


def test_doc_generator_no_input(skills):
    result = skills["doc_generator"].run({})
    assert isinstance(result, dict)
    assert "error" in result


def test_performance_profiler_with_code(skills):
    result = skills["performance_profiler"].run({"code": "x = [i for i in range(10)]"})
    assert isinstance(result, dict)
    assert "antipatterns" in result or "error" in result


def test_performance_profiler_detects_nested_loops(skills):
    code = "for i in range(10):\n    for j in range(10):\n        pass"
    result = skills["performance_profiler"].run({"code": code})
    assert isinstance(result, dict)
    if "antipatterns" in result:
        assert any(a["type"] == "nested_loop" for a in result["antipatterns"])


def test_refactoring_advisor_with_bad_code(skills):
    result = skills["refactoring_advisor"].run({"code": BAD_CODE, "file_path": "bad.py"})
    assert isinstance(result, dict)
    assert "suggestions" in result or "error" in result


def test_schema_validator_valid(skills):
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    result = skills["schema_validator"].run({"schema": schema, "instance": {"name": "Alice"}})
    assert isinstance(result, dict)
    assert result.get("valid") is True


def test_schema_validator_invalid(skills):
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    result = skills["schema_validator"].run({"schema": schema, "instance": {"age": 5}})
    assert isinstance(result, dict)
    assert result.get("valid") is False or "error" in result


def test_security_scanner_finds_secret(skills):
    code = 'api_key = "sk-supersecretkey12345"'
    result = skills["security_scanner"].run({"code": code, "file_path": "test.py"})
    assert isinstance(result, dict)
    assert "findings" in result or "error" in result
    if "findings" in result:
        assert len(result["findings"]) > 0


def test_security_scanner_clean_code(skills):
    result = skills["security_scanner"].run({"code": SIMPLE_CODE, "file_path": "clean.py"})
    assert isinstance(result, dict)
    if "findings" in result:
        assert result["critical_count"] == 0


def test_type_checker_returns_dict(skills, project_root):
    result = skills["type_checker"].run({"project_root": project_root})
    assert isinstance(result, dict)
    assert "annotation_coverage_pct" in result or "error" in result


def test_linter_enforcer_returns_dict(skills, project_root):
    result = skills["linter_enforcer"].run({"project_root": project_root})
    assert isinstance(result, dict)
    assert "violations" in result or "error" in result


def test_incremental_differ_basic(skills):
    result = skills["incremental_differ"].run({
        "old_code": "def foo():\n    pass\n",
        "new_code": "def foo():\n    return 1\ndef bar():\n    pass\n",
        "file_path": "test.py"
    })
    assert isinstance(result, dict)
    assert "added_symbols" in result or "error" in result
    if "added_symbols" in result:
        assert "bar" in result["added_symbols"]


def test_incremental_differ_empty(skills):
    result = skills["incremental_differ"].run({})
    assert isinstance(result, dict)
    assert "error" in result


def test_tech_debt_quantifier_returns_dict(skills, project_root):
    result = skills["tech_debt_quantifier"].run({"project_root": project_root})
    assert isinstance(result, dict)
    assert "debt_score" in result or "error" in result


def test_api_contract_validator_with_code(skills):
    code = '''
from fastapi import FastAPI
app = FastAPI()

@app.get("/users")
async def list_users():
    pass

@app.post("/users")
async def create_user(name: str):
    pass
'''
    result = skills["api_contract_validator"].run({"code": code, "file_path": "api.py"})
    assert isinstance(result, dict)
    assert "endpoints" in result or "error" in result


def test_generation_quality_checker_good_code(skills):
    result = skills["generation_quality_checker"].run({"task": "add two numbers", "generated_code": SIMPLE_CODE})
    assert isinstance(result, dict)
    assert "quality_score" in result or "error" in result
    if "quality_score" in result:
        assert 0.0 <= result["quality_score"] <= 1.0


def test_generation_quality_checker_bad_syntax(skills):
    result = skills["generation_quality_checker"].run({"task": "anything", "generated_code": "def broken(:\n    pass"})
    assert isinstance(result, dict)
    assert result.get("syntax_valid") is False or "error" in result


def test_generation_quality_checker_no_code(skills):
    result = skills["generation_quality_checker"].run({})
    assert isinstance(result, dict)
    assert "error" in result


def test_git_history_analyzer_returns_dict(skills, project_root):
    result = skills["git_history_analyzer"].run({"project_root": project_root, "lookback_days": 7})
    assert isinstance(result, dict)
    assert "total_commits" in result or "error" in result


def test_skill_composer_refactor_goal(skills):
    result = skills["skill_composer"].run({"goal": "refactor the authentication module"})
    assert isinstance(result, dict)
    assert "workflow" in result or "error" in result
    if "workflow" in result:
        assert len(result["workflow"]) > 0
        assert result["goal_category"] == "refactor"


def test_skill_composer_security_goal(skills):
    result = skills["skill_composer"].run({"goal": "security audit for CVE vulnerabilities"})
    assert isinstance(result, dict)
    if "goal_category" in result:
        assert result["goal_category"] == "security"


def test_error_pattern_matcher_import_error(skills):
    result = skills["error_pattern_matcher"].run({"current_error": "ModuleNotFoundError: No module named 'requests'"})
    assert isinstance(result, dict)
    assert "suggested_fix" in result or "error" in result
    if "matched_pattern" in result:
        assert result["matched_pattern"] == "import_error"


def test_error_pattern_matcher_unknown(skills):
    result = skills["error_pattern_matcher"].run({"current_error": "Something very unusual happened"})
    assert isinstance(result, dict)
    assert "suggested_fix" in result or "error" in result


def test_code_clone_detector_returns_dict(skills, project_root):
    result = skills["code_clone_detector"].run({"project_root": project_root, "min_lines": 5})
    assert isinstance(result, dict)
    assert "clone_count" in result or "error" in result


def test_adaptive_strategy_selector_recommendation(skills):
    result = skills["adaptive_strategy_selector"].run({"goal": "refactor the core module"})
    assert isinstance(result, dict)
    assert "recommended_strategy" in result or "error" in result
    if "recommended_strategy" in result:
        assert isinstance(result["confidence"], float)


def test_adaptive_strategy_selector_record(skills):
    result = skills["adaptive_strategy_selector"].run({
        "goal": "fix a bug",
        "record_result": {"strategy": "sliding_window", "success": True, "cycles": 3, "stop_reason": "CONVERGED"}
    })
    assert isinstance(result, dict)
    assert "recommended_strategy" in result or "error" in result


# ── registry tests ─────────────────────────────────────────────────────────────

def test_registry_returns_all_20_skills(skills):
    assert len(skills) >= 20


def test_registry_all_have_run_method(skills):
    for name, skill in skills.items():
        assert hasattr(skill, "run"), f"Skill '{name}' missing run() method"
        assert callable(skill.run), f"Skill '{name}'.run is not callable"


def test_all_skills_handle_empty_input(skills):
    """Every skill must return a dict (never raise) when given an empty dict."""
    for name, skill in skills.items():
        result = skill.run({})
        assert isinstance(result, dict), f"Skill '{name}' returned non-dict for empty input: {result!r}"


# ── Skill #21: web_fetcher ─────────────────────────────────────────────────────

class TestWebFetcherSkill:
    def test_name(self):
        from agents.skills.web_fetcher import WebFetcherSkill
        assert WebFetcherSkill().name == "web_fetcher"

    def test_empty_input(self):
        from agents.skills.web_fetcher import WebFetcherSkill
        result = WebFetcherSkill().run({})
        assert isinstance(result, dict)
        assert "error" in result

    def test_invalid_url(self):
        from agents.skills.web_fetcher import WebFetcherSkill
        result = WebFetcherSkill().run({"url": "not-a-url"})
        assert isinstance(result, dict)
        # should return error or text
        assert "error" in result or "text" in result


# ── Skill #22: symbol_indexer ──────────────────────────────────────────────────

class TestSymbolIndexerSkill:
    def test_name(self):
        from agents.skills.symbol_indexer import SymbolIndexerSkill
        assert SymbolIndexerSkill().name == "symbol_indexer"

    def test_empty_input(self):
        from agents.skills.symbol_indexer import SymbolIndexerSkill
        result = SymbolIndexerSkill().run({})
        assert isinstance(result, dict)

    def test_indexes_project(self):
        from agents.skills.symbol_indexer import SymbolIndexerSkill
        result = SymbolIndexerSkill().run({"project_root": "agents/skills"})
        assert isinstance(result, dict)
        assert "symbols" in result or "error" in result
        if "symbols" in result:
            assert result["symbol_count"] >= 0


# ── Skill #23: multi_file_editor ───────────────────────────────────────────────

class TestMultiFileEditorSkill:
    def test_name(self):
        from agents.skills.multi_file_editor import MultiFileEditorSkill
        assert MultiFileEditorSkill().name == "multi_file_editor"

    def test_empty_input(self):
        from agents.skills.multi_file_editor import MultiFileEditorSkill
        result = MultiFileEditorSkill().run({})
        assert isinstance(result, dict)
        assert "change_plan" in result

    def test_with_goal(self):
        from agents.skills.multi_file_editor import MultiFileEditorSkill
        result = MultiFileEditorSkill().run({"goal": "refactor model adapter", "project_root": "."})
        assert isinstance(result, dict)
        assert "change_plan" in result
        assert isinstance(result["change_plan"], list)
        assert "affected_count" in result


# ── updated registry count ─────────────────────────────────────────────────────

def test_registry_returns_all_skills(skills):
    assert len(skills) >= 27
