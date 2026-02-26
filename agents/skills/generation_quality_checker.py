"""Skill: score AI-generated code quality without an LLM."""
from __future__ import annotations
import ast
import re
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_HARDCODED_PATH_RE = re.compile(r'["\'](?:/home/|/root/|C:\\\\|/tmp/)[^"\']{3,}["\']')
_BARE_EXCEPT_RE = re.compile(r'^\s*except\s*:', re.MULTILINE)


def _has_docstring(tree: ast.Module) -> bool:
    return bool(tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant))


def _get_unused_imports(tree: ast.AST, source: str) -> List[str]:
    unused = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if source.count(name) == 1:
                    unused.append(name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                name = alias.asname or alias.name
                if source.count(name) == 1:
                    unused.append(name)
    return unused


def _intent_match(task: str, source: str, tree: ast.AST) -> float:
    keywords = {w.lower().strip("'\"()") for w in re.split(r"[\s,;:/\\]+", task) if len(w) > 3}
    # Look in function/class names + docstrings
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name.lower())
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            doc = str(node.value.value).lower()
            names.update(doc.split())
    matched = sum(1 for kw in keywords if any(kw in n for n in names))
    return round(min(matched / max(len(keywords), 1), 1.0), 2)


class GenerationQualityCheckerSkill(SkillBase):
    name = "generation_quality_checker"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task: str = input_data.get("task", "")
        code: Optional[str] = input_data.get("generated_code") or input_data.get("code")
        if not code:
            return {"error": "Provide 'generated_code' in input_data"}

        issues: List[Dict] = []
        syntax_valid = True

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {"quality_score": 0.0, "issues": [{"type": "syntax_error", "severity": "critical", "suggestion": str(exc)}], "syntax_valid": False, "intent_match_score": 0.0}

        # Check: has at least one function/class
        has_structure = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(tree))
        if not has_structure:
            issues.append({"type": "no_structure", "severity": "medium", "suggestion": "Generated code has no functions or classes"})

        # Check: module docstring
        if not _has_docstring(tree):
            issues.append({"type": "missing_docstring", "severity": "low", "suggestion": "Add a module-level docstring"})

        # Check: bare except
        if _BARE_EXCEPT_RE.search(code):
            issues.append({"type": "bare_except", "severity": "medium", "suggestion": "Replace bare 'except:' with specific exception types"})

        # Check: hardcoded paths
        if _HARDCODED_PATH_RE.search(code):
            issues.append({"type": "hardcoded_path", "severity": "medium", "suggestion": "Avoid hardcoded filesystem paths; use Path or config"})

        # Check: print() in library code
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                issues.append({"type": "print_in_library", "severity": "low", "suggestion": "Replace print() with proper logging"})
                break

        # Check: unused imports
        unused = _get_unused_imports(tree, code)
        if unused:
            issues.append({"type": "unused_imports", "severity": "low", "suggestion": f"Unused imports: {', '.join(unused[:5])}"})

        intent = _intent_match(task, code, tree) if task else 0.5

        penalty = sum({"critical": 40, "high": 20, "medium": 10, "low": 5}.get(i["severity"], 5) for i in issues)
        quality_score = max(0.0, round((100 - penalty) / 100, 2))

        log_json("INFO", "generation_quality_checker_complete", details={"score": quality_score, "issues": len(issues)})
        return {"quality_score": quality_score, "issues": issues, "syntax_valid": syntax_valid, "intent_match_score": intent, "has_structure": has_structure}
