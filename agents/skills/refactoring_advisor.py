"""Skill: detect code smells and suggest refactoring opportunities."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_SNAKE_RE = __import__("re").compile(r"^[a-z_][a-z0-9_]*$")
_PASCAL_RE = __import__("re").compile(r"^[A-Z][a-zA-Z0-9]*$")


def _count_statements(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node) if isinstance(_, ast.stmt))


def _check_magic_numbers(node: ast.AST, source_lines: List[str]) -> List[Dict]:
    findings = []
    safe = {0, 1, -1, 2, 100, 1000}
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            if n.value not in safe and hasattr(n, "lineno"):
                findings.append({"pattern": "magic_number", "line": n.lineno, "description": f"Magic number {n.value!r} – consider a named constant", "priority": "low"})
    return findings


def _analyze(source: str, file_path: str) -> List[Dict]:
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []
    suggestions = []
    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Long parameter list
            n_args = len(node.args.args) + len(node.args.posonlyargs)
            if n_args > 5:
                suggestions.append({"pattern": "long_param_list", "location": node.name, "line": node.lineno, "description": f"{n_args} parameters – consider a data class or **kwargs", "priority": "medium"})
            # God function
            stmts = _count_statements(node)
            if stmts > 30:
                suggestions.append({"pattern": "god_function", "location": node.name, "line": node.lineno, "description": f"Function has {stmts} statements – consider splitting", "priority": "high"})
            # Deeply nested
            depth = 0
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    depth += 1
            if depth > 3:
                suggestions.append({"pattern": "deep_nesting", "location": node.name, "line": node.lineno, "description": f"Nesting depth ~{depth} – consider early returns or helper functions", "priority": "medium"})
        # Unused imports (basic)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name and source.count(name) == 1:
                    suggestions.append({"pattern": "unused_import", "location": name, "line": node.lineno, "description": f"Import '{name}' may be unused", "priority": "low"})

    suggestions.extend(_check_magic_numbers(tree, lines))
    return suggestions


class RefactoringAdvisorSkill(SkillBase):
    name = "refactoring_advisor"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        file_path: str = input_data.get("file_path", "<string>")
        project_root_str: Optional[str] = input_data.get("project_root")

        all_suggestions: List[Dict] = []

        if code:
            all_suggestions = _analyze(code, file_path)
        elif project_root_str:
            root = Path(project_root_str)
            for f in root.rglob("*.py"):
                if ".git" in f.parts or "__pycache__" in f.parts or "node_modules" in f.parts:
                    continue
                try:
                    src = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rel = str(f.relative_to(root))
                for s in _analyze(src, rel):
                    s["file"] = rel
                    all_suggestions.append(s)
        else:
            return {"error": "Provide 'code' or 'project_root'"}

        high = sum(1 for s in all_suggestions if s.get("priority") == "high")
        score = max(0.0, round(1.0 - len(all_suggestions) / max(len(all_suggestions) + 10, 1), 2))

        log_json("INFO", "refactoring_advisor_complete", details={"suggestions": len(all_suggestions), "high_priority": high})
        return {"suggestions": all_suggestions[:100], "smell_count": len(all_suggestions), "score": score, "high_priority_count": high}
