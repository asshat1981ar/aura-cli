"""Skill: compute cyclomatic complexity and code complexity metrics per function."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_COMPLEXITY_NODES = (ast.If, ast.For, ast.While, ast.With, ast.ExceptHandler, ast.Assert, ast.comprehension)
_BOOL_OP_NODES = (ast.BoolOp,)


def _complexity(node: ast.AST) -> int:
    score = 1
    for child in ast.walk(node):
        if isinstance(child, _COMPLEXITY_NODES):
            score += 1
        elif isinstance(child, _BOOL_OP_NODES):
            score += len(child.values) - 1  # type: ignore[attr-defined]
    return score


def _nesting_depth(node: ast.AST, current: int = 0) -> int:
    _NEST = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)
    max_depth = current
    for child in ast.iter_child_nodes(node):
        d = _nesting_depth(child, current + (1 if isinstance(child, _NEST) else 0))
        max_depth = max(max_depth, d)
    return max_depth


def _analyze_source(source: str, file_path: str) -> List[Dict]:
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return [{"name": "<module>", "error": str(exc)}]
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = _complexity(node)
            depth = _nesting_depth(node)
            lines = (node.end_lineno or node.lineno) - node.lineno + 1  # type: ignore[attr-defined]
            risk = "low"
            if cc > 15 or lines > 80 or depth > 4:
                risk = "high"
            elif cc > 10 or lines > 50 or depth > 3:
                risk = "medium"
            results.append({"name": node.name, "line": node.lineno, "complexity": cc, "lines": lines, "nesting_depth": depth, "risk_level": risk})
    return results


class ComplexityScorerSkill(SkillBase):
    name = "complexity_scorer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        file_path: str = input_data.get("file_path", "<string>")
        project_root_str: Optional[str] = input_data.get("project_root")

        all_results: Dict[str, List[Dict]] = {}

        if code:
            all_results[file_path] = _analyze_source(code, file_path)
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
                all_results[rel] = _analyze_source(src, rel)
        else:
            return {"error": "Provide 'code' or 'project_root' in input_data"}

        all_funcs = [fn for funcs in all_results.values() for fn in funcs if "error" not in fn]
        high_risk = [fn for fn in all_funcs if fn.get("risk_level") == "high"]
        avg_cc = round(sum(fn["complexity"] for fn in all_funcs) / max(len(all_funcs), 1), 2)

        log_json("INFO", "complexity_scorer_complete", details={"functions": len(all_funcs), "high_risk": len(high_risk)})
        return {"functions": all_funcs if code else [], "file_results": {k: v for k, v in all_results.items() if not code}, "file_avg_complexity": avg_cc, "high_risk_count": len(high_risk), "high_risk_functions": high_risk[:20]}
