"""Skill: quantify technical debt across a codebase."""
from __future__ import annotations
import ast
import re
from pathlib import Path
from typing import Any, Dict, List

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_DEBT_TAGS = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|NOQA|TEMP|BUG)\b", re.IGNORECASE)
_EFFORT = {"TODO": 0.5, "FIXME": 1.0, "HACK": 2.0, "XXX": 1.5, "BUG": 2.0, "NOQA": 0.25, "TEMP": 1.0}


def _count_functions(source: str) -> int:
    try:
        tree = ast.parse(source)
        return sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except SyntaxError:
        return 0


class TechDebtQuantifierSkill(SkillBase):
    name = "tech_debt_quantifier"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root = Path(input_data.get("project_root", "."))
        debt_items: List[Dict] = []
        total_lines = 0
        total_files = 0
        total_funcs = 0
        total_todos = 0

        py_files = [f for f in project_root.rglob("*.py") if ".git" not in f.parts and "node_modules" not in f.parts and "__pycache__" not in f.parts]

        for f in py_files:
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = src.splitlines()
            rel = str(f.relative_to(project_root))
            total_files += 1
            total_lines += len(lines)
            total_funcs += _count_functions(src)

            # Comment-based debt
            for i, line in enumerate(lines, 1):
                m = _DEBT_TAGS.search(line)
                if m:
                    tag = m.group(1).upper()
                    total_todos += 1
                    debt_items.append({"type": tag, "file": rel, "line": i, "severity": "medium" if tag in ("TODO", "NOQA") else "high", "effort_days_estimate": _EFFORT.get(tag, 1.0), "snippet": line.strip()[:100]})

            # Long files
            if len(lines) > 300:
                debt_items.append({"type": "LONG_FILE", "file": rel, "line": 1, "severity": "medium", "effort_days_estimate": 1.0, "snippet": f"{len(lines)} lines"})

            # Long functions
            try:
                tree = ast.parse(src)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn_lines = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
                        if fn_lines > 60:
                            debt_items.append({"type": "LONG_FUNCTION", "file": rel, "line": node.lineno, "severity": "medium", "effort_days_estimate": 0.5, "snippet": f"{node.name}: {fn_lines} lines"})
            except SyntaxError:
                pass

            # Missing __init__.py check
            if f.name != "__init__.py":
                parent = f.parent
                if not (parent / "__init__.py").exists():
                    debt_items.append({"type": "MISSING_INIT", "file": rel, "line": 0, "severity": "low", "effort_days_estimate": 0.1, "snippet": f"Directory {parent.name}/ missing __init__.py"})

        # Dedup MISSING_INIT by directory
        seen_dirs = set()
        unique_items = []
        for item in debt_items:
            if item["type"] == "MISSING_INIT":
                d = str(Path(item["file"]).parent)
                if d in seen_dirs:
                    continue
                seen_dirs.add(d)
            unique_items.append(item)

        total_effort = sum(i["effort_days_estimate"] for i in unique_items)
        # Score: 0 (worst) to 100 (best). Penalize by debt density
        density = len(unique_items) / max(total_files, 1)
        debt_score = max(0.0, round(100 - min(density * 20, 100), 1))

        summary = (f"{len(unique_items)} debt items found across {total_files} files "
                   f"({total_todos} TODO/FIXME/HACK). "
                   f"Estimated remediation: {total_effort:.1f} dev-days. Debt score: {debt_score}/100.")

        log_json("INFO", "tech_debt_quantifier_complete", details={"items": len(unique_items), "score": debt_score})
        return {"debt_score": debt_score, "debt_items": unique_items[:200], "total_todos": total_todos, "total_files": total_files, "total_lines": total_lines, "estimated_effort_days": round(total_effort, 1), "summary": summary}
