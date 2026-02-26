"""Skill: check type annotations using mypy or heuristic annotation coverage."""
from __future__ import annotations
import ast
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json


def _annotation_coverage(source: str) -> float:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0.0
    total = 0
    annotated = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total += 1
            if node.returns or any(a.annotation for a in node.args.args):
                annotated += 1
    return round(annotated / max(total, 1) * 100, 1)


def _run_mypy(target: str, cwd: Path) -> Optional[List[Dict]]:
    try:
        result = subprocess.run(
            ["python3", "-m", "mypy", "--ignore-missing-imports", "--no-error-summary", target],
            cwd=str(cwd), capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        errors = []
        for line in output.splitlines():
            m = re.match(r"^(.+?):(\d+): (error|warning|note): (.+)$", line)
            if m:
                errors.append({"file": m.group(1), "line": int(m.group(2)), "level": m.group(3), "error": m.group(4)})
        return errors
    except FileNotFoundError:
        return None
    except Exception:
        return None


class TypeCheckerSkill(SkillBase):
    name = "type_checker"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root_str: Optional[str] = input_data.get("project_root")
        file_path: Optional[str] = input_data.get("file_path")
        cwd = Path(project_root_str or ".")
        target = file_path or "."
        errors = _run_mypy(target, cwd)
        mypy_available = errors is not None

        if not mypy_available:
            # Heuristic fallback
            root = Path(project_root_str or ".")
            coverages = []
            for f in root.rglob("*.py"):
                if ".git" in f.parts or "__pycache__" in f.parts or "node_modules" in f.parts:
                    continue
                try:
                    src = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                coverages.append(_annotation_coverage(src))
            avg = round(sum(coverages) / max(len(coverages), 1), 1)
            return {"type_errors": [], "annotation_coverage_pct": avg, "error_count": 0, "mypy_available": False, "note": "mypy not available; showing annotation coverage"}

        real_errors = [e for e in errors if e["level"] == "error"]
        all_sources = []
        if project_root_str:
            root = Path(project_root_str)
            for f in root.rglob("*.py"):
                if ".git" in f.parts or "__pycache__" in f.parts or "node_modules" in f.parts:
                    continue
                try:
                    src = f.read_text(encoding="utf-8", errors="replace")
                    all_sources.append(_annotation_coverage(src))
                except OSError:
                    pass
        avg_cov = round(sum(all_sources) / max(len(all_sources), 1), 1)

        log_json("INFO", "type_checker_complete", details={"errors": len(real_errors), "mypy": mypy_available})
        return {"type_errors": real_errors[:100], "annotation_coverage_pct": avg_cov, "error_count": len(real_errors), "mypy_available": True}
