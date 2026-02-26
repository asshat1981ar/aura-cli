"""Skill: run flake8 and check naming conventions via AST.

Supports single files, inline code, and full project scans.
Configurable via: max_line_length, ignore_codes, paths (multi-project).
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_PASCAL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_SCREAMING_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", ".venv", "venv", "dist", "build"}

# Common flake8 fix hints
_FIX_HINTS: Dict[str, str] = {
    "E302": "Add two blank lines before top-level definition.",
    "E303": "Remove extra blank lines.",
    "E401": "Put each import on a separate line.",
    "E501": "Break line to stay under max_line_length.",
    "E711": "Use `is None` / `is not None` instead of `== None`.",
    "E712": "Use `is True` / `is False` instead of `== True`.",
    "F401": "Remove unused import or add `# noqa: F401` if intentional.",
    "F811": "Remove duplicate definition.",
    "F841": "Remove unused local variable.",
    "W291": "Strip trailing whitespace.",
    "W293": "Strip whitespace on blank lines.",
    "W605": "Fix invalid escape sequence — use raw string r'...' or double backslash.",
}


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _snake_to_pascal(name: str) -> str:
    return "".join(w.capitalize() for w in name.split("_"))


def _check_naming(source: str, file_path: str) -> List[Dict]:
    """AST-based naming convention checker (N8xx codes)."""
    violations: List[Dict] = []
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if not _SNAKE_RE.match(name) and not name.startswith("__"):
                violations.append({
                    "code": "N802",
                    "file": file_path,
                    "line": node.lineno,
                    "col": 0,
                    "message": f"Function '{name}' should be snake_case",
                    "fix_hint": f"Rename to '{_camel_to_snake(name)}'",
                })

        elif isinstance(node, ast.ClassDef):
            name = node.name
            if not _PASCAL_RE.match(name):
                violations.append({
                    "code": "N801",
                    "file": file_path,
                    "line": node.lineno,
                    "col": 0,
                    "message": f"Class '{name}' should be PascalCase",
                    "fix_hint": f"Rename to '{_snake_to_pascal(name)}'",
                })

        elif isinstance(node, ast.Name) and isinstance(getattr(node, "ctx", None), ast.Store):
            name = node.id
            if (
                not name.startswith("_")
                and not _SNAKE_RE.match(name)
                and not _SCREAMING_RE.match(name)
                and not _PASCAL_RE.match(name)
            ):
                violations.append({
                    "code": "N806",
                    "file": file_path,
                    "line": getattr(node, "lineno", 0),
                    "col": 0,
                    "message": f"Variable '{name}' should be snake_case",
                    "fix_hint": f"Rename to '{name.lower()}'",
                })

    return violations


def _flake8_available() -> bool:
    """Return True if flake8 is importable."""
    try:
        result = subprocess.run(
            ["python3", "-m", "flake8", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_flake8(
    target: str,
    cwd: Path,
    max_line_length: int = 120,
    ignore_codes: Optional[List[str]] = None,
) -> Optional[List[Dict]]:
    """Run flake8 on target path. Returns violations list or None if flake8 unavailable."""
    cmd = [
        "python3", "-m", "flake8",
        f"--max-line-length={max_line_length}",
        "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
    ]
    if ignore_codes:
        cmd.append(f"--extend-ignore={','.join(ignore_codes)}")
    cmd.append(target)

    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=60,
        )
        violations: List[Dict] = []
        for line in (result.stdout + result.stderr).splitlines():
            m = re.match(r"^(.+?):(\d+):(\d+): ([A-Z]\d+) (.+)$", line)
            if m:
                code = m.group(4)
                violations.append({
                    "code": code,
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "col": int(m.group(3)),
                    "message": m.group(5),
                    "fix_hint": _FIX_HINTS.get(code, ""),
                })
        return violations
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _scan_project(
    root: Path,
    max_line_length: int,
    ignore_codes: Optional[List[str]],
) -> Dict[str, Any]:
    """Scan a single project root. Returns aggregated violation data."""
    flake8_violations = _run_flake8(".", root, max_line_length, ignore_codes) or []
    naming_violations: List[Dict] = []

    for f in sorted(root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in f.parts):
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        naming_violations.extend(_check_naming(src, str(f.relative_to(root))))

    all_v = flake8_violations + naming_violations
    errors = sum(1 for v in flake8_violations if v["code"].startswith("E"))
    warnings = sum(1 for v in flake8_violations if v["code"].startswith("W"))
    critical = sum(1 for v in all_v if v["code"] in ("F401", "F811", "W605", "E711"))

    # Top offending files
    file_counts: Dict[str, int] = {}
    for v in all_v:
        file_counts[v.get("file", "?")] = file_counts.get(v.get("file", "?"), 0) + 1
    top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "project_root": str(root),
        "violations": all_v[:200],
        "violation_count": len(all_v),
        "error_count": errors,
        "warning_count": warnings,
        "naming_violation_count": len(naming_violations),
        "critical_count": critical,
        "top_offending_files": [{"file": f, "count": c} for f, c in top_files],
    }


class LinterEnforcerSkill(SkillBase):
    """
    Run flake8 and AST-based naming-convention checks on Python code.

    Input options (use one):
      code + file_path  — lint inline source string
      project_root      — lint entire project
      paths             — list of project roots (multi-project scan)

    Config:
      max_line_length   — default 120
      ignore_codes      — list of flake8 codes to suppress (e.g. ["E501","W503"])
    """

    name = "linter_enforcer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        file_path: Optional[str] = input_data.get("file_path")
        project_root_str: Optional[str] = input_data.get("project_root")
        paths: Optional[List[str]] = input_data.get("paths")  # multi-project
        max_line_length: int = int(input_data.get("max_line_length", 120))
        ignore_codes: Optional[List[str]] = input_data.get("ignore_codes")

        available = _flake8_available()

        # --- Inline code ---
        if code and file_path:
            import tempfile, os
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                flake8_v = _run_flake8(tmp_path, Path("."), max_line_length, ignore_codes) or []
                # Remap temp file path back to file_path label
                for v in flake8_v:
                    v["file"] = file_path
            finally:
                os.unlink(tmp_path)
            naming_v = _check_naming(code, file_path)
            all_v = flake8_v + naming_v
            errors = sum(1 for v in flake8_v if v["code"].startswith("E"))
            warnings = sum(1 for v in flake8_v if v["code"].startswith("W"))
            log_json("INFO", "linter_enforcer_complete", details={"file": file_path, "violations": len(all_v)})
            return {
                "file": file_path,
                "violations": all_v,
                "violation_count": len(all_v),
                "error_count": errors,
                "warning_count": warnings,
                "naming_violation_count": len(naming_v),
                "flake8_available": available,
            }

        # --- Multi-project scan ---
        if paths:
            results = []
            for p in paths:
                root = Path(p)
                if not root.exists():
                    results.append({"project_root": p, "error": "Path not found"})
                    continue
                results.append(_scan_project(root, max_line_length, ignore_codes))
            total = sum(r.get("violation_count", 0) for r in results if "error" not in r)
            log_json("INFO", "linter_enforcer_multi_complete", details={"projects": len(paths), "total_violations": total})
            return {
                "projects_scanned": len(paths),
                "total_violations": total,
                "flake8_available": available,
                "results": results,
            }

        # --- Single project root ---
        root = Path(project_root_str or ".")
        result = _scan_project(root, max_line_length, ignore_codes)
        result["flake8_available"] = available
        log_json("INFO", "linter_enforcer_complete", details={
            "root": str(root),
            "violations": result["violation_count"],
            "errors": result["error_count"],
        })
        return result
