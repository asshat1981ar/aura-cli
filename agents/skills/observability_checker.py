"""Skill: observability checker — logging coverage, structured logging, error tracing."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agents.skills.base import SkillBase
from core.logging_utils import log_json

# Names that indicate a logging call
_LOG_CALL_NAMES: Set[str] = {
    "log_json",                                  # AURA's own structured logger
    "logging.debug", "logging.info", "logging.warning", "logging.error", "logging.critical",
    "logger.debug", "logger.info", "logger.warning", "logger.error", "logger.critical",
    "log.debug", "log.info", "log.warning", "log.error", "log.critical",
}

_BARE_PRINT_RE = re.compile(r"^\s*print\s*\(")


def _is_log_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id in ("log_json", "print"):
        return func.id != "print"
    if isinstance(func, ast.Attribute):
        qualified = f"{getattr(func.value, 'id', '')}.{func.attr}"
        return qualified in _LOG_CALL_NAMES
    return False


def _has_log_call(body: List[ast.stmt]) -> bool:
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call) and _is_log_call(node):
            return True
    return False


def _count_bare_prints(source: str) -> List[Dict]:
    issues = []
    for i, line in enumerate(source.splitlines(), 1):
        if _BARE_PRINT_RE.match(line):
            issues.append({"line": i, "snippet": line.strip()[:100]})
    return issues


def _analyse_file(source: str, file_path: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"file": file_path, "error": f"SyntaxError: {exc}", "functions": [], "issues": []}

    functions = []
    issues = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        fn_name = node.name
        fn_line = node.lineno
        body = node.body
        has_log = _has_log_call(body)
        has_except = any(isinstance(n, ast.ExceptHandler) for n in ast.walk(ast.Module(body=body, type_ignores=[])))

        # Check if except blocks re-raise or log
        except_without_log = False
        for child in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(child, ast.ExceptHandler):
                if not _has_log_call(child.body):
                    # Check it at least re-raises
                    reraises = any(isinstance(n, ast.Raise) for n in ast.walk(ast.Module(body=child.body, type_ignores=[])))
                    if not reraises:
                        except_without_log = True

        fn_info = {
            "name": fn_name,
            "line": fn_line,
            "has_logging": has_log,
            "has_exception_handling": has_except,
            "except_without_log_or_reraise": except_without_log,
        }
        functions.append(fn_info)

        if not has_log and len(body) > 5:
            issues.append({
                "type": "missing_logging",
                "severity": "low",
                "function": fn_name,
                "line": fn_line,
                "detail": f"Function '{fn_name}' has no logging calls (>{len(body)} statements).",
            })
        if except_without_log:
            issues.append({
                "type": "silent_exception",
                "severity": "medium",
                "function": fn_name,
                "line": fn_line,
                "detail": f"Function '{fn_name}' catches exceptions without logging or re-raising.",
            })

    # Bare prints
    bare_prints = _count_bare_prints(source)
    for bp in bare_prints:
        issues.append({
            "type": "bare_print",
            "severity": "low",
            "function": None,
            "line": bp["line"],
            "detail": f"bare print() at line {bp['line']} — use structured logging instead: {bp['snippet']}",
        })

    logged_fns = sum(1 for f in functions if f["has_logging"])
    coverage_pct = round(100 * logged_fns / len(functions), 1) if functions else 100.0

    return {
        "file": file_path,
        "function_count": len(functions),
        "logged_function_count": logged_fns,
        "logging_coverage_pct": coverage_pct,
        "bare_print_count": len(bare_prints),
        "issues": issues,
        "issue_count": len(issues),
    }


class ObservabilityCheckerSkill(SkillBase):
    """Check Python code for logging coverage, silent exceptions, and bare print() calls."""

    name = "observability_checker"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        file_path: str = input_data.get("file_path", "<stdin>")
        project_root: Optional[str] = input_data.get("project_root")

        if code:
            result = _analyse_file(code, file_path)
            return result

        if project_root:
            root = Path(project_root)
            py_files = [
                f for f in root.rglob("*.py")
                if not any(part.startswith(".") or part in ("__pycache__", "node_modules") for part in f.parts)
            ]
            if not py_files:
                return {"error": f"No Python files found under '{project_root}'.", "files_scanned": 0}

            results = []
            for pf in py_files:
                try:
                    src = pf.read_text(encoding="utf-8", errors="replace")
                    results.append(_analyse_file(src, str(pf.relative_to(root))))
                except Exception as exc:
                    log_json("WARN", "observability_checker_read_error", details={"file": str(pf), "error": str(exc)})

            total_fns = sum(r.get("function_count", 0) for r in results)
            total_logged = sum(r.get("logged_function_count", 0) for r in results)
            total_issues = sum(r.get("issue_count", 0) for r in results)
            total_prints = sum(r.get("bare_print_count", 0) for r in results)
            overall_coverage = round(100 * total_logged / total_fns, 1) if total_fns else 100.0

            # Worst files by issue count
            worst = sorted(results, key=lambda r: r.get("issue_count", 0), reverse=True)[:5]

            return {
                "files_scanned": len(results),
                "total_functions": total_fns,
                "total_logged_functions": total_logged,
                "overall_logging_coverage_pct": overall_coverage,
                "total_bare_prints": total_prints,
                "total_issues": total_issues,
                "worst_files": [{"file": r["file"], "issues": r["issue_count"]} for r in worst],
                "results": results,
            }

        return {"error": "Provide 'code' (Python source) or 'project_root' (directory path)."}
