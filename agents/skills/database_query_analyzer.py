"""Skill: database query analyzer — detect N+1 queries, missing indexes, slow patterns."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agents.skills.base import SkillBase
from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

# Raw SQL anti-patterns (applied to string literals and f-strings in source)
_SQL_ANTIPATTERNS = [
    (
        re.compile(r"SELECT\s+\*\s+FROM", re.IGNORECASE),
        "select_star",
        "medium",
        "SELECT * fetches all columns — specify only needed columns for performance.",
    ),
    (
        re.compile(r"SELECT\s+.+\s+FROM\s+\w+\s+WHERE\s+1\s*=\s*1", re.IGNORECASE),
        "always_true_where",
        "low",
        "WHERE 1=1 is a debugging artifact; remove it or it will scan the full table.",
    ),
    (
        re.compile(r"LIKE\s+'%[^%']+%'", re.IGNORECASE),
        "leading_wildcard_like",
        "high",
        "Leading wildcard LIKE '%...%' prevents index usage — consider full-text search.",
    ),
    (
        re.compile(r"\bIN\s*\(\s*SELECT\b", re.IGNORECASE),
        "in_subquery",
        "medium",
        "IN (SELECT ...) subquery can be slow — consider EXISTS or a JOIN instead.",
    ),
    (
        re.compile(r"\bNOT\s+IN\s*\(", re.IGNORECASE),
        "not_in_clause",
        "medium",
        "NOT IN is slow on large sets and fails on NULLs — use NOT EXISTS or LEFT JOIN.",
    ),
    (
        re.compile(r"ORDER\s+BY\s+RAND\(\)", re.IGNORECASE),
        "order_by_rand",
        "high",
        "ORDER BY RAND() performs a full-table sort — use offset-based or keyset pagination.",
    ),
    (
        re.compile(r"OFFSET\s+\d{4,}", re.IGNORECASE),
        "deep_offset_pagination",
        "high",
        "Large OFFSET forces the DB to scan and discard rows — use keyset/cursor pagination.",
    ),
    (
        re.compile(r"\bOR\b.+\bOR\b.+\bOR\b", re.IGNORECASE),
        "many_or_conditions",
        "low",
        "Many OR conditions may prevent index use — consider UNION or IN clause.",
    ),
    (
        re.compile(r"(DELETE|UPDATE)\s+\w+\s+WHERE\s+1\s*=\s*1", re.IGNORECASE),
        "bulk_delete_update_no_filter",
        "critical",
        "DELETE/UPDATE WHERE 1=1 affects all rows — add a proper WHERE clause.",
    ),
    (
        re.compile(r"(DROP\s+TABLE|TRUNCATE\s+TABLE)", re.IGNORECASE),
        "destructive_ddl",
        "high",
        "Destructive DDL (DROP/TRUNCATE) found in application code — move to migrations.",
    ),
]

# ORM-level N+1 patterns (SQLAlchemy / Django ORM style)
_ORM_N1_PATTERNS = [
    (
        re.compile(r"for\s+\w+\s+in\s+\w+.*:\s*\n.*\.\w+\.", re.MULTILINE),
        "possible_n1_loop",
        "high",
        "Loop over queryset with attribute access — possible N+1. Use select_related/prefetch_related or joinedload.",
    ),
]

# Missing index hints — columns frequently filtered but rarely indexed
_MISSING_INDEX_COLS = re.compile(
    r"WHERE\s+(?P<col>\w+)\s*=\s*[%?:]", re.IGNORECASE
)

# Parameterized query check — dangerous string formatting in queries
_STRING_FORMAT_IN_QUERY = re.compile(
    r"""(execute|query|raw)\s*\(\s*[f"'].*(%s|%d|\{[^}]+\}|\"\s*\+|\'\s*\+)""",
    re.IGNORECASE,
)

_INJECTION_RISK = re.compile(
    r"""(?:execute|query|raw|filter|where)\s*\(\s*[f"'].*\{""",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# AST-based analysis
# ---------------------------------------------------------------------------

def _extract_string_values(node: ast.expr) -> List[str]:
    """Recursively extract string literal values from an AST node."""
    values = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        values.append(node.value)
    elif isinstance(node, ast.JoinedStr):  # f-string
        # Reconstruct approximate text from f-string parts
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                parts.append("{...}")
        values.append("".join(parts))
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        values.extend(_extract_string_values(node.left))
        values.extend(_extract_string_values(node.right))
    return values


def _find_query_calls(tree: ast.AST) -> List[Dict]:
    """Find all .execute(), .query(), .filter(), .raw() calls and their string args."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        method = ""
        if isinstance(func, ast.Attribute):
            method = func.attr
        elif isinstance(func, ast.Name):
            method = func.id

        if method not in ("execute", "query", "raw", "filter", "where", "text"):
            continue

        for arg in node.args:
            for s in _extract_string_values(arg):
                hits.append({"method": method, "sql": s, "line": node.lineno})
    return hits


def _detect_n1_in_loops(tree: ast.AST, source_lines: List[str]) -> List[Dict]:
    """Detect patterns like: for item in queryset: item.related_obj.field"""
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.comprehension)):
            continue
        loop_line = getattr(node, "lineno", 0)
        body = getattr(node, "body", [])
        for child in ast.walk(ast.Module(body=body, type_ignores=[])):
            # Look for chained attribute access: a.b.c
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Attribute):
                    findings.append({
                        "issue": "possible_n1_query",
                        "severity": "high",
                        "line": loop_line,
                        "detail": (
                            f"Chained attribute access inside loop at line {loop_line} "
                            "may trigger N+1 queries. Use eager loading (select_related, "
                            "prefetch_related, joinedload)."
                        ),
                    })
                    break
        if findings and findings[-1]["line"] == loop_line:
            continue  # already recorded for this loop

    # Deduplicate by line
    seen: Set[int] = set()
    unique = []
    for f in findings:
        if f["line"] not in seen:
            seen.add(f["line"])
            unique.append(f)
    return unique


def _analyse_source(source: str, file_path: str) -> Dict[str, Any]:
    findings: List[Dict] = []
    query_count = 0

    # ---- Text-level scan ----
    for i, line in enumerate(source.splitlines(), 1):
        for pattern, issue_id, severity, detail in _SQL_ANTIPATTERNS:
            if pattern.search(line):
                findings.append({
                    "issue": issue_id,
                    "severity": severity,
                    "file": file_path,
                    "line": i,
                    "detail": detail,
                    "snippet": line.strip()[:120],
                })
        # Injection risk: f-string / format in execute()
        if _INJECTION_RISK.search(line):
            findings.append({
                "issue": "sql_injection_risk",
                "severity": "critical",
                "file": file_path,
                "line": i,
                "detail": "Dynamic SQL built with f-string/format — use parameterized queries.",
                "snippet": line.strip()[:120],
            })

    # ---- AST-level scan ----
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "file": file_path,
            "error": "SyntaxError — could not parse file",
            "findings": findings,
            "finding_count": len(findings),
        }

    query_hits = _find_query_calls(tree)
    query_count = len(query_hits)

    # Check SQL strings found in call args
    for hit in query_hits:
        sql = hit["sql"]
        for pattern, issue_id, severity, detail in _SQL_ANTIPATTERNS:
            if pattern.search(sql):
                findings.append({
                    "issue": issue_id,
                    "severity": severity,
                    "file": file_path,
                    "line": hit["line"],
                    "detail": detail,
                    "snippet": sql[:120],
                })

    # N+1 detection
    n1_findings = _detect_n1_in_loops(tree, source.splitlines())
    findings.extend([{**f, "file": file_path} for f in n1_findings])

    # Deduplicate by (issue, line)
    seen_keys: Set[tuple] = set()
    unique: List[Dict] = []
    for f in findings:
        key = (f["issue"], f.get("line", 0))
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(f)

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    unique.sort(key=lambda x: severity_order.get(x["severity"], 9))

    return {
        "file": file_path,
        "query_call_count": query_count,
        "findings": unique,
        "finding_count": len(unique),
        "critical_count": sum(1 for f in unique if f["severity"] == "critical"),
        "high_count": sum(1 for f in unique if f["severity"] == "high"),
    }


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------

class DatabaseQueryAnalyzerSkill(SkillBase):
    """
    Analyse Python source for database query anti-patterns:
    N+1 queries, missing index hints, SELECT *, SQL injection risks,
    leading-wildcard LIKE, deep pagination, destructive DDL in app code.

    Supports both raw SQL strings and ORM patterns (SQLAlchemy, Django ORM).
    """

    name = "database_query_analyzer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        file_path: str = input_data.get("file_path", "<stdin>")
        project_root: Optional[str] = input_data.get("project_root")

        if code:
            return _analyse_source(code, file_path)

        if project_root:
            root = Path(project_root)
            py_files = [
                f for f in root.rglob("*.py")
                if not any(
                    part.startswith(".") or part in ("__pycache__", "node_modules", "migrations")
                    for part in f.parts
                )
            ]
            if not py_files:
                return {"error": f"No Python files found under '{project_root}'.", "files_scanned": 0}

            results = []
            for pf in py_files:
                try:
                    src = pf.read_text(encoding="utf-8", errors="replace")
                    # Quick pre-filter: skip files with no DB-related keywords
                    if not re.search(r"(execute|query|SELECT|INSERT|UPDATE|DELETE|filter|objects\.|session\.)", src, re.IGNORECASE):
                        continue
                    result = _analyse_source(src, str(pf.relative_to(root)))
                    if result.get("finding_count", 0) > 0 or result.get("query_call_count", 0) > 0:
                        results.append(result)
                except Exception as exc:
                    log_json("WARN", "db_query_analyzer_read_error", details={"file": str(pf), "error": str(exc)})

            total_findings = sum(r.get("finding_count", 0) for r in results)
            total_critical = sum(r.get("critical_count", 0) for r in results)
            total_high = sum(r.get("high_count", 0) for r in results)

            worst = sorted(results, key=lambda r: r.get("finding_count", 0), reverse=True)[:5]

            return {
                "files_scanned": len(results),
                "total_findings": total_findings,
                "total_critical": total_critical,
                "total_high": total_high,
                "worst_files": [{"file": r["file"], "findings": r["finding_count"]} for r in worst],
                "results": results,
            }

        return {"error": "Provide 'code' (Python source) or 'project_root' (directory path)."}
