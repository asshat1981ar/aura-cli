"""Skill: profile code and detect performance anti-patterns via AST analysis."""
from __future__ import annotations
import ast
import cProfile
import io
import pstats
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_ANTIPATTERNS = [
    ("nested_loop", "Nested for-loops detected – potential O(n²) complexity"),
    ("compile_in_loop", "re.compile() inside a loop – move outside"),
    ("list_in_loop", "List comprehension building result appended in loop – use extend or itertools"),
]


def _detect_antipatterns(source: str) -> List[Dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.loop_depth = 0

        def visit_For(self, node):
            self.loop_depth += 1
            if self.loop_depth > 1:
                findings.append({"type": "nested_loop", "line": node.lineno, "description": _ANTIPATTERNS[0][1]})
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_While(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Call(self, node):
            if self.loop_depth > 0:
                if isinstance(node.func, ast.Attribute) and node.func.attr == "compile":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "re":
                        findings.append({"type": "compile_in_loop", "line": node.lineno, "description": _ANTIPATTERNS[1][1]})
            self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def _profile_snippet(code: str, max_rows: int = 10) -> List[Dict]:
    pr = cProfile.Profile()
    try:
        pr.enable()
        exec(compile(code, "<profiled>", "exec"), {})  # noqa: S102
    except Exception:
        pass
    finally:
        pr.disable()
    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(max_rows)
    output = stream.getvalue()
    hotspots = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0].replace(".", "").isdigit():
            try:
                hotspots.append({"calls": int(float(parts[0])), "cumtime_ms": round(float(parts[3]) * 1000, 2), "func": " ".join(parts[5:])})
            except (ValueError, IndexError):
                continue
    return hotspots[:max_rows]


class PerformanceProfilerSkill(SkillBase):
    name = "performance_profiler"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code: Optional[str] = input_data.get("code")
        if not code:
            return {"error": "Provide 'code' in input_data"}

        antipatterns = _detect_antipatterns(code)
        hotspots = _profile_snippet(code)

        complexity_hints = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    for child in ast.walk(node):
                        if isinstance(child, ast.For) and child is not node:
                            complexity_hints.append("O(n²) – nested loops detected")
                            break
        except SyntaxError:
            pass

        estimated = "O(n²)" if any("nested" in a["type"] for a in antipatterns) else "O(n) or better"

        log_json("INFO", "performance_profiler_complete", details={"antipatterns": len(antipatterns), "hotspots": len(hotspots)})
        return {"hotspots": hotspots, "antipatterns": antipatterns, "estimated_complexity": estimated, "complexity_hints": complexity_hints}
